
import torch
from torch import nn
from torch.nn import init, functional as F
import numpy as np
import matplotlib.pyplot as plt
import collections
from functools import partial

from ts_torch.torch_util_mini import np_ify

# The (A)NP takes as input a `NPRegressionDescription` namedtuple with fields:
#   `query`: a tuple containing ((context_x, context_y), target_x)
#   `target_y`: a tensor containing the ground truth for the targets to be
#     predicted
#   `num_total_points`: A vector containing a scalar that describes the total
#     number of datapoints used (context + target)
#   `num_context_points`: A vector containing a scalar that describes the number
#     of datapoints used as context
# The GPCurvesReader returns the newly sampled data in this format at each
# iteration

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"))


def to_device(data: NPRegressionDescription, device='cuda:0'):
    ((context_x, context_y), target_x) = data.query
    query = ((context_x.to(device), context_y.to(device)), target_x.to(device))
    return NPRegressionDescription(
            query=query,
            target_y=data.target_y.to(device),
            num_total_points=data.num_total_points,
            num_context_points=data.num_context_points)


class GPCurvesReader(object):
    """Generates curves using a Gaussian Process (GP).

    Supports vector inputs (x) and vector outputs (y). Kernel is
    mean-squared exponential, using the x-value l2 coordinate distance scaled by
    some factor chosen randomly in a range. Outputs are independent gaussian
    processes.
    """

    def __init__(self,
                 batch_size,
                 max_num_context,
                 x_size=1,
                 y_size=1,
                 l1_scale=0.6,
                 sigma_scale=1.0,
                 random_kernel_parameters=True,
                 testing=False):
        """Creates a regression dataset of functions sampled from a GP.

        Args:
            batch_size: An integer.
            max_num_context: The max number of observations in the context.
            x_size: Integer >= 1 for length of "x values" vector.
            y_size: Integer >= 1 for length of "y values" vector.
            l1_scale: Float; typical scale for kernel distance function.
            sigma_scale: Float; typical scale for variance.
            random_kernel_parameters: If `True`, the kernel parameters (l1 and sigma)
              will be sampled uniformly within [0.1, l1_scale] and [0.1, sigma_scale].
            testing: Boolean that indicates whether we are testing. If so there are
              more targets for visualization.
        """
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._random_kernel_parameters = random_kernel_parameters
        self._testing = testing

    def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        """Applies the Gaussian kernel to generate curve data.

        Args:
            xdata: Tensor of shape [B, num_total_points, x_size] with
                the values of the x-axis data.
            l1: Tensor of shape [B, y_size, x_size], the scale
                parameter of the Gaussian kernel.
            sigma_f: Tensor of shape [B, y_size], the magnitude
                of the std.
            sigma_noise: Float, std of the noise that we add for stability.

        Returns:
            The kernel, a float tensor of shape
            [B, y_size, num_total_points, num_total_points].
        """
        num_total_points = xdata.shape[1]

        # Expand and take the difference
        xdata1 = torch.unsqueeze(xdata, dim=1)  # [B, 1, num_total_points, x_size]
        xdata2 = torch.unsqueeze(xdata, dim=2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        norm = torch.pow(diff[:, None, :, :, :] / l1[:, :, None, None, :], 2)

        norm = torch.sum(norm, -1)  # [B, data_size, num_total_points, num_total_points]

        # [B, y_size, num_total_points, num_total_points]
        kernel = torch.pow(sigma_f, 2)[:, :, None, None] * torch.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise ** 2) * torch.eye(num_total_points)

        return kernel

    def generate_curves(self):
        """Builds the op delivering the data.

        Generated functions are `float32` with x values between -2 and 2.

        Returns:
            A `CNPRegressionDescription` namedtuple.
        """
        num_context = torch.randint(low=3, high=self._max_num_context, size=[])

        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.
        if self._testing:
            num_target = 400
            num_total_points = num_target
            x_values = torch.unsqueeze(torch.arange(-2., 2., 1./100, dtype=torch.float32), dim=0).repeat(self._batch_size, 1)
            x_values = torch.unsqueeze(x_values, dim=-1)
        # During training the number of target points and their x-positions are
        # selected at random
        else:
            num_target = torch.randint(low=0, high=self._max_num_context - num_context, size=[])
            num_total_points = num_context + num_target
            x_values = torch.FloatTensor(self._batch_size, num_total_points, self._x_size).uniform_(-2, 2)

        # Set kernel parameters
        # Either choose a set of random parameters for the mini-batch
        if self._random_kernel_parameters:
            l1 = torch.FloatTensor(self._batch_size, self._y_size,
                                    self._x_size).uniform_(0.1, self._l1_scale)
            sigma_f = torch.FloatTensor(self._batch_size, self._y_size).uniform_(0.1, self._sigma_scale)

        # Or use the same fixed parameters for all mini-batches
        else:
            l1 = torch.ones([self._batch_size, self._y_size, self._x_size]) * self._l1_scale
            sigma_f = torch.ones([self._batch_size, self._y_size]) * self._sigma_scale

        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        kernel = self._gaussian_kernel(x_values, l1, sigma_f)

        # Calculate Cholesky, using double precision for better stability:
        cholesky = torch.cholesky(kernel.double()).float()

        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = torch.bmm(cholesky.view(-1, num_total_points, num_total_points),
                             torch.randn([self._batch_size * self._y_size, num_total_points, 1])).view(self._batch_size, self._y_size, num_total_points, 1)

        # [batch_size, num_total_points, y_size]
        y_values = torch.transpose(torch.squeeze(y_values, 3), 1, 2)

        if self._testing:
            # Select the targets
            target_x = x_values
            target_y = y_values

            # Select the observations
            idx = torch.randperm(num_target)
            context_x = x_values[:, idx[:num_context]]
            context_y = y_values[:, idx[:num_context]]
        else:
            # Select the targets which will consist of the context points as well as
            # some new target points
            target_x = x_values[:, :num_target + num_context, :]
            target_y = y_values[:, :num_target + num_context, :]

            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=target_x.shape[1],
            num_context_points=num_context)


class BatchLinear(nn.Module):
    def __init__(self, d_in, output_sizes):
        super(BatchLinear, self).__init__()
        self.linears = nn.ModuleList()
        for d_out in output_sizes:
            self.linears.append(nn.Linear(d_in, d_out))
            d_in = d_out

    def forward(self, input):
        batch_size, n_size, filter_size = input.shape
        output = input.reshape(-1, filter_size)
        for linear_layer in self.linears[:-1]:
            output = F.relu(linear_layer(output))
        output = self.linears[-1](output)
        output = output.reshape(batch_size, n_size, -1)

        return output


class DeterministicEncoder(nn.Module):
    """The Deterministic Encoder."""

    def __init__(self, d_in, output_sizes, attention):
        """(A)NP deterministic encoder.

        Args:
        output_sizes: An iterable containing the output sizes of the encoding MLP.
        attention: The attention module.
        """
        super(DeterministicEncoder, self).__init__()
        self.batch_linear = BatchLinear(d_in, output_sizes)
        self._attention = attention

    def forward(self, context_x, context_y, target_x):
        """Encodes the inputs into one representation.

        Args:
            context_x: Tensor of shape [B,observations,d_x]. For this 1D regression
                task this corresponds to the x-values.
            context_y: Tensor of shape [B,observations,d_y]. For this 1D regression
                task this corresponds to the y-values.
            target_x: Tensor of shape [B,target_observations,d_x].
                For this 1D regression task this corresponds to the x-values.

        Returns:
            The encoded representation. Tensor of shape [B,target_observations,d]
        """

        # Concatenate x and y along the filter axes
        encoder_input = torch.cat([context_x, context_y], dim=-1)

        # Pass final axis through MLP
        hidden = self.batch_linear(encoder_input)

        # Apply attention
        hidden = self._attention(context_x, target_x, hidden)

        return hidden


class LatentEncoder(nn.Module):
    """The Latent Encoder."""

    def __init__(self, d_in, output_sizes, num_latents):
        """(A)NP latent encoder.

        Args:
            output_sizes: An iterable containing the output sizes of the encoding MLP.
            num_latents: The latent dimensionality.
        """
        super(LatentEncoder, self).__init__()
        self.batch_linear = BatchLinear(d_in, output_sizes)
        # First apply intermediate relu layer
        interim_dim = (output_sizes[-1] + num_latents) // 2
        self.interim_layer = nn.Linear(output_sizes[-1], interim_dim)
        # Then apply further linear layers to output latent mu and log sigma
        self.mu_layer = nn.Linear(interim_dim, num_latents)
        self.log_sigma_layer = nn.Linear(interim_dim, num_latents)

    def forward(self, x, y):
        """Encodes the inputs into one representation.

        Args:
            x: Tensor of shape [B,observations,d_x]. For this 1D regression
                task this corresponds to the x-values.
            y: Tensor of shape [B,observations,d_y]. For this 1D regression
                task this corresponds to the y-values.

        Returns:
            A normal distribution over tensors of shape [B, num_latents]
        """

        # Concatenate x and y along the filter axes
        encoder_input = torch.cat([x, y], dim=-1)

        # Pass final axis through MLP
        hidden = self.batch_linear(encoder_input)

        # Aggregator: take the mean over all points
        hidden = hidden.mean(dim=1)
        hidden = F.relu(self.interim_layer(hidden))
        mu = self.mu_layer(hidden)
        log_sigma = self.log_sigma_layer(hidden)

        # Compute sigma
        sigma = 0.1 + 0.9 * F.sigmoid(log_sigma)

        return torch.distributions.Normal(loc=mu, scale=sigma)


class Decoder(nn.Module):
    """The Decoder."""

    def __init__(self, d_in, output_sizes):
        """(A)NP decoder.

        Args:
            output_sizes: An iterable containing the output sizes of the decoder MLP
                as defined in `basic.Linear`.
        """

        super(Decoder, self).__init__()
        self.batch_linear = BatchLinear(d_in, output_sizes)

    def forward(self, representation, target_x):
        """Decodes the individual targets.

        Args:
            representation: The representation of the context for target predictions.
                Tensor of shape [B,target_observations,?].
            target_x: The x locations for the target query.
                Tensor of shape [B,target_observations,d_x].

        Returns:
            dist: A multivariate Gaussian over the target points. A distribution over
                tensors of shape [B,target_observations,d_y].
            mu: The mean of the multivariate Gaussian.
                Tensor of shape [B,target_observations,d_x].
            sigma: The standard deviation of the multivariate Gaussian.
                Tensor of shape [B,target_observations,d_x].
        """
        # concatenate target_x and representation
        hidden = torch.cat([representation, target_x], dim=-1)

        # Pass final axis through MLP
        hidden = self.batch_linear(hidden)

        # Get the mean an the variance
        mu, log_sigma = torch.chunk(hidden, 2, dim=-1)

        # Bound the variance
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        # Get the distribution
        # dist = torch.distributions.MultivariateNormal(
        #     loc=mu.view(-1, 1), covariance_matrix=sigma.view(-1, 1))
        dist = torch.distributions.Normal(loc=mu, scale=sigma)

        return dist, mu, sigma


class LatentModel(nn.Module):
    """The (A)NP model."""

    def __init__(self, d_x, d_y, latent_encoder_output_sizes, num_latents,
                 decoder_output_sizes, use_deterministic_path=True,
                 deterministic_encoder_output_sizes=None, attention=None):
        """Initialises the model.

        Args:
            latent_encoder_output_sizes: An iterable containing the sizes of hidden
                layers of the latent encoder.
            num_latents: The latent dimensionality.
            decoder_output_sizes: An iterable containing the sizes of hidden layers of
                the decoder. The last element should correspond to d_y * 2
                (it encodes both mean and variance concatenated)
            use_deterministic_path: a boolean that indicates whether the deterministic
                encoder is used or not.
            deterministic_encoder_output_sizes: An iterable containing the sizes of
                hidden layers of the deterministic encoder. The last one is the size
                of the deterministic representation r.
            attention: The attention module used in the deterministic encoder.
                Only relevant when use_deterministic_path=True.
        """
        super(LatentModel, self).__init__()
        self._latent_encoder = LatentEncoder(d_x+d_y, latent_encoder_output_sizes,
                                             num_latents)
        self._use_deterministic_path = use_deterministic_path
        if use_deterministic_path:
            self._deterministic_encoder = DeterministicEncoder(
                d_x+d_y, deterministic_encoder_output_sizes, attention)
            d_decoder = latent_encoder_output_sizes[-1]+deterministic_encoder_output_sizes[-1] + d_x
        else:
            d_decoder = latent_encoder_output_sizes[-1] + d_x

        self._decoder = Decoder(d_decoder, decoder_output_sizes)

    def forward(self, query, num_targets, target_y=None):
        # query, num_targets, target_y = data_train.query, data_train.num_total_points, data_train.target_y
        """Returns the predicted mean and variance at the target points.

        Args:
            query: Array containing ((context_x, context_y), target_x) where:
                context_x: Tensor of shape [B,num_contexts,d_x].
                    Contains the x values of the context points.
                context_y: Tensor of shape [B,num_contexts,d_y].
                    Contains the y values of the context points.
                target_x: Tensor of shape [B,num_targets,d_x].
                    Contains the x values of the target points.
            num_targets: Number of target points.
            target_y: The ground truth y values of the target y.
                Tensor of shape [B,num_targets,d_y].

        Returns:
            log_p: The log_probability of the target_y given the predicted
                distribution. Tensor of shape [B,num_targets].
            mu: The mean of the predicted distribution.
                Tensor of shape [B,num_targets,d_y].
            sigma: The variance of the predicted distribution.
                Tensor of shape [B,num_targets,d_y].
        """

        (context_x, context_y), target_x = query

        # Pass query through the encoder and the decoder
        prior = self._latent_encoder(context_x, context_y)

        # For training, when target_y is available, use targets for latent encoder.
        # Note that targets contain contexts by design.
        if target_y is None:
            latent_rep = prior.sample()
        # For testing, when target_y unavailable, use contexts for latent encoder.
        else:
            posterior = self._latent_encoder(target_x, target_y)
            latent_rep = posterior.sample()
        latent_rep = torch.unsqueeze(latent_rep, dim=1).repeat([1, num_targets, 1])
        if self._use_deterministic_path:
            deterministic_rep = self._deterministic_encoder(context_x, context_y, target_x)
            representation = torch.cat([deterministic_rep, latent_rep], dim=-1)
        else:
            representation = latent_rep

        dist, mu, sigma = self._decoder(representation, target_x)

        # If we want to calculate the log_prob for training we will make use of the
        # target_y. At test time the target_y is not available so we return None.
        if target_y is not None:
            log_p = dist.log_prob(target_y).squeeze()
            posterior = self._latent_encoder(target_x, target_y)
            kl = torch.distributions.kl_divergence(posterior, prior).sum(dim=-1, keepdims=True)
            kl = kl.repeat([1, num_targets])
            loss = - (log_p - kl / torch.tensor(num_targets).float()).mean()
        else:
            log_p = None
            kl = None
            loss = None

        return mu, sigma, log_p, kl, loss


def uniform_attention(q, v):
    """Uniform attention. Equivalent to np.

    Args:
        q: queries. tensor of shape [B,m,d_k].
        v: values. tensor of shape [B,n,d_v].

    Returns:
        tensor of shape [B,m,d_v].
    """
    total_points = q.shape[1]
    rep = v.mean(dim=1, keepdims=True).repeat([1, total_points, 1])  # [B,1,d_v]
    return rep


def laplace_attention(q, k, v, scale, normalise):
    """Computes laplace exponential attention.

    Args:
        q: queries. tensor of shape [B,m,d_k].
        k: keys. tensor of shape [B,n,d_k].
        v: values. tensor of shape [B,n,d_v].
        scale: float that scales the L1 distance.
        normalise: Boolean that determines whether weights sum to 1.

    Returns:
        tensor of shape [B,m,d_v].
    """
    k = torch.unsqueeze(k, dim=1)  # [B,1,n,d_k]
    q = torch.unsqueeze(q, dim=2)  # [B,m,1,d_k]
    unnorm_weights = - torch.abs((k - q) / scale)  # [B,m,n,d_k]
    unnorm_weights = unnorm_weights.sum(dim=-1)  # [B,m,n]
    if normalise:
        weight_fn = partial(F.softmax, dim=-1)
    else:
        weight_fn = lambda x: 1 + F.tanh(x)
    weights = weight_fn(unnorm_weights)  # [B,m,n]
    rep = torch.bmm(weights, v)  # [B,m,d_v]
    return rep


def dot_product_attention(q, k, v, normalise):
    """Computes dot product attention.

    Args:
        q: queries. tensor of  shape [B,m,d_k].
        k: keys. tensor of shape [B,n,d_k].
        v: values. tensor of shape [B,n,d_v].
        normalise: Boolean that determines whether weights sum to 1.

    Returns:
        tensor of shape [B,m,d_v].
    """
    d_k = q.shape[-1]
    scale = torch.sqrt(torch.tensor(d_k).float())
    unnorm_weights = torch.bmm(q, k.transpose(1, 2)) / scale  # [B,m,n]
    if normalise:
        weight_fn = partial(F.softmax, dim=-1)
    else:
        weight_fn = torch.sigmoid
    weights = weight_fn(unnorm_weights)  # [B,m,n]
    rep = torch.bmm(weights, v)  # [B,m,d_v]
    return rep

class Conv1dWithInit(nn.Conv1d):
    def __init__(self, std, *args, **kwargs):
        super(Conv1dWithInit, self).__init__(*args, **kwargs)
        init.normal_(self.weight, std=std)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, num_heads=8):
        super(MultiHeadAttention, self).__init__()

        head_size = d_v // num_heads

        std_key_init = d_k ** -0.5
        std_value_init = d_v ** -0.5

        self.q_layers = nn.ModuleList()
        self.k_layers = nn.ModuleList()
        self.v_layers = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        for h in range(num_heads):
            self.q_layers.append(Conv1dWithInit(std_key_init, in_channels=d_k, out_channels=head_size, kernel_size=1, bias=False))
            self.k_layers.append(Conv1dWithInit(std_key_init, in_channels=d_k, out_channels=head_size, kernel_size=1, bias=False))
            self.v_layers.append(Conv1dWithInit(std_key_init, in_channels=d_v, out_channels=head_size, kernel_size=1, bias=False))
            self.attn_layers.append(Conv1dWithInit(std_value_init, in_channels=head_size, out_channels=d_v, kernel_size=1, bias=False))

    def forward(self, q, k, v):
        rep = torch.tensor(0.0)
        for i in range(len(self.q_layers)):
            o = dot_product_attention(self.q_layers[i](q.transpose(1, 2)).transpose(1, 2),
                                      self.k_layers[i](k.transpose(1, 2)).transpose(1, 2),
                                      self.v_layers[i](v.transpose(1, 2)).transpose(1, 2),
                                      normalise=True)
            rep = rep + self.attn_layers[i](o.transpose(1, 2)).transpose(1, 2)
        return rep

# def multihead_attention(q, k, v, num_heads=8):
#     """Computes multi-head attention.
#
#     Args:
#       q: queries. tensor of  shape [B,m,d_k].
#       k: keys. tensor of shape [B,n,d_k].
#       v: values. tensor of shape [B,n,d_v].
#       num_heads: number of heads. Should divide d_v.
#
#     Returns:
#       tensor of shape [B,m,d_v].
#     """
#     d_k = q.shape[-1]
#     d_v = v.shape[-1]
#     head_size = d_v // num_heads
#
#     rep = torch.tensor(0.0)
#
#     for h in range(num_heads):
#         q_head = (nn.Conv1d(in_channels=d_k, out_channels=head_size, kernel_size=1, bias=False)(q.transpose(1, 2))).transpose(1, 2)
#         k_head = (nn.Conv1d(in_channels=d_k, out_channels=head_size, kernel_size=1, bias=False)(k.transpose(1, 2))).transpose(1, 2)
#         v_head = (nn.Conv1d(in_channels=d_v, out_channels=head_size, kernel_size=1, bias=False)(v.transpose(1, 2))).transpose(1, 2)
#         o = dot_product_attention(q_head, k_head, v_head, normalise=True)
#         rep = rep + (nn.Conv1d(in_channels=head_size, out_channels=d_v, kernel_size=1, bias=False)(o.transpose(1, 2))).transpose(1, 2)
#
#     return rep


class Attention(nn.Module):
    """The Attention module."""

    def __init__(self, rep, d_x, output_sizes, att_type, scale=1., normalise=True,
                 num_heads=8):
        """Create attention module.

        Takes in context inputs, target inputs and
        representations of each context input/output pair
        to output an aggregated representation of the context data.
        Args:
            rep: transformation to apply to contexts before computing attention.
                One of: ['identity','mlp'].
            output_sizes: list of number of hidden units per layer of mlp.
                Used only if rep == 'mlp'.
            att_type: type of attention. One of the following:
                ['uniform','laplace','dot_product','multihead']
            scale: scale of attention.
            normalise: Boolean determining whether to:
                1. apply softmax to weights so that they sum to 1 across context pts or
                2. apply custom transformation to have weights in [0,1].
            num_heads: number of heads for multihead.
        """
        super(Attention, self).__init__()
        self._rep = rep
        self._output_sizes = output_sizes
        self._type = att_type
        self._scale = scale
        self._normalise = normalise

        if self._type == 'multihead':
            if self._rep == 'identity':
                d_k, d_v = d_x, d_x
            elif self._rep == 'mlp':
                d_k, d_v = output_sizes[-1], output_sizes[-1]

            self._num_heads = num_heads
            self.multihead_attention = MultiHeadAttention(d_k, d_v, num_heads)

        self.batch_linear = BatchLinear(d_x, output_sizes)

    def forward(self, x1, x2, r):
        """Apply attention to create aggregated representation of r.

        Args:
            x1: tensor of shape [B,n1,d_x].
            x2: tensor of shape [B,n2,d_x].
            r: tensor of shape [B,n1,d].

        Returns:
            tensor of shape [B,n2,d]

        Raises:
            NameError: The argument for rep/type was invalid.
        """
        if self._rep == 'identity':
            k, q = (x1, x2)
        elif self._rep == 'mlp':
            # Pass through MLP
            k = self.batch_linear(x1)
            q = self.batch_linear(x2)
        else:
            raise NameError("'rep' not among ['identity','mlp']")

        if self._type == 'uniform':
            rep = uniform_attention(q, r)
        elif self._type == 'laplace':
            rep = laplace_attention(q, k, r, self._scale, self._normalise)
        elif self._type == 'dot_product':
            rep = dot_product_attention(q, k, r, self._normalise)
        elif self._type == 'multihead':
            rep = self.multihead_attention(q, k, r)
        else:
            raise NameError(("'att_type' not among ['uniform','laplace','dot_product'"
                             ",'multihead']"))

        return rep


def plot_functions(ep, target_x, target_y, context_x, context_y, pred_y, std):
    """Plots the predicted mean and variance and the context points.

    Args:
        target_x: An array of shape [B,num_targets,1] that contains the
            x values of the target points.
        target_y: An array of shape [B,num_targets,1] that contains the
            y values of the target points.
        context_x: An array of shape [B,num_contexts,1] that contains
            the x values of the context points.
        context_y: An array of shape [B,num_contexts,1] that contains
            the y values of the context points.
        pred_y: An array of shape [B,num_targets,1] that contains the
            predicted means of the y values at the target points in target_x.
        std: An array of shape [B,num_targets,1] that contains the
            predicted std dev of the y values at the target points in target_x.
    """
    fig = plt.figure()
    # Plot everything
    plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
    plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
    plt.plot(context_x[0], context_y[0], 'ko', markersize=10)
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - std[0, :, 0],
        pred_y[0, :, 0] + std[0, :, 0],
        alpha=0.2,
        facecolor='#65c9f7',
        interpolate=True)

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    plt.grid('off')
    ax = plt.gca()
    fig.savefig('./anp/out/test_{}.png'.format(ep))
    plt.close(fig)


def np_train():
    TRAINING_ITERATIONS = 100000 #@param {type:"number"}
    MAX_CONTEXT_POINTS = 50 #@param {type:"number"}
    PLOT_AFTER = 1000 #@param {type:"number"}
    HIDDEN_SIZE = 128 #@param {type:"number"}
    MODEL_TYPE = 'NP' #@param ['NP','ANP']
    random_kernel_parameters = True #@param {type:"boolean"}

    # Train dataset
    dataset_train = GPCurvesReader(
        batch_size=16, max_num_context=MAX_CONTEXT_POINTS, random_kernel_parameters=random_kernel_parameters)

    # Test dataset
    dataset_test = GPCurvesReader(
        batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True, random_kernel_parameters=random_kernel_parameters)


    # Sizes of the layers of the MLPs for the encoders and decoder
    # The final output layer of the decoder outputs two values, one for the mean and
    # one for the variance of the prediction at the target location
    latent_encoder_output_sizes = [HIDDEN_SIZE]*4
    num_latents = HIDDEN_SIZE
    deterministic_encoder_output_sizes = [HIDDEN_SIZE]*4
    decoder_output_sizes = [HIDDEN_SIZE]*2 + [2]
    use_deterministic_path = True


    # ANP with multihead attention
    if MODEL_TYPE == 'ANP':
        attention = Attention(rep='mlp', d_x=dataset_train._x_size, output_sizes=[HIDDEN_SIZE]*2,
                            att_type='multihead')
    # NP - equivalent to uniform attention
    elif MODEL_TYPE == 'NP':
        attention = Attention(rep='identity', d_x=dataset_train._x_size, output_sizes=deterministic_encoder_output_sizes, att_type='uniform')
    else:
        raise NameError("MODEL_TYPE not among ['ANP,'NP']")

    # Define the model
    model = LatentModel(dataset_train._x_size, dataset_train._y_size, latent_encoder_output_sizes, num_latents,
                        decoder_output_sizes, use_deterministic_path,
                        deterministic_encoder_output_sizes, attention)


    # Set up the optimizer and train step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train and plot
    for it in range(TRAINING_ITERATIONS):

        model.train()

        data_train = dataset_train.generate_curves()

        # Define the loss
        _, _, log_prob, _, loss = model(data_train.query, data_train.num_total_points,
                                        data_train.target_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Plot the predictions in `PLOT_AFTER` intervals
        if it % PLOT_AFTER == 0:
            data_test = dataset_test.generate_curves()
            model.eval()
            _, _, log_prob, _, loss = model(data_train.query, data_train.num_total_points,
                                            data_train.target_y)

            # Get the predicted mean and variance at the target points for the testing set
            mu, sigma, _, _, _ = model(data_test.query, data_test.num_total_points)
            loss_value, pred_y, std_y, target_y, whole_query = loss.detach().numpy(), mu.detach().numpy(), sigma.detach().numpy(), data_test.target_y, data_test.query

            (context_x, context_y), target_x = whole_query
            print('Iteration: {}, loss: {}'.format(it, loss_value))

            # Plot the prediction and the context
            plot_functions(it, target_x, target_y, context_x, context_y, pred_y, std_y)


def anp_train():
    TRAINING_ITERATIONS = 100000 #@param {type:"number"}
    MAX_CONTEXT_POINTS = 50 #@param {type:"number"}
    PLOT_AFTER = 1000 #@param {type:"number"}
    HIDDEN_SIZE = 128 #@param {type:"number"}
    MODEL_TYPE = 'ANP' #@param ['NP','ANP']
    random_kernel_parameters=True #@param {type:"boolean"}

    # Train dataset
    dataset_train = GPCurvesReader(
        batch_size=16, max_num_context=MAX_CONTEXT_POINTS)

    # Test dataset
    dataset_test = GPCurvesReader(
        batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True)


    # Sizes of the layers of the MLPs for the encoders and decoder
    # The final output layer of the decoder outputs two values, one for the mean and
    # one for the variance of the prediction at the target location
    latent_encoder_output_sizes = [HIDDEN_SIZE]*4
    num_latents = HIDDEN_SIZE
    deterministic_encoder_output_sizes = [HIDDEN_SIZE]*4
    decoder_output_sizes = [HIDDEN_SIZE]*2 + [2]
    use_deterministic_path = True

    # ANP with multihead attention
    if MODEL_TYPE == 'ANP':
        attention = Attention(rep='mlp', d_x=dataset_train._x_size, output_sizes=[HIDDEN_SIZE]*2,
                            att_type='dot_product', normalise=False)
    # NP - equivalent to uniform attention
    elif MODEL_TYPE == 'NP':
        attention = Attention(rep='identity', d_x=dataset_train._x_size, output_sizes=None, att_type='uniform')
    else:
        raise NameError("MODEL_TYPE not among ['ANP,'NP']")

    # Define the model
    model = LatentModel(dataset_train._x_size, dataset_train._y_size, latent_encoder_output_sizes, num_latents,
                        decoder_output_sizes, use_deterministic_path,
                        deterministic_encoder_output_sizes, attention)

    model.cuda()

    # Set up the optimizer and train step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train and plot
    for it in range(TRAINING_ITERATIONS):

        model.train()

        data_train = dataset_train.generate_curves()
        data_train = to_device(data_train, 'cuda:0')

        # Define the loss
        _, _, log_prob, _, loss = model(data_train.query, data_train.num_total_points,
                                        data_train.target_y)

        # # Get the predicted mean and variance at the target points for the testing set
        # mu, sigma, _, _, _ = model(data_test.query, data_test.num_total_points)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Plot the predictions in `PLOT_AFTER` intervals
        if it % PLOT_AFTER == 0:
            data_test = dataset_test.generate_curves()
            data_test = to_device(data_test, 'cuda:0')
            model.eval()
            with torch.set_grad_enabled(False):
                _, _, log_prob, _, loss = model(data_train.query, data_train.num_total_points,
                                                data_train.target_y)

                # Get the predicted mean and variance at the target points for the testing set
                mu, sigma, _, _, _ = model(data_test.query, data_test.num_total_points)
            loss_value, pred_y, std_y, target_y, whole_query = loss, mu, sigma, data_test.target_y, data_test.query

            (context_x, context_y), target_x = whole_query
            print('Iteration: {}, loss: {}'.format(it, np_ify(loss_value)))

            # Plot the prediction and the context
            plot_functions(it, np_ify(target_x), np_ify(target_y), np_ify(context_x), np_ify(context_y), np_ify(pred_y), np_ify(std_y))

