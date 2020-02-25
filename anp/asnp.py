
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
import collections
import math
# from tqdm import tqdm
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from ts_torch.torch_util_mini import np_ify
from anp.gp import NPRegressionDescription, GPCurvesReader


class configs:
    def DEFINE_integer(self, attr_nm, value_, desc_):
        setattr(self, attr_nm, value_)

    def DEFINE_float(self, attr_nm, value_, desc_):
        setattr(self, attr_nm, value_)

    def DEFINE_string(self, attr_nm, value_, desc_):
        setattr(self, attr_nm, value_)

flags = configs()
FLAGS = flags

# models
flags.DEFINE_integer('HIDDEN_SIZE', 128, 'hidden unit size of network')
flags.DEFINE_string('MODEL_TYPE', 'NP', "{NP|SNP}")
flags.DEFINE_float('beta', 1.0, 'weight to kl loss term')
# dataset
flags.DEFINE_string('dataset', 'gp', '{gp}')
flags.DEFINE_integer('case', 1, '{1|2|3}')
flags.DEFINE_integer('MAX_CONTEXT_POINTS', 500,
                     'max context size at each time-steps')
flags.DEFINE_integer('LEN_SEQ', 20, 'sequence length')
flags.DEFINE_integer('LEN_GIVEN', 10, 'given context length')
flags.DEFINE_integer('LEN_GEN', 10, 'generalization test sequence length')
# gp dataset
flags.DEFINE_float('l1_min', 0.7, 'l1 initial boundary')
flags.DEFINE_float('l1_max', 1.2, 'l1 initial boundary')
flags.DEFINE_float('l1_vel', 0.03, 'l1 kernel parameter dynamics')
flags.DEFINE_float('sigma_min', 1.0, 'sigma initial boundary')
flags.DEFINE_float('sigma_max', 1.6, 'sigma initial boundary')
flags.DEFINE_float('sigma_vel', 0.05, 'sigma kernel parameter dynamics')
# training
flags.DEFINE_integer('TRAINING_ITERATIONS', 1000000, 'training iteration')
flags.DEFINE_integer('PLOT_AFTER', 1000, 'plot iteration')
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_string('log_folder', 'logs', 'log folder')

dataset_train = GPCurvesReader(
    batch_size=FLAGS.batch_size, max_num_context=FLAGS.MAX_CONTEXT_POINTS,
    len_seq=FLAGS.LEN_SEQ, len_given=FLAGS.LEN_GIVEN,
    len_gen=FLAGS.LEN_GEN,
    l1_min=FLAGS.l1_min, l1_max=FLAGS.l1_max, l1_vel=FLAGS.l1_vel,
    sigma_min=FLAGS.sigma_min, sigma_max=FLAGS.sigma_max,
    sigma_vel=FLAGS.sigma_vel, temporal=True,
    case=FLAGS.case)

data_train = dataset_train.generate_temporal_curves(seed=None)


def bmm(x: torch.Tensor, y:torch.Tensor):
    # x shape: (B1, B2, ..., A, B)
    # y shape: (B1, B2, ..., B, C)
    # out shape: (B1, B2, ..., A, C)
    assert len(x.shape) >= 3
    assert x.shape[:-2] == y.shape[:-2]
    out_shape = list(x.shape[:-1]) + list(y.shape[-1:])
    out = torch.bmm(x.view(-1, x.shape[-2], x.shape[-1]), y.view(-1, y.shape[-2], y.shape[-1]))
    return out.reshape(out_shape)


def to_device(data: NPRegressionDescription, device='cuda:0'):
    ((context_x, context_y), target_x) = data.query
    context_x = [ctx.to(device) for ctx in context_x]
    context_y = [cty.to(device) for cty in context_y]
    target_x = [tgx.to(device) for tgx in target_x]
    target_y = [tgy.to(device) for tgy in data.target_y]
    query = ((context_x, context_y), target_x)
    return NPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=[pt.to(device) for pt in data.num_total_points],
            num_context_points=[pt.to(device) for pt in data.num_context_points],
            hyperparams=[hp.to(device) for hp in data.hyperparams])


def example_lstm():
    lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
    inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

    # initialize the hidden state.
    hidden = (torch.randn(1, 1, 3),
              torch.randn(1, 1, 3))
    for i in inputs:
        # Step through the sequence one element at a time.
        # after each step, hidden contains the hidden state.
        out, hidden = lstm(i.view(1, 1, -1), hidden)

    # alternatively, we can do the entire sequence all at once.
    # the first value returned by LSTM is all of the hidden states throughout
    # the sequence. the second is just the most recent hidden state
    # (compare the last slice of "out" with "hidden" below, they are the same)
    # The reason for this is that:
    # "out" will give you access to all hidden states in the sequence
    # "hidden" will allow you to continue the sequence and backpropagate,
    # by passing it as an argument  to the lstm at a later time
    # Add the extra 2nd dimension
    inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
    out, hidden = lstm(inputs, hidden)
    print(out)
    print(hidden)


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Define the LSTM layer
        self.lstm_cell = nn.LSTMCell(self.input_dim, self.hidden_dim)

    def init_hidden(self, batch_size):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(batch_size, self.hidden_dim),
                torch.zeros(batch_size, self.hidden_dim))

    def forward(self, input, hidden=None):
        # Forward pass through LSTM layer
        # input shape: [batch, input_size]
        # shape of hidden: (a, b), where a and b both
        # have shape (batch_size, hidden_dim).
        if hidden is None:
            hidden = self.init_hidden(input.shape[0])

        hidden = self.lstm_cell(input, hidden)
        # hidden = self.lstm_cell(input.view(self.batch_size, -1), hidden)

        return hidden # (h, c)


class LatentEncoder(nn.Module):
    """
    Latent Encoder [For prior, posterior]
    """

    def __init__(self, num_hidden, num_latent, input_dim=3):
        super(LatentEncoder, self).__init__()
        self.input_projection = Linear(input_dim, num_hidden)
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.penultimate_layer = Linear(num_hidden, num_hidden, w_init='relu')
        self.mu = Linear(num_hidden, num_latent)
        self.log_sigma = Linear(num_hidden, num_latent)

    def forward(self, x, y):
        # concat location (x) and value (y)
        encoder_input = torch.cat([x, y], dim=-1)

        # project vector with dimension 3 --> num_hidden
        encoder_input = self.input_projection(encoder_input)

        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        # mean
        hidden = encoder_input.mean(dim=1)
        hidden = torch.relu(self.penultimate_layer(hidden))

        # get mu and sigma
        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)

        # reparameterization trick
        sigma = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        # return distribution
        # return mu, log_sigma, z
        return torch.distributions.Normal(loc=mu, scale=sigma)


class DeterministicEncoder(nn.Module):
    """
    Deterministic Encoder [r]
    """

    def __init__(self, num_hidden, input_dim=3):
        super(DeterministicEncoder, self).__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.input_projection = Linear(input_dim, num_hidden)
        self.context_projection = Linear(1, num_hidden)
        self.target_projection = Linear(1, num_hidden)

    def forward(self, context_x, context_y, target_x):
        # concat context location (x), context value (y)
        encoder_input = torch.cat([context_x, context_y], dim=-1)

        # project vector with dimension 3 --> num_hidden
        encoder_input = self.input_projection(encoder_input)

        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        # query: target_x, key: context_x, value: representation
        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        # cross attention layer
        for attention in self.cross_attentions:
            query, _ = attention(keys, encoder_input, query)

        return query


class Decoder(nn.Module):
    """
    Decoder for generation
    """

    def __init__(self, num_hidden):
        super(Decoder, self).__init__()
        self.target_projection = Linear(1, num_hidden)
        self.linears = nn.ModuleList([Linear(num_hidden * 3, num_hidden * 3, w_init='relu') for _ in range(3)])
        self.final_projection = Linear(num_hidden * 3, 2)

    def forward(self, r, z, target_x):
        batch_size, num_targets, _ = target_x.size()
        # project vector with dimension 2 --> num_hidden
        target_x = self.target_projection(target_x)

        # concat all vectors (r,z,target_x)
        hidden = torch.cat([torch.cat([r, z], dim=-1), target_x], dim=-1)

        # mlp layers
        for linear in self.linears:
            hidden = torch.relu(linear(hidden))

        # get mu and sigma
        y_pred = self.final_projection(hidden)

        # Get the mean an the variance
        mu, log_sigma = torch.chunk(y_pred, 2, dim=-1)

        # Bound the variance
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        dist = torch.distributions.Normal(loc=mu, scale=sigma)

        return dist, mu, sigma


class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """

    def __init__(self, num_hidden_k):
        """
        :param num_hidden_k: dimension of hidden
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query):
        # Get attention score
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)

        attn = torch.softmax(attn, dim=-1)

        # Dropout
        attn = self.attn_dropout(attn)

        # Get Context Vector
        result = torch.bmm(attn, value)

        return result, attn


class Attention(nn.Module):
    """
    Attention Network
    """

    def __init__(self, d_hidden, d_model, h=4):
        """
        :param d_hidden: dimension of hidden
        :param h: num of heads
        """
        super(Attention, self).__init__()

        self.d_hidden = d_hidden
        self.d_hidden_per_attn = d_hidden // h
        self.h = h

        self.key = Linear(d_hidden, d_hidden, bias=False)
        self.value = Linear(d_model, d_hidden, bias=False)
        self.query = Linear(d_hidden, d_hidden, bias=False)

        self.multihead = MultiheadAttention(self.d_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(d_hidden * 2, d_hidden)

        self.layer_norm = nn.LayerNorm(d_hidden)

    def forward(self, key, value, query):
        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        residual = query

        # Make multihead
        key = self.key(key).view(batch_size, seq_k, self.h, self.d_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_k, self.h, self.d_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.d_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.d_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.d_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.d_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(key, value, query)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q, self.d_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)

        result = torch.cat([residual, result], dim=-1)

        # Concatenate context vector with input (most important)
        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + residual

        # Layer normalization
        result = self.layer_norm(result)

        return result, attns


class LatentModel(nn.Module):
    """
    Latent Model (Attentive Neural Process)
    """

    def __init__(self, num_hidden, d_x=1, d_y=1):
        super(LatentModel, self).__init__()
        self.latent_encoder = LatentEncoder(num_hidden, num_hidden, input_dim=d_x+d_y)
        self.deterministic_encoder = DeterministicEncoder(num_hidden, input_dim=d_x+d_y)
        self.decoder = Decoder(num_hidden)
        self.BCELoss = nn.BCELoss()

    def forward(self, query, target_y=None):

        (context_x, context_y), target_x = query
        num_targets = target_x.size(1)

        # prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y)
        prior = self.latent_encoder(context_x, context_y)

        # For training
        if target_y is not None:
            # posterior_mu, posterior_var, posterior = self.latent_encoder(target_x, target_y)
            # z = posterior
            posterior = self.latent_encoder(target_x, target_y)
            z = posterior.sample()

        # For Generation
        else:
            # z = prior
            z = prior.sample()


        z = z.unsqueeze(1).repeat(1, num_targets, 1)  # [B, T_target, H]
        r = self.deterministic_encoder(context_x, context_y, target_x)  # [B, T_target, H]

        # mu should be the prediction of target y
        dist, mu, sigma = self.decoder(r, z, target_x)

        # For Training
        if target_y is not None:
            log_p = dist.log_prob(target_y).squeeze()

            posterior = self.latent_encoder(target_x, target_y)
            kl = torch.distributions.kl_divergence(posterior, prior).sum(dim=-1, keepdims=True)
            kl = kl.repeat([1, num_targets])
            loss = - (log_p - kl / torch.tensor(num_targets).float()).mean()

            # # get log probability
            # bce_loss = self.BCELoss(torch.sigmoid(mu), target_y)
            #
            # # get KL divergence between prior and posterior
            # kl = self.kl_div(prior_mu, prior_var, posterior_mu, posterior_var)
            #
            # # maximize prob and minimize KL divergence
            # loss = bce_loss + kl

        # For Generation
        else:
            log_p = None
            kl = None
            loss = None

        return mu, sigma, log_p, kl, loss

    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (torch.exp(posterior_var) + (posterior_mu - prior_mu) ** 2) / torch.exp(prior_var) - 1. + (
                    prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div


class ImaginaryContext(nn.Module):
    def __init__(self, d_hidden, d_model):
        super(ImaginaryContext, self).__init__()
        self.d_hidden = d_hidden
        self.d_model = d_model

        # key_inference
        self.ikey_lstm = LSTMCell(d_hidden+d_model, d_hidden)
        self.ikey_hidden_layer = Linear(d_hidden, d_hidden, w_init='relu')
        self.ikey_infer_mu = Linear(d_hidden, d_hidden)
        self.ikey_infer_logsigma = Linear(d_hidden, d_hidden)

        # imagination tracker
        self.itracker_lstm = LSTMCell(d_hidden+d_model, d_model)
        self.itracker_cross_attentions = nn.ModuleList([Attention(d_hidden) for _ in range(2)])
        self.itracker_hidden_layer = Linear(d_hidden, d_hidden, w_init='relu')
        self.itracker_mu = Linear(d_hidden, d_model)
        self.itracker_logsigma = Linear(d_hidden, d_model)

        self.reset_variables()

    def reset_variables(self):
        self.x_im = None
        self.u_im = None
        self.ikey_hidden = None
        self.itracker_hidden = None

    def _initialize_variables(self, batch_size):
        self.x_im = torch.zeros(batch_size, self.d_hidden)
        self.u_im = torch.zeros(batch_size, self.d_model)
        self.ikey_hidden = self.ikey_lstm.init_hidden(batch_size)
        self.itracker_hidden = self.itracker_lstm.init_hidden(batch_size)

    def get_variables(self):
        return self.x_im, self.u_im, self.ikey_hidden, self.itracker_hidden

    def forward(self, x_re_t, r_re_t):
        """
            context_x : context_real x (shape: [batch, num_contexts, d_hidden])
            context_r : "representation of context" = f_orderinv(context_x, context_y)
              (shape: [batch, num_contexts, d_model])
        """
        batch, num_contexts, _ = x_re_t.shape
        x_re_t_lstm = x_re_t.reshape([-1, self.d_hidden])
        r_re_t_lstm = r_re_t.reshape([-1, self.d_model])

        # get t-1 values
        if self.x_im is None:
            self._initialize_variables(batch * num_contexts) # batch_lstm = batch * num_contexts

        x_im_prev, u_im_prev, ikey_hidden_prev, itracker_hidden_prev = self.get_variables()

        # imaginary key inference
        ikey_input = torch.cat([x_im_prev, r_re_t_lstm], dim=-1)
        ikey_hidden = self.ikey_lstm(ikey_input, ikey_hidden_prev)  # ikey_hidden_t: (h_t, c_t)
        ikey_hidden_t = torch.relu(self.ikey_hidden_layer(ikey_hidden[0]))

        # x_t ~ N(f_mu(h_t), f_sigma(h_t))
        ikey_mu = self.ikey_infer_mu(ikey_hidden_t)
        ikey_log_sigma = self.ikey_infer_logsigma(ikey_hidden_t)
        ikey_sigma = torch.exp(0.5 * ikey_log_sigma)
        x_im_dist = torch.distributions.Normal(loc=ikey_mu, scale=ikey_sigma)

        # (x_im_t shape:[batch_lstm, d_hidden]: projection된 input으로 보자. mu, sigma layer에 반영)
        x_im_t = x_im_dist.sample()

        # imagination tracker
        itracker_input = torch.cat([x_im_t, u_im_prev], dim=-1)
        itracker_hidden = self.itracker_lstm(itracker_input, itracker_hidden_prev)  # itracker_hidden: (h_t, c_t)

        # query: x_im only (shape: [batch_lstm, d_hidden] => [batch, n_context, d_hidden])
        # key: x_im + x_re (shape: [batch_lstm, d_hidden] => [batch, n_context*2, d_hidden])
        # value: h_im + v_re (shape: [batch_lstm, d_model] => [batch, n_context*2, d_model])
        key_ = torch.cat([x_re_t_lstm, x_im_t], dim=0).reshape([batch, -1, self.d_hidden])
        value_ = torch.cat([r_re_t_lstm, itracker_hidden[0]], dim=0).reshape([batch, -1, self.d_model])
        query_ = x_im_t.reshape([batch, -1, self.d_hidden])
        for attn in self.itracker_cross_attentions:
            query_, _ = attn(key_, value_, query_)  # key / value / query

        # x_t ~ N(f_mu(a_t), f_sigma(a_t))
        itracker_mu = self.itracker_mu(query_)
        itracker_log_sigma = self.itracker_logsigma(query_)
        itracker_sigma = torch.exp(0.5 * itracker_log_sigma)
        u_im_dist = torch.distributions.Normal(loc=itracker_mu, scale=itracker_sigma)

        # (u_im_t shape:[batch, d_model]
        u_im_t = u_im_dist.sample()

        # set prev variables
        self.x_im = x_im_t
        self.u_im = u_im_t
        self.ikey_hidden = ikey_hidden
        self.itracker_hidden = itracker_hidden

        return x_im_dist, u_im_dist, x_im_t, u_im_t


class CommonLatentEncoder(nn.Module):
    def __init__(self, d_x, d_y, d_hidden, d_model):
        super(CommonLatentEncoder, self).__init__()
        self.input_projection = Linear(d_x + d_y, d_hidden)
        self.hidden_layer = nn.ModuleList([Linear(d_hidden, d_hidden) for _ in range(2)])
        self.output_projection = Linear(d_hidden, d_model)

    def forward(self, x_re_t, y_re_t):
        input_re_t = torch.cat([x_re_t, y_re_t], dim=-1)
        context = self.input_projection(input_re_t)
        for h_layer in self.hidden_layer:
            context = torch.relu(h_layer(context))

        return context, self.output_projection(context)


class GlobalLatentEncoder(nn.Module):
    def __init__(self, d_hidden, d_model):
        super(GlobalLatentEncoder, self).__init__()
        self.hidden_layer = nn.ModuleList([Linear(d_hidden, d_hidden) for _ in range(2)])
        self.output_projection = Linear(d_hidden, d_model)

    def forward(self, context_re_t, context_im_t):

        for h_layer in self.hidden_layer:
            context_re_t = torch.relu(h_layer(context_re_t))

        context_re_t = self.output_projection(context_re_t)
        context_t = torch.cat([context_re_t, context_im_t], dim=0).mean(dim=0)
        context_t =
        pass


class QueryDepLatentEncoder(nn.Module):
    def __init__(self, d_hidden, d_model):
        super(QueryDepLatentEncoder, self).__init__()
        self.re_encoder = CommonLatentEncoder(d_x, d_y, d_hidden, d_model)

    def forward(self, x_re_t, y_re_t):



class ASNP(nn.Module):
    def __init__(self, d_x, d_y, d_hidden, d_model):
        super(ASNP, self).__init__()
        self.common_latent_encoder = CommonLatentEncoder(d_x, d_y, d_hidden, d_model)
        self.global_latent_encoder = GlobalLatentEncoder()
        self.query_latent_encoder = QueryDepLatentEncoder()
        self.decoder = Decoder()
        self.imaginary_context = ImaginaryContext()

    def forward(self, context, target_y):
        (x_re_t, y_re_t), target_x = context
        context_re_t = self.common_latent_encoder(x_re_t, y_re_t)

        pass


def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = 0.001 * warmup_step ** 0.5 * min(step_num * warmup_step ** -1.5, step_num ** -0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


def main2():
    TRAINING_ITERATIONS = 100000 #@param {type:"number"}
    MAX_CONTEXT_POINTS = 50 #@param {type:"number"}
    PLOT_AFTER = 1000 #@param {type:"number"}

    # Train dataset
    dataset_train = GPCurvesReader(
        batch_size=16, max_num_context=MAX_CONTEXT_POINTS)

    # Test dataset
    dataset_test = GPCurvesReader(
        batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True)


    # Sizes of the layers of the MLPs for the encoders and decoder
    # The final output layer of the decoder outputs two values, one for the mean and
    # one for the variance of the prediction at the target location


    model = LatentModel(128).cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for it in range(TRAINING_ITERATIONS):

        data_train = dataset_train.generate_curves()
        data_train = to_device(data_train, 'cuda:0')
        model.train()
        # Define the loss
        query, target_y = data_train.query,  data_train.target_y
        _, _, log_prob, _, loss = model(data_train.query,  data_train.target_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Plot the predictions in `PLOT_AFTER` intervals
        if it % PLOT_AFTER == 0:
            data_test = dataset_test.generate_curves()
            data_test = to_device(data_test, 'cuda:0')
            model.eval()
            with torch.set_grad_enabled(False):
                _, _, log_prob, _, loss = model(data_train.query, data_train.target_y)

                # Get the predicted mean and variance at the target points for the testing set
                mu, sigma, _, _, _ = model(data_test.query)
            loss_value, pred_y, std_y, target_y, whole_query = loss, mu, sigma, data_test.target_y, data_test.query

            (context_x, context_y), target_x = whole_query
            print('Iteration: {}, loss: {}'.format(it, np_ify(loss_value)))

            # Plot the prediction and the context
            plot_functions(it, np_ify(target_x), np_ify(target_y), np_ify(context_x), np_ify(context_y), np_ify(pred_y), np_ify(std_y))



def main():
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=False, )

    epochs = 200
    model = LatentModel(128).cuda()
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter()
    global_step = 0
    for epoch in range(epochs):
        dloader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn, shuffle=True, num_workers=16)
        pbar = tqdm(dloader)
        for i, data in enumerate(pbar):
            global_step += 1
            adjust_learning_rate(optim, global_step)
            context_x, context_y, target_x, target_y = data
            context_x = context_x.cuda()
            context_y = context_y.cuda()
            target_x = target_x.cuda()
            target_y = target_y.cuda()

            # pass through the latent model
            y_pred, kl, loss = model(context_x, context_y, target_x, target_y)

            # Training step
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Logging
            writer.add_scalars('training_loss', {
                'loss': loss,
                'kl': kl.mean(),

            }, global_step)

        # save model by each epoch
        # torch.save({'model': model.state_dict(),
        #         'optimizer': optim.state_dict()},
        #        os.path.join('./checkpoint', 'checkpoint_%d.pth.tar' % (epoch + 1)))

