import collections
import numpy as np
# import tensorflow as tf
import torch
NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points",
     "hyperparams"))


class GPCurvesReader(object):
    """Generates curves using a Gaussian Process (GP).
    Supports vector inputs (x) and vector outputs (y). Kernel is
    mean-squared exponential, using the x-value l2 coordinate distance scaled
    by some factor chosen randomly in a range. Outputs are
    independent gaussian processes.
    """

    def __init__(self,
               batch_size,
               max_num_context,
               x_size=1,
               y_size=1,
               testing=False,
               len_seq=10,
               len_given=5,
               len_gen=10,
               l1_min=0.7,
               l1_max=1.2,
               l1_vel=0.05,
               sigma_min=1.0,
               sigma_max=1.6,
               sigma_vel=0.05,
               temporal=False,
               case=1,
               ):
        """Creates a regression dataset of functions sampled from a GP.
        Args:
            batch_size: An integer.
            max_num_context: The max number of observations in the context.
            x_size: Integer >= 1 for length of "x values" vector.
            y_size: Integer >= 1 for length of "y values" vector.
            l1_scale: Float; typical scale for kernel distance function.
            sigma_scale: Float; typical scale for variance.
            testing: Boolean that indicates whether we are testing.
                    If so there are more targets for visualization.
        """
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._testing = testing
        self._len_seq = len_seq
        self._len_given = len_given
        self._len_gen = len_gen
        self._l1_min = l1_min
        self._l1_max = l1_max
        self._l1_vel = l1_vel
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._sigma_vel = sigma_vel
        self._temporal = temporal
        self._case = case

        self._noise_factor = 0.1

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

    def generate_temporal_curves(self, seed=None):
        # Set kernel parameters
        # Either choose a set of random parameters for the mini-batch
        l1 = torch.FloatTensor(self._batch_size, self._y_size,
                               self._x_size).uniform_(self._l1_min, self._l1_max)

        sigma_f = torch.FloatTensor(self._batch_size, self._y_size).uniform_(self._sigma_min, self._sigma_max)

        l1_vel = torch.FloatTensor(self._batch_size, self._y_size,
                               self._x_size).uniform_(-1*self._l1_vel, self._l1_vel)

        sigma_f_vel = torch.FloatTensor(self._batch_size, self._y_size).uniform_(-1*self._sigma_vel, self._sigma_vel)

        if self._testing:
            num_total_points = 400
        else:
            num_total_points = 100

        y_value_base = torch.randn([self._batch_size, self._y_size, num_total_points, 1])

        curve_list = []
        if (self._case == 2) or (self._case == 3):
            # sparse time or long term tracking
            idx = torch.randperm(self._len_seq)[:(self._len_given)]
        for t in range(self._len_seq):
            if self._case == 1:    # using len_given
                if t < self._len_given:
                    num_context = torch.randint(low=5, high=self._max_num_context, size=[])

                else:
                    num_context = torch.tensor(0)
                    #num_context = tf.constant(1)
            if self._case == 2:    # sparse time
                nc_cond = torch.where(idx == t)[0].reshape([-1])
                if len(nc_cond) == 0:
                    num_context = torch.tensor(0)
                else:
                    num_context = torch.randint(size=[], low=5, high=self._max_num_context)

            if self._case==3:    # long term tracking
                nc_cond = torch.where(idx == t)[0].reshape([-1])
                if len(nc_cond) == 0:
                    num_context = torch.tensor(0)
                else:
                    num_context = torch.tensor(1)

            if self._temporal:
                encoded_t = None
            else:
                encoded_t = 0.25 + 0.5*t/self._len_seq
            curve_list.append(self.generate_curves(l1, sigma_f, num_context,
                                                   y_value_base,
                                                   encoded_t))
            vel_noise = l1_vel * self._noise_factor * torch.randn([self._batch_size,self._y_size, self._x_size])
            l1 += l1_vel + vel_noise
            vel_noise = sigma_f_vel * self._noise_factor *  torch.randn([self._batch_size, self._x_size])
            sigma_f += sigma_f_vel + vel_noise

        if self._testing:
            for t in range(self._len_seq,self._len_seq+self._len_gen):
                num_context = torch.tensor(0)

                if self._temporal:
                    encoded_t = None
                else:
                    encoded_t = 0.25 + 0.5*t/self._len_seq
                curve_list.append(self.generate_curves(l1, sigma_f,
                                                   num_context,
                                                   y_value_base,
                                                   encoded_t))
                vel_noise = l1_vel * self._noise_factor * torch.randn([
                                self._batch_size,self._y_size, self._x_size])
                l1 += l1_vel + vel_noise
                vel_noise = sigma_f_vel*self._noise_factor*torch.randn(
                                [self._batch_size, self._x_size])
                sigma_f += sigma_f_vel + vel_noise

        context_x_list, context_y_list = [], []
        target_x_list, target_y_list = [], []
        num_total_points_list = []
        num_context_points_list = []
        for t in range(len(curve_list)):
            (context_x, context_y), target_x = curve_list[t].query
            target_y = curve_list[t].target_y
            num_total_points_list.append(curve_list[t].num_total_points)
            num_context_points_list.append(curve_list[t].num_context_points)
            context_x_list.append(context_x)
            context_y_list.append(context_y)
            target_x_list.append(target_x)
            target_y_list.append(target_y)

        query = ((context_x_list, context_y_list), target_x_list)

        return NPRegressionDescription(
                query=query,
                target_y=target_y_list,
                num_total_points=num_total_points_list,
                num_context_points=num_context_points_list,
                hyperparams=[torch.tensor(0)])

    def generate_curves(self, l1, sigma_f, num_context=3,
                      y_value_base=None, encoded_t=None):
        """Builds the op delivering the data.
        Generated functions are `float32` with x values between -2 and 2.
        Returns:
            A `CNPRegressionDescription` namedtuple.
        """

        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.

        if self._testing:
            num_total_points = 400
            num_target = num_total_points
            x_values = torch.unsqueeze(torch.arange(-4., 4., 1. / 50, dtype=torch.float32), dim=0).repeat(self._batch_size, 1)
        else:
            num_total_points = 100
            maxval = self._max_num_context - num_context + 1

            num_target = torch.randint(low=1, high=maxval, size=[])
            x_values = torch.unsqueeze(torch.arange(-4., 4., 1. / 12.5, dtype=torch.float32), dim=0).repeat(self._batch_size, 1)

        x_values = torch.unsqueeze(x_values, dim=-1)
        # During training the number of target points and their x-positions are

        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        kernel = self._gaussian_kernel(x_values, l1, sigma_f)

        # Calculate Cholesky, using double precision for better stability:
        cholesky = torch.cholesky(kernel.double()).float()

        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]

        y_values = torch.bmm(cholesky.view(-1, num_total_points, num_total_points), y_value_base.view(-1, num_total_points, 1))
        y_values = y_values.view(self._batch_size, self._y_size, num_total_points)

        # [batch_size, num_total_points, y_size]
        y_values = torch.transpose(y_values, 1, 2)

        if self._testing:
            # Select the targets
            target_x = x_values
            target_y = y_values

            if encoded_t is not None:
                target_x = torch.cat([
                     target_x,
                     torch.ones([self._batch_size, num_total_points, 1]) * encoded_t
                     ], dim=-1)

            # Select the observations
            idx = torch.randperm(num_target)
            context_x = x_values[:, idx[:num_context]]
            context_y = y_values[:, idx[:num_context]]

            if encoded_t is not None:
                context_x = torch.cat([
                    context_x,
                    torch.ones([self._batch_size, num_context, 1]) * encoded_t
                    ], dim=-1)

        else:
            # Select the targets which will consist of the context points
            # as well as some new target points
            idx = torch.randperm(num_total_points)
            target_x = x_values[:, idx[:num_target + num_context]]
            target_y = y_values[:, idx[:num_target + num_context]]

            if encoded_t is not None:
                target_x = torch.cat([
                    target_x,
                    torch.ones([self._batch_size, min(num_total_points, num_target + num_context), 1])
                    * encoded_t], dim=-1)

            # Select the observations
            context_x = x_values[:, idx[:num_context]]
            context_y = y_values[:, idx[:num_context]]

            if encoded_t is not None:
                context_x = torch.cat([
                    context_x,
                    torch.ones([self._batch_size, min(num_total_points, num_context), 1]) * encoded_t
                    ], dim=-1)

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=target_x.shape[1],
            num_context_points=num_context,
            hyperparams=[torch.tensor(0)])