import tensorflow as tf

from tensorflow_probability.python.distributions import Distribution, Normal


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample(n)
        if return_pre_tanh_value:
            return tf.math.tanh(z), z
        else:
            return tf.math.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = tf.math.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - tf.math.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample()

        if return_pretanh_value:
            return tf.math.tanh(z), z
        else:
            return tf.math.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal(
                tf.zeros_like(self.normal_mean),
                tf.ones_like(self.normal_std)
            ).sample()
        )

        if return_pretanh_value:
            return tf.math.tanh(z), z
        else:
            return tf.math.tanh(z)
