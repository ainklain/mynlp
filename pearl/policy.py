import abc
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense


from pearl.base import Policy
from pearl.network import MLP
from pearl.core.util import Wrapper
from pearl.tf2_core import np_ify

# from myrl.distribution.distribution import TanhNormal
from pearl.distribution import TanhNormal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class ExplorationPolicy(Policy, metaclass=abc.ABCMeta):
    def set_num_steps_total(self, t):
        pass


class SerializablePolicy(Policy, metaclass=abc.ABCMeta):
    """
    Policy that can be serialized.
    """

    def get_param_values(self):
        return None

    def set_param_values(self, values):
        pass

    """
    Parameters should be passed as np arrays in the two functions below.
    """

    def get_param_values_np(self):
        return None

    def set_param_values_np(self, values):
        pass


class TanhGaussianPolicy(MLP, ExplorationPolicy):
    """
    Usage:
    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.
    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            latent_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        # self.save_init_params(locals())
        super(TanhGaussianPolicy, self).__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.latent_dim = latent_dim
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = Dense(action_dim,
                                         kernel_initializer=tf.keras.initializers.RandomUniform(-init_w, init_w),
                                         bias_initializer=tf.keras.initializers.RandomUniform(-init_w, init_w))
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs, deterministic=False):
        actions = self.get_actions(obs, deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs, deterministic=False):
        outputs = self.call(obs, deterministic=deterministic)[0]
        return np_ify(outputs)

    def call(
            self,
            obs,
            reparameterize=False,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """

        x = obs
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        mean = self.output_layer(x)
        if self.std is None:
            log_std = self.last_fc_log_std(x)
            log_std = tf.clip_by_value(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = tf.math.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = tf.math.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob, pre_tanh_value,
        )


class MakeDeterministic(Wrapper, Policy):
    def __init__(self, stochastic_policy):
        super(MakeDeterministic, self).__init__(stochastic_policy)
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)

    def get_actions(self, observations):
        return self.stochastic_policy.get_actions(observations,
                                                  deterministic=True)
