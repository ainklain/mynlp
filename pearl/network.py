"""
General networks for pytorch.
Algorithm-specific networks should go else-where.
"""

from tf_additional.normalization import LayerNormalization

import tensorflow as tf
from tensorflow.keras import layers, activations, Model
from tensorflow.keras.layers import Dense, LSTM

from pearl.base import Policy
from pearl import tf2_util as tfu
from pearl.tf2_core import TF2Module


def identity(x):
    return x


class DenseWithLayerNorm(layers.Layer):
    def __init__(self, *args, layer_norm=True, activation=activations.relu, **kwargs):
        super().__init__()
        self.layer_norm = layer_norm
        self.layer = Dense(*args, activation='linear', **kwargs)
        self.layer_normalization = LayerNormalization()
        self.activation = activation

    def call(self, x):
        x = self.layer(x)
        if self.layer_norm:
            x = self.layer_normalization(x)
        return self.activation(x)


class MLP(Model, TF2Module):
    def __init__(self,
                 hidden_sizes,
                 output_size,
                 input_size,
                 init_w=3e-3,
                 hidden_activation=activations.relu,
                 output_activation=activations.linear,
                 hidden_init='glorot_uniform',
                 b_init_value=0.1,
                 layer_norm=False,
                 **kwargs
                 ):
        super(MLP, self).__init__(**kwargs)
        # self.save_init_params(locals())

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_layers = list()

        for i, h in enumerate(hidden_sizes):
            if i < len(hidden_sizes) - 1:
                add_layer_norm = layer_norm
            else:
                add_layer_norm = False

            self.hidden_layers.append(
                DenseWithLayerNorm(h,
                                   layer_norm=add_layer_norm,
                                   activation=hidden_activation,
                                   kernel_initializer=hidden_init,
                                   bias_initializer=tf.keras.initializers.Constant(b_init_value))
            )

        self.output_layer = Dense(output_size,
                                   activation='linear',
                                   kernel_initializer=tf.keras.initializers.RandomUniform(-init_w, init_w),
                                   bias_initializer=tf.keras.initializers.RandomUniform(-init_w, init_w))
        self.output_activation = output_activation

    def call(self, inputs, return_preactivations=False, output_layer=True):
        x = inputs
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)

        if output_layer:
            preactivation = self.output_layer(x)
            out = self.output_activation(preactivation)
            if return_preactivations:
                return out, preactivation
            else:
                return out
        else:
            return x


class FlattenMLP(MLP):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def call(self, *inputs, **kwargs):
        flat_inputs = tf.concat(inputs, axis=1)
        return super().call(flat_inputs, **kwargs)


class MLPPolicy(MLP, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer=None,
            **kwargs
    ):
        # self.save_init_params(locals())
        super(MLPPolicy, self).__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def call(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().call(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMLPPolicy(MLPPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super(TanhMLPPolicy, self).__init__(*args, output_activation=activations.tanh, **kwargs)


class MLPEncoder(FlattenMLP):
    '''
    encode context via MLP
    '''

    def reset(self, num_tasks=1):
        pass


class RecurrentEncoder(FlattenMLP):
    '''
    encode context via recurrent network
    '''

    def __init__(self,
                 *args,
                 **kwargs
    ):
        # self.save_init_params(locals())
        super(RecurrentEncoder, self).__init__(*args, **kwargs)
        self.hidden_dim = self.hidden_sizes[-1]
        self.hidden = tf.zeros([1, 1, self.hidden_dim])

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)

        self.lstm = LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

    def call(self, inputs, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = inputs.shape()
        out = tf.reshape(inputs, [task * seq, feat])

        for hidden_layer in self.hidden_layers:
            out = hidden_layer(out)

        out = tf.reshape(out, [task, seq, -1])
        out, (hn, cn) = self.lstm(out, (self.hidden, torch.zeros(self.hidden.size()).to(ptu.device)))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out[:, -1, :]

        preactivation = self.output_layer(out)
        out = self.output_activation(preactivation)
        if return_preactivations:
            return out, preactivation
        else:
            return out

        # embed with MLP
        # for i, fc in enumerate(self.fcs):
        #     out = fc(out)
        #     out = self.hidden_activation(out)
        #
        # out = out.view(task, seq, -1)
        # out, (hn, cn) = self.lstm(out, (self.hidden, torch.zeros(self.hidden.size()).to(ptu.device)))
        # self.hidden = hn
        # # take the last hidden state to predict z
        # out = out[:, -1, :]
        #
        # # output layer
        # preactivation = self.last_fc(out)
        # output = self.output_activation(preactivation)
        # if return_preactivations:
        #     return output, preactivation
        # else:
        #     return output

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)
