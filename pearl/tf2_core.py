import abc
import numpy as np

import tensorflow as tf
from tensorflow import Variable, Tensor

import pearl.tf2_util as tfu
from pearl.core.serializable import Serializable



class TF2Module(Serializable, metaclass=abc.ABCMeta):
    def __init__(self):
        super(TF2Module, self).__init__()

    def get_param_values(self):
        return self.get_weights()

    def set_param_values(self, param_values):
        self.set_weights(param_values)

    def get_param_values_np(self):
        weight_list = self.get_weights()
        np_list = list()
        for i, tensor in enumerate(weight_list):
            np_list[i] = tensor.numpy()
        return np_list

    def set_param_values_np(self, param_values):
        tf2_list = list()
        for i, tensor in enumerate(param_values):
            tf2_list[i] = tf.convert_to_tensor(tensor, dtype=tf.float32)
        self.set_weights(tf2_list)

    def copy(self):
        copy = Serializable.clone(self)
        tfu.copy_model_params_from_to(self, copy)
        return copy

    def save_init_params(self, locals):
        """
        Should call this FIRST THING in the __init__ method if you ever want
        to serialize or clone this network.
        Usage:
        ```
        def __init__(self, ...):
            self.init_serialization(locals())
            ...
        ```
        :param locals:
        :return:
        """
        Serializable.quick_init(self, locals)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.set_param_values(d["params"])

    def regularizable_parameters(self):
        """
        Return generator of regularizable parameters. Right now, all non-flat
        vectors are assumed to be regularizabled, presumably because only
        biases are flat.
        :return:
        """
        for param in self.parameters():
            if len(param.size()) > 1:
                yield param

    def eval_np(self, *args, **kwargs):
        """
        Eval this module with a numpy interface
        Same as a call to __call__ except all Variable input/outputs are
        replaced with numpy equivalents.
        Assumes the output is either a single object or a tuple of objects.
        """
        tf2_args = tuple(tf2_ify(x) for x in args)
        tf2_kwargs = {k: tf2_ify(v) for k, v in kwargs.items()}
        outputs = self.__call__(*tf2_args, **tf2_kwargs)
        if isinstance(outputs, tuple):
            return tuple(np_ify(x) for x in outputs)
        else:
            return np_ify(outputs)


def tf2_ify(np_array_or_other):
    if isinstance(np_array_or_other, np.ndarray):
        return tf.convert_to_tensor(np_array_or_other)
    else:
        return np_array_or_other


def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, Variable) or isinstance(tensor_or_other, Tensor):
        return tensor_or_other.numpy()
    else:
        return tensor_or_other
