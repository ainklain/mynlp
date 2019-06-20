import tensorflow as tf
import numpy as np
from tensorflow import Variable


def soft_update_from_to(source, target, tau):
    soft_weight = list()
    for target_param, param in zip(target.get_weights(), source.get_weights()):
        soft_weight.append(target_param * (1.0 - tau) + param * tau)

    target.set_weights(soft_weight)


def copy_model_params_from_to(source, target):
    target.set_weights(source.get_weights())
    # for target_param, param in zip(target.parameters(), source.parameters()):
    #     target_param.data.copy_(param.data)


def elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return tf.convert_to_tensor(elem_or_tuple, dtype=tf.float32)


def filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v

def np_to_tf2_batch(np_batch):
    return {
        k: elem_or_tuple_to_variable(x)
        for k, x in filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }



def fanin_init(inputs):
    size = inputs.shape()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    # return tensor.data.uniform_(-bound, bound)
    return tf.keras.initializers.RandomUniform(-bound, bound)
#
#
# def fanin_init_weights_like(tensor):
#     size = tensor.size()
#     if len(size) == 2:
#         fan_in = size[0]
#     elif len(size) > 2:
#         fan_in = np.prod(size[1:])
#     else:
#         raise Exception("Shape must be have dimension at least 2.")
#     bound = 1. / np.sqrt(fan_in)
#     new_tensor = FloatTensor(tensor.size())
#     new_tensor.uniform_(-bound, bound)
#     return new_tensor

