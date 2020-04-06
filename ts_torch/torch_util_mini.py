
import numpy as np
import torch


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def torch_ify(np_array_or_other):
    if isinstance(np_array_or_other, np.ndarray):
        return from_numpy(np_array_or_other)
    else:
        return np_array_or_other


def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, torch.autograd.Variable):
        return get_numpy(tensor_or_other)
    else:
        return tensor_or_other


def to_device(device, list_to_device):
    assert isinstance(list_to_device, list)

    for i, value_ in enumerate(list_to_device):
        if isinstance(value_, dict):
            for key in value_.keys():
                value_[key] = value_[key].to(device)
        elif isinstance(value_, torch.Tensor):
            list_to_device[i] = value_.to(device)
        else:
            raise NotImplementedError

    return list_to_device