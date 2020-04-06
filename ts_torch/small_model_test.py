
import builtins

try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile


import numpy as np
import os
import torch
from torch import nn, optim

from ts_torch.config_torch_mini import Config
from ts_mini.features_mini import FeatureNew
from ts_torch.data_process_torch_mini import DataScheduler, Noise
from ts_torch.model_torch_mini import TSModel
from ts_torch.performance_torch_mini import Performance
from ts_torch import torch_util_mini as tu


def example2():
    # model_predictor_list = ['nmy', 'nmsize']
    model_predictor_list = ['nmy', 'pos_20', 'nmstd']

    # model_predictor_list = ['nmir', 'nmy', 'nmsize', 'pos_20', 'nmstd']
    features_structure = {'regression': {'logp_base': {'logp': [0],
                                                       'logy': [20, 60, 120, 250],
                                                       'std': [20, 60, 120],
                                                       'stdnew': [20, 60],
                                                       'mdd': [20, 60, 120],
                                                       'fft': [100, 3],
                                                       'nmlogy': [20, 60],
                                                       'nmstd': [20, 60],
                                                       'nmy': [20, 60],
                                                       'nmir': [20, 60]
                                                       },
                    'size_base': {'nmsize': [0]},
                    'turnover_base': {'nmturnover': [0], 'tsturnover': [0]},
                        # 'wlogy_base': {'nmwlogy': [0], 'wlogy': [0]},
                        # 'wlogyrnk_base': {'nmwlogyrnk': [0]},

                    },
     'classification': {'logp_base': {'pos': [20, 60, 120, 250]}}}

    return model_predictor_list, features_structure


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


def run():
    i = 10;
    country = 'kr';
    model_predictor_list, features_structure = example2()
    # use_swa = True
    configs = Config(use_macro=False, use_swa=False)

    k_days = 20;
    pred = model_predictor_list[0];
    # country = 'kr';
    configs.set_datatype(country + '_stock')
    configs.sampling_days = k_days
    configs.set_kdays(k_days, pred=pred, model_predictor_list=model_predictor_list,
                      features_structure=features_structure)
    configs.n_heads = 8
    configs.f_name = '{}_{}_nswa_profile_test_00{}'.format(country, k_days, i)
    num_epochs = 5

    configs.early_stopping_count = 1
    configs.learning_rate = 5e-4
    configs.update_comment = 'single pred per task'
    config_str = configs.export()

    features_cls = FeatureNew(configs)
    ds = DataScheduler(configs, features_cls)
    ds.set_idx(8250)
    # ds.test_end_idx = min(ds.test_end_idx + 250, ds.data_generator.max_length - ds.configs.k_days - 1)
    ds.test_end_idx = min(ds.test_end_idx + 250, ds.data_generator.max_length - 1)

    os.makedirs(os.path.join(ds.data_out_path), exist_ok=True)

    model = TSModel(configs, features_cls, weight_scheme=configs.weight_scheme)

    performer = Performance(configs)
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
    scheduler = None

    # train test
    ds.train(model, optimizer, scheduler, performer, num_epochs=num_epochs)

    # ds test
    early_stopping_count = ds.configs.early_stopping_count
    ep = 0
    train_loss = ds.step_epoch(ep, model, optimizer, scheduler=scheduler, is_train=True)


    # model test
    mode = 'train'
    model.train()
    dataloader, features_list = ds._dataloader(mode)
    total_loss = 0
    i = 0
    features, labels, add_infos = next(iter(dataloader))

    features, labels, add_infos = to_device(tu.device, [features, labels, add_infos])
    with torch.set_grad_enabled(True):
        features_with_noise = {'input': features['input'], 'output': features['output']}
        labels_with_noise = labels
        labels_mtl_noise = ds.labels_torch(features_list, labels_with_noise, add_infos)
        features_with_noise = Noise.adversarial_noise(features_with_noise, labels_mtl_noise, model)

        labels_mtl = ds.labels_torch(features_list, labels_with_noise, add_infos)
        pred, loss_each = model.forward_with_loss(features_with_noise, labels_mtl)

        losses = 0
        for key in loss_each.keys():
            losses += loss_each[key].mean()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses
