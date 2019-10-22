
import numpy as np
import os
import pandas as pd
import pickle
import time

# v2.1 test
from ts_mini.config_mini import Config
from ts_mini.features_mini import FeatureNew
from ts_mini.data_process_v2_1_mini import DataGeneratorDynamic
configs = Config()
features_cls = FeatureNew(configs)

dg_st = time.time()
dg = DataGeneratorDynamic(features_cls, configs.data_type, configs.univ_type)
dg_et = time.time()
print("dg set: {} sec".format(dg_et - dg_st))

# date_i = 6250

data_path = './data/{}_{}'.format(configs.univ_type, configs.sampling_days)
os.makedirs(data_path, exist_ok=True)

def define_keys():
    feature_keys = ['logy_{}'.format(n) for n in [20, 60, 120]]
    feature_keys += ['std_{}'.format(n) for n in [20, 60, 120]]
    feature_keys += ['stdnew_{}'.format(n) for n in [20, 60, 120]]
    feature_keys += ['mdd_{}'.format(n) for n in [20, 60, 120]]
    feature_keys += ['pos_{}'.format(n) for n in [20, 60]]

    label_keys = ['logy_{}'.format(n) for n in [20, 60, 120]]
    label_keys += ['stdnew_{}'.format(n) for n in [20, 60, 120]]
    label_keys += ['pos_{}'.format(n) for n in [20, 60]]

    return feature_keys, label_keys

feature_keys, label_keys = define_keys()
# load and save data
et = time.time()
# for date_i in range(4000, len(dg.date_), configs.sampling_days):
input_enc, output_dec, target_dec = [], [], []
for date_i in range(4000, 4100, configs.sampling_days):
    file_nm = os.path.join(data_path, '{}.pkl'.format(date_i))
    if os.path.exists(file_nm):
        result = pickle.load(file_nm)
    else:
        result = dg.sample_data(date_i)
        if result is False:
            continue
        print(date_i, dg.date_[date_i])
        pickle.dump(result, open(os.path.join(data_path, '{}.pkl'.format(date_i)), 'wb'))

    features_dict, labels_dict, spot_dict = result

    n_assets = len(spot_dict['asset_list'])
    n_features = len(feature_keys)
    M = configs.m_days // configs.sampling_days + 1

    question = np.stack([features_dict[key] for key in feature_keys], axis=-1)
    question = np.transpose(question, axes=(1, 0, 2))
    assert question.shape == [n_assets, M, n_features]

    answer = np.zeros([n_assets, 2, n_features], dtype=np.float32)

    answer[:, 0, :] = question[:, -1, :]  # temporary
    answer[:, 1, :] = np.stack([labels_dict[key] for key in label_keys if labels_dict[key] is not None else np.zeros(n_assets)], axis=-1)

    assert len(size_adjusted_factor) == n_asset
    assert len(size_adjusted_factor_mktcap) == n_asset

    input_enc, output_dec, target_dec = question[:], answer[:, :-1, :], answer[:, 1:, :]
    additional_info['size_value'] = size_adjusted_factor[:]
    additional_info['mktcap'] = size_adjusted_factor_mktcap[:]
    assert len(additional_info['assets_list']) == len(input_enc)

    assert np.sum(input_enc[:, -1:, :] - output_dec[:, :, :]) == 0







