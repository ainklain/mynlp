
import numpy as np
import os
import pandas as pd
import pickle
import time

# v2.1 test
from ts_mini.config_mini import Config
from ts_mini.features_mini import FeatureNew
from ts_mini.data_process_v2_1_mini import DataGeneratorDynamic, DataScheduler
from ts_mini.model_mini import TSModel

configs = Config()
features_cls = FeatureNew(configs)


# scheduler test

ds = DataScheduler(configs, features_cls)
ds.set_idx(6500)
trainset = ds._dataset('train')
evalset = ds._dataset('eval')

model = TSModel(configs, features_cls, weight_scheme=configs.weight_scheme)

ds.train(model, trainset, evalset)











#data generator TEST

dg_st = time.time()
dg = DataGeneratorDynamic(features_cls, configs.data_type, configs.univ_type)
dg_et = time.time()
print("dg set: {} sec".format(dg_et - dg_st))

# date_i = 6250

data_path = './data/{}_{}'.format(configs.univ_type, configs.sampling_days)
os.makedirs(data_path, exist_ok=True)

# feature_keys, label_keys = define_keys()
key_lists = define_keys()
# load and save data
# for date_i in range(4000, len(dg.date_), configs.sampling_days):
input_enc, output_dec, target_dec, additional_info = [], [], [], []
for date_i in range(4000, len(dg.date_), configs.sampling_days):
    loop_st = time.time()
    file_nm = os.path.join(data_path, '{}.pkl'.format(date_i))
    if os.path.exists(file_nm):
        result = pickle.load(open(file_nm, 'rb'))
    else:
        result = dg.sample_data(date_i)
        if result is False:
            continue
        pickle.dump(result, open(os.path.join(data_path, '{}.pkl'.format(date_i)), 'wb'))

    features_dict, labels_dict, spot_dict = result

    n_assets = len(spot_dict['asset_list'])
    n_features = len(key_lists)
    M = configs.m_days // configs.sampling_days + 1

    question = np.stack([features_dict[key] for key in key_lists], axis=-1)
    question = np.transpose(question, axes=(1, 0, 2))
    assert question.shape == (n_assets, M, n_features)

    answer = np.zeros([n_assets, 2, n_features], dtype=np.float32)

    answer[:, 0, :] = question[:, -1, :]  # temporary
    answer[:, 1, :] = np.stack([labels_dict[key] if labels_dict[key] is not None else np.zeros(n_assets) for key in key_lists], axis=-1)


    input_enc.append(question[:])
    output_dec.append(answer[:, :-1, :])
    target_dec.append(answer[:, 1:, :])
    additional_info.append(spot_dict)

    loop_et = time.time()

    print('[{} / {}] 1loop: {} sec'.format(date_i, dg.date_[date_i], loop_et - loop_st))








