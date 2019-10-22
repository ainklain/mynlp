
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

# load and save data
et = time.time()
for date_i in range(4000, len(dg.date_), configs.sampling_days):
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

    feature_keys = ['logy_{}'.format(n) for n in [20, 60, 120]]
    feature_keys += ['std_{}'.format(n) for n in [20, 60, 120]]
    feature_keys += ['stdnew_{}'.format(n) for n in [20, 60, 120]]
    feature_keys += ['mdd_{}'.format(n) for n in [20, 60, 120]]
    feature_keys += ['pos_{}'.format(n) for n in [20, 60]]

    label_keys = ['logy_{}'.format(n) for n in [20, 60, 120]]
    label_keys += ['stdnew_{}'.format(n) for n in [20, 60, 120]]
    label_keys += ['pos_{}'.format(n) for n in [20, 60]]

    features_dict[feature_keys[0]]






