
import numpy as np
import os
import pandas as pd
import pickle
import time

# v2.1 test
from ts_mini.config_mini import Config
from ts_mini.features_mini import FeatureNew, Performance
from ts_mini.data_process_v2_1_mini import DataGeneratorDynamic, DataScheduler
from ts_mini.model_v2_0_mini import TSModel

configs = Config()
features_cls = FeatureNew(configs)


k_days = 20; w_scheme = 'mw'; univ_type='selected'; pred='cslogy'; balancing_method='nothing';head=8
configs.set_kdays(k_days)
configs.balancing_method = balancing_method
configs.f_name = 'kr_mw_rand_{}_{}_{}_{}_h{}_v2_06'.format(k_days, univ_type, balancing_method, pred, head)
configs.train_steps = 100
configs.eval_steps = 100
configs.save_steps = 100
configs.attention_head_size = head
configs.early_stopping_count = 2
configs.weight_scheme = 'mw'  # mw / ew
config_str = configs.export()

# scheduler test

ds = DataScheduler(configs, features_cls)
performer = Performance(configs)
model = TSModel(configs, features_cls, weight_scheme=configs.weight_scheme)

os.makedirs(os.path.join(ds.data_out_path, configs.f_name), exist_ok=True)
with open(os.path.join(ds.data_out_path, configs.f_name, 'config.txt'), 'w') as f:
    f.write(config_str)

if os.path.exists(os.path.join(ds.data_out_path, configs.f_name, configs.f_name + '.pkl')):
    model.load_model(os.path.join(ds.data_out_path, configs.f_name, configs.f_name))

ds.set_idx(5000)
ii = 0
jj = 0

trainset = ds._dataset('train')
evalset = ds._dataset('eval')
testset_insample = ds._dataset('test_insample')
testset = ds._dataset('test')

while not ds.done:
    if ii > 100 or (ii > 1 and model.eval_loss > 10000):
        jj += 1
        ii = 0
        ds.next()

        print("jj: {}".format(jj))
        trainset = ds._dataset('train')
        evalset = ds._dataset('eval')
        testset_insample = ds._dataset('test_insample')
        testset = ds._dataset('test')

    # if trainset is None:
    #     trainset = ds._dataset('train')
    #     evalset = ds._dataset('eval')

    if ii > 0:
        is_trained = ds.train(model, trainset, evalset
                              , model_name=os.path.join(ds.data_out_path, configs.f_name, configs.f_name)
                              , epoch=True)

        if is_trained is not False:
            model.save_model(os.path.join(ds.data_out_path, configs.f_name, configs.f_name, str(ds.base_idx), configs.f_name))

    ds.test(performer, model, testset,
            use_label=True,
            out_dir=os.path.join(ds.data_out_path, configs.f_name, str(jj), 'test'),
            file_nm='test_{}.png'.format(ii),
            ylog=False,
            # save_type='csv',
            table_nm='kr_weekly_score_temp',
            t_stepsize=configs.k_days // configs.sampling_days)

    ii += 1


