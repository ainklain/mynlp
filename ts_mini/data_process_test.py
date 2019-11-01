
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


k_days = 5; w_scheme = 'mw'; univ_type='selected'; pred='cslogy'; balancing_method='nothing';head=2
configs.balancing_method = balancing_method
configs.f_name = 'kr_mw_rand_{}_{}_{}_{}_h{}_v2_03'.format(k_days, univ_type, balancing_method, pred, head)
configs.train_steps = 100
configs.eval_steps = 100
configs.save_steps = 100
configs.attention_head_size = 8
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
            time_step=configs.k_days // configs.sampling_days)

    ii += 1









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
for date_i in range(1500, len(dg.date_), configs.sampling_days):
    loop_st = time.time()
    file_nm = os.path.join(data_path, '{}.pkl'.format(date_i))
    if os.path.exists(file_nm):
        result = pickle.load(open(file_nm, 'rb'))
        ori_pf = dg.features_cls.possible_func[:]
        dg.features_cls.possible_func = ['cslogy', 'csstd']
        result2 = dg.sample_data(date_i)
        dg.features_cls.possible_func = ori_pf

        features_dict, labels_dict, spot_dict = result
        features_dict2, labels_dict2, spot_dict2 = result2
        for key in spot_dict.keys():
            if key == 'asset_list':
                assert spot_dict[key] == spot_dict2[key]
            else:
                assert (spot_dict[key].values == spot_dict2[key].values).all()

        for key in features_dict2.keys():
            features_dict[key] = features_dict2[key]
            labels_dict[key] = labels_dict2[key]

        pickle.dump((features_dict, labels_dict, spot_dict), open(os.path.join(data_path, '{}.pkl'.format(date_i)), 'wb'))
    else:
        result = dg.sample_data(date_i)
        if result is False:
            continue
        pickle.dump(result, open(os.path.join(data_path, '{}.pkl'.format(date_i)), 'wb'))

    loop_et = time.time()

    print('[{} / {}] 1loop: {} sec'.format(date_i, dg.date_[date_i], loop_et - loop_st))

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









