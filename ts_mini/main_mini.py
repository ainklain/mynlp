
from ts_mini.config_mini import Config
from ts_mini.model_mini import TSModel
from ts_mini.features_mini import Feature
from ts_mini.data_process_v2_0_mini import DataScheduler

import os
import numpy as np

def main(k_days, pred, univ_type, balancing_method):
    # k_days = 20; w_scheme = 'mw'; univ_type='selected'; pred='std'; balancing_method='once'
    ts_configs = Config()
    ts_configs.set_kdays(k_days, pred=pred)
    if k_days == 5:
        ts_configs.m_days = 60
    else:
        ts_configs.m_days = 120

    ts_configs.balancing_method = balancing_method
    ts_configs.f_name = 'kr_mw_{}_{}_{}_{}_002'.format(k_days, univ_type, balancing_method, pred)  #: kr every
    ts_configs.train_steps = 10000
    ts_configs.eval_steps = 500
    ts_configs.early_stopping_count = 5
    ts_configs.weight_scheme = 'mw'  # mw / ew
    config_str = ts_configs.export()
    # get data for all assets and dates
    features_cls = Feature(ts_configs)

    ds = DataScheduler(ts_configs, features_cls, data_type='kr_stock', univ_type=univ_type)
    model = TSModel(ts_configs, features_cls, weight_scheme=ts_configs.weight_scheme)
    # ts_configs.f_name = 'kr_mtl_dg_dynamic_2_0_90'  #: kr every

    os.makedirs(os.path.join(ds.data_out_path, ts_configs.f_name), exist_ok=True)
    with open(os.path.join(ds.data_out_path, ts_configs.f_name, 'config.txt'), 'w') as f:
        f.write(config_str)


    if os.path.exists(os.path.join(ds.data_out_path, ts_configs.f_name, ts_configs.f_name + '.pkl')):
        model.load_model(os.path.join(ds.data_out_path, ts_configs.f_name, ts_configs.f_name))

    ds.set_idx(6000)
    ds.test_end_idx = ds.base_idx + 1000
    ii = 0
    while not ds.done:
        is_trained = ds.train(model,
                 train_steps=ts_configs.train_steps,
                 eval_steps=ts_configs.eval_steps,
                 save_steps=200,
                 early_stopping_count=ts_configs.early_stopping_count,
                 model_name=os.path.join(ds.data_out_path, ts_configs.f_name, ts_configs.f_name))

        if is_trained is not False:
            model.save_model(os.path.join(ds.data_out_path, ts_configs.f_name, ts_configs.f_name, str(ds.base_idx), ts_configs.f_name))

            ds.test(model,
                    use_label=True,
                    out_dir=os.path.join(ds.data_out_path, ts_configs.f_name, 'test'),
                    file_nm='test_{}.png'.format(ii),
                    ylog=False,
                    save_type='csv',
                    table_nm='kr_weekly_score_temp',
                    time_step=ts_configs.k_days // ts_configs.sampling_days)

        ds.next()
        ii += 1

i = 0
for k_days in [20, 5, 10]:
    for pred in ['pos', 'std', 'mdd']:
        for univ_type in ['selected', 'all']:
            for balancing_method in ['once', 'each']:
                i += 1
                # if i <= 1:
                #     continue
                print(univ_type, pred, k_days, balancing_method)
                main(k_days, pred, univ_type, balancing_method)

# if __name__ == '__main__':
#     main()
