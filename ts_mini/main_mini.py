
from ts_mini.config_mini import Config
from ts_mini.model_mini import TSModel
from ts_mini.features_mini import Feature
from ts_mini.data_process_v2_0_mini import DataScheduler

import os
import numpy as np

def main():
    ts_configs = Config()
    ts_configs.set_kdays(20)

    ts_configs.f_name = 'kr_model_mw_20_wo_beta_day1'  #: kr every
    ts_configs.train_steps = 10000
    ts_configs.eval_steps = 200
    ts_configs.early_stopping_count = 5
    ts_configs.weight_scheme = 'mw'  # mw / ew
    config_str = ts_configs.export()
    # get data for all assets and dates
    features_cls = Feature(ts_configs)

    ds = DataScheduler(ts_configs, features_cls, data_type='kr_stock', univ_type='all')
    model = TSModel(ts_configs, features_cls, weight_scheme=ts_configs.weight_scheme)
    # ts_configs.f_name = 'kr_mtl_dg_dynamic_2_0_90'  #: kr every

    os.makedirs(os.path.join(ds.data_out_path, ts_configs.f_name), exist_ok=True)
    with open(os.path.join(ds.data_out_path, ts_configs.f_name, 'config.txt'), 'w') as f:
        f.write(config_str)

    if os.path.exists(os.path.join(ds.data_out_path, ts_configs.f_name, ts_configs.f_name + '.pkl')):
        model.load_model(os.path.join(ds.data_out_path, ts_configs.f_name, ts_configs.f_name))

    # ds.set_idx(5600)
    ds.set_idx(5000)
    ds.test_end_idx = ds.base_idx + 1000
    ii = 0
    while not ds.done:
        ds.train(model,
                 train_steps=ts_configs.train_steps,
                 eval_steps=ts_configs.eval_steps,
                 save_steps=200,
                 early_stopping_count=ts_configs.early_stopping_count,
                 model_name=os.path.join(ds.data_out_path, ts_configs.f_name, ts_configs.f_name))

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

# if __name__ == '__main__':
#     main()
