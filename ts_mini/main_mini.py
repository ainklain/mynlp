
from ts_mini.config_mini import Config
from ts_mini.model_mini import TSModel
from ts_mini.data_process_v2_0_mini import DataScheduler

import os

def main():
    ts_configs = Config()
    # get data for all assets and dates
    ds = DataScheduler(ts_configs, data_type='kr_stock')

    model = TSModel(ts_configs)
    ts_configs.f_name = 'kr_mtl_dg_dynamic_2_1_large_250'  #: kr every

    if os.path.exists(os.path.join(ds.data_out_path, ts_configs.f_name, ts_configs.f_name + '.pkl')):
        model.load_model(os.path.join(ds.data_out_path, ts_configs.f_name, ts_configs.f_name))

    ds.set_idx(4000)
    ds.test_end_idx = ds.base_idx + 1000
    ii = 0
    while not ds.done:
        ds.train(model,
                 train_steps=10000,
                 eval_steps=10,
                 save_steps=200,
                 early_stopping_count=20,
                 model_name=os.path.join(ds.data_out_path, ts_configs.f_name, ts_configs.f_name))

        model.save_model(os.path.join(ds.data_out_path, ts_configs.f_name, ts_configs.f_name, str(ds.base_idx), ts_configs.f_name))
        ds.test(model,
                use_label=True,
                out_dir=os.path.join(ds.data_out_path, ts_configs.f_name, 'test'),
                file_nm='test_{}.png'.format(ii),
                ylog=False,
                save_type='csv',
                table_nm='kr_weekly_score_temp')

        ds.next()
        ii += 1

