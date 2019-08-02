
# from dbmanager import SqlManager
from ts_mini.utils_mini import *
# from ts_mini.features_mini import processing # processing_split, labels_for_mtl

import pandas as pd
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split


class DataScheduler:
    def __init__(self, configs, features_cls, data_type='kr_stock', univ_type='all'):
        # make a directory for outputs
        self.data_out_path = os.path.join(os.getcwd(), configs.data_out_path)
        os.makedirs(self.data_out_path, exist_ok=True)

        # self.data_generator = DataGenerator(data_type)    # infocode
        self.data_generator = DataGeneratorDynamic(features_cls, data_type, univ_type)    # infocode

        self.train_set_length = configs.train_set_length
        self.retrain_days = configs.retrain_days
        self.m_days = configs.m_days
        self.k_days = configs.k_days
        self.sampling_days = configs.sampling_days

        self.train_batch_size = configs.batch_size
        self.eval_batch_size = 256
        self.trainset_rate = configs.trainset_rate

        self.features_cls = features_cls
        self._initialize()

    def _initialize(self):
        self.base_idx = self.train_set_length

        self.train_begin_idx = 0
        self.eval_begin_idx = int(self.base_idx * self.trainset_rate)
        self.test_begin_idx = self.base_idx - self.m_days
        self.test_end_idx = self.base_idx + self.retrain_days

    def set_idx(self, base_idx):
        self.base_idx = base_idx

        self.train_begin_idx = np.max([0, base_idx - self.train_set_length])
        self.eval_begin_idx = int(self.train_set_length * self.trainset_rate) + self.train_begin_idx
        self.test_begin_idx = self.base_idx - self.m_days
        self.test_end_idx = self.base_idx + self.retrain_days

    def get_data_params(self, mode='train'):
        dg = self.data_generator
        data_params = dict()
        data_params['sampling_days'] = self.sampling_days
        data_params['m_days'] = self.m_days
        data_params['k_days'] = self.k_days
        data_params['calc_length'] = 250
        data_params['univ_idx'] = self.test_begin_idx
        if mode == 'train':
            start_idx = self.train_begin_idx + self.m_days
            end_idx = self.eval_begin_idx - self.k_days
            data_params['balance_class'] = True
            data_params['label_type'] = 'trainable_label'   # trainable: calc_length 반영
        elif mode == 'eval':
            start_idx = self.eval_begin_idx + self.m_days
            end_idx = self.test_begin_idx - self.k_days
            data_params['balance_class'] = True
            data_params['label_type'] = 'trainable_label'   # trainable: calc_length 반영
        elif mode == 'test':
            start_idx = self.test_begin_idx + self.m_days
            # start_idx = self.test_begin_idx
            end_idx = self.test_end_idx
            data_params['balance_class'] = False
            data_params['label_type'] = 'test_label'        # test: 예측하고자 하는 것만 반영 (k_days)
        elif mode == 'predict':
            start_idx = self.test_begin_idx + self.m_days
            # start_idx = self.test_begin_idx
            end_idx = self.test_end_idx
            data_params['balance_class'] = False
            data_params['label_type'] = None            # label 없이 과거데이터만으로 스코어 산출
        else:
            raise NotImplementedError

        print("start idx:{} ({}) / end idx: {} ({})".format(start_idx, dg.date_[start_idx], end_idx, dg.date_[end_idx]))

        return start_idx, end_idx, data_params

    def _dataset(self, mode='train'):
        input_enc, output_dec, target_dec = [], [], []
        features_list = []
        additional_infos = []
        start_idx, end_idx, data_params = self.get_data_params(mode)

        for i, d in enumerate(range(start_idx, end_idx, self.sampling_days)):
            _sampled_data = self.data_generator.sample_inputdata_split_new(d, **data_params)
            if _sampled_data is False:
                continue
            else:
                tmp_ie, tmp_od, tmp_td, features_list, additional_info = _sampled_data
            input_enc.append(tmp_ie)
            output_dec.append(tmp_od)
            target_dec.append(tmp_td)
            additional_infos.append(additional_info)

        if len(input_enc) == 0:
            return False

        if not mode in ['test', 'predict']:
            input_enc = np.concatenate(input_enc, axis=0)
            output_dec = np.concatenate(output_dec, axis=0)
            target_dec = np.concatenate(target_dec, axis=0)

        start_date = self.data_generator.date_[start_idx]
        end_date = self.data_generator.date_[end_idx]
        return input_enc, output_dec, target_dec, features_list, additional_infos, start_date, end_date

    def train(self,
              model,
              train_steps=1,
              eval_steps=10,
              save_steps=50,
              early_stopping_count=10,
              model_name='ts_model_v1.0'):

        # make directories for graph results (both train and test one)
        train_out_path = os.path.join(self.data_out_path, model_name, '{}'.format(self.base_idx))
        os.makedirs(train_out_path, exist_ok=True)

        _train_dataset = self._dataset('train')
        _eval_dataset = self._dataset('eval')
        if _train_dataset is False or _eval_dataset is False:
            print('[train] no train/eval data')
            return False

        train_input_enc, train_output_dec, train_target_dec, features_list, _, _, _ = _train_dataset
        eval_input_enc, eval_output_dec, eval_target_dec, _, _, _, _ = _eval_dataset

        assert np.sum(train_input_enc[:, -1, :] - train_output_dec[:, 0, :]) == 0
        assert np.sum(eval_input_enc[:, -1, :] - eval_output_dec[:, 0, :]) == 0

        # K > 1인 경우 미래데이터 안 땡겨쓰게.
        train_new_output = np.zeros_like(train_output_dec)
        train_new_output[:, 0, :] = train_output_dec[:, 0, :]
        eval_new_output = np.zeros_like(eval_output_dec)
        eval_new_output[:, 0, :] = eval_output_dec[:, 0, :]

        train_dataset = dataset_process(train_input_enc, train_new_output, train_target_dec, batch_size=self.train_batch_size)
        eval_dataset = dataset_process(eval_input_enc, eval_new_output, eval_target_dec, batch_size=self.eval_batch_size, iter_num=1)
        print("train step: {}  eval step: {}".format(len(train_input_enc) // self.train_batch_size,
                                                     len(eval_input_enc) // self.eval_batch_size))
        for i, (features, labels) in enumerate(train_dataset.take(train_steps)):
            print_loss = False
            if i % save_steps == 0:
                model.save_model(model_name)

            if i % eval_steps == 0:
                print_loss = True
                model.evaluate_mtl(eval_dataset, features_list, steps=len(eval_input_enc) // self.eval_batch_size)

                print("[t: {} / i: {}] min_eval_loss:{} / count:{}".format(self.base_idx, i, model.eval_loss, model.eval_count))
                if model.eval_count >= early_stopping_count:
                    print("[t: {} / i: {}] train finished.".format(self.base_idx, i))
                    model.weight_to_optim()
                    model.save_model(model_name)
                    break

            labels_mtl = self.features_cls.labels_for_mtl(features_list, labels)
            model.train_mtl(features, labels_mtl, print_loss=print_loss)

    def test(self, model, use_label=True, out_dir=None, file_nm='out.png', ylog=False, save_type=None, table_nm=None, time_step=1):
        if out_dir is None:
            test_out_path = os.path.join(self.data_out_path, '{}/test'.format(self.base_idx))
        else:
            test_out_path = out_dir

        os.makedirs(test_out_path, exist_ok=True)
        if file_nm is None:
            save_file_name = '{}/{}'.format(test_out_path, '_all.png')
        else:
            save_file_name = '{}/{}'.format(test_out_path, file_nm)

        if use_label:
            _dataset_list = self._dataset('test')
            if _dataset_list is False:
                print('[test] no test data')
                return False
            self.features_cls.predict_plot_mtl_cross_section_test(model, _dataset_list,  save_dir=save_file_name, ylog=ylog, time_step=time_step)

        if save_type is not None:
            _dataset_list = self._dataset('predict')
            if _dataset_list is False:
                print('[predict] no test data')
                return False

            if save_type == 'db':
                self.save_score_to_db(model, _dataset_list, table_nm=table_nm)
            elif save_type == 'csv':
                self.save_score_to_csv(model, _dataset_list, out_dir=test_out_path)

    def save_score_to_csv(self, model, dataset_list, out_dir=None):
        input_enc_list, output_dec_list, _, _, additional_infos, start_date, _ = dataset_list
        df_infos = pd.DataFrame(columns={'start_d', 'base_d', 'infocode', 'score'})
        for i, (input_enc_t, output_dec_t) in enumerate(zip(input_enc_list, output_dec_list)):
            assert np.sum(input_enc_t[:, -1, :] - output_dec_t[:, 0, :]) == 0
            assert np.sum(output_dec_t[:, 1:, :]) == 0
            features = {'input': input_enc_t, 'output': output_dec_t}
            predictions = model.predict_mtl(features)
            df_infos = pd.concat([df_infos, pd.DataFrame({
                'start_d': start_date,
                'base_d': additional_infos[i]['date'],
                'infocode': additional_infos[i]['assets_list'],
                'score': predictions[self.features_cls.pred_feature][:, 0, 0]})], ignore_index=True, sort=True)
        df_infos.to_csv(os.path.join(out_dir, 'out_{}.csv'.format(str(start_date))))

    def save_score_to_db(self, model, dataset_list, table_nm='kr_weekly_score_temp'):
        if table_nm is None:
            table_nm = 'kr_weekly_score_temp'

        input_enc_list, output_dec_list, _, _, additional_infos, start_date, _ = dataset_list
        df_infos = pd.DataFrame(columns={'start_d', 'base_d', 'infocode', 'score'})
        for i, (input_enc_t, output_dec_t) in enumerate(zip(input_enc_list, output_dec_list)):
            assert np.sum(input_enc_t[:, -1, :] - output_dec_t[:, 0, :]) == 0
            assert np.sum(output_dec_t[:, 1:, :]) == 0
            features = {'input': input_enc_t, 'output': output_dec_t}
            predictions = model.predict_mtl(features)
            df_infos = pd.concat([df_infos, pd.DataFrame({
                'start_d': start_date,
                'base_d': additional_infos[i]['date'],
                'infocode': additional_infos[i]['assets_list'],
                'score': predictions[self.features_cls.pred_feature][:, 0, 0]})], ignore_index=True, sort=True)

            # db insert
            # sqlm = SqlManager()
            # sqlm.set_db_name('passive')
            # sqlm.db_insert(df_infos[['start_d', 'base_d', 'infocode', 'score']], table_nm, fast_executemany=True)

    def next(self):
        self.base_idx += self.retrain_days
        self.train_begin_idx += self.retrain_days
        self.eval_begin_idx += self.retrain_days
        self.test_begin_idx += self.retrain_days
        self.test_end_idx = min(self.test_end_idx + self.retrain_days, self.data_generator.max_length - self.k_days - 1)

    def get_date(self):
        return self.date_[self.base_d]

    @property
    def date_(self):
        return self.data_generator.date_

    @property
    def done(self):
        # if self.test_end_idx > self.data_generator.max_length:
        if self.test_end_idx <= self.test_begin_idx:
            return True
        else:
            return False


class DataSchedulerCrossSection:
    def __init__(self, configs, data_type='kr_stock'):
        # make a directory for outputs
        self.data_out_path = os.path.join(os.getcwd(), configs.data_out_path)
        os.makedirs(self.data_out_path, exist_ok=True)

        # self.data_generator = DataGenerator(data_type)    # infocode
        self.data_generator = DataGeneratorDynamic(data_type)    # infocode

        self.train_set_length = configs.train_set_length
        self.retrain_days = configs.retrain_days
        self.m_days = configs.m_days
        self.k_days = configs.k_days
        self.sampling_days = configs.sampling_days

        self.train_batch_size = configs.batch_size
        self.trainset_rate = configs.trainset_rate
        self._initialize()

    def _initialize(self):
        self.base_idx = self.train_set_length

        self.train_begin_idx = 0
        # self.eval_begin_idx = int(self.base_idx * self.trainset_rate)
        self.test_begin_idx = self.base_idx - self.m_days
        self.test_end_idx = self.base_idx + self.retrain_days

    def set_idx(self, base_idx):
        self.base_idx = base_idx

        self.train_begin_idx = np.max([0, base_idx - self.train_set_length])
        # self.eval_begin_idx = int(self.train_set_length * self.trainset_rate) + self.train_begin_idx
        self.test_begin_idx = self.base_idx - self.m_days
        self.test_end_idx = self.base_idx + self.retrain_days

    def get_data_params(self, mode='train'):
        dg = self.data_generator
        data_params = dict()
        data_params['sampling_days'] = self.sampling_days
        data_params['m_days'] = self.m_days
        data_params['k_days'] = self.k_days
        data_params['univ_idx'] = self.test_begin_idx
        # data_params['univ_idx'] = None
        if mode == 'train':
            start_idx = self.train_begin_idx + self.m_days
            end_idx = self.test_begin_idx - self.k_days
            data_params['balance_class'] = True
            data_params['use_label'] = True
        elif mode == 'eval':
            start_idx = self.eval_begin_idx + self.m_days
            end_idx = self.test_begin_idx - self.k_days
            data_params['balance_class'] = True
            data_params['use_label'] = True
        elif mode == 'test':
            start_idx = self.test_begin_idx + self.m_days
            # start_idx = self.test_begin_idx
            end_idx = self.test_end_idx
            data_params['balance_class'] = False
            data_params['use_label'] = True
        elif mode == 'predict':
            start_idx = self.test_begin_idx + self.m_days
            # start_idx = self.test_begin_idx
            end_idx = self.test_end_idx
            data_params['balance_class'] = False
            data_params['use_label'] = False
        else:
            raise NotImplementedError

        print("start idx:{} ({}) / end idx: {} ({})".format(start_idx, dg.date_[start_idx], end_idx, dg.date_[end_idx]))

        return start_idx, end_idx, data_params

    def _dataset(self, mode='train'):
        input_enc, output_dec, target_dec = [], [], []
        features_list = []
        additional_infos = []
        start_idx, end_idx, data_params = self.get_data_params(mode)

        for i, d in enumerate(range(start_idx, end_idx, self.sampling_days)):
            _sampled_data = self.data_generator.sample_inputdata_split(d, **data_params)
            if _sampled_data is False:
                continue
            else:
                tmp_ie, tmp_od, tmp_td, features_list, additional_info = _sampled_data
            input_enc.append(tmp_ie)
            output_dec.append(tmp_od)
            target_dec.append(tmp_td)
            additional_infos.append(additional_info)

        if len(input_enc) == 0:
            return False

        if not mode in ['test', 'predict']:
            input_enc = np.concatenate(input_enc, axis=0)
            output_dec = np.concatenate(output_dec, axis=0)
            target_dec = np.concatenate(target_dec, axis=0)

        start_date = self.data_generator.date_[start_idx]
        end_date = self.data_generator.date_[end_idx]
        return input_enc, output_dec, target_dec, features_list, additional_infos, start_date, end_date

    def train(self,
              model,
              train_steps=1,
              eval_steps=10,
              save_steps=50,
              early_stopping_count=10,
              model_name='ts_model_v1.0'):

        # make directories for graph results (both train and test one)
        train_out_path = os.path.join(self.data_out_path, model_name, '{}'.format(self.base_idx))
        os.makedirs(train_out_path, exist_ok=True)

        _train_dataset = self._dataset('train')
        _eval_dataset = self._dataset('eval')
        if _train_dataset is False or _eval_dataset is False:
            print('[train] no train/eval data')
            return False

        train_input_enc, train_output_dec, train_target_dec, features_list, _, _, _ = _train_dataset
        eval_input_enc, eval_output_dec, eval_target_dec, _, _, _, _ = _eval_dataset

        assert np.sum(train_input_enc[:, -1, :] - train_output_dec[:, 0, :]) == 0
        assert np.sum(eval_input_enc[:, -1, :] - eval_output_dec[:, 0, :]) == 0

        # K > 1인 경우 미래데이터 안 땡겨쓰게.
        train_new_output = np.zeros_like(train_output_dec)
        train_new_output[:, 0, :] = train_output_dec[:, 0, :]
        eval_new_output = np.zeros_like(eval_output_dec)
        eval_new_output[:, 0, :] = eval_output_dec[:, 0, :]

        train_dataset = dataset_process(train_input_enc, train_new_output, train_target_dec, batch_size=self.train_batch_size)
        eval_dataset = dataset_process(eval_input_enc, eval_new_output, eval_target_dec, batch_size=self.train_batch_size, iter_num=1)
        for i, (features, labels) in enumerate(train_dataset.take(train_steps)):
            print_loss = False
            if i % save_steps == 0:
                model.save_model(model_name)

            if i % eval_steps == 0:
                print_loss = True
                model.evaluate_mtl(eval_dataset, features_list, steps=50)

                print("[t: {} / i: {}] min_eval_loss:{} / count:{}".format(self.base_idx, i, model.eval_loss, model.eval_count))
                if model.eval_count >= early_stopping_count:
                    print("[t: {} / i: {}] train finished.".format(self.base_idx, i))
                    model.weight_to_optim()
                    model.save_model(model_name)
                    break

            labels_mtl = self.features_cls.labels_for_mtl(features_list, labels)
            model.train_mtl(features, labels_mtl, print_loss=print_loss)

    def test(self, model, use_label=True, out_dir=None, file_nm='out.png', ylog=False, save_type=None, table_nm=None):
        if out_dir is None:
            test_out_path = os.path.join(self.data_out_path, '{}/test'.format(self.base_idx))
        else:
            test_out_path = out_dir

        os.makedirs(test_out_path, exist_ok=True)
        if file_nm is None:
            save_file_name = '{}/{}'.format(test_out_path, '_all.png')
        else:
            save_file_name = '{}/{}'.format(test_out_path, file_nm)

        if use_label:
            _dataset_list = self._dataset('test')
            predict_plot_mtl_cross_section_test(model, _dataset_list,  save_dir=save_file_name, ylog=ylog, eval_type='pos')

        if save_type is not None:
            _dataset_list = self._dataset('predict')
            if save_type == 'db':
                self.save_score_to_db(model, _dataset_list, table_nm=table_nm)
            elif save_type == 'csv':
                self.save_score_to_csv(model, _dataset_list, out_dir=test_out_path)


    def save_score_to_csv(self, model, dataset_list, out_dir=None):
        input_enc_list, output_dec_list, _, _, additional_infos, start_date, _ = dataset_list
        df_infos = pd.DataFrame(columns={'start_d', 'base_d', 'infocode', 'score'})
        for i, (input_enc_t, output_dec_t) in enumerate(zip(input_enc_list, output_dec_list)):
            assert np.sum(input_enc_t[:, -1, :] - output_dec_t[:, 0, :]) == 0
            assert np.sum(output_dec_t[:, 1:, :]) == 0
            features = {'input': input_enc_t, 'output': output_dec_t}
            predictions = model.predict_mtl(features)
            df_infos = pd.concat([df_infos, pd.DataFrame({
                'start_d': start_date,
                'base_d': additional_infos[i]['date'],
                'infocode': additional_infos[i]['assets_list'],
                'score': predictions['pos'][:, 0, 0]})], ignore_index=True, sort=True)
        df_infos.to_csv(os.path.join(out_dir, 'out_{}.csv'.format(str(start_date))))

    def save_score_to_db(self, model, dataset_list, table_nm='kr_weekly_score_temp'):
        if table_nm is None:
            table_nm = 'kr_weekly_score_temp'

        input_enc_list, output_dec_list, _, _, additional_infos, start_date, _ = dataset_list
        df_infos = pd.DataFrame(columns={'start_d', 'base_d', 'infocode', 'score'})
        for i, (input_enc_t, output_dec_t) in enumerate(zip(input_enc_list, output_dec_list)):
            assert np.sum(input_enc_t[:, -1, :] - output_dec_t[:, 0, :]) == 0
            assert np.sum(output_dec_t[:, 1:, :]) == 0
            features = {'input': input_enc_t, 'output': output_dec_t}
            predictions = model.predict_mtl(features)
            df_infos = pd.concat([df_infos, pd.DataFrame({
                'start_d': start_date,
                'base_d': additional_infos[i]['date'],
                'infocode': additional_infos[i]['assets_list'],
                'score': predictions['pos'][:, 0, 0]})], ignore_index=True, sort=True)

            # db insert
            # sqlm = SqlManager()
            # sqlm.set_db_name('passive')
            # sqlm.db_insert(df_infos[['start_d', 'base_d', 'infocode', 'score']], table_nm, fast_executemany=True)

    def next(self):
        self.base_idx += self.retrain_days
        self.train_begin_idx += self.retrain_days
        self.eval_begin_idx += self.retrain_days
        self.test_begin_idx += self.retrain_days
        self.test_end_idx = min(self.test_end_idx + self.retrain_days, self.data_generator.max_length - self.k_days - 1)

    def get_date(self):
        return self.date_[self.base_d]

    @property
    def date_(self):
        return self.data_generator.date_

    @property
    def done(self):
        # if self.test_end_idx > self.data_generator.max_length:
        if self.test_end_idx <= self.test_begin_idx:
            return True
        else:
            return False



class DataGeneratorDynamic:
    def __init__(self, features_cls, data_type='kr_stock', univ_type='all'):
        if data_type == 'kr_stock':
            data_path = './data/kr_close_y_90.csv'
            data_df_temp = pd.read_csv(data_path)
            data_df_temp = data_df_temp[data_df_temp.infocode > 0]

            date_temp = data_df_temp[['date_', 'infocode']].groupby('date_').count()
            date_temp = date_temp[date_temp.infocode >= 10]
            date_temp.columns = ['cnt']

            self.date_ = list(date_temp.index)
            self.data_df = pd.merge(date_temp, data_df_temp, on='date_')  # 최소 10종목 이상 존재 하는 날짜만
            self.data_df['y'] = self.data_df['y'] + 1
            self.data_df['cum_y'] = self.data_df[['date_', 'infocode', 'y']].groupby('infocode').cumprod(axis=0)

            self.univ_type = univ_type
            if univ_type == 'all':
                self.data_code = pd.read_csv('./data/kr_sizeinfo_90.csv')
                self.data_code = self.data_code[self.data_code.infocode > 0]
            elif univ_type == 'selected':
                self.data_code = pd.read_csv('./data/kr_univ_monthly.csv')
                self.data_code = self.data_code[self.data_code.infocode > 0]
            self.base_d = None

            self.features_cls = features_cls

    def _set_df_pivoted(self, univ_idx):
        date_arr = self.data_code.eval_d.unique()
        if np.sum(date_arr <= self.date_[univ_idx]) == 0:
            return False

        base_d = max(date_arr[date_arr <= self.date_[univ_idx]])

        if self.base_d != base_d:
            print('base_d changed {} -> {}'.format(self.base_d, base_d))
            self.base_d = base_d

            univ_list = list(self.data_code[self.data_code.eval_d == base_d]['infocode'].to_numpy(dtype=np.int32))

            df_pivoted = self.data_df[['date_', 'infocode', 'cum_y']].pivot(index='date_', columns='infocode')
            df_pivoted.columns = df_pivoted.columns.droplevel(0).to_numpy(dtype=np.int32)
            univ_list_selected = sorted(list(set(univ_list).intersection(set(df_pivoted.columns))))

            self.df_pivoted = df_pivoted[univ_list_selected]
            self.df_size = self.data_code[self.data_code.eval_d == base_d][['infocode', 'mktcap']].set_index('infocode').loc[univ_list_selected, :]
            self.df_size['rnk'] = self.df_size.mktcap.rank() / len(self.df_size)
        return True

    def sample_inputdata_split_new(self, base_idx, sampling_days=5, m_days=60, k_days=20, calc_length=250, balance_class=True,label_type='trainable_label', univ_idx=None):
        if label_type != 'trainable_label':
            balance_class = False

        if univ_idx is None:
            univ_idx = base_idx
        is_data_exist = self._set_df_pivoted(univ_idx)

        # 미래데이터 원천 제거
        if not is_data_exist:
            return False

        df_selected_data = self.df_pivoted[(self.df_pivoted.index >= self.date_[base_idx - m_days - calc_length])
                                           & (self.df_pivoted.index <= self.date_[base_idx])]

        # 현재기준 데이터 정제
        df_for_data = df_selected_data.ix[:, np.sum(~df_selected_data.isna(), axis=0) >= len(df_selected_data.index) * 0.9]  # 90% 이상 데이터 존재
        df_for_data.ffill(axis=0, inplace=True)
        df_for_data.bfill(axis=0, inplace=True)
        df_for_data = df_for_data.ix[:, np.sum(df_for_data.isna(), axis=0) == 0]    # 맨 앞쪽 NA 제거

        if df_for_data.empty:
            return False

        additional_info = {'date': self.date_[base_idx], 'assets_list': list(df_for_data.columns)}
        features_list, features_sampled_data, _ = self.features_cls.processing_split_new(df_for_data, m_days=m_days, k_days=k_days, sampling_days=sampling_days, calc_length=calc_length, label_type=None)
        # features_for_data = features_for_data[calc_length:]

        M = m_days // sampling_days

        assert features_sampled_data.shape[0] == M

        if label_type == 'trainable_label':
            # 미래데이터 포함 라벨 생성
            df_selected_label = self.df_pivoted[(self.df_pivoted.index >= self.date_[base_idx - m_days - calc_length])
                                                & (self.df_pivoted.index <= self.date_[base_idx + k_days + calc_length])]

            # 현재기준으로 정제된 종목 기준 라벨 데이터 생성 및 정제
            df_for_label = df_selected_label.loc[:, df_for_data.columns]
            df_for_label.ffill(axis=0, inplace=True)
            df_for_label.bfill(axis=0, inplace=True)
            df_for_label = df_for_label.ix[:, np.sum(df_for_label.isna(), axis=0) == 0]    # 맨 앞쪽 NA 제거

            _, features_data_for_label, features_sampled_label = self.features_cls.processing_split_new(df_for_label, m_days=m_days, k_days=k_days, sampling_days=sampling_days, calc_length=calc_length, label_type='trainable_label')
            # features_for_label = features_for_label[calc_length:]

            assert np.sum(features_sampled_data - features_data_for_label) == 0
        elif label_type == 'test_label':
            df_selected_label = self.df_pivoted[(self.df_pivoted.index >= self.date_[base_idx - m_days - calc_length])
                                                & (self.df_pivoted.index <= self.date_[base_idx + k_days])]

            # 현재기준으로 정제된 종목 기준 라벨 데이터 생성 및 정제
            df_for_label = df_selected_label.loc[:, df_for_data.columns]
            df_for_label.ffill(axis=0, inplace=True)
            df_for_label.bfill(axis=0, inplace=True)
            df_for_label = df_for_label.ix[:, np.sum(df_for_label.isna(), axis=0) == 0]    # 맨 앞쪽 NA 제거

            _, features_data_for_label, features_sampled_label = self.features_cls.processing_split_new(df_for_label, m_days=m_days, k_days=k_days, sampling_days=sampling_days, calc_length=calc_length, label_type='test_label')
            # features_for_label = features_for_label[calc_length:]

        _, n_asset, n_feature = features_sampled_data.shape
        question = np.zeros([n_asset, M, n_feature], dtype=np.float32)

        question[:] = np.transpose(features_sampled_data, [1, 0, 2])
        if label_type == 'trainable_label':
            answer = np.zeros([n_asset, 2, n_feature], dtype=np.float32)
            answer[:, :2, :] = np.transpose(features_sampled_label, [1, 0, 2])
            assert np.sum(answer[:, 0, :] - question[:, -1, :]) == 0
        elif label_type == 'test_label':
            label_idx = features_list.index(self.features_cls.label_feature)

            answer = np.zeros([n_asset, 2, 1], dtype=np.float32)
            answer[:, :2, :] = np.transpose(features_sampled_label, [1, 0, 2])
            assert np.sum(answer[:, 0, 0] - question[:, -1, label_idx]) == 0
        else:
            answer = np.zeros([n_asset, 2, n_feature], dtype=np.float32)
            answer[:, 0, :] = question[:, -1, :]

        # _, n_asset, n_feature = features_sampled_data.shape
        # question = np.zeros([n_asset, M, n_feature], dtype=np.float32)
        # answer = np.zeros([n_asset, K+1, n_feature], dtype=np.float32)
        #
        # question[:] = np.transpose(features_sampled_data[:M], [1, 0, 2])
        # if use_label:
        #     answer_data = features_sampled_label[-(K + 1):]
        #     answer[:, :len(answer_data), :] = np.transpose(answer_data, [1, 0, 2])
        #     assert np.sum(answer[:, 0, :] - question[:, -1, :]) == 0
        # else:
        #     answer[:, 0, :] = question[:, -1, :]

        if balance_class:
            idx_label = features_list.index(self.features_cls.label_feature)
            where_p = (answer[:, 1, idx_label] > 0)
            where_n = (answer[:, 1, idx_label] <= 0)
            n_max = np.max([np.sum(where_p), np.sum(where_n)])
            idx_pos = np.concatenate([np.random.choice(np.where(where_p)[0], np.sum(where_p), replace=False),
                                      np.random.choice(np.where(where_p)[0], n_max - np.sum(where_p), replace=True)])
            idx_neg = np.concatenate([np.random.choice(np.where(where_n)[0], np.sum(where_n), replace=False),
                                      np.random.choice(np.where(where_n)[0], n_max - np.sum(where_n), replace=True)])

            idx_bal = np.concatenate([idx_pos, idx_neg])
            input_enc, output_dec, target_dec = question[idx_bal], answer[idx_bal, :-1, :], answer[idx_bal, 1:, :]
            # num_min_class = np.min([np.sum(answer[:, 1, idx_y] > 0), np.sum(answer[:, 1, idx_y] <= 0)])
            # idx_pos = np.random.choice(np.where(answer[:, 1, 0] > 0)[0], num_min_class, replace=False)
            # idx_neg = np.random.choice(np.where(answer[:, 1, 0] <= 0)[0], num_min_class, replace=False)
            # idx_bal = np.concatenate([idx_pos, idx_neg])
            # input_enc, output_dec, target_dec = question[idx_bal], answer[idx_bal, :-1, :], answer[idx_bal, 1:, :]
        else:
            input_enc, output_dec, target_dec = question[:], answer[:, :-1, :], answer[:, 1:, :]
            assert len(additional_info['assets_list']) == len(input_enc)

        if label_type == 'test_label':
            label_idx = features_list.index(self.features_cls.label_feature)
            assert np.sum(input_enc[:, -1, label_idx] - output_dec[:, 0, 0]) == 0
        else:
            assert np.sum(input_enc[:, -1, :] - output_dec[:, 0, :]) == 0
        return input_enc, output_dec, target_dec, features_list, additional_info

    def sample_inputdata_split(self, base_idx, sampling_days=5, m_days=60, k_days=20, balance_class=True, use_label=True):
        if use_label is False:
            balance_class = False

        self._set_df_pivoted(base_idx)
        # 미래데이터 원천 제거
        calc_length = 250
        df_selected_data = self.df_pivoted[(self.df_pivoted.index >= self.date_[base_idx - m_days - calc_length])
                                           & (self.df_pivoted.index <= self.date_[base_idx])]

        # 현재기준 데이터 정제
        df_for_data = df_selected_data.ix[:, np.sum(~df_selected_data.isna(), axis=0) >= len(df_selected_data.index) * 0.9]  # 90% 이상 데이터 존재
        df_for_data.ffill(axis=0, inplace=True)
        df_for_data.bfill(axis=0, inplace=True)
        df_for_data = df_for_data.ix[:, np.sum(df_for_data.isna(), axis=0) == 0]    # 맨 앞쪽 NA 제거

        if df_for_data.empty:
            return False

        additional_info = {'date': self.date_[base_idx], 'assets_list': list(df_for_data.columns)}
        features_list, features_for_data = self.features_cls.processing_split(df_for_data, m_days=m_days, k_days=k_days, calc_length=calc_length)
        # features_for_data = features_for_data[calc_length:]

        assert features_for_data.shape[0] == m_days + 1

        M = m_days // sampling_days
        K = k_days // sampling_days

        features_sampled_data = features_for_data[::sampling_days][1:]  # 0, 5, 10, ... 데이터 중 0 데이터 제거 (의미없음)
        assert features_sampled_data.shape[0] == M

        if use_label:
            # 미래데이터 포함 라벨 생성
            df_selected_label = self.df_pivoted[(self.df_pivoted.index >= self.date_[base_idx-m_days - calc_length])
                                                & (self.df_pivoted.index <= self.date_[base_idx+k_days])]

            # 현재기준으로 정제된 종목 기준 라벨 데이터 생성 및 정제
            df_for_label = df_selected_label.loc[:, df_for_data.columns]
            df_for_label.ffill(axis=0, inplace=True)
            df_for_label.bfill(axis=0, inplace=True)
            df_for_label = df_for_label.ix[:, np.sum(df_for_label.isna(), axis=0) == 0]    # 맨 앞쪽 NA 제거

            _, features_for_label = self.features_cls.processing_split(df_for_label, m_days=m_days, k_days=k_days, calc_length=calc_length)
            # features_for_label = features_for_label[calc_length:]

            assert features_for_label.shape[0] == m_days + k_days + 1
            assert np.sum(features_for_data - features_for_label[:(m_days + 1), :, :]) == 0

            features_sampled_label = features_for_label[::sampling_days][1:]  # 0, 5, 10, ... 데이터 중 0 데이터 제거 (의미없음)
            assert features_sampled_label.shape[0] == M + K

        _, n_asset, n_feature = features_sampled_data.shape
        question = np.zeros([n_asset, M, n_feature], dtype=np.float32)
        answer = np.zeros([n_asset, K+1, n_feature], dtype=np.float32)

        question[:] = np.transpose(features_sampled_data[:M], [1, 0, 2])
        if use_label:
            answer_data = features_sampled_label[-(K + 1):]
            answer[:, :len(answer_data), :] = np.transpose(answer_data, [1, 0, 2])
            assert np.sum(answer[:, 0, :] - question[:, -1, :]) == 0
        else:
            answer[:, 0, :] = question[:, -1, :]

        if balance_class:
            idx_label = features_list.index(self.features_cls.label_feature)
            where_p = (answer[:, 1, idx_label] > 0)
            where_n = (answer[:, 1, idx_label] <= 0)
            n_max = np.max([np.sum(where_p), np.sum(where_n)])
            idx_pos = np.concatenate([np.random.choice(np.where(where_p)[0], np.sum(where_p), replace=False),
                                      np.random.choice(np.where(where_p)[0], n_max - np.sum(where_p), replace=True)])
            idx_neg = np.concatenate([np.random.choice(np.where(where_n)[0], np.sum(where_n), replace=False),
                                      np.random.choice(np.where(where_n)[0], n_max - np.sum(where_n), replace=True)])

            idx_bal = np.concatenate([idx_pos, idx_neg])
            input_enc, output_dec, target_dec = question[idx_bal], answer[idx_bal, :-1, :], answer[idx_bal, 1:, :]
            # num_min_class = np.min([np.sum(answer[:, 1, idx_y] > 0), np.sum(answer[:, 1, idx_y] <= 0)])
            # idx_pos = np.random.choice(np.where(answer[:, 1, 0] > 0)[0], num_min_class, replace=False)
            # idx_neg = np.random.choice(np.where(answer[:, 1, 0] <= 0)[0], num_min_class, replace=False)
            # idx_bal = np.concatenate([idx_pos, idx_neg])
            # input_enc, output_dec, target_dec = question[idx_bal], answer[idx_bal, :-1, :], answer[idx_bal, 1:, :]
        else:
            input_enc, output_dec, target_dec = question[:], answer[:, :-1, :], answer[:, 1:, :]
            assert len(additional_info['assets_list']) == len(input_enc)

        assert np.sum(input_enc[:, -1, :] - output_dec[:, 0, :]) == 0
        return input_enc, output_dec, target_dec, features_list, additional_info

    def sample_inputdata(self, base_idx, codes_list=None, sampling_days=5, m_days=60, k_days=20,
                         balance_class=True,
                         **kwargs):

        self._set_df_pivoted(base_idx)
        calc_length = 250
        df_selected = self.df_pivoted[(self.df_pivoted.index >= self.date_[base_idx-m_days-calc_length])
                                      & (self.df_pivoted.index <= self.date_[base_idx+k_days])]
        df_not_null = df_selected.ix[:, np.sum(~df_selected.isna(), axis=0) >= len(df_selected.index) * 0.9]  # 90% 이상 데이터 존재
        df_not_null.ffill(axis=0, inplace=True)
        df_not_null.bfill(axis=0, inplace=True)
        df_not_null = df_not_null.ix[:, np.sum(df_not_null.isna(), axis=0) == 0]    # 맨 앞쪽 NA 제거

        if codes_list is not None:
            assert type(codes_list) == list
            codes_exist = []
            for code in codes_list:
                if code in df_not_null.columns:
                    codes_exist.append(code)

            if len(codes_exist) >= 1:
                df_not_null = df_not_null[codes_exist]
            else:
                return False

        if df_not_null.empty:
            return False

        additional_info = {'date': self.date_[base_idx], 'assets_list': list(df_not_null.columns)}

        features_list, features_data = processing(df_not_null, m_days=m_days, calc_length=calc_length)
        features_data = features_data[calc_length:]

        assert features_data.shape[0] == m_days + k_days + 1

        M = m_days // sampling_days
        K = k_days // sampling_days

        features_sampled_data = features_data[::sampling_days][1:]  # 0, 5, 10, ... 데이터 중 0 데이터 제거 (의미없음)
        assert features_sampled_data.shape[0] == M + K
        _, n_asset, n_feature = features_sampled_data.shape
        question = np.zeros([n_asset, M, n_feature], dtype=np.float32)
        answer = np.zeros([n_asset, K+1, n_feature], dtype=np.float32)

        question[:] = np.transpose(features_sampled_data[:M], [1, 0, 2])

        answer_data = features_sampled_data[-(K + 1):]
        answer[:, :len(answer_data), :] = np.transpose(answer_data, [1, 0, 2])
        if balance_class:
            num_min_class = np.min([np.sum(answer[:, 1, 0] > 0), np.sum(answer[:, 1, 0] <= 0)])
            idx_pos = np.random.choice(np.where(answer[:, 1, 0] > 0)[0], num_min_class, replace=False)
            idx_neg = np.random.choice(np.where(answer[:, 1, 0] <= 0)[0], num_min_class, replace=False)
            idx_bal = np.concatenate([idx_pos, idx_neg])
            input_enc, output_dec, target_dec = question[idx_bal], answer[idx_bal, :-1, :], answer[idx_bal, 1:, :]
        else:
            input_enc, output_dec, target_dec = question[:], answer[:, :-1, :], answer[:, 1:, :]
            assert len(additional_info['assets_list']) == len(input_enc)

        assert np.sum(input_enc[:, -1, :] - output_dec[:, 0, :]) == 0
        return input_enc, output_dec, target_dec, features_list, additional_info

    @property
    def max_length(self):
        return len(self.date_)


def rearrange(input, output, target):
    features = {"input": input, "output": output}
    return features, target


# 학습에 들어가 배치 데이터를 만드는 함수이다.
def dataset_process(train_input_enc, train_output_dec, train_target_dec, batch_size, shuffle=True, iter_num=None):
    # Dataset을 생성하는 부분으로써 from_tensor_slices부분은
    # 각각 한 문장으로 자른다고 보면 된다.
    # train_input_enc, train_output_dec, train_target_dec
    # 3개를 각각 한문장으로 나눈다.
    dataset = tf.data.Dataset.from_tensor_slices((train_input_enc, train_output_dec, train_target_dec))
    # 전체 데이터를 섞는다.
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=len(train_input_enc))
    # 배치 인자 값이 없다면  에러를 발생 시킨다.
    assert batch_size is not None, "train batchSize must not be None"
    # from_tensor_slices를 통해 나눈것을
    # 배치크기 만큼 묶어 준다.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # 데이터 각 요소에 대해서 rearrange 함수를
    # 통해서 요소를 변환하여 맵으로 구성한다.
    dataset = dataset.map(rearrange)
    # repeat()함수에 원하는 에포크 수를 넣을수 있으면
    # 아무 인자도 없다면 무한으로 이터레이터 된다.
    if iter_num is None:
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat(iter_num)
    # make_one_shot_iterator를 통해 이터레이터를
    # 만들어 준다.
    # 이터레이터를 통해 다음 항목의 텐서
    # 개체를 넘겨준다.
    return dataset
