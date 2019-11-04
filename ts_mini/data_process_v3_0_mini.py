
import pandas as pd
import tensorflow as tf
import numpy as np
import os
from functools import partial
import time
from tqdm import tqdm
import pickle

def log_y_nd(log_p, n):
    if len(log_p.shape) == 2:
        return np.r_[log_p[:n, :] - log_p[:1, :], log_p[n:, :] - log_p[:-n, :]]
    elif len(log_p.shape) == 1:
        return np.r_[log_p[:n] - log_p[:1], log_p[n:] - log_p[:-n]]
    else:
        raise NotImplementedError


def fft(log_p, n, m_days, k_days):
    assert (len(log_p) == (m_days + k_days + 1)) or (len(log_p) == (m_days + 1))

    log_p_fft = np.fft.fft(log_p[:(m_days + 1)], axis=0)
    log_p_fft[n:-n] = 0
    return np.real(np.fft.ifft(log_p_fft, m_days + k_days + 1, axis=0))[:len(log_p)]


def std_nd(log_p, n):
    y = np.exp(log_y_nd(log_p, 1)) - 1.
    stdarr = np.zeros_like(y)
    for t in range(1, len(y)):
        stdarr[t, :] = np.std(y[max(0, t - n):(t + 1), :], axis=0)

    return stdarr


def std_nd_new(log_p, n):
    y = np.exp(log_y_nd(log_p, n)) - 1.
    stdarr = np.zeros_like(y)
    for t in range(1, len(y)):
        stdarr[t, :] = np.std(y[max(0, t - n * 12):(t + 1), :][::n], axis=0)

    return stdarr


def mdd_nd(log_p, n):
    mddarr = np.zeros_like(log_p)
    for t in range(len(log_p)):
        mddarr[t, :] = log_p[t, :] - np.max(log_p[max(0, t - n):(t + 1), :], axis=0)

    return mddarr


def arr_to_cs(arr):
    order = arr.argsort(axis=1)
    return_value = order.argsort(axis=1) / np.max(order, axis=1, keepdims=True)
    return return_value


def arr_to_normal(arr):
    return_value = (arr - np.mean(arr, axis=1, keepdims=True)) / np.std(arr, axis=1, ddof=1, keepdims=True)
    return return_value


def numpy_fill(arr):
    '''Solution provided by Divakar.'''
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out


def cleansing_missing_value(df_selected, n_allow_missing_value=5, to_log=True):
    mask = np.sum(df_selected.isna(), axis=0) <= n_allow_missing_value
    df = df_selected.ix[:, mask].ffill().bfill()
    df = df / df.iloc[0]
    if to_log:
        df = np.log(df)

    return df


class DataContainer:
    def __init__(self, date_):
        self.date_ = date_
        self.codes = list()
        self.dataset = list()


class Config:
    def __init__(self):
        self.m_days = 60
        self.k_days = 20
        self.calc_length = 250
        self.delay_days = 1
        self.sampling_days = 5

    def generate_name(self):
        return "M{}_K{}".format(self.m_days, self.k_days)


class Feature:
    def __init__(self, configs):
        self.name = configs.generate_name()
        self.calc_length = configs.calc_length
        self.m_days = configs.m_days
        self.k_days = configs.k_days
        self.delay_days = configs.delay_days
        self.sampling_days = configs.sampling_days
        self.possible_func = ['logy', 'std', 'stdnew', 'pos', 'mdd', 'fft']

    def split_data_label(self, data_arr):
        # log_p_arr shape : (m_days + k_days + 1, all dates)
        assert len(data_arr) == self.m_days + self.k_days + self.delay_days + 1

        data_ = data_arr[:(self.m_days + 1)]
        label_ = data_arr[self.m_days:][self.delay_days:]
        return data_, label_

    def calc_func(self, arr, feature_nm, debug=False):
        # 아래 함수 추가할때마다 추가해줄것...

        func_nm, nd = feature_nm.split('_')
        n = int(nd)

        calc_length, m_days, k_days, delay_days = self.calc_length, self.m_days, self.k_days, self.delay_days
        k_days_adj = k_days + delay_days

        assert arr.shape[0] == calc_length + m_days + k_days_adj + 1
        if debug:
            # label 데이터 제거 후 산출
            arr_debug = arr[:-k_days_adj]

        # arr default: logp
        if func_nm == 'logy':
            result = log_y_nd(arr, n)[calc_length:]
            if debug:
                result_debug = log_y_nd(arr_debug, n)[calc_length:]
        elif func_nm == 'std':
            result = std_nd(arr, n)[calc_length:]
            if debug:
                result_debug = std_nd(arr_debug, n)[calc_length:]
        elif func_nm == 'stdnew':
            result = std_nd_new(arr, n)[calc_length:]
            if debug:
                result_debug = std_nd_new(arr_debug, n)[calc_length:]
        elif func_nm == 'pos':
            result = np.sign(log_y_nd(arr, n)[calc_length:])
            if debug:
                result_debug = np.sign(log_y_nd(arr_debug, n)[calc_length:])
        elif func_nm == 'mdd':
            # arr: data without calc data
            result = mdd_nd(arr[calc_length:], n)
            if debug:
                result_debug = mdd_nd(arr_debug[calc_length:], n)
        elif func_nm == 'fft':
            # arr: data without calc data
            result = fft(arr[calc_length:], n, m_days, k_days_adj)
            if debug:
                result_debug = fft(arr_debug[calc_length:], n, m_days, k_days_adj)

        feature, label = self.split_data_label(result)
        if debug:
            n_error = np.sum(feature - result_debug)
            if n_error != 0:
                print("[debug: {}] data not matched.".format(func_nm))
                raise AssertionError

        return feature[::self.sampling_days], label[::self.sampling_days]

    def calc_features(self, log_p_arr, transpose=False, debug=False):
        if transpose:
            log_p_arr = np.transpose(log_p_arr)

        # log_p_arr shape : (days per each date, codes_list)
        # features_dict = dict()
        # labels_dict = dict()
        data_dict = dict()
        for func_nm in self.possible_func:
        # for func_nm in ['logy']:
            for n in [5, 10, 20, 60, 120]:
                nm = '{}_{}'.format(func_nm, n)
                # features_dict[nm], labels_dict[nm] = self.calc_func(log_p_arr, nm, debug)
                data_dict[nm] = dict()
                data_dict[nm]['feature'], data_dict[nm]['label'] = self.calc_func(log_p_arr, nm, debug)

        # if transpose:
        #     for key in features_dict.keys():
        #         features_dict[key] = np.transpose(features_dict[key])
        if transpose:
            for key in data_dict.keys():
                data_dict[key]['feature'] = np.transpose(data_dict[key]['feature'])
                data_dict[key]['label'] = np.transpose(data_dict[key]['label'])

        return data_dict


class Code:
    def __init__(self, df_p):
        self.name = df_p.name
        self._initialize(df_p)

    def _initialize(self, df_p):
        self.begin_d = df_p.first_valid_index()
        self.end_d = df_p.last_valid_index()
        self.date_i = df_p.index[(df_p.index >= self.begin_d) & (df_p.index <= self.end_d)].to_list()
        value_i = df_p.loc[self.date_i].to_numpy(dtype=np.float32)
        logp_i = np.log(value_i)
        self.logp_i = logp_i - logp_i[0]

    def is_vector(self, arr):
        if len(arr.shape) == 1:
            return True
        else:
            return False

    def split_by_nd(self, nd):
        arr_1d = self.logp_i
        # 1차원 데이터
        assert self.is_vector(arr_1d)

        # 최소한 nd보단 긴 길이
        assert len(arr_1d) >= nd + 1

        arr_2d = []
        for i in range(nd, len(arr_1d)):
            arr_2d.append(arr_1d[(i-nd):(i+1)])

        assert len(arr_2d) == len(self.date_i[nd:])
        return np.stack(arr_2d), self.date_i[nd:]

    def cleansing_missing_value(self, data_for_calc, date_index, n_allow_missing_value=5):
        mask = np.sum(np.isnan(data_for_calc), axis=1) <= n_allow_missing_value
        data_for_calc = numpy_fill(data_for_calc[mask])     # [forward fill] np version.
        date_index = list(np.array(date_index)[mask])
        return data_for_calc, date_index

    def prepare_data_for_calc(self, feature_cls):
        m_days = feature_cls.m_days
        k_days = feature_cls.k_days
        calc_length = feature_cls.calc_length
        delay_days = feature_cls.delay_days
        k_days_adj = k_days + delay_days

        logp_for_calc, date_index = self.split_by_nd(calc_length + m_days + k_days_adj)
        logp_for_calc, date_index = self.cleansing_missing_value(logp_for_calc, date_index, int((m_days + k_days_adj) * 0.1))
        self.feature_i, self.label_i = feature_cls.calc_features(logp_for_calc, transpose=True)
        self.date_index = date_index


class CodesInDate:
    def __init__(self, name, data_dir='./data'):
        self._initialize(name)

    def _initialize(self, name):
        if os.path.exists(os.path.join(self.data_dir, '{}.pkl'.format(name))):
            self.load(name)
        else:
            data_path = './data/kr_close_y_90.csv'
            data_df_temp = pd.read_csv(data_path)
            data_df_temp = data_df_temp[data_df_temp.infocode > 0]

            date_temp = data_df_temp[['date_', 'infocode']].groupby('date_').count()
            date_temp = date_temp[date_temp.infocode >= 10]
            date_temp.columns = ['cnt']

            date_ = list(date_temp.index)
            data_df = pd.merge(date_temp, data_df_temp, on='date_')  # 최소 10종목 이상 존재 하는 날짜만
            data_df['y'] = data_df['y'] + 1
            data_df['cum_y'] = data_df[['date_', 'infocode', 'y']].groupby('infocode').cumprod(axis=0)

            df_pivoted_all = data_df[['date_', 'infocode', 'cum_y']].pivot(index='date_', columns='infocode')
            df_pivoted_all.columns = df_pivoted_all.columns.droplevel(0).to_numpy(dtype=np.int32)
            self.df_pivoted = df_pivoted_all

    def a(self):
        df_pivoted = self.df_pivoted
        feature_cls = Feature(Config())
        m_days = feature_cls.m_days
        k_days = feature_cls.k_days
        calc_length = feature_cls.calc_length
        delay_days = feature_cls.delay_days
        k_days_adj = k_days + delay_days
        len_data = calc_length + m_days
        len_label = k_days_adj

        date_ = list(df_pivoted.index)

        tt = time.time()
        for date_i in range(len_data, len(date_)):
            if date_i % 100 == 0:
                prev_t = tt
                tt = time.time()
                print('i {}, time: {}'.format(date_i, tt - prev_t))
            start_d = date_[(date_i - len_data)]
            end_d = date_[min(date_i + len_label, len(date_) - 1)]

            select_where = ((df_pivoted.index >= start_d) & (df_pivoted.index <= end_d))
            df_logp = cleansing_missing_value(df_pivoted.ix[select_where, :], n_allow_missing_value=5, to_log=True)

            data_i = feature_cls.calc_features(df_logp.to_numpy(dtype=np.float32), transpose=False)
            base_dir = './data/preprocessed/{}/'.format(date_i)
            os.makedirs(base_dir, exist_ok=True)
            for feature_nm in data_i.keys():
                pickle.dump(data_i[feature_nm], open(os.path.join(base_dir, '{}.pkl'.format(feature_nm)), 'wb'))

            obj = {'feature': feature_i, 'label': label_i, 'asset_list': df_logp.columns.to_list()}



def prepare_all_code():
    data_path = './data/kr_close_y_90.csv'
    data_df_temp = pd.read_csv(data_path)
    data_df_temp = data_df_temp[data_df_temp.infocode > 0]

    date_temp = data_df_temp[['date_', 'infocode']].groupby('date_').count()
    date_temp = date_temp[date_temp.infocode >= 10]
    date_temp.columns = ['cnt']

    date_ = list(date_temp.index)
    data_df = pd.merge(date_temp, data_df_temp, on='date_')  # 최소 10종목 이상 존재 하는 날짜만
    data_df['y'] = data_df['y'] + 1
    data_df['cum_y'] = data_df[['date_', 'infocode', 'y']].groupby('infocode').cumprod(axis=0)

    df_pivoted_all = data_df[['date_', 'infocode', 'cum_y']].pivot(index='date_', columns='infocode')
    df_pivoted_all.columns = df_pivoted_all.columns.droplevel(0).to_numpy(dtype=np.int32)

    feature_cls = Feature(Config())
    code_list = []

    for i in tqdm(range(len(df_pivoted_all.columns))):
        code = df_pivoted_all.columns[i]
        print(code)
        code_list.append(Code(df_pivoted_all.loc[:, code]))
        code_list[i].prepare_data_for_calc(feature_cls)




class DataScheduler:
    def __init__(self, configs, features_cls, data_type='kr_stock', univ_type='all'):
        # make a directory for outputs
        self.data_out_path = os.path.join(os.getcwd(), configs.data_out_path)
        os.makedirs(self.data_out_path, exist_ok=True)

        # self.data_generator = DataGenerator(data_type)    # infocode
        self.data_generator = DataGeneratorDynamic(features_cls, data_type, univ_type=univ_type, use_beta=configs.use_beta, delayed_days=configs.delayed_days)    # infocode

        self.train_set_length = configs.train_set_length
        self.retrain_days = configs.retrain_days
        self.m_days = configs.m_days
        self.k_days = configs.k_days
        self.sampling_days = configs.sampling_days
        self.balancing_method = configs.balancing_method

        self.train_batch_size = configs.batch_size
        self.eval_batch_size = 256
        self.trainset_rate = configs.trainset_rate

        self.features_cls = features_cls
        self._initialize()

    def _initialize(self):
        self.base_idx = self.train_set_length

        self.train_begin_idx = 0
        self.eval_begin_idx = int(self.train_set_length * self.trainset_rate)
        self.test_begin_idx = self.base_idx - self.m_days
        self.test_end_idx = self.base_idx + self.retrain_days

    def set_idx(self, base_idx):
        self.base_idx = base_idx

        self.train_begin_idx = np.max([0, base_idx - self.train_set_length])
        self.eval_begin_idx = int(self.train_set_length * self.trainset_rate) + np.max([0, base_idx - self.train_set_length])
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
        # data_params['univ_idx'] = None
        if mode == 'train':
            start_idx = self.train_begin_idx + self.m_days
            end_idx = self.eval_begin_idx - self.k_days
            data_params['balance_class'] = True
            data_params['label_type'] = 'trainable_label'   # trainable: calc_length 반영
            decaying_factor = 0.99   # 기간별 샘플 중요도
        elif mode == 'eval':
            start_idx = self.eval_begin_idx + self.m_days
            end_idx = self.test_begin_idx - self.k_days
            data_params['balance_class'] = True
            data_params['label_type'] = 'trainable_label'   # trainable: calc_length 반영
            decaying_factor = 1.   # 기간별 샘플 중요도
        elif mode == 'test':
            start_idx = self.test_begin_idx + self.m_days
            # start_idx = self.test_begin_idx
            end_idx = self.test_end_idx
            data_params['balance_class'] = False
            data_params['label_type'] = 'test_label'        # test: 예측하고자 하는 것만 반영 (k_days)
            decaying_factor = 1.   # 기간별 샘플 중요도
        elif mode == 'test_insample':
            start_idx = self.train_begin_idx + self.m_days
            # start_idx = self.test_begin_idx
            end_idx = self.test_begin_idx - self.k_days
            data_params['balance_class'] = False
            data_params['label_type'] = 'test_label'        # test: 예측하고자 하는 것만 반영 (k_days)
            decaying_factor = 1.   # 기간별 샘플 중요도
        elif mode == 'predict':
            start_idx = self.test_begin_idx + self.m_days
            # start_idx = self.test_begin_idx
            end_idx = self.test_end_idx
            data_params['balance_class'] = False
            data_params['label_type'] = None            # label 없이 과거데이터만으로 스코어 산출
            decaying_factor = 1.   # 기간별 샘플 중요도
        else:
            raise NotImplementedError

        print("start idx:{} ({}) / end idx: {} ({})".format(start_idx, dg.date_[start_idx], end_idx, dg.date_[end_idx]))

        return start_idx, end_idx, data_params, decaying_factor

    def _dataset(self, mode='train'):
        input_enc, output_dec, target_dec = [], [], []  # test/predict 인경우 list, train/eval인 경우 array
        features_list = []
        additional_infos_list = []  # test/predict 인경우 list, train/eval인 경우 dict
        sampling_wgt = []  # time decaying factor
        start_idx, end_idx, data_params, decaying_factor = self.get_data_params(mode)

        n_loop = np.ceil((end_idx - start_idx) / self.sampling_days)
        for i, d in enumerate(range(start_idx, end_idx, self.sampling_days)):
            if self.balancing_method in ['once', 'nothing']:
                _sampled_data = self.data_generator.sample_inputdata_split_new3(d, **data_params)
            elif self.balancing_method == 'each':
                _sampled_data = self.data_generator.sample_inputdata_split_new2(d, **data_params)
            else:
                raise NotImplementedError

            if _sampled_data is False:
                continue
            else:
                tmp_ie, tmp_od, tmp_td, features_list, additional_info = _sampled_data
                additional_info['importance_wgt'] = np.array([decaying_factor ** (n_loop - i - 1) for _ in range(len(tmp_ie))], dtype=np.float32)

            input_enc.append(tmp_ie)
            output_dec.append(tmp_od)
            target_dec.append(tmp_td)
            additional_infos_list.append(additional_info)

        if len(input_enc) == 0:
            return False

        if mode in ['train', 'eval']:
            additional_infos = dict()
            input_enc = np.concatenate(input_enc, axis=0)
            output_dec = np.concatenate(output_dec, axis=0)
            target_dec = np.concatenate(target_dec, axis=0)

            size_value = np.concatenate([additional_info['size_value'] for additional_info in additional_infos_list], axis=0)
            mktcap = np.concatenate([additional_info['mktcap'] for additional_info in additional_infos_list], axis=0)
            importance_wgt = np.concatenate([additional_info['importance_wgt'] for additional_info in additional_infos_list], axis=0)

            if self.balancing_method == 'once':
                idx_label = features_list.index(self.features_cls.label_feature)
                where_p = (np.squeeze(target_dec)[:, idx_label] > 0)
                where_n = (np.squeeze(target_dec)[:, idx_label] <= 0)
                n_max = np.max([np.sum(where_p), np.sum(where_n)])
                idx_pos = np.concatenate([np.random.choice(np.where(where_p)[0], np.sum(where_p), replace=False),
                                          np.random.choice(np.where(where_p)[0], n_max - np.sum(where_p),
                                                           replace=True)])
                idx_neg = np.concatenate([np.random.choice(np.where(where_n)[0], np.sum(where_n), replace=False),
                                          np.random.choice(np.where(where_n)[0], n_max - np.sum(where_n),
                                                           replace=True)])

                idx_bal = np.concatenate([idx_pos, idx_neg])
                input_enc, output_dec, target_dec = input_enc[idx_bal], output_dec[idx_bal], target_dec[idx_bal]
                additional_infos['size_value'] = size_value[idx_bal]
                additional_infos['mktcap'] = mktcap[idx_bal]
                importance_wgt['importance_wgt'] = importance_wgt[idx_bal]
            else:
                additional_infos['size_value'] = size_value[:]
                additional_infos['mktcap'] = mktcap[:]
                additional_infos['importance_wgt'] = importance_wgt[:]
        else:
            additional_infos = additional_infos_list


        start_date = self.data_generator.date_[start_idx]
        end_date = self.data_generator.date_[end_idx]
        return input_enc, output_dec, target_dec, features_list, additional_infos, start_date, end_date

    def train(self,
              model,
              trainset=None,
              evalset=None,
              train_steps=1,
              eval_steps=10,
              save_steps=50,
              early_stopping_count=10,
              model_name='ts_model_v1.0'):

        # make directories for graph results (both train and test one)
        train_out_path = os.path.join(self.data_out_path, model_name, '{}'.format(self.base_idx))
        os.makedirs(train_out_path, exist_ok=True)

        if trainset is None:
            _train_dataset = self._dataset('train')
        else:
            _train_dataset = trainset

        if evalset is None:
            _eval_dataset = self._dataset('eval')
        else:
            _eval_dataset = evalset

        if _train_dataset is False or _eval_dataset is False:
            print('[train] no train/eval data')
            return False

        train_input_enc, train_output_dec, train_target_dec, features_list, train_add_infos, _, _ = _train_dataset
        eval_input_enc, eval_output_dec, eval_target_dec, _, eval_add_infos, _, _ = _eval_dataset

        assert np.sum(train_input_enc[:, -1, :] - train_output_dec[:, 0, :]) == 0
        assert np.sum(eval_input_enc[:, -1, :] - eval_output_dec[:, 0, :]) == 0

        train_size_value = train_add_infos['size_value']
        train_importance_wgt = train_add_infos['importance_wgt']
        eval_size_value = eval_add_infos['size_value']
        eval_importance_wgt = eval_add_infos['importance_wgt']

        # train_size_value = np.concatenate([add_info['size_value'] for add_info in train_add_infos], axis=0)
        # eval_size_value = np.concatenate([add_info['size_value'] for add_info in eval_add_infos], axis=0)

        # K > 1인 경우 미래데이터 안 땡겨쓰게.
        train_new_output = np.zeros_like(train_output_dec)
        eval_new_output = np.zeros_like(eval_output_dec)
        # if weight_scheme == 'ew':
        #     train_new_output[:, 0, :] = train_output_dec[:, 0, :]
        #     eval_new_output[:, 0, :] = eval_output_dec[:, 0, :]
        # elif weight_scheme == 'mw':
        train_new_output[:, 0, :] = train_output_dec[:, 0, :] + train_size_value[:, 0, :]
        eval_new_output[:, 0, :] = eval_output_dec[:, 0, :] + eval_size_value[:, 0, :]

        train_dataset = dataset_process(train_input_enc, train_new_output, train_target_dec, train_size_value, batch_size=self.train_batch_size, importance_wgt=train_importance_wgt)
        eval_dataset = dataset_process(eval_input_enc, eval_new_output, eval_target_dec, eval_size_value, batch_size=self.eval_batch_size, importance_wgt=eval_importance_wgt, iter_num=1)
        print("train step: {}  eval step: {}".format(len(train_input_enc) // self.train_batch_size,
                                                     len(eval_input_enc) // self.eval_batch_size))
        for i, (features, labels, size_values, importance_wgt) in enumerate(train_dataset.take(train_steps)):
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

            features_with_noise = {'input': None, 'output': features['output']}


            # add random noise
            if np.random.random() <= 0.4:
                # normal with mu=0 and sig=sigma
                sample_sigma = tf.math.reduce_std(features['input'], axis=[0, 1], keepdims=True)
                eps = sample_sigma * tf.random.normal(features['input'].shape, mean=0, stddev=1)
            else:
                eps = 0

            features_with_noise['input'] = features['input'] + eps


            # randomly masked input data
            if np.random.random() <= 0.1:
                t_size = features['input'].shape[1]
                mask = np.ones_like(features['input'])
                masked_idx = np.random.choice(t_size, size=int(t_size * 0.2), replace=False)
                for mask_i in masked_idx:
                    mask[:, mask_i, :] = 0

                    features_with_noise['input'] = features_with_noise['input'] * mask

            labels_mtl = self.features_cls.labels_for_mtl(features_list, labels, size_values, importance_wgt)
            model.train_mtl(features_with_noise, labels_mtl, print_loss=print_loss)

    def test(self, model, dataset=None, use_label=True, out_dir=None, file_nm='out.png', ylog=False, save_type=None, table_nm=None, time_step=1):
        if out_dir is None:
            test_out_path = os.path.join(self.data_out_path, '{}/test'.format(self.base_idx))
        else:
            test_out_path = out_dir

        os.makedirs(test_out_path, exist_ok=True)
        if use_label:
            if dataset is None:
                _dataset_list = self._dataset('test')
            else:
                _dataset_list = dataset

            if _dataset_list is False:
                print('[test] no test data')
                return False
            self.features_cls.predict_plot_mtl_cross_section_test(model, _dataset_list,  save_dir=test_out_path, file_nm=file_nm, ylog=ylog, time_step=time_step)
            self.features_cls.predict_plot_mtl_cross_section_test_long(model, _dataset_list, save_dir=test_out_path + "2", file_nm=file_nm, ylog=ylog, t_stepsize=time_step, invest_rate=0.8)
            self.features_cls.predict_plot_mtl_cross_section_test_long(model, _dataset_list, save_dir=test_out_path + "3", file_nm=file_nm, ylog=ylog, t_stepsize=time_step, invest_rate=0.6)

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
        size_value_list = [add_info['size_value'] for add_info in additional_infos]
        df_infos = pd.DataFrame(columns={'start_d', 'base_d', 'infocode', 'score'})
        for i, (input_enc_t, output_dec_t, size_value) in enumerate(zip(input_enc_list, output_dec_list, size_value_list)):
            assert np.sum(input_enc_t[:, -1, :] - output_dec_t[:, 0, :]) == 0
            assert np.sum(output_dec_t[:, 1:, :]) == 0
            new_output_t = np.zeros_like(output_dec_t)
            new_output_t[:, 0, :] = output_dec_t[:, 0, :] + size_value[:, 0, :]

            features = {'input': input_enc_t, 'output': new_output_t}
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
        size_value_list = [add_info['size_value'] for add_info in additional_infos]
        df_infos = pd.DataFrame(columns={'start_d', 'base_d', 'infocode', 'score'})
        for i, (input_enc_t, output_dec_t, size_value) in enumerate(zip(input_enc_list, output_dec_list, size_value_list)):
            assert np.sum(input_enc_t[:, -1, :] - output_dec_t[:, 0, :]) == 0
            assert np.sum(output_dec_t[:, 1:, :]) == 0
            new_output_t = np.zeros_like(output_dec_t)
            new_output_t[:, 0, :] = output_dec_t[:, 0, :] + size_value[:, 0, :]
            features = {'input': input_enc_t, 'output': new_output_t}
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



from ts_mini.utils_mini import *
# from ts_mini.features_mini import processing # processing_split, labels_for_mtl

import pandas as pd
import pickle
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
        self.data_generator = DataGeneratorDynamic(features_cls, data_type, univ_type=univ_type, use_beta=configs.use_beta, delayed_days=configs.delayed_days)    # infocode

        self.train_set_length = configs.train_set_length
        self.retrain_days = configs.retrain_days
        self.m_days = configs.m_days
        self.k_days = configs.k_days
        self.sampling_days = configs.sampling_days
        self.balancing_method = configs.balancing_method

        self.train_batch_size = configs.batch_size
        self.eval_batch_size = 256
        self.trainset_rate = configs.trainset_rate

        self.features_cls = features_cls
        self._initialize()

    def _initialize(self):
        self.base_idx = self.train_set_length

        self.train_begin_idx = 0
        self.eval_begin_idx = int(self.train_set_length * self.trainset_rate)
        self.test_begin_idx = self.base_idx - self.m_days
        self.test_end_idx = self.base_idx + self.retrain_days

    def set_idx(self, base_idx):
        self.base_idx = base_idx

        self.train_begin_idx = np.max([0, base_idx - self.train_set_length])
        self.eval_begin_idx = int(self.train_set_length * self.trainset_rate) + np.max([0, base_idx - self.train_set_length])
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
        # data_params['univ_idx'] = None

    def _dataset2(self, mode='train'):
        input_enc, output_dec, target_dec = [], [], []  # test/predict 인경우 list, train/eval인 경우 array
        features_list = []
        additional_infos_list = []  # test/predict 인경우 list, train/eval인 경우 dict
        sampling_wgt = []  # time decaying factor
        start_idx, end_idx, data_params, decaying_factor = self.get_data_params(mode)

        n_loop = np.ceil((end_idx - start_idx) / self.sampling_days)
        for i, d in enumerate(range(start_idx, end_idx, self.sampling_days)):
            if self.balancing_method in ['once', 'nothing']:
                _sampled_data = self.data_generator.sample_inputdata_split_new3(d, **data_params)
            else:
                raise NotImplementedError

            if _sampled_data is False:
                continue
            else:
                tmp_ie, tmp_od, tmp_td, features_list, additional_info = _sampled_data
                additional_info['importance_wgt'] = np.array([decaying_factor ** (n_loop - i - 1) for _ in range(len(tmp_ie))], dtype=np.float32)

            input_enc.append(tmp_ie)
            output_dec.append(tmp_od)
            target_dec.append(tmp_td)
            additional_infos_list.append(additional_info)

        if len(input_enc) == 0:
            return False

        additional_infos = additional_infos_list


        start_date = self.data_generator.date_[start_idx]
        end_date = self.data_generator.date_[end_idx]
        return input_enc, output_dec, target_dec, features_list, additional_infos, start_date, end_date

    def _dataset(self, mode='train'):
        input_enc, output_dec, target_dec = [], [], []  # test/predict 인경우 list, train/eval인 경우 array
        features_list = []
        additional_infos_list = []  # test/predict 인경우 list, train/eval인 경우 dict
        sampling_wgt = []  # time decaying factor
        start_idx, end_idx, data_params, decaying_factor = self.get_data_params(mode)

        n_loop = np.ceil((end_idx - start_idx) / self.sampling_days)
        for i, d in enumerate(range(start_idx, end_idx, self.sampling_days)):
            if self.balancing_method in ['once', 'nothing']:
                _sampled_data = self.data_generator.sample_inputdata_split_new3(d, **data_params)
            elif self.balancing_method == 'each':
                _sampled_data = self.data_generator.sample_inputdata_split_new2(d, **data_params)
            else:
                raise NotImplementedError

            if _sampled_data is False:
                continue
            else:
                tmp_ie, tmp_od, tmp_td, features_list, additional_info = _sampled_data
                additional_info['importance_wgt'] = np.array([decaying_factor ** (n_loop - i - 1) for _ in range(len(tmp_ie))], dtype=np.float32)

            input_enc.append(tmp_ie)
            output_dec.append(tmp_od)
            target_dec.append(tmp_td)
            additional_infos_list.append(additional_info)

        if len(input_enc) == 0:
            return False

        if mode in ['train', 'eval']:
            additional_infos = dict()
            input_enc = np.concatenate(input_enc, axis=0)
            output_dec = np.concatenate(output_dec, axis=0)
            target_dec = np.concatenate(target_dec, axis=0)

            size_value = np.concatenate([additional_info['size_value'] for additional_info in additional_infos_list], axis=0)
            mktcap = np.concatenate([additional_info['mktcap'] for additional_info in additional_infos_list], axis=0)
            importance_wgt = np.concatenate([additional_info['importance_wgt'] for additional_info in additional_infos_list], axis=0)

            if self.balancing_method == 'once':
                idx_label = features_list.index(self.features_cls.label_feature)
                where_p = (np.squeeze(target_dec)[:, idx_label] > 0)
                where_n = (np.squeeze(target_dec)[:, idx_label] <= 0)
                n_max = np.max([np.sum(where_p), np.sum(where_n)])
                idx_pos = np.concatenate([np.random.choice(np.where(where_p)[0], np.sum(where_p), replace=False),
                                          np.random.choice(np.where(where_p)[0], n_max - np.sum(where_p),
                                                           replace=True)])
                idx_neg = np.concatenate([np.random.choice(np.where(where_n)[0], np.sum(where_n), replace=False),
                                          np.random.choice(np.where(where_n)[0], n_max - np.sum(where_n),
                                                           replace=True)])

                idx_bal = np.concatenate([idx_pos, idx_neg])
                input_enc, output_dec, target_dec = input_enc[idx_bal], output_dec[idx_bal], target_dec[idx_bal]
                additional_infos['size_value'] = size_value[idx_bal]
                additional_infos['mktcap'] = mktcap[idx_bal]
                importance_wgt['importance_wgt'] = importance_wgt[idx_bal]
            else:
                additional_infos['size_value'] = size_value[:]
                additional_infos['mktcap'] = mktcap[:]
                additional_infos['importance_wgt'] = importance_wgt[:]
        else:
            additional_infos = additional_infos_list


        start_date = self.data_generator.date_[start_idx]
        end_date = self.data_generator.date_[end_idx]
        return input_enc, output_dec, target_dec, features_list, additional_infos, start_date, end_date

    def train(self,
              model,
              trainset=None,
              evalset=None,
              train_steps=1,
              eval_steps=10,
              save_steps=50,
              early_stopping_count=10,
              model_name='ts_model_v1.0'):

        # make directories for graph results (both train and test one)
        train_out_path = os.path.join(self.data_out_path, model_name, '{}'.format(self.base_idx))
        os.makedirs(train_out_path, exist_ok=True)

        if trainset is None:
            _train_dataset = self._dataset('train')
        else:
            _train_dataset = trainset

        if evalset is None:
            _eval_dataset = self._dataset('eval')
        else:
            _eval_dataset = evalset

        if _train_dataset is False or _eval_dataset is False:
            print('[train] no train/eval data')
            return False

        train_input_enc, train_output_dec, train_target_dec, features_list, train_add_infos, _, _ = _train_dataset
        eval_input_enc, eval_output_dec, eval_target_dec, _, eval_add_infos, _, _ = _eval_dataset

        assert np.sum(train_input_enc[:, -1, :] - train_output_dec[:, 0, :]) == 0
        assert np.sum(eval_input_enc[:, -1, :] - eval_output_dec[:, 0, :]) == 0

        train_size_value = train_add_infos['size_value']
        train_importance_wgt = train_add_infos['importance_wgt']
        eval_size_value = eval_add_infos['size_value']
        eval_importance_wgt = eval_add_infos['importance_wgt']

        # train_size_value = np.concatenate([add_info['size_value'] for add_info in train_add_infos], axis=0)
        # eval_size_value = np.concatenate([add_info['size_value'] for add_info in eval_add_infos], axis=0)

        # K > 1인 경우 미래데이터 안 땡겨쓰게.
        train_new_output = np.zeros_like(train_output_dec)
        eval_new_output = np.zeros_like(eval_output_dec)
        # if weight_scheme == 'ew':
        #     train_new_output[:, 0, :] = train_output_dec[:, 0, :]
        #     eval_new_output[:, 0, :] = eval_output_dec[:, 0, :]
        # elif weight_scheme == 'mw':
        train_new_output[:, 0, :] = train_output_dec[:, 0, :] + train_size_value[:, 0, :]
        eval_new_output[:, 0, :] = eval_output_dec[:, 0, :] + eval_size_value[:, 0, :]

        train_dataset = dataset_process(train_input_enc, train_new_output, train_target_dec, train_size_value, batch_size=self.train_batch_size, importance_wgt=train_importance_wgt)
        eval_dataset = dataset_process(eval_input_enc, eval_new_output, eval_target_dec, eval_size_value, batch_size=self.eval_batch_size, importance_wgt=eval_importance_wgt, iter_num=1)
        print("train step: {}  eval step: {}".format(len(train_input_enc) // self.train_batch_size,
                                                     len(eval_input_enc) // self.eval_batch_size))
        for i, (features, labels, size_values, importance_wgt) in enumerate(train_dataset.take(train_steps)):
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

            features_with_noise = {'input': None, 'output': features['output']}


            # add random noise
            if np.random.random() <= 0.4:
                # normal with mu=0 and sig=sigma
                sample_sigma = tf.math.reduce_std(features['input'], axis=[0, 1], keepdims=True)
                eps = sample_sigma * tf.random.normal(features['input'].shape, mean=0, stddev=1)
            else:
                eps = 0

            features_with_noise['input'] = features['input'] + eps


            # randomly masked input data
            if np.random.random() <= 0.1:
                t_size = features['input'].shape[1]
                mask = np.ones_like(features['input'])
                masked_idx = np.random.choice(t_size, size=int(t_size * 0.2), replace=False)
                for mask_i in masked_idx:
                    mask[:, mask_i, :] = 0

                    features_with_noise['input'] = features_with_noise['input'] * mask

            labels_mtl = self.features_cls.labels_for_mtl(features_list, labels, size_values, importance_wgt)
            model.train_mtl(features_with_noise, labels_mtl, print_loss=print_loss)

    def test(self, model, dataset=None, use_label=True, out_dir=None, file_nm='out.png', ylog=False, save_type=None, table_nm=None, time_step=1):
        if out_dir is None:
            test_out_path = os.path.join(self.data_out_path, '{}/test'.format(self.base_idx))
        else:
            test_out_path = out_dir

        os.makedirs(test_out_path, exist_ok=True)
        if use_label:
            if dataset is None:
                _dataset_list = self._dataset('test')
            else:
                _dataset_list = dataset

            if _dataset_list is False:
                print('[test] no test data')
                return False
            self.features_cls.predict_plot_mtl_cross_section_test(model, _dataset_list,  save_dir=test_out_path, file_nm=file_nm, ylog=ylog, time_step=time_step)
            self.features_cls.predict_plot_mtl_cross_section_test_long(model, _dataset_list, save_dir=test_out_path + "2", file_nm=file_nm, ylog=ylog, t_stepsize=time_step, invest_rate=0.8)
            self.features_cls.predict_plot_mtl_cross_section_test_long(model, _dataset_list, save_dir=test_out_path + "3", file_nm=file_nm, ylog=ylog, t_stepsize=time_step, invest_rate=0.6)

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
        size_value_list = [add_info['size_value'] for add_info in additional_infos]
        df_infos = pd.DataFrame(columns={'start_d', 'base_d', 'infocode', 'score'})
        for i, (input_enc_t, output_dec_t, size_value) in enumerate(zip(input_enc_list, output_dec_list, size_value_list)):
            assert np.sum(input_enc_t[:, -1, :] - output_dec_t[:, 0, :]) == 0
            assert np.sum(output_dec_t[:, 1:, :]) == 0
            new_output_t = np.zeros_like(output_dec_t)
            new_output_t[:, 0, :] = output_dec_t[:, 0, :] + size_value[:, 0, :]

            features = {'input': input_enc_t, 'output': new_output_t}
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
        size_value_list = [add_info['size_value'] for add_info in additional_infos]
        df_infos = pd.DataFrame(columns={'start_d', 'base_d', 'infocode', 'score'})
        for i, (input_enc_t, output_dec_t, size_value) in enumerate(zip(input_enc_list, output_dec_list, size_value_list)):
            assert np.sum(input_enc_t[:, -1, :] - output_dec_t[:, 0, :]) == 0
            assert np.sum(output_dec_t[:, 1:, :]) == 0
            new_output_t = np.zeros_like(output_dec_t)
            new_output_t[:, 0, :] = output_dec_t[:, 0, :] + size_value[:, 0, :]
            features = {'input': input_enc_t, 'output': new_output_t}
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



def convert_to_tensor(df_pivoted_all):
    dataset = [DataContainer(date_) for date_ in df_pivoted_all.index]
    infocode = 181
    for infocode in df_pivoted_all.columns[:2]:
        cls = Code(infocode)
        df = df_pivoted_all.loc[:, infocode]
        begin_d, end_d = df.first_valid_index(), df.last_valid_index()
        date_i = df.index[(df.index >= begin_d) & (df.index <= end_d)].to_list()
        value_i = df.loc[date_i].ffill().to_numpy(dtype=np.float32)
        assert len(date_i) == len(value_i)

        for i in range(date_i.index(begin_d), date_i.index(end_d)):
            value_i


class Universe:
    def __init__(self, univ_nm='selected'):
        # eval_d / infocode / size_port / mktcap
        self._set_data(univ_nm)

    def _set_size_data(self):
        # load size data
        size_data = pd.read_csv('./data/kr_mktcap_daily.csv')
        size_data.columns = ['eval_d', 'infocode', 'mktcap', 'size_port']
        size_data = size_data[size_data.infocode > 0]
        size_data['mktcap'] = size_data['mktcap'] / 1000.
        self.size_data = size_data

    def _set_data(self, univ_nm):
        if univ_nm == 'all':
            data_code = pd.read_csv('./data/kr_sizeinfo_90.csv')
            self.data_code = data_code[data_code.infocode > 0]
        elif univ_nm == 'selected':
            self._set_size_data()
            # size_data = pd.read_csv('./data/kr_sizeinfo_90.csv')
            date_ = pd.read_csv('./data/date.csv')
            data_code = pd.read_csv('./data/kr_univ_monthly.csv')
            data_code = data_code[data_code.infocode > 0]

            w_date = pd.merge(data_code, date_, left_on='eval_d', right_on='work_m')
            data_code_w_size = pd.merge(w_date, self.size_data, left_on=['infocode', 'work_m'], right_on=['infocode', 'eval_d'])
            self.data_code = data_code_w_size.ix[:, ['eval_m', 'infocode', 'size_port', 'mktcap']]
            self.data_code.columns = ['eval_d', 'infocode', 'size_port', 'mktcap']

univ =Universe('all')

class DataGeneratorDynamic:
    def __init__(self, features_cls, data_type='kr_stock', univ_type='all', use_beta=True, delayed_days=0):
        if data_type == 'kr_stock':
            data_path = './data/kr_close_y_90.csv'
            data_df_temp = pd.read_csv(data_path)
            data_df_temp = data_df_temp[data_df_temp.infocode > 0]

            date_temp = data_df_temp[['date_', 'infocode']].groupby('date_').count()
            date_temp = date_temp[date_temp.infocode >= 10]
            date_temp.columns = ['cnt']

            date_ = list(date_temp.index)
            data_df = pd.merge(date_temp, data_df_temp, on='date_')  # 최소 10종목 이상 존재 하는 날짜만
            data_df['y'] = data_df['y'] + 1
            data_df['cum_y'] = data_df[['date_', 'infocode', 'y']].groupby('infocode').cumprod(axis=0)

            df_pivoted_all = data_df[['date_', 'infocode', 'cum_y']].pivot(index='date_', columns='infocode')
            df_pivoted_all.columns = df_pivoted_all.columns.droplevel(0).to_numpy(dtype=np.int32)

            self.delayed_days = delayed_days

    def _set_df_pivoted(self, base_idx, univ_idx):
        date_arr = self.data_code.eval_d.unique()
        if univ_idx is None:
            univ_idx = base_idx

        if (np.sum(date_arr <= self.date_[base_idx]) == 0) or (np.sum(date_arr <= self.date_[univ_idx]) == 0):
            return False

        size_list = list(self.size_data[self.size_data.eval_d == self.date_[base_idx]]['infocode'].to_numpy(dtype=np.int32))

        base_d = max(date_arr[date_arr <= self.date_[base_idx]])

        if self.base_d != base_d:
            # print('base_d changed {} -> {}'.format(self.base_d, base_d))
            self.base_d = base_d

            univ_d = max(date_arr[date_arr <= self.date_[univ_idx]])
            univ_list = list(self.data_code[self.data_code.eval_d == univ_d]['infocode'].to_numpy(dtype=np.int32))
            base_list = list(self.data_code[self.data_code.eval_d == base_d]['infocode'].to_numpy(dtype=np.int32))

            # df_pivoted = self.data_df[['date_', 'infocode', 'cum_y']].pivot(index='date_', columns='infocode')
            # df_pivoted.columns = df_pivoted.columns.droplevel(0).to_numpy(dtype=np.int32)
            if self.use_beta:
                univ_list_selected = sorted(list(set.intersection(set(univ_list),
                                                                  set(base_list),
                                                                  set(size_list),
                                                                  set(self.df_pivoted_all.columns),
                                                                  set(self.df_beta_all.columns),
                                                                  set(self.df_ivol_all.columns))))

                self.df_beta = self.df_beta_all[univ_list_selected]
                self.df_ivol = self.df_ivol_all[univ_list_selected]
            else:
                univ_list_selected = sorted(list(set.intersection(set(univ_list), set(base_list), set(size_list), set(self.df_pivoted_all.columns))))

            self.df_pivoted = self.df_pivoted_all[univ_list_selected]
            # self.df_size = self.data_code[self.data_code.eval_d == base_d][['infocode', 'mktcap']].set_index('infocode').loc[univ_list_selected, :]
            # self.df_size['rnk'] = self.df_size.mktcap.rank() / len(self.df_size)
            self.df_size = self.size_data[self.size_data.eval_d == self.date_[base_idx]][['infocode', 'mktcap']].set_index('infocode').loc[univ_list_selected, :]
            self.df_size['rnk'] = self.df_size.mktcap.rank() / len(self.df_size)
            assert self.df_pivoted.shape[1] == self.df_size.shape[0]

        return True

    def make_market_idx(self, df_for_data, mktcap, m_days, sampling_days, calc_length, label_type, delayed_days, additional_dict):
        log_p = np.log(df_for_data.values, dtype=np.float32)
        log_p = log_p - log_p[0, :]
        mkt_idx = np.sum(log_p * mktcap.reshape([1, -1]), axis=1) / np.sum(mktcap)
        mkt_df = pd.DataFrame(mkt_idx, index=df_for_data.index, columns=['mkt'])

        features_list, features_sampled_data, _ = self.features_cls.processing_split_new(mkt_df,
                                                                                         m_days=m_days,
                                                                                         sampling_days=sampling_days,
                                                                                         calc_length=calc_length,
                                                                                         label_type=None,
                                                                                         delayed_days=self.delayed_days,
                                                                                         additional_dict=additional_dict)

        return features_list, features_sampled_data

    def sample_inputdata_split_new3(self, base_idx, sampling_days=5, m_days=60, k_days=20, calc_length=250
                                    , univ_idx=None):

        is_data_exist = self._set_df_pivoted(base_idx, univ_idx)

        if not is_data_exist:
            return False

        # 미래데이터 원천 제거
        df_selected_data = self.df_pivoted[(self.df_pivoted.index >= self.date_[base_idx - m_days - calc_length])
                                           & (self.df_pivoted.index <= self.date_[base_idx])]

        # 현재기준 데이터 정제
        df_for_data = df_selected_data.ix[:, np.sum(~df_selected_data.isna(), axis=0) >= len(df_selected_data.index) * 0.9]  # 90% 이상 데이터 존재
        df_for_data.ffill(axis=0, inplace=True)
        df_for_data.bfill(axis=0, inplace=True)
        df_for_data = df_for_data.ix[:, np.sum(df_for_data.isna(), axis=0) == 0]    # 맨 앞쪽 NA 제거

        if df_for_data.empty:
            return False


        additional_info = {'date': self.date_[base_idx], 'inv_date': self.date_[base_idx + self.delayed_days], 'assets_list': list(df_for_data.columns)}
        additional_dict = None

        size_adjusted_factor = np.array(self.df_size.loc[df_for_data.columns].rnk, dtype=np.float32).reshape([-1, 1, 1])
        size_adjusted_factor_mktcap = np.array(self.df_size.loc[df_for_data.columns].mktcap, dtype=np.float32).reshape([-1, 1, 1])
        assert df_for_data.shape[-1] == size_adjusted_factor_mktcap.shape[0]

        features_list, features_sampled_data, _ = self.features_cls.processing_split_new(df_for_data,
                                                                                         m_days=m_days,
                                                                                         # k_days=k_days,
                                                                                         sampling_days=sampling_days,
                                                                                         calc_length=calc_length,
                                                                                         label_type=None,
                                                                                         delayed_days=self.delayed_days,
                                                                                         additional_dict=additional_dict)

        M = m_days // sampling_days

        assert features_sampled_data.shape[0] == M

        # ##### 라벨
        if label_type in ['trainable_label', 'test_label']:
            # 1 day adj.
            df_selected_label = self.df_pivoted[(self.df_pivoted.index >= self.date_[base_idx - m_days - calc_length])
                                                & (self.df_pivoted.index <= self.date_[base_idx + (k_days + self.delayed_days)])]  # 하루 뒤 데이터

            # 현재기준으로 정제된 종목 기준 라벨 데이터 생성 및 정제
            df_for_label = df_selected_label.loc[:, df_for_data.columns]
            df_for_label.ffill(axis=0, inplace=True)
            df_for_label.bfill(axis=0, inplace=True)
            df_for_label = df_for_label.ix[:, np.sum(df_for_label.isna(), axis=0) == 0]    # 맨 앞쪽 NA 제거

            _, features_data_for_label, features_sampled_label = self.features_cls.processing_split_new(df_for_label,
                                                                                                        m_days=m_days,
                                                                                                        # k_days=k_days,
                                                                                                        sampling_days=sampling_days,
                                                                                                        calc_length=calc_length,
                                                                                                        label_type=label_type,
                                                                                                        delayed_days=self.delayed_days,
                                                                                                        additional_dict=additional_dict)

            assert np.sum(features_sampled_data - features_data_for_label) == 0

        _, n_asset, n_feature = features_sampled_data.shape
        question = np.zeros([n_asset, M, n_feature], dtype=np.float32)
        answer = np.zeros([n_asset, 2, n_feature], dtype=np.float32)

        question[:] = np.transpose(features_sampled_data, [1, 0, 2])
        if label_type == 'trainable_label':
            answer[:, :2, :] = np.transpose(features_sampled_label, [1, 0, 2])
            answer[:, 0, :] = question[:, -1, :]    # temporary
            assert np.sum(answer[:, 0, :] - question[:, -1, :]) == 0
        elif label_type == 'test_label':
            label_idx = features_list.index(self.features_cls.label_feature)
            answer[:, 0, :] = question[:, -1, :]
            answer[:, 1, label_idx] = np.transpose(features_sampled_label, [1, 0, 2])[:, 1, 0]
            assert features_sampled_label.shape[-1] == 1
        else:
            answer[:, 0, :] = question[:, -1, :]

        assert len(size_adjusted_factor) == n_asset
        assert len(size_adjusted_factor_mktcap) == n_asset

        input_enc, output_dec, target_dec = question[:], answer[:, :-1, :], answer[:, 1:, :]
        additional_info['size_value'] = size_adjusted_factor[:]
        additional_info['mktcap'] = size_adjusted_factor_mktcap[:]
        assert len(additional_info['assets_list']) == len(input_enc)

        assert np.sum(input_enc[:, -1:, :] - output_dec[:, :, :]) == 0

        return input_enc, output_dec, target_dec, features_list, additional_info

    @property
    def max_length(self):
        return len(self.date_)


def rearrange(input, output, target, size_value, importance_wgt):
    features = {"input": input, "output": output}
    return features, target, size_value, importance_wgt


# 학습에 들어가 배치 데이터를 만드는 함수이다.
def dataset_process(input_enc, output_dec, target_dec, size_value, batch_size, importance_wgt=None, shuffle=True, iter_num=None):
    # Dataset을 생성하는 부분으로써 from_tensor_slices부분은
    # 각각 한 문장으로 자른다고 보면 된다.
    # train_input_enc, train_output_dec, train_target_dec
    # 3개를 각각 한문장으로 나눈다.
    dataset = tf.data.Dataset.from_tensor_slices((input_enc, output_dec, target_dec, size_value, importance_wgt))

    # 전체 데이터를 섞는다.
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=len(input_enc))
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
