from timeseries.utils import *
from timeseries.features import processing

import pandas as pd
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split



def data_split():
    data_path = './data/kr_data.csv'
    data_df = pd.read_csv(data_path)
    data_pivoted = pd.pivot(data_df, index='date_', columns='infocode')

    for col in data_pivoted.columns.levels[0]:
        df_ = data_pivoted[[col]]
        df_.columns = df_.columns.droplevel(0)
        df_.to_csv('./data/kr_{}.csv'.format(col))


class DataScheduler:
    def __init__(self, configs, is_infocode=True):
        # make a directory for outputs
        self.data_out_path = os.path.join(os.getcwd(), configs.data_out_path)
        os.makedirs(self.data_out_path, exist_ok=True)

        self.is_infocode = is_infocode
        if is_infocode:
            self.data_generator = DataGenerator_v3(configs.data_path)    # infocode
        else:
            self.data_generator = DataGenerator_v2(configs.data_path)  # bbticker

        self.train_set_length = configs.train_set_length
        self.retrain_days = configs.retrain_days
        self.m_days = configs.m_days
        self.k_days = configs.k_days
        self.sampling_days = configs.sampling_days
        self.max_seq_len_in = configs.max_sequence_length_in
        self.max_seq_len_out = configs.max_sequence_length_out

        self.train_batch_size = configs.batch_size
        self.train_rate = configs.train_rate

        self._initialize()

    def _initialize(self):
        self.base_idx = self.train_set_length

        self.train_begin_idx = 0
        self.eval_begin_idx = int(self.base_idx * self.train_rate)
        self.test_begin_idx = self.base_idx - self.m_days
        self.test_end_idx = self.base_idx + self.retrain_days

    def set_idx(self, base_idx):
        self.base_idx = base_idx

        self.train_begin_idx = np.max([0, base_idx - self.train_set_length])
        self.eval_begin_idx = int(self.train_set_length * self.train_rate) + self.train_begin_idx
        self.test_begin_idx = self.base_idx - self.m_days
        self.test_end_idx = self.base_idx + self.retrain_days

    def _dataset_custom(self, start_idx, end_idx, step_size, k_days=None, codes_list=None):
        if k_days is None:
            k_days = self.k_days
        input_enc, output_dec, target_dec = [], [], []

        code_dict = self.get_code_dict(codes_list)

        print("start idx:{} ({}) / end idx: {} ({})".format(
            start_idx,
            self.data_generator.date_[start_idx],
            end_idx,
            self.data_generator.date_[end_idx]))
        for i, d in enumerate(range(start_idx, end_idx, step_size)):
            _sampled_data = self.data_generator.sample_inputdata(d,
                                                     sampling_days=self.sampling_days,
                                                     m_days=self.m_days,
                                                     k_days=k_days,
                                                     max_seq_len_in=self.max_seq_len_in,
                                                     max_seq_len_out=self.max_seq_len_out,
                                                                 **code_dict)
            if _sampled_data is False:
                return False
            else:
                tmp_ie, tmp_od, tmp_td, features_list = _sampled_data
            input_enc.append(tmp_ie)
            output_dec.append(tmp_od)
            target_dec.append(tmp_td)
        input_enc = np.concatenate(input_enc, axis=0)
        output_dec = np.concatenate(output_dec, axis=0)
        target_dec = np.concatenate(target_dec, axis=0)

        return input_enc, output_dec, target_dec, features_list

    def _dataset(self, mode='train', codes_list=None):
        input_enc, output_dec, target_dec = [], [], []
        features_list = []
        if mode == 'train':
            start_idx = self.train_begin_idx + self.m_days
            end_idx = self.eval_begin_idx - self.k_days
            step_size = self.sampling_days
            k_days = self.k_days
        elif mode == 'eval':
            start_idx = self.eval_begin_idx + self.m_days
            end_idx = self.test_begin_idx - self.k_days
            step_size = self.sampling_days
            k_days = self.k_days
        elif mode == 'test':
            start_idx = self.test_begin_idx + self.m_days
            end_idx = self.test_end_idx
            step_size = self.sampling_days
            # k_days = self.sampling_days
            k_days = self.k_days
        else:
            raise NotImplementedError

        code_dict = self.get_code_dict(codes_list)

        print("start idx:{} ({}) / end idx: {} ({})".format(
            start_idx,
            self.data_generator.date_[start_idx],
            end_idx,
            self.data_generator.date_[end_idx]))
        for i, d in enumerate(range(start_idx, end_idx, step_size)):
            _sampled_data = self.data_generator.sample_inputdata(d,
                                                     sampling_days=self.sampling_days,
                                                     m_days=self.m_days,
                                                     k_days=k_days,
                                                     max_seq_len_in=self.max_seq_len_in,
                                                     max_seq_len_out=self.max_seq_len_out,
                                                                 **code_dict)
            if _sampled_data is False:
                return False
            else:
                tmp_ie, tmp_od, tmp_td, features_list = _sampled_data
            input_enc.append(tmp_ie)
            output_dec.append(tmp_od)
            target_dec.append(tmp_td)
        input_enc = np.concatenate(input_enc, axis=0)
        output_dec = np.concatenate(output_dec, axis=0)
        target_dec = np.concatenate(target_dec, axis=0)

        return input_enc, output_dec, target_dec, features_list

    def train(self,
            model,
            train_steps=1,
            eval_steps=10,
            save_steps=50,
            early_stopping_count=10,
            model_name='ts_model_v1.0'):

        # make directories for graph results (both train and test one)
        train_out_path = os.path.join(self.data_out_path, '{}'.format(self.base_idx))
        os.makedirs(train_out_path, exist_ok=True)

        train_input_enc, train_output_dec, train_target_dec, features_list = self._dataset('train')
        eval_input_enc, eval_output_dec, eval_target_dec, _ = self._dataset('eval')

        train_dataset = dataset_process(train_input_enc, train_output_dec, train_target_dec, batch_size=self.train_batch_size)
        train_dataset_plot = dataset_process(train_input_enc, train_output_dec, train_target_dec, batch_size=1)
        eval_dataset = dataset_process(train_input_enc, train_output_dec, train_target_dec, batch_size=1)
        for i, (features, labels) in enumerate(train_dataset.take(train_steps)):
            print_loss = False
            if i % save_steps == 0:
                model.save_model(model_name)
                predict_plot(model, train_dataset_plot, features_list, 250,
                             save_dir='{}/train_{}.png'.format(train_out_path, i))
                predict_plot(model, eval_dataset, features_list, 250,
                             save_dir='{}/eval_{}.png'.format(train_out_path, i))

            if i % eval_steps == 0:
                print_loss = True
                model.evaluate(eval_dataset, steps=20)
                print("[t: {} / i: {}] min_eval_loss:{} / count:{}".format(self.base_idx, i, model.eval_loss, model.eval_count))
                if model.eval_count >= early_stopping_count:
                    print("[t: {} / i: {}] train finished.".format(self.base_idx, i))
                    model.weight_to_optim()
                    model.save_model(model_name)
                    break

            model.train(features, labels, print_loss=print_loss)

    def get_code_dict(self, codes_list):
        if codes_list is not None:
            print(codes_list)

            if type(codes_list) != list:
                codes_list = [codes_list]

        if self.is_infocode is True:
            code_dict = {'infocodes': codes_list}
        else:
            code_dict = {'bbtickers': codes_list}

        return code_dict

    def test(self, model, actor=None):
        test_out_path = os.path.join(self.data_out_path, '{}/test'.format(self.base_idx))
        os.makedirs(test_out_path, exist_ok=True)
        dg = self.data_generator
        codes_list = list(dg.df_pivoted.columns[~dg.df_pivoted.ix[self.base_idx].isna()])
        dataset_list = list()
        for code_ in codes_list:
            code_dict = self.get_code_dict(code_)
            _dataset = self._dataset('test', **code_dict)
            if _dataset is False:
                continue
            else:
                input_enc, output_dec, target_dec, features_list = _dataset

            assert np.sum(input_enc[:, -1, :] - output_dec[:, 0, :]) == 0
            new_output = np.zeros_like(output_dec)
            new_output[:, 0, :] = output_dec[:, 0, :]

            if actor is not None:
                for t in range(self.max_seq_len_out):
                    if t > 0:
                        new_output[:, t, :] = obs[:, t - 1, :]
                    features_pred = {'input': input_enc, 'output': new_output}
                    obs = model.predict(features_pred)

                test_dataset = dataset_process(input_enc, new_output, target_dec, batch_size=1, mode='test')

                predict_plot_with_actor(model, actor, test_dataset, features_list,
                             size=self.retrain_days // self.sampling_days,
                             save_dir='{}/actor_{}.png'.format(test_out_path, code_))
            else:

                test_dataset = dataset_process(input_enc, new_output, target_dec, batch_size=1, mode='test')
                # test_dataset = dataset_process(input_enc, output_dec, target_dec, batch_size=1, mode='test')

                predict_plot(model, test_dataset, features_list,
                             size=self.retrain_days // self.sampling_days,
                             save_dir='{}/{}.png'.format(test_out_path, code_))

            dataset_list.append(test_dataset)
        return dataset_list, features_list

    def train_tickers(self,
                      model,
                      codes_list,
                      train_steps=1,
                      eval_steps=10,
                      save_steps=50,
                      early_stopping_count=10,
                      model_name='ts_model_v1.0'):
        # make directories for graph results (both train and test one)
        train_out_path = os.path.join(self.data_out_path, '{}'.format(self.base_idx))
        os.makedirs(train_out_path, exist_ok=True)

        code_dict = self.get_code_dict(codes_list)

        train_input_enc, train_output_dec, train_target_dec, features_list = self._dataset('train', **code_dict)
        eval_input_enc, eval_output_dec, eval_target_dec, _ = self._dataset('eval', **code_dict)

        train_dataset = dataset_process(train_input_enc, train_output_dec, train_target_dec, batch_size=self.train_batch_size)
        train_dataset_plot = dataset_process(train_input_enc, train_output_dec, train_target_dec, batch_size=1)
        eval_dataset = dataset_process(train_input_enc, train_output_dec, train_target_dec, batch_size=1)
        for i, (features, labels) in enumerate(train_dataset.take(train_steps)):
            print_loss = False
            if i % save_steps == 0:
                model.save_model(model_name)
                predict_plot(model, train_dataset_plot, features_list, 250,
                             save_dir='{}/train_{}.png'.format(train_out_path, i))
                predict_plot(model, eval_dataset, features_list, 250,
                             save_dir='{}/eval_{}.png'.format(train_out_path, i))

            if i % eval_steps == 0:
                print_loss = True
                model.evaluate(eval_dataset, steps=20)
                print("[t: {} / i: {}] min_eval_loss:{} / count:{}".format(self.base_idx, i, model.eval_loss, model.eval_count))
                if model.eval_count >= early_stopping_count:
                    print("[t: {} / i: {}] train finished.".format(self.base_idx, i))
                    model.weight_to_optim()
                    break

            model.train(features, labels, print_loss=print_loss)

    def test_bbticker(self, model, bbticker):
        test_out_path = os.path.join(self.data_out_path, '{}/test'.format(self.base_idx))
        os.makedirs(test_out_path, exist_ok=True)
        input_enc, output_dec, target_dec, features_list = self._dataset('test', bbtickers=[bbticker])
        test_dataset = dataset_process(input_enc, output_dec, target_dec, batch_size=1, mode='test')
        predict_plot(model, test_dataset, features_list, size=self.retrain_days // self.sampling_days,
                     save_dir='{}/{}.png'.format(test_out_path, bbticker))
        print('{}/{}.png'.format(test_out_path, bbticker))
        return input_enc, output_dec, target_dec, features_list

    def next(self):
        self.base_idx += self.retrain_days
        self.train_begin_idx += self.retrain_days
        self.eval_begin_idx += self.retrain_days
        self.test_begin_idx = self.base_idx - self.m_days
        self.test_end_idx = self.base_idx + self.retrain_days

    def get_date(self):
        return self.date_[self.base_d]

    @property
    def date_(self):
        return self.data_generator.date_

    @property
    def done(self):
        if self.test_end_idx > self.data_generator.max_length:
            return True
        else:
            return False



class DataGenerator_v3:
    # v3: korea stocks data with fft
    def __init__(self, data_path):
        data_path = './data/kr_close_.csv'
        data_df = pd.read_csv(data_path, index_col=0)
        self.data_df = data_df[np.sum(~data_df.isna(), axis=1) >= 10]
        self.date_ = list(self.data_df.index)

    def get_full_dataset(self, start_d, end_d):
        df_selected = self.data_df[(self.data_df.index > start_d) & (self.data_df.index <= end_d)]
        return df_selected.ix[:, np.sum(df_selected.isna(), axis=0) == 0]

    def sample_inputdata(self, base_idx, infocodes=None, sampling_days=5, m_days=60, k_days=20,
                         max_seq_len_in=12,
                         max_seq_len_out=4):

        features_list, features_data = processing(self.data_df, self.date_[base_idx-m_days], self.date_[base_idx+k_days], infocodes)

        assert features_data.shape[0] == m_days + k_days + 1

        M = m_days // sampling_days
        K = k_days // sampling_days

        features_sampled_data = features_data[::5]
        _, n_asset, n_feature = features_sampled_data.shape
        question = np.zeros([n_asset, max_seq_len_in, n_feature], dtype=np.float32)
        answer = np.zeros([n_asset, max_seq_len_out+1, n_feature], dtype=np.float32)

        question[:] = np.transpose(features_sampled_data[:M], [1, 0, 2])

        answer_data = features_sampled_data[-(K + 1):]
        answer[:, :len(answer_data), :] = np.transpose(answer_data, [1, 0, 2])

        input_enc, output_dec, target_dec = question[:], answer[:, :-1, :], answer[:, 1:, :]
        return input_enc, output_dec, target_dec, features_list

    @property
    def max_length(self):
        return len(self.date_)



class DataGenerator_v2:
    # v2: bloomberg data
    def __init__(self, data_path):
        # data_path ='./timeseries/asset_data.csv'
        data_df = pd.read_csv(data_path)
        self.df_pivoted = data_df.pivot(index='eval_d', columns='bbticker')
        self.df_pivoted.columns = self.df_pivoted.columns.droplevel(0)
        self.date_ = list(self.df_pivoted.index)

    def get_full_dataset(self, start_d, end_d):
        df_selected = self.df_pivoted[(self.df_pivoted.index > start_d) & (self.df_pivoted.index <= end_d)]
        return df_selected.ix[:, np.sum(df_selected.isna(), axis=0) == 0]

    def sample_inputdata(self, base_idx, bbtickers=None, sampling_days=5, m_days=60, k_days=20,
                         max_seq_len_in=12,
                         max_seq_len_out=4):
        # df_selected = self.df_pivoted[self.df_pivoted.index <= base_d][-(m_days+1):]
        # print("base_idx:{} / date: {}".format(base_idx, self.date_[base_idx]))
        df_selected = self.df_pivoted[(self.df_pivoted.index <= self.date_[base_idx+k_days])
                                      & (self.df_pivoted.index >= self.date_[base_idx-m_days])]
        dataset_df = df_selected.ix[:, np.sum(df_selected.isna(), axis=0) == 0]

        if bbtickers is not None:
            assert type(bbtickers) == list
            bbtickers_exist = list()
            for bbticker in bbtickers:
                if bbticker in dataset_df.columns:
                    bbtickers_exist.append(bbticker)
            if len(bbtickers_exist) >= 1:
                dataset_df = dataset_df[bbtickers_exist]
            else:
                return False

        prc = np.array(dataset_df, dtype=np.float32)

        fc = FeatureCalculator(prc, sampling_days)
        features = fc.generate_features()

        assert prc.shape[0] == m_days + k_days + 1

        M = m_days // sampling_days
        K = k_days // sampling_days

        # (timeseries, assets, features)
        features_list_linear, columns_list_linear = dict_to_list(features, features.keys() - ['log_cum_y'])
        features_list_normalize, columns_list_normal = dict_to_list(features, ['log_cum_y'])
        features_list_normalize = normalize(features_list_normalize,  M=M)

        features_normalized = np.concatenate([features_list_linear, features_list_normalize], axis=-1)
        features_list = columns_list_linear + columns_list_normal

        assert features_normalized.shape[0] == M + K

        _, n_asset, n_feature = features_normalized.shape
        question = np.zeros([n_asset, max_seq_len_in, n_feature], dtype=np.float32)
        answer = np.zeros([n_asset, max_seq_len_out+1, n_feature], dtype=np.float32)

        question[:] = np.transpose(features_normalized[:M], [1, 0, 2])

        answer_data = features_normalized[-(K + 1):]
        answer[:, :len(answer_data), :] = np.transpose(answer_data, [1, 0, 2])

        input_enc, output_dec, target_dec = question[:], answer[:, :-1, :], answer[:, 1:, :]
        return input_enc, output_dec, target_dec, features_list

    @property
    def max_length(self):
        return len(self.date_)


class DataGenerator:
    def __init__(self, data_path):
        # data_path ='./timeseries/asset_data.csv'
        data_df = pd.read_csv(data_path, index_col=0)
        self.df_pivoted = data_df.pivot(index='eval_d', columns='bbticker')
        self.df_pivoted.columns = self.df_pivoted.columns.droplevel(0)
        self.date_ = list(self.df_pivoted.index)

    def get_full_dataset(self, start_d, end_d):
        df_selected = self.df_pivoted[(self.df_pivoted.index > start_d) & (self.df_pivoted.index <= end_d)]
        return df_selected.ix[:, np.sum(df_selected.isna(), axis=0) == 0]

    def generate_dataset_v2(self, start_d, end_d, sampling_days=5, m_days=60, k_days=20, seed=1234):
        full_data_selected = self.get_full_dataset(start_d, end_d)
        train_input, train_label, features_list = self.data_to_factor(full_data_selected, sampling_days, m_days, k_days, seed)

        return train_input, train_label, features_list

    def data_to_factor_v2(self, dataset_df, sampling_days=5, m_days=60, k_days=20, seed=1234):
        assert m_days % sampling_days == 0
        assert k_days % sampling_days == 0
        prc = np.array(dataset_df, dtype=np.float32)

        fc = FeatureCalculator(prc, sampling_days)
        features = fc.generate_features()

        # (timeseries, assets, features)
        features_list_normalize, columns_list_normal = dict_to_list(features, ['log_cum_y'])
        features_list_linear, columns_list_linear = dict_to_list(features, features.keys() - ['log_cum_y'])

        M = m_days // sampling_days
        K = k_days // sampling_days

        # len(features['log_y'])
        for i, t in enumerate(range(-M + 1, K + 1)):
            sub_features_linear = features_list_linear[i]
            sub_features_normalize = normalize(features_list_normalize[(t - M): (t + K)])

            sub_features = np.concatenate([sub_features_linear, sub_features_normalize], axis=-1)
            features_list = columns_list_linear + columns_list_normal

            question_t = np.transpose(sub_features[:M], [1, 0, 2])
            answer_t = np.transpose(sub_features[M - 1:], [1, 0, 2])
            if i == 0:
                question = question_t[:]
                answer = answer_t[:]
            else:
                question = np.concatenate([question, question_t], axis=0)
                answer = np.concatenate([answer, answer_t], axis=0)

        idx = np.arange(len(question))
        if seed > 0:
            np.random.seed(seed)
            np.random.shuffle(idx)

        return question[idx], answer[idx], features_list


    def generate_dataset(self, start_d, end_d, sampling_days=5, m_days=60, k_days=20, train_rate=0.6, seed=1234):
        full_data_selected = self.get_full_dataset(start_d, end_d)

        train_dataset = full_data_selected.iloc[:int(len(full_data_selected) * train_rate)]
        eval_dataset = full_data_selected.iloc[int(len(full_data_selected) * train_rate):]

        train_input, train_label, features_list = self.data_to_factor(train_dataset, sampling_days, m_days, k_days, seed)
        eval_input, eval_label, _ = self.data_to_factor(eval_dataset, sampling_days, m_days, k_days, seed)

        return train_input, train_label, eval_input, eval_label, features_list

    def data_to_factor(self, dataset_df, sampling_days=5, m_days=60, k_days=20, seed=1234):
        assert m_days % sampling_days == 0
        assert k_days % sampling_days == 0
        prc = np.array(dataset_df, dtype=np.float32)

        fc = FeatureCalculator(prc, sampling_days)
        features = fc.generate_features()

        # (batches, assets, features)
        features_list_normalize, columns_list_normal = dict_to_list(features, ['log_cum_y'])
        features_list_linear, columns_list_linear = dict_to_list(features, features.keys() - ['log_cum_y'])

        M = m_days // sampling_days
        K = k_days // sampling_days
        for i, t in enumerate(range(M, len(features['log_y']) - K)):
            sub_features_linear = features_list_linear[(t - M): (t + K)]
            sub_features_normalize = normalize(features_list_normalize[(t - M): (t + K)])

            sub_features = np.concatenate([sub_features_linear, sub_features_normalize], axis=-1)
            features_list = columns_list_linear + columns_list_normal

            question_t = np.transpose(sub_features[:M], [1, 0, 2])
            answer_t = np.transpose(sub_features[M - 1:], [1, 0, 2])
            if i == 0:
                question = question_t[:]
                answer = answer_t[:]
            else:
                question = np.concatenate([question, question_t], axis=0)
                answer = np.concatenate([answer, answer_t], axis=0)

        idx = np.arange(len(question))
        if seed > 0:
            np.random.seed(seed)
            np.random.shuffle(idx)

        return question[idx], answer[idx], features_list

    @property
    def max_length(self):
        return len(self.date_)


def std_arr(arr_x, n):
    stdarr = np.zeros_like(arr_x)
    for t in range(1, len(arr_x)):
        stdarr[t] = np.std(arr_x[max(0, t-n):(t+1)])

    return stdarr


def mdd_arr(logcumarr_x, n):
    mddarr = np.zeros_like(logcumarr_x)
    for t in range(len(logcumarr_x)):
        mddarr[t] = logcumarr_x[t] - np.max(logcumarr_x[max(0, t-n):(t+1)])

    return mddarr


def load_data(data_path, name='kospi', token_length=5):
    # 판다스를 통해서 데이터를 불러온다.
    data_df = pd.read_csv(data_path, header=0)
    # 질문과 답변 열을 가져와 question과 answer에 넣는다.

    y_1d = np.array(data_df[name], dtype=np.float32)
    log_cum_y = np.cumsum(np.log(1. + y_1d))

    features = dict()
    # daily returns
    features['y'] = np.concatenate([log_cum_y[:token_length],
                                    log_cum_y[token_length:] - log_cum_y[:-token_length]])[::token_length]

    # cumulative returns
    features['log_cum_y'] = np.cumsum(np.log(1. + features['y']))

    # positive
    features['positive'] = (features['y'] >= 0) * np.array(1., dtype=np.float32) - (features['y'] < 0) * np.array(1., dtype=np.float32)

    # moving average
    for n in [5, 20, 60, 120]:
        if n == token_length:
            continue
        features['y_{}d'.format(n)] = np.concatenate([log_cum_y[:n], log_cum_y[n:] - log_cum_y[:-n]])[::token_length]

    # std
    for n in [20, 60, 120]:
        features['std{}d'.format(n)] = std_arr(y_1d, n)[::token_length]
        features['mdd{}d'.format(n)] = mdd_arr(log_cum_y, n)[::token_length]

    # (batches, assets, features)
    features_list_normalize, columns_list_normal = dict_to_list(features, ['log_cum_y'])
    features_list_linear, columns_list_linear = dict_to_list(features, features.keys() - ['log_cum_y'])
    question = list()
    answer = list()

    M = 60 // token_length
    K = 20 // token_length
    for i in range(M, len(features['log_y']) - K):
        sub_features_linear = features_list_linear[(i - M): (i + K)]
        sub_features_normalize = normalize(features_list_normalize[(i - M): (i + K)])

        sub_features = np.concatenate([sub_features_linear, sub_features_normalize], axis=-1)
        columns_list = columns_list_linear + columns_list_normal
        question.append(sub_features[:M])
        answer.append(sub_features[M-1:])

    train_input, eval_input, train_label, eval_label = train_test_split(question, answer, test_size=0.33, random_state=123)
    # 그 값을 리턴한다.
    return train_input, train_label, eval_input, eval_label, columns_list


def rearrange(input, output, target):
    features = {"input": input, "output": output}
    return features, target


# 학습에 들어가 배치 데이터를 만드는 함수이다.
def dataset_process(train_input_enc, train_output_dec, train_target_dec, batch_size, mode='train'):
    # Dataset을 생성하는 부분으로써 from_tensor_slices부분은
    # 각각 한 문장으로 자른다고 보면 된다.
    # train_input_enc, train_output_dec, train_target_dec
    # 3개를 각각 한문장으로 나눈다.
    dataset = tf.data.Dataset.from_tensor_slices((train_input_enc, train_output_dec, train_target_dec))
    # 전체 데이터를 섞는다.
    if mode == 'train':
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
    if batch_size == 1:
        dataset = dataset.repeat(1)
    else:
        dataset = dataset.repeat()
    # make_one_shot_iterator를 통해 이터레이터를
    # 만들어 준다.
    # 이터레이터를 통해 다음 항목의 텐서
    # 개체를 넘겨준다.
    return dataset




