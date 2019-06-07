
import pandas as pd
import tensorflow as tf
import numpy as np
import os

from sklearn.model_selection import train_test_split


def normalize(arr_x, eps=1e-6):
    return (arr_x - np.mean(arr_x, axis=0)) / (np.std(arr_x, axis=0) + eps)


def dict_to_list(dict, key_list=None):
    arr = list()
    if key_list is None:
        key_list = dict.keys()

    for key in key_list:
        arr.append(dict[key])

    return np.stack(arr, axis=-1), list(key_list)


class FeatureCalculator:
    def __init__(self, prc, sampling_freq):
        self.sampling_freq = sampling_freq
        self.log_cum_y = np.log(prc / prc[0, :])
        self.y_1d = self.get_y_1d()
        self.log_y = self.moving_average(sampling_freq, sampling_freq=sampling_freq)

    def get_y_1d(self, eps=1e-6):
        return np.concatenate([self.log_cum_y[0:1, :], np.exp(self.log_cum_y[1:, :] - self.log_cum_y[:-1, :] + eps) - 1.], axis=0)

    def moving_average(self, n, sampling_freq=1):
        return np.concatenate([self.log_cum_y[:n, :], self.log_cum_y[n:, :] - self.log_cum_y[:-n, :]], axis=0)[::sampling_freq, :]

    def positive(self):
        return (self.log_y >= 0) * np.array(1., dtype=np.float32) - (self.log_y < 0) * np.array(1., dtype=np.float32)

    def get_std(self, n, sampling_freq=1):
        stdarr = np.zeros_like(self.y_1d)
        for t in range(1, len(self.y_1d)):
            stdarr[t, :] = np.std(self.y_1d[max(0, t - n):(t + 1), :], axis=0)
        return stdarr[::sampling_freq, :]

    def get_mdd(self, n, sampling_freq=1):
        mddarr = np.zeros_like(self.log_cum_y)
        for t in range(len(self.log_cum_y)):
            mddarr[t, :] = self.log_cum_y[t, :] - np.max(self.log_cum_y[max(0, t - n):(t + 1), :])

        return mddarr[::sampling_freq, :]

    def generate_features(self):
        features = dict()
        features['log_y'] = self.log_y
        features['log_cum_y'] = self.log_cum_y[::self.sampling_freq]
        features['positive'] = self.positive()
        for n in [20, 60, 120]:
            features['y_{}d'.format(n)] = self.moving_average(n, self.sampling_freq)
            features['std{}d'.format(n)] = self.get_std(n, self.sampling_freq)
            features['mdd{}d'.format(n)] = self.get_mdd(n, self.sampling_freq)

        # remove redundant values at t=0 (y=0, cum_y=0, ...)
        for key in features.keys():
            features[key] = features[key][1:]

        return features


class DataScheduler:
    def __init__(self, configs):
        # make a directory for outputs
        self.data_out_path = os.path.join(os.getcwd(), configs.data_out_path)
        os.makedirs(self.data_out_path, exist_ok=True)

        self.data_generator = DataGenerator_v2(configs.data_path)

        self.train_set_length = configs.train_set_length
        self.retrain_days = configs.retrain_days
        self.m_days = configs.m_days
        self.k_days = configs.k_days
        self.sampling_days = configs.sampling_days

        self.train_rate = configs.train_rate

        self._initialize()

    def _initialize(self):
        self.base_d = self.train_set_length

        self.train_begin_d = 0
        self.eval_begin_d = int(self.base_d * self.train_rate)
        self.test_begin_d = self.base_d - self.m_days
        self.test_end_d = self.base_d + self.retrain_days

    def _encoding_set(self, input, label, mode='train'):
        input_enc = input[:]
        output_enc = label[:, :-1, :]

        if mode == 'test':
            target_enc = np.zeros_like(output_enc)
        else:
            target_enc = label[:, 1:, :]

        return

    def run(self, mode='train', steps=1):
        # make directories for graph results (both train and test one)
        os.makedirs(os.path.join(self.data_out_path, '/{}/'.format(self.end_d)), exist_ok=True)
        os.makedirs(os.path.join(self.data_out_path, '/{}/test/'.format(self.end_d)), exist_ok=True)
        if mode == 'train':
            train_input, train_label, features_list = self.data_generator.generate_dataset_v2(
                self.train_begin_d, self.eval_begin_d,
                sampling_days=self.sampling_days,
                m_days=self.m_days,
                k_days=self.k_days)

        elif mode == 'eval':
            eval_input, eval_label, features_list = self.data_generator.generate_dataset_v2(
                self.eval_begin_d, self.base_d,
                sampling_days=self.sampling_days,
                m_days=self.m_days,
                k_days=self.k_days)

        elif mode == 'test':
            test_input, test_label, features_list = self.data_generator.generate_dataset_v2(
                self.test_begin_d, self.test_end_d,
                sampling_days=self.sampling_days,
                m_days=self.m_days,
                k_days=0)
        else:
            raise NotImplementedError

    def _run(self):
        pass

    def get_date(self):
        return self.date_[self.base_d]

    @property
    def date_(self):
        return self.data_generator.date_

    @property
    def done(self):
        if self.end_d == self.dg.max_length:
            return True
        else:
            return False


class DataGenerator_v2:
    def __init__(self, data_path):
        # data_path ='./timeseries/asset_data.csv'
        data_df = pd.read_csv(data_path, index_col=0)
        self.df_pivoted = data_df.pivot(index='eval_d', columns='bbticker')
        self.df_pivoted.columns = self.df_pivoted.columns.droplevel(0)
        self.date_ = list(self.df_pivoted.index)

    def get_full_dataset(self, start_d, end_d):
        df_selected = self.df_pivoted[(self.df_pivoted.index > start_d) & (self.df_pivoted.index <= end_d)]
        return df_selected.ix[:, np.sum(df_selected.isna(), axis=0) == 0]

    def sample_inputdata(self, base_d, sampling_days=5, m_days=60):
        df_selected = self.df_pivoted[self.df_pivoted.index <= base_d][-(m_days+1):]
        dataset_df = df_selected.ix[:, np.sum(df_selected.isna(), axis=0) == 0]

        prc = np.array(dataset_df, dtype=np.float32)

        fc = FeatureCalculator(prc, sampling_days)
        features = fc.generate_features()

        # (timeseries, assets, features)
        features_list_linear, columns_list_linear = dict_to_list(features, features.keys() - ['log_cum_y'])
        features_list_normalize, columns_list_normal = dict_to_list(features, ['log_cum_y'])
        features_list_normalize = normalize(features_list_normalize)

        assert features_list_linear.shape[0] == m_days / sampling_days

        features_normalized = np.concatenate([features_list_linear, features_list_normalize], axis=-1)
        features_list = columns_list_linear + columns_list_normal

        question = np.transpose(features_normalized[:], [1, 0, 2])
        answer =  np.transpose(np.concatenate([features_normalized[-1:],
                                               np.zeros_like(features_normalized[-1:])]), [1, 0, 2])


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
            sub_features_linear = features_list_linear[]
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
    if mode == 'train':
        dataset = dataset.repeat()
    elif mode == 'test':
        dataset = dataset.repeat(1)
    else:
        dataset = dataset.repeat()
    # make_one_shot_iterator를 통해 이터레이터를
    # 만들어 준다.
    # 이터레이터를 통해 다음 항목의 텐서
    # 개체를 넘겨준다.
    return dataset



