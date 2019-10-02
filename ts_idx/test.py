
import numpy as np
import pandas as pd


def log_y_nd(log_p, n):
    assert len(log_p.shape) == 2

    return np.r_[log_p[:n, :] - log_p[:1, :], log_p[n:, :] - log_p[:-n, :]]


def fft(log_p, n, m_days, k_days):
    assert (len(log_p) == (m_days + k_days + 1)) or (len(log_p) == (m_days + 1))

    log_p_fft = np.fft.fft(log_p[:(m_days + 1)], axis=0)
    log_p_fft[n:-n] = 0
    return np.real(np.fft.ifft(log_p_fft, m_days + k_days + 1, axis=0))[:len(log_p)]


def std_nd(log_p, n):
    y = np.exp(log_y_nd(log_p, 1)) - 1.
    stdarr = np.zeros_like(y)
    for t in range(1, len(y)):
        stdarr[t, :] = np.std(y[max(0, t - n):(t + 1), :], ddof=1, axis=0)

    return stdarr


def std_nd_new(log_p, n):
    y = np.exp(log_y_nd(log_p, n)) - 1.
    stdarr = np.zeros_like(y)
    for t in range(1, len(y)):
        stdarr[t, :] = np.std(y[max(0, t - n * 12):(t + 1), :][::n], ddof=1, axis=0)

    return stdarr


def mdd_nd(log_p, n):
    mddarr = np.zeros_like(log_p)
    for t in range(len(log_p)):
        mddarr[t, :] = log_p[t, :] - np.max(log_p[max(0, t - n):(t + 1), :], axis=0)

    return mddarr


def positional_encoding(seq_size, dim):
    encoded_vec = np.array([pos / np.power(10000, 2 * i / dim) for pos in range(seq_size) for i in range(dim)], dtype=np.float32)
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return encoded_vec.reshape([seq_size, dim])


class DataGeneratorIndex:
    def __init__(self):
        data_path = './data/data_for_metarl.csv'
        data_df = pd.read_csv(data_path)
        data_df.set_index('eval_d', inplace=True)
        date_ = list(data_df.index)

        feature = dict()
        logy = np.log(1. + data_df.values)
        logp = np.cumsum(logy, axis=0) - logy[0]

        for n in [5, 20, 60]:
            feature['logy_{}'.format(n)] = log_y_nd(logp, n)
            feature['std_{}'.format(n)] = std_nd(logp, n)
            feature['mdd_{}'.format(n)] = mdd_nd(logp, n)
        feature['stdnew_{}'.format(5)] = std_nd_new(logp, n)
        feature['pos_{}'.format(5)] = np.sign(feature['logy_{}'.format(5)])


        min_d = 100
        time_steps = 5
        lookback = 60

        dataset = {}
        for i in range(min_d, len(date_) - time_steps, time_steps):
            dataset[date_[i]] = {'data': [], 'label': None}
            dataset[date_[i]]['label'] = [int(feature['pos_5'][i+time_steps][0] > 0), int(feature['pos_5'][i+time_steps][0] < 0)]
            for j in range(0, lookback, time_steps):
                tmp = logy[(i - j - time_steps):(i - j)]
                for key in feature.keys():
                    tmp = np.r_[tmp, feature[key][(i - j - 1):(i - j), :]]

                if j == 0:
                    tmp_all = tmp.flatten()
                else:
                    tmp_all = np.c_[tmp_all, tmp.flatten()]
            dataset[date_[i]]['data'] = tmp_all.transpose()

        seq_size, dim = dataset[date_[min_d]]['data'].shape
        pos_encoding = positional_encoding(seq_size, dim)

        train_start_d = date_[min_d]
        train_end_d = date_[1000]
        eval_start_d = date_[1000]
        eval_end_d = date_[1200]


        train_set = list()
        n_task = 10
        support_data = [[] for _ in range(n_task)]
        target_data = [[] for _ in range(n_task)]

        tasks = np.random.randint(min_d + 120, 1000 - 20, n_task)
        for i, task in enumerate(tasks):
            support_start_d = task - 120
            support_end_d = task
            target_end_d = task + 20
            for d in dataset.keys():
                if d >= date_[support_start_d] and d < date_[support_end_d]:
                    support_data[i].append(dataset[d])
                elif d < date_[target_end_d]:
                    target_data[i].append(dataset[d])





import tensorflow as tf
import sys

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Embedding

class FeedForward(Model):
    def __init__(self, dim_out, num_units, out_activation='linear'):
        super().__init__()
        self.in_layer = Dense(num_units, activation=tf.nn.relu)
        self.out_layer = Dense(dim_out, activation=out_activation)

    def call(self, inputs):
        return self.out_layer(self.in_layer(inputs))







