
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data_path = './data/kr_close_.csv'
# data_df = pd.read_csv(data_path, index_col=0)
# df =data_df


def log_y_nd(log_p, n):
    assert len(log_p.shape) == 2

    return np.r_[log_p[:n, :] - log_p[:1, :], log_p[n:, :] - log_p[:-n, :]]


def fft(log_p, n, m_days, k_days):
    assert (len(log_p) == (m_days + k_days + 1)) or (len(log_p) == (m_days + 1))

    log_p_fft = np.fft.fft(log_p[:(m_days+1)], axis=0)
    log_p_fft[n:-n] = 0
    return np.real(np.fft.ifft(log_p_fft, m_days + k_days + 1, axis=0))[:len(log_p)]


def std_nd(log_p, n):
    y = np.exp(log_y_nd(log_p, 1)) - 1.
    stdarr = np.zeros_like(y)
    for t in range(1, len(y)):
        stdarr[t, :] = np.std(y[max(0, t - n):(t + 1), :], axis=0)

    return stdarr


def mdd_nd(log_p, n):
    mddarr = np.zeros_like(log_p)
    for t in range(len(log_p)):
        mddarr[t, :] = log_p[t, :] - np.max(log_p[max(0, t - n):(t + 1), :])

    return mddarr


def labels_for_mtl(features_list, labels):
    labels_mtl = {'ret': np.stack([labels[:, :, features_list.index('log_y')],
                                   labels[:, :, features_list.index('log_20y')],
                                   labels[:, :, features_list.index('log_60y')],
                                   labels[:, :, features_list.index('log_120y')]], axis=-1),
                  'pos': (np.concatenate([labels[:, :, features_list.index('positive')].numpy() > 0,
                                          labels[:, :, features_list.index('positive')].numpy() <= 0], axis=1)
                          * 1.).reshape([-1, 1, 2]),
                  'pos20': (np.concatenate([labels[:, :, features_list.index('positive20')].numpy() > 0,
                                            labels[:, :, features_list.index('positive20')].numpy() <= 0], axis=1)
                            * 1.).reshape([-1, 1, 2]),
                  'std': np.stack([labels[:, :, features_list.index('std_20')],
                                   labels[:, :, features_list.index('std_60')],
                                   labels[:, :, features_list.index('std_120')]], axis=-1),
                  'mdd': np.stack([labels[:, :, features_list.index('mdd_20')],
                                   labels[:, :, features_list.index('mdd_60')]], axis=-1),
                  'fft': np.stack([labels[:, :, features_list.index('fft_3com')],
                                   labels[:, :, features_list.index('fft_100com')]], axis=-1)}

    return labels_mtl


def processing_split(df_not_null, m_days, k_days):
    # if type(df.columns) == pd.MultiIndex:
    #     df.columns = df.columns.droplevel(0)

    log_p = np.log(df_not_null.values, dtype=np.float32)
    log_p = log_p - log_p[0, :]

    log_5y = log_y_nd(log_p, 5)
    log_20y = log_y_nd(log_p, 20)
    log_60y = log_y_nd(log_p, 60)
    log_120y = log_y_nd(log_p, 120)
    # log_240y = log_y_nd(log_p, 240)

    fft_3com = fft(log_p, 3, m_days, k_days)
    fft_6com = fft(log_p, 6, m_days, k_days)
    fft_100com = fft(log_p, 100, m_days, k_days)

    std_20 = std_nd(log_p, 20)
    std_60 = std_nd(log_p, 60)
    std_120 = std_nd(log_p, 120)

    mdd_20 = mdd_nd(log_p, 20)
    mdd_60 = mdd_nd(log_p, 60)
    mdd_120 = mdd_nd(log_p, 120)

    pos = np.sign(log_5y)
    pos20 = np.sign(log_20y)
    pos60 = np.sign(log_60y)
    cum_log_y = np.cumsum(log_5y, axis=0)

    features_list = ['log_y', 'log_20y', 'log_60y', 'log_120y',
                'fft_3com', 'fft_100com', 'std_20', 'std_60', 'std_120',
                'mdd_20', 'mdd_60', 'positive', 'positive20', 'positive60']

    features_data = np.stack([log_20y, log_5y, log_60y, log_120y,
                              fft_3com, fft_100com, std_20, std_60, std_120,
                              mdd_20, mdd_60, pos, pos20, pos60], axis=-1)

    assert len(features_list) == features_data.shape[-1]
    # feature_df = pd.DataFrame(np.transpose(features_data[:, :, 0]), columns=features_list)
    return features_list, features_data



def processing(df_not_null, m_days):
    # if type(df.columns) == pd.MultiIndex:
    #     df.columns = df.columns.droplevel(0)

    log_p = np.log(df_not_null.values, dtype=np.float32)
    log_p = log_p - log_p[0, :]

    log_5y = log_y_nd(log_p, 5)
    log_20y = log_y_nd(log_p, 20)
    log_60y = log_y_nd(log_p, 60)
    log_120y = log_y_nd(log_p, 120)
    # log_240y = log_y_nd(log_p, 240)

    fft_3com = fft(log_p, 3, m_days)
    fft_6com = fft(log_p, 6, m_days)
    fft_100com = fft(log_p, 100, m_days)

    std_20 = std_nd(log_p, 20)
    std_60 = std_nd(log_p, 60)
    std_120 = std_nd(log_p, 120)

    mdd_20 = mdd_nd(log_p, 20)
    mdd_60 = mdd_nd(log_p, 60)
    mdd_120 = mdd_nd(log_p, 120)

    pos = np.sign(log_5y)
    pos20 = np.sign(log_20y)
    pos60 = np.sign(log_60y)
    cum_log_y = np.cumsum(log_5y, axis=0)

    features_list = ['log_y', 'log_20y', 'log_60y', 'log_120y',
                'fft_3com', 'fft_100com', 'std_20', 'std_60', 'std_120',
                'mdd_20', 'mdd_60', 'positive', 'positive20', 'positive60']

    features_data = np.stack([log_5y, log_20y, log_60y, log_120y,
                              fft_3com, fft_100com, std_20, std_60, std_120,
                              mdd_20, mdd_60, pos, pos20, pos60], axis=-1)

    assert len(features_list) == features_data.shape[-1]
    # feature_df = pd.DataFrame(np.transpose(features_data[:, :, 0]), columns=features_list)
    return features_list, features_data


def getWeights(d, size):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d-k+1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def fracDiff(features_arr, d, thres=.1):
    n_row, n_col = features_arr.shape
    w = getWeights(d, n_row)
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_>thres].shape[0]
    frac_diff_arr = np.zeros_like(features_arr)
    for i_col in range(n_col):
        featuresF, arr_ = features_arr[:, i_col:(i_col+1)], np.zeros([len(features_arr), 1])
        for i_row in range(skip, n_row):
            if not np.isfinite(features_arr[i_row, i_col]):
                continue
            arr_[i_row, :] = np.dot(w[-(i_row+1):, :].T, featuresF[:(i_row+1)])[0, 0]
            frac_diff_arr[:, i_col:(i_col+1)] = arr_[:]
        # frac_diff_arr = pd.concat(frac_diff_arr, axis=1)
    return df


class FeatureCalculator:
    # example:
    #
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
        features = OrderedDict()
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




