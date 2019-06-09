
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

def normalize(arr_x, eps=1e-6, M=None):
    if M is None:
        return (arr_x - np.mean(arr_x, axis=0)) / (np.std(arr_x, axis=0) + eps)
    else:
        # return (arr_x - np.mean(arr_x, axis=0)) / (np.std(arr_x, axis=0) + eps)
        return (arr_x - np.mean(arr_x[:M], axis=0)) / (np.std(arr_x[:M], axis=0) + eps)


def dict_to_list(dict, key_list=None):
    arr = list()
    ordered_key_list = list()

    if key_list is None:
        key_list = list(dict.keys())

    for key in dict.keys():
        if key in key_list:
            ordered_key_list.append(key)
            arr.append(dict[key])

    return np.stack(arr, axis=-1), ordered_key_list


def predict_plot(model, dataset, columns_list, size=250, save_dir='out.jpg'):

    cost_rate = 0.000
    idx_y = columns_list.index('log_y')
    idx_pos = columns_list.index('positive')

    true_y = np.zeros(size)
    pred_both = np.zeros_like(true_y)
    pred_pos = np.zeros_like(true_y)
    pred_y = np.zeros_like(true_y)
    pred_avg = np.zeros_like(true_y)

    prev_w_both = 0
    prev_w_pos = 0
    prev_w_y = 0
    for j, (features, labels) in enumerate(dataset.take(size)):
        predictions = model.predict(features)
        true_y[j] = labels[0, 0, idx_y]
        if predictions[0, 0, idx_y] > 0:
            pred_y[j] = labels[0, 0, idx_y] - cost_rate * (1. - prev_w_y)
            prev_w_y = 1
        else:
            pred_y[j] = - cost_rate * prev_w_y
            prev_w_y = 0
        if predictions[0, 0, idx_pos] > 0:
            pred_pos[j] = labels[0, 0, idx_y] - cost_rate * (1. - prev_w_pos)
            prev_w_pos = 1
        else:
            pred_pos[j] = - cost_rate * prev_w_pos
            prev_w_pos = 0
        if (predictions[0, 0, idx_y] > 0) and (predictions[0, 0, idx_pos] > 0):
            pred_both[j] = labels[0, 0, idx_y] - cost_rate * (1. - prev_w_both)
            prev_w_both = 1
        else:
            pred_both[j] = - cost_rate * prev_w_both
            prev_w_both = 0

        pred_avg[j] = (pred_y[j] + pred_pos[j]) / 2.

    data = pd.DataFrame({'true_y': np.cumsum(np.log(1. + true_y)),
                         'pred_both': np.cumsum(np.log(1. + pred_both)),
                         'pred_pos': np.cumsum(np.log(1. + pred_pos)),
                         'pred_y': np.cumsum(np.log(1. + pred_y)),
                         'pred_avg': np.cumsum(np.log(1. + pred_avg))})

    fig = plt.figure()
    plt.plot(data)
    plt.legend(data.columns)
    fig.savefig(save_dir)
    plt.close(fig)


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


def getWeights(d, size):
    w=[1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d-k+1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

def plotWeights(dRange, nPlots, size):
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = getWeights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::1], columns=[d])
        w = w.join(w_, how='outer')
    ax = w.plot()
    ax.legend(loc='upper left'); mpl.show()
    return

def fracDiff(series, d, thres=.01):
    w = getWeights(d, series.shape[0])
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_>thres].shape[0]
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc in range(skip, seriesF.shape[0]):
            loc= seriesF.index[iloc]
            if not np.isinite(series.loc[loc, name]):
                continue
            df_[loc] = np.dot(w[-(iloc+1):, :].T, seriesF.loc[:loc])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df

def fracDiff_FFD(series, d, thres=1e-5):
    # w = getWeights_FFD(d, thres)
    w = getWeights(d, thres)
    width = len(w) - 1
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isinfinite(series.loc[loc1, name]):
                continue
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df