
import numpy as np
import os
import pandas as pd
from matplotlib import cm, pyplot as plt


def log_y_nd(log_p, n):
    if len(log_p.shape) == 2:
        return np.r_[log_p[:n, :] - log_p[:1, :], log_p[n:, :] - log_p[:-n, :]]
    elif len(log_p.shape) == 1:
        return np.r_[log_p[:n] - log_p[:1], log_p[n:] - log_p[:-n]]
    else:
        raise NotImplementedError


def fft(log_p, n, m_days, k_days):
    assert (len(log_p) == (m_days + k_days + 1)) or (len(log_p) >= (m_days + 1))
    if (len(log_p) < (m_days + k_days + 1)) and (len(log_p) > (m_days + 1)):
        log_p = log_p[:(m_days + 1)]

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
    # cross sectional normalize
    return_value = (arr - np.nanmean(arr, axis=1, keepdims=True)) / (np.nanstd(arr, axis=1, ddof=1, keepdims=True) + 1e-6)
    return return_value


def arr_to_normal_ts(arr, m_days, calc_length):
    assert len(arr) >= (calc_length + m_days + 1)
    arr_insample = arr[:(calc_length + m_days + 1)]

    # time series normalize
    return_value = (arr - np.nanmean(arr_insample, axis=0, keepdims=True)) / (np.nanstd(arr_insample, axis=0, ddof=1, keepdims=True) + 1e-6)
    return return_value


def numpy_fill(arr):
    '''Solution provided by Divakar.'''
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out


class FeatureNew:
    def __init__(self, configs):
        self.name = configs.generate_name()
        self.calc_length = configs.calc_length
        self.calc_length_label = configs.calc_length_label
        self.m_days = configs.m_days
        self.k_days = configs.k_days
        self.delay_days = configs.delay_days
        self.sampling_days = configs.sampling_days
        self.possible_func = configs.possible_func
        # v1.0 호환
        self.features_structure = configs.features_structure

    def split_data_label(self, data_arr):
        # log_p_arr shape : (m_days + calc_len_label + 1, all dates)
        # log_p_arr shape : (m_days + k_days + 1, all dates)
        # assert len(data_arr) == self.m_days + self.k_days + self.delay_days + 1
        assert len(data_arr) >= self.m_days + 1

        data_ = data_arr[:(self.m_days + 1)]
        if len(data_arr) < self.m_days + self.k_days + self.delay_days + 1:
            print('[FeatureNew.split_data_label] Not enough data for making label.')
            label_ = None
        else:
            label_ = data_arr[self.m_days:][self.delay_days:]
        return data_, label_

    # def calc_func_size(self, arr):
    #     calc_length, m_days, k_days, delay_days = self.calc_length, self.m_days, self.k_days, self.delay_days
    #     k_days_adj = k_days + delay_days
    #     result = arr_to_normal(arr[calc_length:])
    #     result[np.isnan(result)] = 0   # TODO: 임시로 nan값 0처리
    #     feature, label = self.split_data_label(result)
    #
    #     if label is None:
    #         label_ = None
    #     # label 산출을 위한 최소한의 데이터가 없는 경우 (ex. predict)
    #     else:
    #         if len(label) <= 1:
    #             label_ = None
    #         else:
    #             label_ = label[1]
    #
    #     return feature[::self.sampling_days], label_

    def calc_func(self, arr, feature_nm, debug=False):

        func_nm, nd = feature_nm.split('_')
        n = int(nd)

        calc_length, m_days, k_days, delay_days = self.calc_length, self.m_days, self.k_days, self.delay_days
        k_days_adj = k_days + delay_days

        # assert arr.shape[0] == calc_length + m_days + k_days_adj + 1
        assert arr.shape[0] >= calc_length + m_days + 1
        if debug:
            # label 데이터 제거 후 산출
            arr_debug = arr[:(calc_length + m_days + 1)]

        # arr default: logp  TODO: arr Type ISSUE
        if func_nm in ['logp', 'tsturnover', 'tsnormal']:
            result = arr_to_normal_ts(arr, m_days, calc_length)[calc_length:]
            if debug:
                result_debug = arr_to_normal_ts(arr_debug, m_days, calc_length)[calc_length:]
        elif func_nm == 'tsp':
            arr = np.exp(arr)
            result = arr_to_normal_ts(arr, m_days, calc_length)[calc_length:]
            if debug:
                arr_debug = np.exp(arr_debug)
                result_debug = arr_to_normal_ts(arr_debug, m_days, calc_length)[calc_length:]
        elif func_nm == 'nmy':
            result = log_y_nd(arr, n)[calc_length:]
            result = arr_to_normal(np.exp(result) - 1.)
            if debug:
                result_debug = log_y_nd(arr_debug, n)[calc_length:]
                result_debug = arr_to_normal(np.exp(result_debug) - 1.)
        elif func_nm == 'logy':
            result = log_y_nd(arr, n)[calc_length:]
            if debug:
                result_debug = log_y_nd(arr_debug, n)[calc_length:]
        elif func_nm == 'std':
            result = std_nd(arr, n)[calc_length:]
            if debug:
                result_debug = std_nd(arr_debug, n)[calc_length:]
        elif func_nm == 'ir':
            result = (np.exp(log_y_nd(arr, n)[calc_length:])-1) / (std_nd(arr, n)[calc_length:] + 1e-6)
            if debug:
                result_debug = (np.exp(log_y_nd(arr_debug, n)[calc_length:])-1) / (std_nd(arr_debug, n)[calc_length:] + 1e-6)
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
            result = fft(arr[calc_length:][:(m_days + k_days_adj + 1)], n, m_days, k_days_adj)
            if debug:
                result_debug = fft(arr_debug[calc_length:], n, m_days, k_days_adj)
        elif func_nm == 'cslogy':
            # arr: data without calc data
            result = arr_to_cs(log_y_nd(arr, n)[calc_length:])
            if debug:
                result_debug = arr_to_cs(log_y_nd(arr_debug, n)[calc_length:])
        elif func_nm == 'csstd':
            # arr: data without calc data
            result = arr_to_cs(std_nd_new(arr, n)[calc_length:])
            if debug:
                result_debug = arr_to_cs(std_nd_new(arr_debug, n)[calc_length:])
        elif func_nm == 'nmlogy':
            # arr: data without calc data
            result = arr_to_normal(log_y_nd(arr, n)[calc_length:])
            if debug:
                result_debug = arr_to_normal(log_y_nd(arr_debug, n)[calc_length:])
        elif func_nm == 'nmstd':
            # arr: data without calc data
            result = arr_to_normal(std_nd_new(arr, n)[calc_length:])
            if debug:
                result_debug = arr_to_normal(std_nd_new(arr_debug, n)[calc_length:])
        elif func_nm == 'nmir':
            result = arr_to_normal(np.exp(log_y_nd(arr, n)[calc_length:]) / (std_nd(arr, n)[calc_length:] + 1e-6))
            if debug:
                result_debug = arr_to_normal(np.exp(log_y_nd(arr_debug, n)[calc_length:]) / (std_nd(arr_debug, n)[calc_length:] + 1e-6))
        elif func_nm == 'nmirnew':
            result = arr_to_normal(np.exp(log_y_nd(arr, n)[calc_length:]) / (std_nd_new(arr, n)[calc_length:] + 1e-6))
            if debug:
                result_debug = arr_to_normal(np.exp(log_y_nd(arr_debug, n)[calc_length:]) / (std_nd_new(arr_debug, n)[calc_length:] + 1e-6))

        # arr : size_arr   TODO: arr Type ISSUE
        elif func_nm in ['nmsize', 'nmturnover', 'nmivol', 'csnormal', 'nmwlogy', 'nmwlogyrnk']:
            result = arr_to_normal(arr[calc_length:])
            result[np.isnan(result)] = 0  # TODO: 임시로 nan값 0처리
            if debug:
                result_debug = arr_to_normal(arr_debug[calc_length:])
                result_debug[np.isnan(result_debug)] = 0  # TODO: 임시로 nan값 0처리

        elif func_nm in ['value', 'wlogy']:
            result = arr[calc_length:]
            result[np.isnan(result)] = 0  # TODO: 임시로 nan값 0처리
            if debug:
                result_debug = arr_debug[calc_length:]
                result_debug[np.isnan(result_debug)] = 0  # TODO: 임시로 nan값 0처리

        feature, label = self.split_data_label(result)
        if debug:
            n_error = np.sum(np.abs(feature - result_debug) >= 1e-5)
            if n_error != 0:
                print("[debug: {} nd: {}] data not matched.".format(func_nm, nd))
                raise AssertionError

        # 라벨에 대한 고민. 5 sampling_days에서 logy_20을 구하는 경우,
        # 5일 뒤의 logy_20일지 20일 뒤의 logy_20일지..
        # 첫번째의 경우 label[::k_days], 두번째의 경우 label[::n]
        if label is None:
            label_ = None
        # label 산출을 위한 최소한의 데이터가 없는 경우 (ex. predict)
        elif func_nm == 'fft':
            if len(label) <= k_days:
                label_ = None
            else:
                label_ = label[k_days]
        else:
            if len(label) <= n:
                label_ = None
            else:
                label_ = label[n]
        return feature[::self.sampling_days], label_

    def calc_features(self, arr, debug=False, calc_list=None):
        # type_ in ['logp', 'size', 'turnover']

        # log_p_arr shape : [n_days per each base_d, codes_list]
        features_dict = dict()
        labels_dict = dict()
        if calc_list is None:       # logp base features 전체 계산
            for func_nm in self.possible_func['logp_base']:
                if func_nm in ['logp']:
                    nm = '{}_{}'.format(func_nm, 0)
                    features_dict[nm], labels_dict[nm] = self.calc_func(arr, nm, debug)
                elif func_nm == 'fft':
                    for n in [3, 6, 100]:
                        nm = '{}_{}'.format(func_nm, n)
                        features_dict[nm], labels_dict[nm] = self.calc_func(arr, nm, debug)
                else:
                    for n in [5, 10, 20, 60, 120, 250]:
                        nm = '{}_{}'.format(func_nm, n)
                        features_dict[nm], labels_dict[nm] = self.calc_func(arr, nm, debug)
        else:                       # 일부 features 계산 (매크로데이터의 경우 다 계산할 필요 X)
            for nm in calc_list:
                features_dict[nm], labels_dict[nm] = self.calc_func(arr, nm, debug)

        return features_dict, labels_dict

    def get_weighted_arr(self, logp_arr, wgt_arr):
        delayed_wgt_arr = np.ones_like(wgt_arr)
        delayed_wgt_arr[self.k_days:, :] = wgt_arr[:-self.k_days, :]
        wlogy = np.exp(log_y_nd(logp_arr, self.k_days)) * wgt_arr
        # wstd = std_nd(logp_arr, self.k_days) * wgt_arr
        # wstdnew = std_nd_new(logp_arr, self.k_days) * wgt_arr
        return wlogy #, wstd, wstdnew

    def labels_for_mtl(self, features_list, labels, size_factor, importance_wgt):
        labels_mtl = dict()
        for cls in self.features_structure.keys():
            for key in self.features_structure[cls].keys():
                n_arr = self.features_structure[cls][key]
                if cls == 'classification':    # classification
                    for n in n_arr:
                        feature_nm = '{}_{}'.format(key, n)
                        labels_mtl[feature_nm] = np.stack([labels[:, :, features_list.index(feature_nm)] > 0,
                                                           labels[:, :, features_list.index(feature_nm)] <= 0],
                                                          axis=-1) * 1.
                else:
                    labels_mtl[key] = np.stack([labels[:, :, features_list.index("{}_{}".format(key, n))]
                                                for n in n_arr], axis=-1)

                    # if key == 'cslogy':
                    #     labels_mtl['cslogy_idx'] = [features_list.index("{}_{}".format(key, n)) for n in n_arr]

        labels_mtl['size_factor'] = size_factor
        labels_mtl['importance_wgt'] = importance_wgt

        return labels_mtl


