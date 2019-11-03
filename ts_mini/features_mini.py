
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


def weight_scale(score, method='L_60'):
    method = method.lower()
    m_args = method.split('_')

    scale = np.zeros_like(score)
    # score값에 따라 기본 ew/mw weight * scale 해주는 값
    if m_args[0] == 'bm':
        scale[:] = 1.
    else:
        rank_ = np.argsort(-score)  # 값이 큰 순서
        if m_args[0] == 'l':
            # 상위 n_percent% 투자
            # ex) 'L_60', 'L_80', ...
            assert len(m_args[1:]) == 1
            n_percent = int(m_args[1]) / 100
            lower_bound = 0
            upper_bound = int(len(rank_) * n_percent)
            scale[rank_[lower_bound:upper_bound]] = 1.
        elif m_args[0] == 'ls':
            # 롱숏 ntile분위, 각분위 당 wgt_diff 씩 조정 배분
            # ex) 'ls_5_20', 'LS_4_10', ...
            assert len(m_args[1:]) == 2
            ntile = int(m_args[1])
            wgt_diff = int(m_args[2]) / 100
            for i in range(ntile):
                lower_bound = int(len(rank_) * (i / ntile))
                upper_bound = int(len(rank_) * ((i + 1.) / ntile))
                scale[rank_[lower_bound:upper_bound]] = (1 + wgt_diff * (ntile - 1) / 2) - wgt_diff * i
        elif m_args[0] == 'each':
            ntile = int(m_args[1])
            scale = np.zeros([len(score), ntile])
            for i in range(ntile):
                lower_bound = int(len(rank_) * (i / ntile))
                upper_bound = int(len(rank_) * ((i + 1.) / ntile))
                scale[rank_[lower_bound:upper_bound], i] = 1.

    return scale


class FeatureNew:
    def __init__(self, configs):
        self.name = configs.generate_name()
        self.calc_length = configs.calc_length
        self.calc_length_label = configs.calc_length_label
        self.m_days = configs.m_days
        self.k_days = configs.k_days
        self.delay_days = configs.delay_days
        self.sampling_days = configs.sampling_days
        # 아래 함수 추가할때마다 추가해줄것...
        self.possible_func = ['logy', 'std', 'stdnew', 'pos', 'mdd', 'fft', 'cslogy', 'csstd']
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

        feature, label = self.split_data_label(result)
        if debug:
            n_error = np.sum(feature - result_debug)
            if n_error != 0:
                print("[debug: {}] data not matched.".format(func_nm))
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

    def calc_features(self, log_p_arr, transpose=False, debug=False):
        if transpose:
            log_p_arr = np.transpose(log_p_arr)

        # log_p_arr shape : (days per each date, codes_list)
        features_dict = dict()
        labels_dict = dict()
        for func_nm in self.possible_func:
            if func_nm == 'fft':
                for n in [3, 6, 100]:
                    nm = '{}_{}'.format(func_nm, n)
                    features_dict[nm], labels_dict[nm] = self.calc_func(log_p_arr, nm, debug)
            else:
                for n in [5, 10, 20, 60, 120]:
                    nm = '{}_{}'.format(func_nm, n)
                    features_dict[nm], labels_dict[nm] = self.calc_func(log_p_arr, nm, debug)

        if transpose:
            for key in features_dict.keys():
                features_dict[key] = np.transpose(features_dict[key])

            for key in labels_dict.keys():
                labels_dict[key] = np.transpose(labels_dict[key])

        return features_dict, labels_dict

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

                    if key == 'cslogy':
                        labels_mtl['cslogy_idx'] = [features_list.index("{}_{}".format(key, n)) for n in n_arr]

        labels_mtl['size_factor'] = size_factor
        labels_mtl['importance_wgt'] = importance_wgt

        return labels_mtl


class Performance:
    def __init__(self, configs):
        self.configs = configs
        self.label_feature = configs.label_feature
        self.pred_feature = configs.pred_feature
        self.cost_rate = configs.cost_rate

    def define_variables(self, t_steps, assets, f_keys=None):
        var_dict = dict(y=np.zeros([t_steps, 1]),
                        turnover=np.zeros([t_steps, 1]),
                        total_cost=np.zeros([t_steps, 1]),
                        y_w_cost=np.zeros([t_steps, 1]),
                        wgt=pd.DataFrame({'old': np.zeros_like(assets), 'new': np.zeros_like(assets)},
                                         index=assets,
                                         dtype=np.float32))  # column 0: old wgt, 1: new wgt
        if f_keys is not None:
            for key in f_keys:
                var_dict[key] = np.zeros([t_steps, 1])

        return var_dict

    def define_variables_ntile(self, t_steps, assets, n_tile, f_keys=None):
        var_dict = dict(y=np.zeros([t_steps, 1]),
                        y_each=np.zeros([t_steps, n_tile]),
                        turnover=np.zeros([t_steps, 1]),
                        total_cost=np.zeros([t_steps, 1]),
                        y_w_cost=np.zeros([t_steps, 1]),
                        wgt=pd.DataFrame({'old': np.zeros_like(assets), 'new': np.zeros_like(assets)},
                                         index=assets,
                                         dtype=np.float32))  # column 0: old wgt, 1: new wgt
        if f_keys is not None:
            for key in f_keys:
                var_dict[key] = np.zeros([t_steps, 1])
                var_dict[key + '_each'] = np.zeros([t_steps, n_tile])

        return var_dict


    # call by reference (var_dict를 파라미터로 받아서 업데이트)
    def calculate_cost(self, t, var_dict, assets, label_y):
        # nickname
        wgt_ = var_dict['wgt']

        var_dict['turnover'][t] = np.sum(np.abs(wgt_['new'] - wgt_['old']))
        var_dict['total_cost'][t] = var_dict['total_cost'][t - 1] + var_dict['turnover'][t] * self.cost_rate
        wgt_.loc[:, 'old'] = 0.0
        wgt_.loc[assets, 'old'] = ((1 + label_y) * wgt_.loc[assets, 'new']) / np.sum((1 + label_y) * wgt_.loc[assets, 'new'])
        var_dict['y_w_cost'][t] = np.sum(label_y * wgt_.loc[assets, 'new']) - var_dict['turnover'][t] * self.cost_rate

    def predict_plot_mtl_cross_section_test(self, model, dataset_list, save_dir
                                            , file_nm='test.png'
                                            , ylog=False
                                            , t_stepsize=1
                                            , ls_method='ls_5_20'
                                            , plot_all_features=True):
        m_args = ls_method.split('_')
        if m_args[0] == 'ls':
            n_tile = int(m_args[1])
            def_variables = self.define_variables_ntile
            kw = {'n_tile': n_tile}
        else:
            n_tile = -1
            def_variables = self.define_variables
            kw = {}

        if dataset_list is False:
            return False

        ie_list, od_list, td_list, features_list, add_infos, start_d, end_d = dataset_list
        size_factor_list = [np.array(add_info['size_factor'], dtype=np.float32).reshape([-1, 1, 1]) for add_info in add_infos]
        mktcap_list = [np.array(add_info['size_factor_mktcap'], dtype=np.float32).reshape([-1, 1, 1]) for add_info in add_infos]
        assets_list = [add_info['asset_list'] for add_info in add_infos]

        all_assets_list = list()
        for assets in assets_list:
            all_assets_list = sorted(list(set(all_assets_list + assets)))

        idx_y = features_list.index(self.label_feature)
        t_steps = int(np.ceil(len(ie_list) / t_stepsize)) + 1

        # define variables to save values
        if plot_all_features:
            model_keys = list(model.predictor.keys())
            features_for_plot = ['main'] + model_keys
        else:
            model_keys = None
            features_for_plot = ['main']

        ew_dict = dict(bm=self.define_variables(t_steps=t_steps, assets=all_assets_list),
                       model=def_variables(t_steps=t_steps, assets=all_assets_list, f_keys=model_keys, **kw))
        mw_dict = dict(bm=self.define_variables(t_steps=t_steps, assets=all_assets_list),
                       model=def_variables(t_steps=t_steps, assets=all_assets_list, f_keys=model_keys, **kw))

        # nickname
        bm_ew = ew_dict['bm']
        bm_mw = mw_dict['bm']
        model_ew = ew_dict['model']
        model_mw = mw_dict['model']

        for i, (ie_t, od_t, td_t, size_, mktcap, assets) \
                in enumerate(zip(ie_list, od_list, td_list, size_factor_list, mktcap_list, assets_list)):
            if i % t_stepsize != 0:
                continue
            t = i // t_stepsize + 1

            # data format
            assert np.sum(ie_t[:, -1, idx_y] - od_t[:, 0, idx_y]) == 0
            new_output_t = np.zeros_like(od_t)
            new_output_t[:, 0, :] = od_t[:, 0, :] + size_[:, 0, :]
            new_output_t2 = np.zeros_like(od_t)
            new_output_t2[:, 0, :] = od_t[:, 0, :] + mktcap[:, 0, :]

            features = {'input': ie_t, 'output': new_output_t}
            labels = td_t
            label_y = labels[:, 0, idx_y]
            mc = mktcap[:, 0, 0]

            assets = np.array(assets)

            # ############ For BenchMark ############

            bm_ew['y'][t] = np.mean(label_y)
            bm_mw['y'][t] = np.sum(label_y * mc) / np.sum(mc)

            # cost calculation
            bm_ew['wgt'].loc[assets, 'new'] = 1. / len(assets)
            self.calculate_cost(t, bm_ew, assets, label_y)

            bm_mw['wgt'].loc[assets, 'new'] = mc / np.sum(mc)
            self.calculate_cost(t, bm_mw, assets, label_y)

            # ############ For Model ############
            # prediction
            predictions = model.predict_mtl(features)
            value_ = dict()

            value_['cslogy'] = predictions['cslogy'][:, 0, 0]
            for f_ in features_for_plot:
                f_for_y = ('y' if f_ == 'main' else f_)
                if f_ == 'main':
                    value_['main'] = predictions[self.pred_feature][:, 0, 0]
                else:
                    value_[f_] = predictions[f_][:, 0, 0]

                if m_args[0] == 'ls':
                    # ntile 별 수익률
                    scale_n = weight_scale(value_[f_], method='each_{}'.format(n_tile))
                    model_ew[f_for_y + '_each'][t, :] = np.matmul(label_y, scale_n) / np.sum(scale_n, axis=0)
                    model_mw[f_for_y + '_each'][t, :] = np.matmul(label_y * mc, scale_n) / np.matmul(mc, scale_n)
                    # or np.sum((label_y * mc).reshape([-1, 1]) * scale, axis=0) / np.sum(mc.reshape(-1, 1) * scale, axis=0)

                    # pf 수익률
                    scale = weight_scale(value_[f_], method=ls_method)
                elif m_args[0] == 'l':
                    scale1 = weight_scale(value_[f_], method=ls_method)
                    scale = scale1 * weight_scale(value_['cslogy'], method=ls_method)

                model_ew[f_for_y][t] = np.sum(label_y * scale) / np.sum(scale)
                model_mw[f_for_y][t] = np.sum(label_y * mc * scale) / np.sum(mc * scale)

                # cost calculation
                model_ew['wgt'].loc[assets, 'new'] = scale / np.sum(scale)
                self.calculate_cost(t, model_ew, assets, label_y)

                model_mw['wgt'].loc[assets, 'new'] = mc * scale / np.sum(mc * scale)
                self.calculate_cost(t, model_mw, assets, label_y)

        for f_ in features_for_plot:
            f_for_y = ('y' if f_ == 'main' else f_)
            if m_args[0] == 'ls':
                y_arr = np.concatenate([bm_ew['y'], bm_mw['y']
                                        , model_ew[f_for_y]
                                        , model_mw[f_for_y]
                                        , model_ew[f_for_y + '_each']
                                        , model_mw[f_for_y + '_each']], axis=-1)
                data = pd.DataFrame(np.cumprod(1. + y_arr, axis=0)
                                    , columns=['bm_ew', 'bm_mw', 'model_ew', 'model_mw']
                                              + ['model_e{}'.format(i+1) for i in range(n_tile)]
                                              + ['model_m{}'.format(i+1) for i in range(n_tile)])
            else:
                y_arr = np.concatenate([bm_ew['y'], bm_mw['y'], model_ew[f_for_y], model_mw[f_for_y]], axis=-1)
                data = pd.DataFrame(np.cumprod(1. + y_arr, axis=0), columns=['bm_ew', 'bm_mw', 'model_ew', 'model_mw'])
            data['diff'] = np.cumprod(1. + model_ew[f_for_y] - bm_ew['y'])
            data['diff_mw'] = np.cumprod(1. + model_mw[f_for_y] - bm_mw['y'])

            if f_ == 'main':
                data['bm_cost'] = bm_mw['total_cost']
                data['bm_turnover'] = bm_mw['turnover']
                data['bm_y_w_cost'] = np.cumprod(1. + bm_mw['y_w_cost'], axis=0) - 1.
                data['model_cost'] = model_mw['total_cost']
                data['model_turnover'] = model_mw['turnover']
                data['model_y_w_cost'] = np.cumprod(1. + model_mw['y_w_cost'], axis=0) - 1.
                data['diff_w_cost'] = np.cumprod(1. + model_mw['y_w_cost'] - bm_mw['y_w_cost'], axis=0) - 1.

            # ################################ figure 1
            if m_args[0] == 'ls':
                # equal fig
                fig = plt.figure()
                fig.suptitle('{} ~ {}'.format(start_d, end_d))
                if f_ == 'main':
                    grid = plt.GridSpec(ncols=2, nrows=3, figure=fig)
                    ax1 = fig.add_subplot(grid[0, 0])
                    ax2 = fig.add_subplot(grid[0, 1])
                    ax3 = fig.add_subplot(grid[1, 0])
                    ax4 = fig.add_subplot(grid[1, 1])
                    ax5 = fig.add_subplot(grid[2, :])
                else:
                    ax1 = fig.add_subplot(221)
                    ax2 = fig.add_subplot(222)
                    ax3 = fig.add_subplot(223)
                    ax4 = fig.add_subplot(224)
                ax1.plot(data[['bm_ew', 'model_ew', 'model_e1', 'model_e{}'.format(n_tile)]])
                box = ax1.get_position()
                ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax1.legend(['true_y', 'long-short', 'long', 'short'], loc='center left', bbox_to_anchor=(1, 0.5))
                if ylog:
                    ax1.set_yscale('log', basey=2)

                ax2.plot(data[['bm_ew'] + ['model_e{}'.format(i + 1) for i in range(n_tile)]])
                box = ax2.get_position()
                ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax2.legend(['true_y'] + ['q{}'.format(i + 1) for i in range(n_tile)], loc='center left',
                           bbox_to_anchor=(1, 0.5))
                ax2.set_yscale('log', basey=2)

                # value fig
                ax3.plot(data[['bm_mw', 'model_mw', 'model_m1', 'model_m{}'.format(n_tile)]])
                box = ax3.get_position()
                ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax3.legend(['true_y(mw)', 'long-short', 'long', 'short'], loc='center left',
                           bbox_to_anchor=(1, 0.5))
                if ylog:
                    ax3.set_yscale('log', basey=2)

                ax4.plot(data[['bm_mw'] + ['model_m{}'.format(i + 1) for i in range(n_tile)]])
                box = ax4.get_position()
                ax4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax4.legend(['true_y(mw)'] + ['q{}'.format(i + 1) for i in range(n_tile)], loc='center left',
                           bbox_to_anchor=(1, 0.5))
                ax4.set_yscale('log', basey=2)

                if f_ == 'main':
                    data[['bm_y_w_cost', 'model_y_w_cost']].plot(ax=ax5, colormap=cm.Set2)
                    box = ax5.get_position()
                    ax5.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax5.legend(['bm_y_w_cost', 'model_y_w_cost'], loc='center left', bbox_to_anchor=(1, 0.8))
                    if ylog:
                        ax5.set_yscale('log', basey=2)

                    ax5_2 = ax5.twinx()
                    data[['diff_w_cost']].plot(ax=ax5_2, colormap=cm.jet)
                    box = ax5_2.get_position()
                    ax5_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax5_2.legend(['diff_w_cost'], loc='center left', bbox_to_anchor=(1, 0.2))
                    if ylog:
                        ax5_2.set_yscale('log', basey=2)

                if file_nm is None:
                    save_file_name = '{}/{}'.format(save_dir, '_all.png')
                else:
                    save_dir_v = os.path.join(save_dir, f_)
                    os.makedirs(save_dir_v, exist_ok=True)
                    file_nm_v = file_nm.replace(file_nm[-4:], "_{}{}".format(f_, file_nm[-4:]))
                    save_file_name = '{}/{}'.format(save_dir_v, file_nm_v)

                fig.savefig(save_file_name)
                # print("figure saved. (dir: {})".format(save_file_name))
                plt.close(fig)

            elif m_args[0] == 'l':
                # equal fig
                fig = plt.figure()
                fig.suptitle('{} ~ {}'.format(start_d, end_d))
                if f_ == 'main':
                    ax1 = fig.add_subplot(311)
                    ax2 = fig.add_subplot(312)
                    ax3 = fig.add_subplot(313)
                    # ax4 = fig.add_subplot(414)
                else:
                    ax1 = fig.add_subplot(211)
                    ax2 = fig.add_subplot(212)

                data[['bm_ew', 'model_ew']].plot(ax=ax1, colormap=cm.Set2)
                box = ax1.get_position()
                ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax1.legend(['bm_ew', 'model_ew'], loc='center left', bbox_to_anchor=(1, 0.8))
                if ylog:
                    ax1.set_yscale('log', basey=2)

                ax1_2 = ax1.twinx()
                data[['diff']].plot(ax=ax1_2, colormap=cm.jet)
                box = ax1_2.get_position()
                ax1_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax1_2.legend(['diff'], loc='center left', bbox_to_anchor=(1, 0.2))
                if ylog:
                    ax1_2.set_yscale('log', basey=2)

                data[['bm_mw', 'model_mw']].plot(ax=ax2, colormap=cm.Set2)
                box = ax2.get_position()
                ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax2.legend(['bm_mw', 'model_mw'], loc='center left', bbox_to_anchor=(1, 0.8))
                if ylog:
                    ax2.set_yscale('log', basey=2)

                ax2_2 = ax2.twinx()
                data[['diff_mw']].plot(ax=ax2_2, colormap=cm.jet)
                box = ax2_2.get_position()
                ax2_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax2_2.legend(['diff_mw'], loc='center left', bbox_to_anchor=(1, 0.2))
                if ylog:
                    ax2_2.set_yscale('log', basey=2)

                if f_ == 'main':
                    data[['bm_y_w_cost', 'model_y_w_cost']].plot(ax=ax3, colormap=cm.Set2)
                    box = ax3.get_position()
                    ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax3.legend(['bm_y_w_cost', 'model_y_w_cost'], loc='center left', bbox_to_anchor=(1, 0.8))
                    if ylog:
                        ax3.set_yscale('log', basey=2)

                    ax3_2 = ax3.twinx()
                    data[['diff_w_cost']].plot(ax=ax3_2, colormap=cm.jet)
                    box = ax3_2.get_position()
                    ax3_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax3_2.legend(['diff_w_cost'], loc='center left', bbox_to_anchor=(1, 0.2))
                    if ylog:
                        ax3_2.set_yscale('log', basey=2)

                if file_nm is None:
                    save_file_name = '{}/{}'.format(save_dir, '_all.png')
                else:
                    save_dir_v = os.path.join(save_dir, f_)
                    os.makedirs(save_dir_v, exist_ok=True)
                    file_nm_v = file_nm.replace(file_nm[-4:], "_{}{}".format(f_, file_nm[-4:]))
                    save_file_name = '{}/{}'.format(save_dir_v, file_nm_v)

                fig.savefig(save_file_name)
                # print("figure saved. (dir: {})".format(save_file_name))
                plt.close(fig)

    def predict_plot_mtl_cross_section_test_long(self, model, dataset_list, save_dir
                                                 , file_nm='test.png'
                                                 , ylog=False
                                                 , t_stepsize=1
                                                 , invest_rate=0.8
                                                 , plot_all_features=True):
        if dataset_list is False:
            return False

        # load data
        ie_list, od_list, td_list, features_list, add_infos, start_d, end_d = dataset_list
        size_factor_list = [np.array(add_info['size_factor'], dtype=np.float32).reshape([-1, 1, 1]) for add_info in add_infos]
        mktcap_list = [np.array(add_info['size_factor_mktcap'], dtype=np.float32).reshape([-1, 1, 1]) for add_info in add_infos]
        assets_list = [add_info['asset_list'] for add_info in add_infos]

        all_assets_list = list()
        for assets in assets_list:
            all_assets_list = sorted(list(set(all_assets_list + assets)))

        # define parameter
        idx_y = features_list.index(self.label_feature)
        t_steps = int(np.ceil(len(ie_list) / t_stepsize)) + 1

        # define variables to save values
        if plot_all_features:
            model_keys = list(model.predictor.keys())
            plot_list = ['main'] + model_keys
        else:
            model_keys = None
            plot_list = ['main']

        ew_dict = dict(bm=self.define_variables(t_steps=t_steps, assets=all_assets_list),
                       model=self.define_variables(t_steps=t_steps, assets=all_assets_list, f_keys=model_keys))
        mw_dict = dict(bm=self.define_variables(t_steps=t_steps, assets=all_assets_list),
                       model=self.define_variables(t_steps=t_steps, assets=all_assets_list, f_keys=model_keys))

        # nickname
        bm_ew = ew_dict['bm']
        bm_mw = mw_dict['bm']
        model_ew = ew_dict['model']
        model_mw = mw_dict['model']

        for i, (ie_t, od_t, td_t, size_, mktcap, assets) \
                in enumerate(zip(ie_list, od_list, td_list, size_factor_list, mktcap_list, assets_list)):
            if i % t_stepsize != 0:
                continue
            t = i // t_stepsize + 1

            # data format
            assert np.sum(ie_t[:, -1, idx_y] - od_t[:, 0, idx_y]) == 0
            new_output_t = np.zeros_like(od_t)
            new_output_t[:, 0, :] = od_t[:, 0, :] + size_[:, 0, :]
            new_output_t2 = np.zeros_like(od_t)
            new_output_t2[:, 0, :] = od_t[:, 0, :] + mktcap[:, 0, :]

            features = {'input': ie_t, 'output': new_output_t}
            labels = td_t
            label_y = labels[:, 0, idx_y]
            mc = mktcap[:, 0, 0]

            # ############ For BenchMark ############

            bm_ew['y'][t] = np.mean(label_y)
            bm_mw['y'][t] = np.sum(label_y * mc) / np.sum(mc)

            # cost calculation
            assets = np.array(assets)
            self.calculate_cost(t, bm_mw, assets, label_y, mc)

            # ############ For Model ############
            # prediction
            predictions = model.predict_mtl(features)
            value_ = dict()

            for feature in plot_list:
                if feature == 'main':
                    value_['main'] = predictions[self.pred_feature][:, 0, 0]
                else:
                    value_[feature] = predictions[feature][:, 0, 0]

            low_crit_cslogy, high_crit_cslogy = np.percentile(value_['cslogy'], q=[100 * (1 - invest_rate), 100])

            for v in plot_list:
                if v in ['logy', 'cslogy', 'fft', 'pos_5', 'pos_20', 'pos_60', 'pos_120', 'main']:
                    low_q, high_q = 100 * (1 - invest_rate), 100
                else:
                    low_q, high_q = 0, 100 * invest_rate

                low_crit, high_crit = np.percentile(value_[v], q=[low_q, high_q])

                # 단일 조건
                # crit1 = ((value_[v] >= low_crit) & (value_[v] < high_crit)).numpy()

                # cslogy와 중복조건
                crit2 = ((value_[v] >= low_crit) & (value_[v] < high_crit)
                         & (value_['cslogy'] >= low_crit_cslogy) & (value_['cslogy'] < high_crit_cslogy)).numpy()

                if v == 'main':
                    model_ew['y'][t] = np.mean(label_y[crit2])
                    model_mw['y'][t] = np.sum(label_y[crit2] * mc[crit2]) / np.sum(mc[crit2])
                    # cost calculation
                    self.calculate_cost(t, model_mw, assets, label_y, mc, crit=crit2)
                else:
                    model_ew[v][t] = np.mean(label_y[crit2])
                    model_mw[v][t] = np.sum(label_y[crit2] * mc[crit2]) / np.sum(mc[crit2])

        for v_ in plot_list:
            v_for_y = ('y' if v_ == 'main' else v_)
            y_arr = np.concatenate([bm_ew['y'], bm_mw['y'], model_ew[v_for_y], model_mw[v_for_y]], axis=-1)
            data = pd.DataFrame(np.cumprod(1. + y_arr, axis=0), columns=['bm_ew', 'bm_mw', 'model_ew', 'model_mw'])
            data['diff'] = np.cumprod(1. + model_ew[v_for_y] - bm_ew['y'])
            data['diff_mw'] = np.cumprod(1. + model_mw[v_for_y] - bm_mw['y'])

            if v_ == 'main':
                data['bm_cost'] = bm_mw['total_cost']
                data['bm_turnover'] = bm_mw['turnover']
                data['bm_y_w_cost'] = np.cumprod(1. + bm_mw['y_w_cost'], axis=0) - 1.
                data['model_cost'] = model_mw['total_cost']
                data['model_turnover'] = model_mw['turnover']
                data['model_y_w_cost'] = np.cumprod(1. + model_mw['y_w_cost'], axis=0) - 1.
                data['diff_w_cost'] = np.cumprod(1. + model_mw['y_w_cost'] - bm_mw['y_w_cost'], axis=0) - 1.

            # ################################ figure 1
            # equal fig
            fig = plt.figure()
            fig.suptitle('{} ~ {}'.format(start_d, end_d))
            if v_ == 'main':
                ax1 = fig.add_subplot(311)
                ax2 = fig.add_subplot(312)
                ax3 = fig.add_subplot(313)
                # ax4 = fig.add_subplot(414)
            else:
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)

            data[['bm_ew', 'model_ew']].plot(ax=ax1, colormap=cm.Set2)
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax1.legend(['bm_ew', 'model_ew'], loc='center left', bbox_to_anchor=(1, 0.8))
            if ylog:
                ax1.set_yscale('log', basey=2)

            ax1_2 = ax1.twinx()
            data[['diff']].plot(ax=ax1_2, colormap=cm.jet)
            box = ax1_2.get_position()
            ax1_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax1_2.legend(['diff'], loc='center left', bbox_to_anchor=(1, 0.2))
            if ylog:
                ax1_2.set_yscale('log', basey=2)

            data[['bm_mw', 'model_mw']].plot(ax=ax2, colormap=cm.Set2)
            box = ax2.get_position()
            ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax2.legend(['bm_mw', 'model_mw'], loc='center left', bbox_to_anchor=(1, 0.8))
            if ylog:
                ax2.set_yscale('log', basey=2)

            ax2_2 = ax2.twinx()
            data[['diff_mw']].plot(ax=ax2_2, colormap=cm.jet)
            box = ax2_2.get_position()
            ax2_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax2_2.legend(['diff_mw'], loc='center left', bbox_to_anchor=(1, 0.2))
            if ylog:
                ax2_2.set_yscale('log', basey=2)

            if v_ == 'main':
                data[['bm_y_w_cost', 'model_y_w_cost']].plot(ax=ax3, colormap=cm.Set2)
                box = ax3.get_position()
                ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax3.legend(['bm_y_w_cost', 'model_y_w_cost'], loc='center left', bbox_to_anchor=(1, 0.8))
                if ylog:
                    ax3.set_yscale('log', basey=2)

                ax3_2 = ax3.twinx()
                data[['diff_w_cost']].plot(ax=ax3_2, colormap=cm.jet)
                box = ax3_2.get_position()
                ax3_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax3_2.legend(['diff_w_cost'], loc='center left', bbox_to_anchor=(1, 0.2))
                if ylog:
                    ax3_2.set_yscale('log', basey=2)

            if file_nm is None:
                save_file_name = '{}/{}'.format(save_dir, '_all.png')
            else:
                save_dir_v = os.path.join(save_dir, v_)
                os.makedirs(save_dir_v, exist_ok=True)
                file_nm_v = file_nm.replace(file_nm[-4:], "_{}{}".format(v_, file_nm[-4:]))
                save_file_name = '{}/{}'.format(save_dir_v, file_nm_v)

            fig.savefig(save_file_name)
            # print("figure saved. (dir: {})".format(save_file_name))
            plt.close(fig)



class Feature:
    def __init__(self, configs):
        self._init_features(configs)

    def _init_features(self, configs):
        # dict: classification
        # self.structure = dict()
        # for cls in configs.features_summary.keys():
        #     for key in configs.features_summary[cls].keys():
        #         n_arr, unit = configs.features_summary[cls][key]
        #         if cls == 'regression':
        #             self.structure[key] = [str(n) + unit for n in n_arr]
        #         elif cls == 'classification':
        #             for n in n_arr:
        #                 self.structure["{}_{}".format(key, n) + unit] = str(n) + unit
        self.features_structure = configs.features_structure
        self.model_predictor_list = configs.model_predictor_list
        # self.structure = {
        #     # 'logy': ['5d', '20d', '60d', '120d', '250d'],
        #     'logy': ['5d', '20d', '60d', '120d'],
        #     'pos_5d': '5d',
        #     'pos_20d': '20d',
        #     'pos_60d': '60d',
        #     'std': ['20d', '60d', '120d'],
        #     # 'mdd': ['20d', '60d', '120d'],
        #     'mdd': ['20d', '60d'],
        #     'fft': ['3com', '100com']
        # }
        # self.model_predictor_list = ['logy', 'pos_5d', 'pos_20d', 'std', 'mdd', 'fft']
        self.label_feature = configs.label_feature
        self.pred_feature = configs.pred_feature

        self.cost_rate = configs.cost_rate

    def labels_for_mtl(self, features_list, labels, size_value, importance_wgt):
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

                    if key == 'cslogy':
                        labels_mtl['cslogy_idx'] = [features_list.index("{}_{}".format(key, n)) for n in n_arr]

        labels_mtl['size_value'] = size_value
        labels_mtl['importance_wgt'] = importance_wgt

        return labels_mtl

    def processing_split_new(self, df_not_null, m_days, sampling_days, calc_length=0, label_type=None,
                             delayed_days=0, additional_dict=None):
        # if type(df.columns) == pd.MultiIndex:
        #     df.columns = df.columns.droplevel(0)
        features_data_dict = dict()
        features_label_dict = dict()
        log_p = np.log(df_not_null.values, dtype=np.float32)

        main_class, sub_class = self.label_feature.split('_')
        n_days = int(sub_class)
        k_days_adj = n_days + delayed_days

        if label_type is None:
            assert len(log_p) == ((calc_length + m_days) + 1)

            log_p_wo_calc = log_p[calc_length:]
            assert len(log_p_wo_calc) == (m_days + 1)

            log_p = log_p - log_p[0, :]
            log_p_wo_calc = log_p_wo_calc - log_p_wo_calc[0, :]

            for cls in self.features_structure.keys():
                for key in self.features_structure[cls]:
                    features_data_dict[key] = dict()
                    for nd in self.features_structure[cls][key]:
                        if key in ['logy']:
                            features_data_dict[key][str(nd)] = log_y_nd(log_p, nd)[calc_length:][:(m_days+1)]
                        elif key == 'std':
                            features_data_dict[key][str(nd)] = std_nd(log_p, nd)[calc_length:][:(m_days + 1)]
                        elif key == 'stdnew':
                            features_data_dict[key][str(nd)] = std_nd_new(log_p, nd)[calc_length:][:(m_days + 1)]
                        elif key == 'pos':
                            features_data_dict[key][str(nd)] = np.sign(features_data_dict['logy'][str(nd)])
                        elif key == 'mdd':
                            features_data_dict[key][str(nd)] = mdd_nd(log_p_wo_calc, nd)[:(m_days + 1)]
                        elif key == 'fft':
                            features_data_dict[key][str(nd)] = fft(log_p_wo_calc, nd, m_days, k_days_adj)[:(m_days + 1)]
                        elif key == 'cslogy':
                            features_data_dict[key][str(nd)] = arr_to_cs(log_y_nd(log_p, nd)[calc_length:][:(m_days+1)])
                        elif key == 'csstd':
                            features_data_dict[key][str(nd)] = arr_to_cs(std_nd_new(log_p, nd)[calc_length:][:(m_days + 1)])

        else:
            # 1 day adj.
            log_p_wo_calc = log_p[calc_length:][:(k_days_adj + m_days + 1)]
            assert len(log_p_wo_calc) == (k_days_adj + m_days + 1)

            log_p = log_p - log_p[0, :]
            log_p_wo_calc = log_p_wo_calc - log_p_wo_calc[0, :]

            for cls in self.features_structure.keys():
                for key in self.features_structure[cls]:
                    features_data_dict[key] = dict()
                    for nd in self.features_structure[cls][key]:
                        if key in ['logy']:
                            features_data_dict[key][str(nd)] = log_y_nd(log_p, nd)[calc_length:][:(m_days+1)]
                        elif key == 'std':
                            features_data_dict[key][str(nd)] = std_nd(log_p, nd)[calc_length:][:(m_days + 1)]
                        elif key == 'stdnew':
                            features_data_dict[key][str(nd)] = std_nd_new(log_p, nd)[calc_length:][:(m_days + 1)]
                        elif key == 'pos':
                            features_data_dict[key][str(nd)] = np.sign(features_data_dict['logy'][str(nd)])
                        elif key == 'mdd':
                            features_data_dict[key][str(nd)] = mdd_nd(log_p_wo_calc, nd)[:(m_days + 1)]
                        elif key == 'fft':
                            features_data_dict[key][str(nd)] = fft(log_p_wo_calc, nd, m_days, k_days_adj)[:(m_days + 1)]
                        elif key == 'cslogy':
                            features_data_dict[key][str(nd)] = arr_to_cs(log_y_nd(log_p, nd)[calc_length:][:(m_days+1)])
                        elif key == 'csstd':
                            features_data_dict[key][str(nd)] = arr_to_cs(std_nd_new(log_p, nd)[calc_length:][:(m_days + 1)])

            if label_type == 'trainable_label':
                # 1 day adj.
                assert len(log_p) == ((calc_length + m_days) + k_days_adj + 1)
                # label part

                for cls in self.features_structure.keys():
                    for key in self.features_structure[cls]:
                        features_label_dict[key] = dict()
                        for nd in self.features_structure[cls][key]:
                            n_freq = np.min([n_days, nd]) + delayed_days
                            if key == 'logy':
                                features_label_dict[key][str(nd)] = log_y_nd(log_p, nd)[(calc_length+m_days):][:(n_freq + 1)][::n_freq]
                            elif key == 'std':
                                features_label_dict[key][str(nd)] = std_nd(log_p, nd)[(calc_length+m_days):][:(n_freq + 1)][::n_freq]
                            elif key == 'stdnew':
                                features_label_dict[key][str(nd)] = std_nd_new(log_p, nd)[(calc_length+m_days):][:(n_freq + 1)][::n_freq]
                            elif key == 'pos':
                                features_label_dict[key][str(nd)] = np.sign(features_label_dict['logy'][str(nd)])
                            elif key == 'mdd':
                                features_label_dict[key][str(nd)] = mdd_nd(log_p_wo_calc, nd)[m_days:][::k_days_adj]
                            elif key == 'fft':
                                features_label_dict[key][str(nd)] = fft(log_p_wo_calc, nd, m_days, k_days_adj)[m_days:][::k_days_adj]
                            elif key == 'cslogy':
                                features_label_dict[key][str(nd)] = arr_to_cs(log_y_nd(log_p, nd)[(calc_length+m_days):][:(n_freq + 1)][::n_freq])
                                # tmp = log_y_nd(log_p, nd)[(calc_length+m_days):][:(n_freq + 1)][::n_freq]
                                # order = tmp.argsort(axis=1)
                                # features_label_dict[key][str(nd)] = order.argsort(axis=1) / np.max(order, axis=1).reshape([-1, 1])
                            elif key == 'csstd':
                                features_label_dict[key][str(nd)] = arr_to_cs(std_nd_new(log_p, nd)[(calc_length+m_days):][:(n_freq + 1)][::n_freq])
                                # tmp = std_nd(log_p, nd)[(calc_length+m_days):][:(n_freq + 1)][::n_freq]
                                # order = tmp.argsort(axis=1)
                                # features_label_dict[key][str(nd)] = order.argsort(axis=1) / np.max(order, axis=1).reshape([-1, 1])

            elif label_type == 'test_label':
                assert len(log_p) == ((calc_length + m_days) + k_days_adj + 1)

                if main_class == 'logy':
                    # 1 day adj.
                    features_label_list = log_y_nd(log_p, n_days)[(calc_length+m_days):][:(k_days_adj + 1)][::k_days_adj]
                else:
                    print('[Feature class]label_type: {} Not Implemented for {}'.format(label_type, main_class))
                    raise NotImplementedError

        features_list = list()
        features_data = list()
        features_label = list()

        for cls in self.features_structure.keys():
            for key in self.features_structure[cls].keys():
                for nd in self.features_structure[cls][key]:
                    features_list.append('{}_{}'.format(key, nd))
                    features_data.append(features_data_dict[key][str(nd)])
                    if label_type == 'trainable_label':
                        features_label.append(features_label_dict[key][str(nd)])

        if additional_dict is not None:
            for key in additional_dict.keys():
                data_raw = np.array(additional_dict[key])[calc_length:][:(m_days + 1)]
                # 1 day adj.
                label_raw = np.array(additional_dict[key])[(calc_length+m_days):][:(k_days_adj + 1)][::k_days_adj]

                # winsorize
                q_bottom = np.percentile(data_raw, q=1)
                q_top = np.percentile(data_raw, q=99)
                data_raw[data_raw > q_top] = q_top
                data_raw[data_raw < q_bottom] = q_bottom
                label_raw[label_raw > q_top] = q_top
                label_raw[label_raw < q_bottom] = q_bottom

                features_list.append(key)
                features_data.append((data_raw - data_raw.mean()) / data_raw.std(ddof=1))
                features_label.append((label_raw - data_raw.mean()) / data_raw.std(ddof=1))

        features_data = np.stack(features_data, axis=-1)[::sampling_days][1:]

        if label_type == 'trainable_label':
            features_label = np.stack(features_label, axis=-1)
        elif label_type == 'test_label':
            features_label = np.expand_dims(features_label_list, axis=-1)

        assert len(features_list) == features_data.shape[-1]
        # feature_df = pd.DataFrame(np.transpose(features_data[:, :, 0]), columns=features_list)
        return features_list, features_data, features_label

    def predict_plot_mtl_cross_section_test(self, model, dataset_list, save_dir, file_nm='test.png', ylog=False, time_step=1):
        if dataset_list is False:
            return False
        else:
            input_enc_list, output_dec_list, target_dec_list, features_list, additional_infos, start_date, end_date = dataset_list

        idx_y = features_list.index(self.label_feature)

        true_y = np.zeros([int(np.ceil(len(input_enc_list) / time_step)) + 1, 1])
        true_y_mw = np.zeros([int(np.ceil(len(input_enc_list) / time_step)) + 1, 1])

        n_tile = 5
        pred_arr = dict()
        pred_arr_mw = dict()
        pred_arr['main'] = np.zeros([len(true_y), n_tile])
        pred_arr_mw['main'] = np.zeros([len(true_y), n_tile])

        pred_arr['ori'] = np.zeros([len(true_y), n_tile])
        pred_arr_mw['ori'] = np.zeros([len(true_y), n_tile])

        pred_arr['cap'] = np.zeros([len(true_y), n_tile])
        pred_arr_mw['cap'] = np.zeros([len(true_y), n_tile])

        turnover_true_mw = np.zeros_like(true_y_mw)
        total_cost_true_mw = np.zeros_like(true_y_mw)
        truy_y_mw_adj = np.zeros_like(true_y_mw)

        turnover_main_mw = np.zeros_like(true_y_mw)
        total_cost_main_mw = np.zeros_like(true_y_mw)
        pred_main_mw_adj = np.zeros_like(true_y_mw)

        for key in model.predictor.keys():
            pred_arr[key] = np.zeros([len(true_y), n_tile])
            pred_arr_mw[key] = np.zeros([len(true_y), n_tile])

        size_value_list = [add_info['size_value'] for add_info in additional_infos]
        mktcap_list = [add_info['mktcap'] for add_info in additional_infos]
        assets_list = [add_info['assets_list'] for add_info in additional_infos]

        all_assets_list = list()
        for assets in assets_list:
            all_assets_list = list(set(all_assets_list + assets))

        wgt_for_cost = pd.DataFrame({'old_wgt_true': np.zeros_like(all_assets_list),
                                     'new_wgt_true': np.zeros_like(all_assets_list),
                                     'old_wgt_main': np.zeros_like(all_assets_list),
                                     'new_wgt_main': np.zeros_like(all_assets_list),
                                     }, index=sorted(all_assets_list), dtype=np.float32)

        for i, (input_enc_t, output_dec_t, target_dec_t, size_value, mktcap, assets) \
                in enumerate(zip(input_enc_list, output_dec_list, target_dec_list, size_value_list, mktcap_list, assets_list)):
            if i % time_step != 0:
                continue
            t = i // time_step + 1
            assert np.sum(input_enc_t[:, -1, idx_y] - output_dec_t[:, 0, idx_y]) == 0
            new_output_t = np.zeros_like(output_dec_t)
            new_output_t[:, 0, :] = output_dec_t[:, 0, :] + size_value[:, 0, :]
            new_output_t2 = np.zeros_like(output_dec_t)
            new_output_t2[:, 0, :] = output_dec_t[:, 0, :] + mktcap[:, 0, :]

            features = {'input': input_enc_t, 'output': new_output_t}
            labels = target_dec_t

            true_y[t] = np.mean(labels[:, 0, idx_y])
            true_y_mw[t] = np.sum(labels[:, 0, idx_y] * mktcap[:, 0, 0]) / np.sum(mktcap[:, 0, 0])

            # cost calculation
            assets = np.array(assets)
            wgt_for_cost.ix[:, 'new_wgt_true'] = 0.0
            wgt_for_cost.ix[assets, 'new_wgt_true'] = mktcap[:, 0, 0] / np.sum(mktcap[:, 0, 0])
            turnover_true_mw[t] = np.sum(np.abs(wgt_for_cost['new_wgt_true'] - wgt_for_cost['old_wgt_true']))
            total_cost_true_mw[t] = total_cost_true_mw[t-1] + turnover_true_mw[t] * self.cost_rate
            wgt_for_cost.ix[:, 'old_wgt_true'] = 0.0
            wgt_for_cost.ix[assets, 'old_wgt_true'] = ((1 + labels[:, 0, idx_y]) * mktcap[:, 0, 0]) / np.sum((1 + labels[:, 0, idx_y]) * mktcap[:, 0, 0])
            truy_y_mw_adj[t] = np.sum(labels[:, 0, idx_y] * mktcap[:, 0, 0]) / np.sum(mktcap[:, 0, 0]) - turnover_true_mw[t] * self.cost_rate

            predictions = model.predict_mtl(features)
            value_ = dict()
            value_['main'] = predictions[self.pred_feature][:, 0, 0]

            for feature in model.predictor.keys():
                value_[feature] = predictions[feature][:, 0, 0]

            predictions_ori = model.predict_mtl({'input': input_enc_t, 'output': output_dec_t})
            value_['ori'] = predictions_ori[self.pred_feature][:, 0, 0]

            predictions_cap = model.predict_mtl({'input': input_enc_t, 'output': new_output_t2})
            value_['cap'] = predictions_cap[self.pred_feature][:, 0, 0]

            for i_tile in range(n_tile):
                low_q, high_q = 100 * (1. - (1.+i_tile) / n_tile),  100 * (1. - i_tile / n_tile)

                for v in value_.keys():
                    low_crit, high_crit = np.percentile(value_[v], q=[low_q, high_q])
                    if high_q == 100:
                        crit_ = (value_[v] >= low_crit)
                    else:
                        crit_ = ((value_[v] >= low_crit) & (value_[v] < high_crit))

                    if v in ['logy', 'cslogy', 'fft', 'pos_5', 'pos_20', 'pos_60', 'pos_120', 'main', 'cap', 'ori']:
                        pred_arr[v][t, i_tile] = np.mean(labels[crit_, 0, idx_y])
                        pred_arr_mw[v][t, i_tile] = np.sum(labels[crit_, 0, idx_y] * mktcap[crit_, 0, 0]) / np.sum(mktcap[crit_, 0, 0])
                    else:
                        pred_arr[v][t, -(i_tile+1)] = np.mean(labels[crit_, 0, idx_y])
                        pred_arr_mw[v][t, -(i_tile+1)] = np.sum(labels[crit_, 0, idx_y] * mktcap[crit_, 0, 0]) / np.sum(mktcap[crit_, 0, 0])

            # cost calculation
            wgt_for_cost.ix[:, 'new_wgt_main'] = 0.0
            for i_tile in range(n_tile):
                low_q, high_q = 100 * (1. - (1. + i_tile) / n_tile), 100 * (1. - i_tile / n_tile)
                low_crit, high_crit = np.percentile(value_['main'], q=[low_q, high_q])
                if high_q == 100:
                    crit_main = (value_['main'] >= low_crit)
                else:
                    crit_main = ((value_['main'] >= low_crit) & (value_['main'] < high_crit))

                wgt_for_cost.ix[assets[crit_main], 'new_wgt_main'] = mktcap[crit_main, 0, 0] * (1 + (2-i_tile) * 0.2)

            wgt_for_cost['new_wgt_main'] = wgt_for_cost['new_wgt_main'] / np.sum(wgt_for_cost['new_wgt_main'])
            turnover_main_mw[t] = np.sum(
                np.abs(wgt_for_cost['new_wgt_main'] - wgt_for_cost['old_wgt_main']))
            total_cost_main_mw[t] = total_cost_main_mw[t - 1] + turnover_main_mw[t] * self.cost_rate
            wgt_for_cost.ix[:, 'old_wgt_main'] = 0.0
            wgt_for_cost.ix[assets, 'old_wgt_main'] = ((1 + labels[:, 0, idx_y]) * wgt_for_cost.ix[assets, 'new_wgt_main']) \
                                                      / np.sum((1 + labels[:, 0, idx_y]) * wgt_for_cost.ix[assets, 'new_wgt_main'])
            pred_main_mw_adj[t] = np.sum(labels[:, 0, idx_y] * wgt_for_cost.ix[assets, 'new_wgt_main']) - turnover_main_mw[t] * self.cost_rate


        for v_ in value_.keys():
            data = pd.DataFrame(np.cumprod(1. + np.concatenate([true_y, true_y_mw, pred_arr[v_], pred_arr_mw[v_]], axis=-1), axis=0),
                                columns=['true_y', 'true_y_mw'] + ['pred_q{}'.format(i + 1) for i in range(n_tile)]
                                        + ['pred_q{}_mw'.format(i + 1) for i in range(n_tile)])
            data['pred_ls'] = np.cumprod(1. + np.mean(pred_arr[v_][:, :1], axis=1) - np.mean(pred_arr[v_][:, -1:], axis=1))
            data['pred_ls2'] = np.cumprod(1. + np.mean(pred_arr[v_][:, :2], axis=1) - np.mean(pred_arr[v_][:, -2:], axis=1))
            data['pred_ls_mw'] = np.cumprod(1. + np.mean(pred_arr_mw[v_][:, :1], axis=1) - np.mean(pred_arr_mw[v_][:, -1:], axis=1))
            data['pred_ls_mw2'] = np.cumprod(1. + np.mean(pred_arr_mw[v_][:, :2], axis=1) - np.mean(pred_arr_mw[v_][:, -2:], axis=1))

            if v_ == 'main':
                data['true_cost'] = total_cost_true_mw
                data['true_turnover'] = turnover_true_mw
                data['true_y_mw_adj'] = np.cumprod(1. + truy_y_mw_adj, axis=0) - 1.
                data['main_cost'] = total_cost_main_mw
                data['main_turnover'] = turnover_main_mw
                data['main_y_mw_adj'] = np.cumprod(1. + pred_main_mw_adj, axis=0) - 1.
                data['diff_adj'] = np.cumprod(1. + pred_main_mw_adj - truy_y_mw_adj, axis=0) - 1.


            # ################################ figure 1
            # equal fig
            fig = plt.figure()
            fig.suptitle('{} ~ {}'.format(start_date, end_date))
            if v_ == 'main':
                grid = plt.GridSpec(ncols=2, nrows=3, figure=fig)
                ax1 = fig.add_subplot(grid[0, 0])
                ax2 = fig.add_subplot(grid[0, 1])
                ax3 = fig.add_subplot(grid[1, 0])
                ax4 = fig.add_subplot(grid[1, 1])
                ax5 = fig.add_subplot(grid[2, :])
            else:
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                ax3 = fig.add_subplot(223)
                ax4 = fig.add_subplot(224)
            ax1.plot(data[['true_y', 'pred_ls', 'pred_ls2', 'pred_q1', 'pred_q{}'.format(n_tile)]])
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax1.legend(['true_y', 'long-short', 'long-short2', 'long', 'short'], loc='center left', bbox_to_anchor=(1, 0.5))
            if ylog:
                ax1.set_yscale('log', basey=2)

            ax2.plot(data[['true_y'] + ['pred_q{}'.format(i + 1) for i in range(n_tile)]])
            box = ax2.get_position()
            ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax2.legend(['true_y'] + ['q{}'.format(i + 1) for i in range(n_tile)], loc='center left', bbox_to_anchor=(1, 0.5))
            ax2.set_yscale('log', basey=2)

            # value fig
            ax3.plot(data[['true_y_mw', 'pred_ls_mw', 'pred_ls_mw2', 'pred_q1_mw', 'pred_q{}_mw'.format(n_tile)]])
            box = ax3.get_position()
            ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax3.legend(['true_y(mw)', 'long-short', 'long-short2', 'long', 'short'], loc='center left', bbox_to_anchor=(1, 0.5))
            if ylog:
                ax3.set_yscale('log', basey=2)

            ax4.plot(data[['true_y_mw'] + ['pred_q{}_mw'.format(i + 1) for i in range(n_tile)]])
            box = ax4.get_position()
            ax4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax4.legend(['true_y(mw)'] + ['q{}'.format(i + 1) for i in range(n_tile)], loc='center left', bbox_to_anchor=(1, 0.5))
            ax4.set_yscale('log', basey=2)

            if v_ == 'main':
                data[['true_y_mw_adj', 'main_y_mw_adj']].plot(ax=ax5, colormap=cm.Set2)
                box = ax5.get_position()
                ax5.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax5.legend(['true_y_mw_adj', 'main_y_mw_adj'], loc='center left', bbox_to_anchor=(1, 0.8))
                if ylog:
                    ax5.set_yscale('log', basey=2)

                ax5_2 = ax5.twinx()
                data[['diff_adj']].plot(ax=ax5_2, colormap=cm.jet)
                box = ax5_2.get_position()
                ax5_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax5_2.legend(['diff_adj'], loc='center left', bbox_to_anchor=(1, 0.2))
                if ylog:
                    ax5_2.set_yscale('log', basey=2)

            if file_nm is None:
                save_file_name = '{}/{}'.format(save_dir, '_all.png')
            else:
                save_dir_v = os.path.join(save_dir, v_)
                os.makedirs(save_dir_v, exist_ok=True)
                file_nm_v = file_nm.replace(file_nm[-4:], "_{}{}".format(v_, file_nm[-4:]))
                save_file_name = '{}/{}'.format(save_dir_v, file_nm_v)

            fig.savefig(save_file_name)
            # print("figure saved. (dir: {})".format(save_file_name))
            plt.close(fig)

    def predict_plot_mtl_cross_section_test_long(self, model, dataset_list, save_dir, file_nm='test.png', ylog=False, time_step=1, invest_rate=0.8):
        if dataset_list is False:
            return False
        else:
            input_enc_list, output_dec_list, target_dec_list, features_list, additional_infos, start_date, end_date = dataset_list

        idx_y = features_list.index(self.label_feature)

        true_y = np.zeros([int(np.ceil(len(input_enc_list) / time_step)) + 1, 1])
        true_y_mw = np.zeros([int(np.ceil(len(input_enc_list) / time_step)) + 1, 1])

        turnover_true_mw = np.zeros_like(true_y_mw)
        total_cost_true_mw = np.zeros_like(true_y_mw)
        truy_y_mw_adj = np.zeros_like(true_y_mw)

        pred_arr = dict()
        pred_arr_mw = dict()
        pred_arr['main'] = np.zeros([len(true_y), 2])
        pred_arr_mw['main'] = np.zeros([len(true_y), 2])

        turnover_main_mw = np.zeros_like(true_y_mw)
        total_cost_main_mw = np.zeros_like(true_y_mw)
        pred_main_mw_adj = np.zeros_like(true_y_mw)


        for key in model.predictor.keys():
            pred_arr[key] = np.zeros([len(true_y), 2])
            pred_arr_mw[key] = np.zeros([len(true_y), 2])

        size_value_list = [add_info['size_value'] for add_info in additional_infos]
        mktcap_list = [add_info['mktcap'] for add_info in additional_infos]
        assets_list = [add_info['assets_list'] for add_info in additional_infos]

        all_assets_list = list()
        for assets in assets_list:
            all_assets_list = list(set(all_assets_list + assets))

        wgt_for_cost = pd.DataFrame({'old_wgt_true': np.zeros_like(all_assets_list),
                                     'new_wgt_true': np.zeros_like(all_assets_list),
                                     'old_wgt_main': np.zeros_like(all_assets_list),
                                     'new_wgt_main': np.zeros_like(all_assets_list),
                                     }, index=sorted(all_assets_list), dtype=np.float32)

        for i, (input_enc_t, output_dec_t, target_dec_t, size_value, mktcap, assets) \
                in enumerate(zip(input_enc_list, output_dec_list, target_dec_list, size_value_list, mktcap_list, assets_list)):
            if i % time_step != 0:
                continue
            t = i // time_step + 1
            assert np.sum(input_enc_t[:, -1, idx_y] - output_dec_t[:, 0, idx_y]) == 0
            new_output_t = np.zeros_like(output_dec_t)
            new_output_t[:, 0, :] = output_dec_t[:, 0, :] + size_value[:, 0, :]
            new_output_t2 = np.zeros_like(output_dec_t)
            new_output_t2[:, 0, :] = output_dec_t[:, 0, :] + mktcap[:, 0, :]

            features = {'input': input_enc_t, 'output': new_output_t}
            labels = target_dec_t

            true_y[t] = np.mean(labels[:, 0, idx_y])
            true_y_mw[t] = np.sum(labels[:, 0, idx_y] * mktcap[:, 0, 0]) / np.sum(mktcap[:, 0, 0])

            # cost calculation
            assets = np.array(assets)
            wgt_for_cost.ix[:, 'new_wgt_true'] = 0.0
            wgt_for_cost.ix[assets, 'new_wgt_true'] = mktcap[:, 0, 0] / np.sum(mktcap[:, 0, 0])
            turnover_true_mw[t] = np.sum(np.abs(wgt_for_cost['new_wgt_true'] - wgt_for_cost['old_wgt_true']))
            total_cost_true_mw[t] = total_cost_true_mw[t-1] + turnover_true_mw[t] * self.cost_rate
            wgt_for_cost.ix[:, 'old_wgt_true'] = 0.0
            wgt_for_cost.ix[assets, 'old_wgt_true'] = ((1 + labels[:, 0, idx_y]) * mktcap[:, 0, 0]) / np.sum((1 + labels[:, 0, idx_y]) * mktcap[:, 0, 0])
            truy_y_mw_adj[t] = np.sum(labels[:, 0, idx_y] * mktcap[:, 0, 0]) / np.sum(mktcap[:, 0, 0]) - turnover_true_mw[t] * self.cost_rate

            predictions = model.predict_mtl(features)
            value_ = dict()
            value_['main'] = predictions[self.pred_feature][:, 0, 0]

            for feature in model.predictor.keys():
                value_[feature] = predictions[feature][:, 0, 0]

            low_crit_cslogy, high_crit_cslogy = np.percentile(value_['cslogy'], q=[100 * (1 - invest_rate), 100])

            for v in value_.keys():
                if v in ['logy', 'cslogy', 'fft', 'pos_5', 'pos_20', 'pos_60', 'pos_120', 'main']:
                    low_q, high_q = 100 * (1 - invest_rate), 100
                else:
                    low_q, high_q = 0, 100 * invest_rate

                low_crit, high_crit = np.percentile(value_[v], q=[low_q, high_q])

                crit1 = ((value_[v] >= low_crit) & (value_[v] < high_crit)).numpy()
                crit2 = ((value_[v] >= low_crit) & (value_[v] < high_crit)
                         & (value_['cslogy'] >= low_crit_cslogy) & (value_['cslogy'] < high_crit_cslogy)).numpy()

                pred_arr[v][t, 0] = np.mean(labels[crit1, 0, idx_y])
                pred_arr_mw[v][t, 0] = np.sum(labels[crit1, 0, idx_y] * mktcap[crit1, 0, 0]) / np.sum(mktcap[crit1, 0, 0])
                pred_arr[v][t, 1] = np.mean(labels[crit2, 0, idx_y])
                pred_arr_mw[v][t, 1] = np.sum(labels[crit2, 0, idx_y] * mktcap[crit2, 0, 0]) / np.sum(mktcap[crit2, 0, 0])
                if v == 'main':
                    # cost calculation
                    wgt_for_cost.ix[:, 'new_wgt_main'] = 0.0
                    wgt_for_cost.ix[assets[crit2], 'new_wgt_main'] = mktcap[crit2, 0, 0] / np.sum(mktcap[crit2, 0, 0])
                    turnover_main_mw[t] = np.sum(np.abs(wgt_for_cost['new_wgt_main'] - wgt_for_cost['old_wgt_main']))
                    total_cost_main_mw[t] = total_cost_main_mw[t-1] + turnover_main_mw[t] * self.cost_rate
                    wgt_for_cost.ix[:, 'old_wgt_main'] = 0.0
                    wgt_for_cost.ix[assets, 'old_wgt_main'] = ((1 + labels[:, 0, idx_y]) * wgt_for_cost.ix[assets, 'new_wgt_main']) \
                                                              / np.sum((1 + labels[:, 0, idx_y]) * wgt_for_cost.ix[assets, 'new_wgt_main'])
                    pred_main_mw_adj[t] = np.sum(labels[:, 0, idx_y] * wgt_for_cost.ix[assets, 'new_wgt_main']) - turnover_main_mw[t] * self.cost_rate

        for v_ in value_.keys():
            data = pd.DataFrame(np.cumprod(1. + np.concatenate([true_y, true_y_mw, pred_arr[v_], pred_arr_mw[v_]], axis=-1), axis=0),
                                columns=['true_y', 'true_y_mw', 'pred1', 'pred2', 'pred1_mw', 'pred2_mw'])

            data['diff1'] = np.cumprod(1. + pred_arr[v_][:, :1] - true_y)
            data['diff2'] = np.cumprod(1. + pred_arr[v_][:, -1:] - true_y)
            data['diff1_mw'] = np.cumprod(1. + pred_arr_mw[v_][:, :1] - true_y_mw)
            data['diff2_mw'] = np.cumprod(1. + pred_arr_mw[v_][:, -1:] - true_y_mw)

            if v_ == 'main':
                data['true_cost'] = total_cost_true_mw
                data['true_turnover'] = turnover_true_mw
                data['true_y_mw_adj'] = np.cumprod(1. + truy_y_mw_adj, axis=0) - 1.
                data['main_cost'] = total_cost_main_mw
                data['main_turnover'] = turnover_main_mw
                data['main_y_mw_adj'] = np.cumprod(1. + pred_main_mw_adj, axis=0) - 1.
                data['diff_adj'] = np.cumprod(1. + pred_main_mw_adj - truy_y_mw_adj, axis=0) - 1.


            # ################################ figure 1
            # equal fig
            fig = plt.figure()
            fig.suptitle('{} ~ {}'.format(start_date, end_date))
            if v_ == 'main':
                ax1 = fig.add_subplot(311)
                ax2 = fig.add_subplot(312)
                ax3 = fig.add_subplot(313)
                # ax4 = fig.add_subplot(414)
            else:
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)

            data[['true_y', 'pred1', 'pred2']].plot(ax=ax1, colormap=cm.Set2)
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax1.legend(['true_y', 'pred1', 'pred2'], loc='center left', bbox_to_anchor=(1, 0.8))
            if ylog:
                ax1.set_yscale('log', basey=2)

            ax1_2 = ax1.twinx()
            data[['diff1', 'diff2']].plot(ax=ax1_2, colormap=cm.jet)
            box = ax1_2.get_position()
            ax1_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax1_2.legend(['diff1', 'diff2'], loc='center left', bbox_to_anchor=(1, 0.2))
            if ylog:
                ax1_2.set_yscale('log', basey=2)


            data[['true_y_mw', 'pred1_mw', 'pred2_mw']].plot(ax=ax2, colormap=cm.Set2)
            box = ax2.get_position()
            ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax2.legend(['true_y_mw', 'pred1_mw', 'pred2_mw'], loc='center left', bbox_to_anchor=(1, 0.8))
            if ylog:
                ax2.set_yscale('log', basey=2)

            ax2_2 = ax2.twinx()
            data[['diff1_mw', 'diff2_mw']].plot(ax=ax2_2, colormap=cm.jet)
            box = ax2_2.get_position()
            ax2_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax2_2.legend(['diff1_mw', 'diff2_mw'], loc='center left', bbox_to_anchor=(1, 0.2))
            if ylog:
                ax2_2.set_yscale('log', basey=2)


            if v_ == 'main':
                data[['true_y_mw_adj', 'main_y_mw_adj']].plot(ax=ax3, colormap=cm.Set2)
                box = ax3.get_position()
                ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax3.legend(['true_y_mw_adj', 'main_y_mw_adj'], loc='center left', bbox_to_anchor=(1, 0.8))
                if ylog:
                    ax3.set_yscale('log', basey=2)

                ax3_2 = ax3.twinx()
                data[['diff_adj']].plot(ax=ax3_2, colormap=cm.jet)
                box = ax3_2.get_position()
                ax3_2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax3_2.legend(['diff_adj'], loc='center left', bbox_to_anchor=(1, 0.2))
                if ylog:
                    ax3_2.set_yscale('log', basey=2)


            if file_nm is None:
                save_file_name = '{}/{}'.format(save_dir, '_all.png')
            else:
                save_dir_v = os.path.join(save_dir, v_)
                os.makedirs(save_dir_v, exist_ok=True)
                file_nm_v = file_nm.replace(file_nm[-4:], "_{}{}".format(v_, file_nm[-4:]))
                save_file_name = '{}/{}'.format(save_dir_v, file_nm_v)

            fig.savefig(save_file_name)
            # print("figure saved. (dir: {})".format(save_file_name))
            plt.close(fig)




