
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        stdarr[t, :] = np.std(y[max(0, t - n):(t + 1), :], axis=0)

    return stdarr


def mdd_nd(log_p, n):
    mddarr = np.zeros_like(log_p)
    for t in range(len(log_p)):
        mddarr[t, :] = log_p[t, :] - np.max(log_p[max(0, t - n):(t + 1), :])

    return mddarr


def arr_to_cs(arr):
    order = arr.argsort(axis=1)
    return_value = order.argsort(axis=1) / np.max(order, axis=1, keepdims=True)
    return return_value

def arr_to_normal(arr):
    return_value = (arr - np.mean(arr, axis=1, keepdims=True)) / np.std(arr, axis=1, ddof=1, keepdims=True)
    return return_value


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

    def labels_for_mtl(self, features_list, labels, size_value):
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

        return labels_mtl

    def processing_split_new(self, df_not_null, m_days, k_days, sampling_days, calc_length=0, label_type=None,
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
                        elif key == 'pos':
                            features_data_dict[key][str(nd)] = np.sign(features_data_dict['logy'][str(nd)])
                        elif key == 'mdd':
                            features_data_dict[key][str(nd)] = mdd_nd(log_p_wo_calc, nd)[:(m_days + 1)]
                        elif key == 'fft':
                            features_data_dict[key][str(nd)] = fft(log_p_wo_calc, nd, m_days, k_days_adj)[:(m_days + 1)]
                        elif key == 'cslogy':
                            features_data_dict[key][str(nd)] = arr_to_cs(log_y_nd(log_p, nd)[calc_length:][:(m_days+1)])
                        elif key == 'csstd':
                            features_data_dict[key][str(nd)] = arr_to_cs(std_nd(log_p, nd)[calc_length:][:(m_days + 1)])

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
                        elif key == 'pos':
                            features_data_dict[key][str(nd)] = np.sign(features_data_dict['logy'][str(nd)])
                        elif key == 'mdd':
                            features_data_dict[key][str(nd)] = mdd_nd(log_p_wo_calc, nd)[:(m_days + 1)]
                        elif key == 'fft':
                            features_data_dict[key][str(nd)] = fft(log_p_wo_calc, nd, m_days, k_days_adj)[:(m_days + 1)]
                        elif key == 'cslogy':
                            features_data_dict[key][str(nd)] = arr_to_cs(log_y_nd(log_p, nd)[calc_length:][:(m_days+1)])
                        elif key == 'csstd':
                            features_data_dict[key][str(nd)] = arr_to_cs(std_nd(log_p, nd)[calc_length:][:(m_days + 1)])

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
                                features_label_dict[key][str(nd)] = arr_to_cs(std_nd(log_p, nd)[(calc_length+m_days):][:(n_freq + 1)][::n_freq])
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

    def predict_plot_mtl_cross_section_test(self, model, dataset_list, save_dir='out.png', ylog=False, time_step=1):
        if dataset_list is False:
            return False
        else:
            input_enc_list, output_dec_list, target_dec_list, features_list, additional_infos, start_date, end_date = dataset_list

        idx_y = features_list.index(self.label_feature)

        true_y = np.zeros([int(np.ceil(len(input_enc_list) / time_step)) + 1, 1])
        true_y_mw = np.zeros([int(np.ceil(len(input_enc_list) / time_step)) + 1, 1])

        n_tile = 5
        pred_arr = np.zeros([len(true_y), n_tile])
        pred_arr_mw = np.zeros([len(true_y), n_tile])

        pred_arr_ori = np.zeros([len(true_y), n_tile])
        pred_arr_ori_mw = np.zeros([len(true_y), n_tile])

        pred_arr_cap = np.zeros([len(true_y), n_tile])
        pred_arr_cap_mw = np.zeros([len(true_y), n_tile])

        # pred_q1 = np.zeros_like(true_y)
        # pred_q2 = np.zeros_like(true_y)
        # pred_q3 = np.zeros_like(true_y)
        # pred_q4 = np.zeros_like(true_y)
        # pred_q5 = np.zeros_like(true_y)
        #
        # pred_q1_mw = np.zeros_like(true_y)
        # pred_q2_mw = np.zeros_like(true_y)
        # pred_q3_mw = np.zeros_like(true_y)
        # pred_q4_mw = np.zeros_like(true_y)
        # pred_q5_mw = np.zeros_like(true_y)

        size_value_list = [add_info['size_value'] for add_info in additional_infos]
        mktcap_list = [add_info['mktcap'] for add_info in additional_infos]
        for i, (input_enc_t, output_dec_t, target_dec_t, size_value, mktcap) in enumerate(zip(input_enc_list, output_dec_list, target_dec_list, size_value_list, mktcap_list)):
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

            predictions = model.predict_mtl(features)
            value_ = predictions[self.pred_feature][:, 0, 0]

            predictions_ori = model.predict_mtl({'input': input_enc_t, 'output': output_dec_t})
            value_ori = predictions_ori[self.pred_feature][:, 0, 0]

            predictions_cap = model.predict_mtl({'input': input_enc_t, 'output': new_output_t2})
            value_cap = predictions_cap[self.pred_feature][:, 0, 0]

            for i_tile in range(n_tile):
                low_q, high_q = 100 * (1. - (1.+i_tile) / n_tile),  100 * (1. - i_tile / n_tile)

                low_crit, high_crit = np.percentile(value_, q=[low_q, high_q])
                if high_q == 100:
                    crit_ = (value_ >= low_crit)
                else:
                    crit_ = ((value_ >= low_crit) & (value_ < high_crit))

                pred_arr[t, i_tile] = np.mean(labels[crit_, 0, idx_y])
                pred_arr_mw[t, i_tile] = np.sum(labels[crit_, 0, idx_y] * mktcap[crit_, 0, 0]) / np.sum(mktcap[crit_, 0, 0])

                low_crit_ori, high_crit_ori = np.percentile(value_ori, q=[low_q, high_q])
                crit_ori = ((value_ >= low_crit_ori) & (value_ < high_crit_ori))
                pred_arr_ori[t, i_tile] = np.mean(labels[crit_ori, 0, idx_y])
                pred_arr_ori_mw[t, i_tile] = np.sum(labels[crit_ori, 0, idx_y] * mktcap[crit_ori, 0, 0]) / np.sum(mktcap[crit_ori, 0, 0])

                low_crit_cap, high_crit_cap = np.percentile(value_cap, q=[low_q, high_q])
                crit_cap = ((value_ >= low_crit_cap) & (value_ < high_crit_cap))
                pred_arr_cap[t, i_tile] = np.mean(labels[crit_cap, 0, idx_y])
                pred_arr_cap_mw[t, i_tile] = np.sum(labels[crit_cap, 0, idx_y] * mktcap[crit_cap, 0, 0]) / np.sum(mktcap[crit_cap, 0, 0])

            # q1_crit, q2_crit, q3_crit, q4_crit = np.percentile(value_, q=[80, 60, 40, 20])
            # crit1 = (value_ >= q1_crit)
            # crit2 = ((value_ >= q2_crit) & (value_ < q1_crit))
            # crit3 = ((value_ >= q3_crit) & (value_ < q2_crit))
            # crit4 = ((value_ >= q4_crit) & (value_ < q3_crit))
            # crit5 = (value_ < q4_crit)
            # pred_q1[t] = np.mean(labels[crit1, 0, idx_y])
            # pred_q2[t] = np.mean(labels[crit2, 0, idx_y])
            # pred_q3[t] = np.mean(labels[crit3, 0, idx_y])
            # pred_q4[t] = np.mean(labels[crit4, 0, idx_y])
            # pred_q5[t] = np.mean(labels[crit5, 0, idx_y])
            #
            # pred_q1_mw[t] = np.sum(labels[crit1, 0, idx_y] * mktcap[crit1, 0, 0]) / np.sum(mktcap[crit1, 0, 0])
            # pred_q2_mw[t] = np.sum(labels[crit2, 0, idx_y] * mktcap[crit2, 0, 0]) / np.sum(mktcap[crit2, 0, 0])
            # pred_q3_mw[t] = np.sum(labels[crit3, 0, idx_y] * mktcap[crit3, 0, 0]) / np.sum(mktcap[crit3, 0, 0])
            # pred_q4_mw[t] = np.sum(labels[crit4, 0, idx_y] * mktcap[crit4, 0, 0]) / np.sum(mktcap[crit4, 0, 0])
            # pred_q5_mw[t] = np.sum(labels[crit5, 0, idx_y] * mktcap[crit5, 0, 0]) / np.sum(mktcap[crit5, 0, 0])

        data = pd.DataFrame(np.cumprod(1. + np.concatenate([true_y, true_y_mw, pred_arr, pred_arr_mw], axis=-1),axis=0),
                            columns=['true_y', 'true_y_mw']
                                    + ['pred_q{}'.format(i + 1) for i in range(n_tile)]
                                    + ['pred_q{}_mw'.format(i + 1) for i in range(n_tile)])
        data['pred_ls'] = np.cumprod(1. + np.mean(pred_arr[:, :1], axis=1) - np.mean(pred_arr[:, -1:], axis=1))
        data['pred_ls2'] = np.cumprod(1. + np.mean(pred_arr[:, :2], axis=1) - np.mean(pred_arr[:, -2:], axis=1))
        data['pred_ls_mw'] = np.cumprod(1. + np.mean(pred_arr_mw[:, :1], axis=1) - np.mean(pred_arr_mw[:, -1:], axis=1))
        data['pred_ls_mw2'] = np.cumprod(1. + np.mean(pred_arr_mw[:, :2], axis=1) - np.mean(pred_arr_mw[:, -2:], axis=1))

        data_ori = pd.DataFrame(np.cumprod(1. + np.concatenate([true_y, true_y_mw, pred_arr_ori, pred_arr_ori_mw], axis=-1),axis=0),
                            columns=['true_y', 'true_y_mw']
                                    + ['pred_q{}'.format(i + 1) for i in range(n_tile)]
                                    + ['pred_q{}_mw'.format(i + 1) for i in range(n_tile)])
        data_ori['pred_ls'] = np.cumprod(1. + np.mean(pred_arr_ori[:, :1], axis=1) - np.mean(pred_arr_ori[:, -1:], axis=1))
        data_ori['pred_ls2'] = np.cumprod(1. + np.mean(pred_arr_ori[:, :2], axis=1) - np.mean(pred_arr_ori[:, -2:], axis=1))
        data_ori['pred_ls_mw'] = np.cumprod(1. + np.mean(pred_arr_ori_mw[:, :1], axis=1) - np.mean(pred_arr_ori_mw[:, -1:], axis=1))
        data_ori['pred_ls_mw2'] = np.cumprod(1. + np.mean(pred_arr_ori_mw[:, :2], axis=1) - np.mean(pred_arr_ori_mw[:, -2:], axis=1))

        data_cap = pd.DataFrame(np.cumprod(1. + np.concatenate([true_y, true_y_mw, pred_arr_cap, pred_arr_cap_mw], axis=-1),axis=0),
                            columns=['true_y', 'true_y_mw']
                                    + ['pred_q{}'.format(i + 1) for i in range(n_tile)]
                                    + ['pred_q{}_mw'.format(i + 1) for i in range(n_tile)])
        data_cap['pred_ls'] = np.cumprod(1. + np.mean(pred_arr_cap[:, :1], axis=1) - np.mean(pred_arr_cap[:, -1:], axis=1))
        data_cap['pred_ls2'] = np.cumprod(1. + np.mean(pred_arr_cap[:, :2], axis=1) - np.mean(pred_arr_cap[:, -2:], axis=1))
        data_cap['pred_ls_mw'] = np.cumprod(1. + np.mean(pred_arr_cap_mw[:, :1], axis=1) - np.mean(pred_arr_cap_mw[:, -1:], axis=1))
        data_cap['pred_ls_mw2'] = np.cumprod(1. + np.mean(pred_arr_cap_mw[:, :2], axis=1) - np.mean(pred_arr_cap_mw[:, -2:], axis=1))
        # data = pd.DataFrame({'true_y': np.cumprod(1. + true_y),
        #                      'pred_ls': np.cumprod(1. + pred_q1 - pred_q5),
        #                      'pred_q1': np.cumprod(1. + pred_q1),
        #                      'pred_q2': np.cumprod(1. + pred_q2),
        #                      'pred_q3': np.cumprod(1. + pred_q3),
        #                      'pred_q4': np.cumprod(1. + pred_q4),
        #                      'pred_q5': np.cumprod(1. + pred_q5),
        #                      'true_y_mw': np.cumprod(1. + true_y_mw),
        #                      'pred_ls_mw': np.cumprod(1. + pred_q1_mw - pred_q5_mw),
        #                      'pred_q1_mw': np.cumprod(1. + pred_q1_mw),
        #                      'pred_q2_mw': np.cumprod(1. + pred_q2_mw),
        #                      'pred_q3_mw': np.cumprod(1. + pred_q3_mw),
        #                      'pred_q4_mw': np.cumprod(1. + pred_q4_mw),
        #                      'pred_q5_mw': np.cumprod(1. + pred_q5_mw),
        # })


        # ################################ figure 1
        # equal fig
        fig = plt.figure()
        fig.suptitle('{} ~ {}'.format(start_date, end_date))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        ax1.plot(data[['true_y', 'pred_ls', 'pred_ls2', 'pred_q1', 'pred_q5']])
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax1.legend(['true_y', 'long-short', 'long-short2', 'long', 'short'], loc='center left', bbox_to_anchor=(1, 0.5))
        if ylog:
            ax1.set_yscale('log', basey=2)

        ax2.plot(data[['true_y', 'pred_q1', 'pred_q2', 'pred_q3', 'pred_q4', 'pred_q5']])
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax2.legend(['true_y', 'q1', 'q2', 'q3', 'q4', 'q5'], loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.set_yscale('log', basey=2)

        # value fig
        ax3.plot(data[['true_y_mw', 'pred_ls_mw', 'pred_ls_mw2', 'pred_q1_mw', 'pred_q5_mw']])
        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax3.legend(['true_y(mw)', 'long-short', 'long-short2', 'long', 'short'], loc='center left', bbox_to_anchor=(1, 0.5))
        if ylog:
            ax3.set_yscale('log', basey=2)

        ax4.plot(data[['true_y_mw', 'pred_q1_mw', 'pred_q2_mw', 'pred_q3_mw', 'pred_q4_mw', 'pred_q5_mw']])
        box = ax4.get_position()
        ax4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax4.legend(['true_y(mw)', 'q1', 'q2', 'q3', 'q4', 'q5'], loc='center left', bbox_to_anchor=(1, 0.5))
        ax4.set_yscale('log', basey=2)

        fig.savefig(save_dir)
        print("figure saved. (dir: {})".format(save_dir))
        plt.close(fig)

        # ################################ figure 2
        # equal fig
        fig = plt.figure()
        fig.suptitle('{} ~ {}'.format(start_date, end_date))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        ax1.plot(data_ori[['true_y', 'pred_ls', 'pred_ls2', 'pred_q1', 'pred_q5']])
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax1.legend(['true_y', 'long-short', 'long-short2', 'long', 'short'], loc='center left', bbox_to_anchor=(1, 0.5))
        if ylog:
            ax1.set_yscale('log', basey=2)

        ax2.plot(data_ori[['true_y', 'pred_q1', 'pred_q2', 'pred_q3', 'pred_q4', 'pred_q5']])
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax2.legend(['true_y', 'q1', 'q2', 'q3', 'q4', 'q5'], loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.set_yscale('log', basey=2)

        # value fig
        ax3.plot(data_ori[['true_y_mw', 'pred_ls_mw', 'pred_ls_mw2', 'pred_q1_mw', 'pred_q5_mw']])
        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax3.legend(['true_y(mw)', 'long-short', 'long-short2', 'long', 'short'], loc='center left', bbox_to_anchor=(1, 0.5))
        if ylog:
            ax3.set_yscale('log', basey=2)

        ax4.plot(data_ori[['true_y_mw', 'pred_q1_mw', 'pred_q2_mw', 'pred_q3_mw', 'pred_q4_mw', 'pred_q5_mw']])
        box = ax4.get_position()
        ax4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax4.legend(['true_y(mw)', 'q1', 'q2', 'q3', 'q4', 'q5'], loc='center left', bbox_to_anchor=(1, 0.5))
        ax4.set_yscale('log', basey=2)

        save_dir_ori = save_dir.replace(save_dir[-4:], '_ori'+save_dir[-4:])
        fig.savefig(save_dir_ori)
        print("figure saved. (dir: {})".format(save_dir_ori))
        plt.close(fig)

        # ################################ figure 3
        # equal fig
        fig = plt.figure()
        fig.suptitle('{} ~ {}'.format(start_date, end_date))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        ax1.plot(data_cap[['true_y', 'pred_ls', 'pred_ls2', 'pred_q1', 'pred_q5']])
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax1.legend(['true_y', 'long-short', 'long-short2', 'long', 'short'], loc='center left', bbox_to_anchor=(1, 0.5))
        if ylog:
            ax1.set_yscale('log', basey=2)

        ax2.plot(data_cap[['true_y', 'pred_q1', 'pred_q2', 'pred_q3', 'pred_q4', 'pred_q5']])
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax2.legend(['true_y', 'q1', 'q2', 'q3', 'q4', 'q5'], loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.set_yscale('log', basey=2)

        # value fig
        ax3.plot(data_cap[['true_y_mw', 'pred_ls_mw', 'pred_ls_mw2', 'pred_q1_mw', 'pred_q5_mw']])
        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax3.legend(['true_y(mw)', 'long-short', 'long-short2', 'long', 'short'], loc='center left', bbox_to_anchor=(1, 0.5))
        if ylog:
            ax3.set_yscale('log', basey=2)

        ax4.plot(data_cap[['true_y_mw', 'pred_q1_mw', 'pred_q2_mw', 'pred_q3_mw', 'pred_q4_mw', 'pred_q5_mw']])
        box = ax4.get_position()
        ax4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax4.legend(['true_y(mw)', 'q1', 'q2', 'q3', 'q4', 'q5'], loc='center left', bbox_to_anchor=(1, 0.5))
        ax4.set_yscale('log', basey=2)

        save_dir_cap = save_dir.replace(save_dir[-4:], '_cap'+save_dir[-4:])
        fig.savefig(save_dir_cap)
        print("figure saved. (dir: {})".format(save_dir_cap))
        plt.close(fig)




