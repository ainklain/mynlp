
import numpy as np
import os
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


def std_nd_new(log_p, n):
    y = np.exp(log_y_nd(log_p, n)) - 1.
    stdarr = np.zeros_like(y)
    for t in range(1, len(y)):
        stdarr[t, :] = np.std(y[max(0, t - n * 12):(t + 1), :][::n], axis=0)

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

        for key in model.predictor.keys():
            pred_arr[key] = np.zeros([len(true_y), n_tile])
            pred_arr_mw[key] = np.zeros([len(true_y), n_tile])

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

                    if v in ['logy', 'cslogy', 'fft', 'pos_5', 'pos_20', 'pos_60', 'pos_120']:
                        pred_arr[v][t, i_tile] = np.mean(labels[crit_, 0, idx_y])
                        pred_arr_mw[v][t, i_tile] = np.sum(labels[crit_, 0, idx_y] * mktcap[crit_, 0, 0]) / np.sum(mktcap[crit_, 0, 0])
                    else:
                        pred_arr[v][t, -(i_tile+1)] = np.mean(labels[crit_, 0, idx_y])
                        pred_arr_mw[v][t, -(i_tile+1)] = np.sum(labels[crit_, 0, idx_y] * mktcap[crit_, 0, 0]) / np.sum(mktcap[crit_, 0, 0])

        for v_ in value_.keys():
            data = pd.DataFrame(np.cumprod(1. + np.concatenate([true_y, true_y_mw, pred_arr[v_], pred_arr_mw[v_]], axis=-1), axis=0),
                                columns=['true_y', 'true_y_mw'] + ['pred_q{}'.format(i + 1) for i in range(n_tile)]
                                        + ['pred_q{}_mw'.format(i + 1) for i in range(n_tile)])
            data['pred_ls'] = np.cumprod(1. + np.mean(pred_arr[v_][:, :1], axis=1) - np.mean(pred_arr[v_][:, -1:], axis=1))
            data['pred_ls2'] = np.cumprod(1. + np.mean(pred_arr[v_][:, :2], axis=1) - np.mean(pred_arr[v_][:, -2:], axis=1))
            data['pred_ls_mw'] = np.cumprod(1. + np.mean(pred_arr_mw[v_][:, :1], axis=1) - np.mean(pred_arr_mw[v_][:, -1:], axis=1))
            data['pred_ls_mw2'] = np.cumprod(1. + np.mean(pred_arr_mw[v_][:, :2], axis=1) - np.mean(pred_arr_mw[v_][:, -2:], axis=1))


            # ################################ figure 1
            # equal fig
            fig = plt.figure()
            fig.suptitle('{} ~ {}'.format(start_date, end_date))
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

    def predict_plot_mtl_cross_section_test_long(self, model, dataset_list, save_dir, file_nm='test.png', ylog=False, time_step=1):
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
        pred_arr['main'] = np.zeros([len(true_y), 2])
        pred_arr_mw['main'] = np.zeros([len(true_y), 2])

        for key in model.predictor.keys():
            pred_arr[key] = np.zeros([len(true_y), 2])
            pred_arr_mw[key] = np.zeros([len(true_y), 2])

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
            value_ = dict()
            value_['main'] = predictions[self.pred_feature][:, 0, 0]

            for feature in model.predictor.keys():
                value_[feature] = predictions[feature][:, 0, 0]


            low_crit_cslogy, high_crit_cslogy = np.percentile(value_['cslogy'], q=[100 / n_tile, 100])

            for v in value_.keys():
                if v in ['logy', 'cslogy', 'fft', 'pos_5', 'pos_20', 'pos_60', 'pos_120']:
                    low_q, high_q = 100 / n_tile, 100
                else:
                    low_q, high_q = 0, 100 * (1 - 1 / n_tile)

                low_crit, high_crit = np.percentile(value_[v], q=[low_q, high_q])

                crit1 = ((value_[v] >= low_crit) & (value_[v] < high_crit))
                crit2 = ((value_[v] >= low_crit) & (value_[v] < high_crit)
                         & (value_['cslogy'] >= low_crit_cslogy) & (value_['cslogy'] < high_crit_cslogy))

                pred_arr[v][t, 0] = np.mean(labels[crit1, 0, idx_y])
                pred_arr_mw[v][t, 0] = np.sum(labels[crit1, 0, idx_y] * mktcap[crit1, 0, 0]) / np.sum(mktcap[crit1, 0, 0])
                pred_arr[v][t, 1] = np.mean(labels[crit2, 0, idx_y])
                pred_arr_mw[v][t, 1] = np.sum(labels[crit2, 0, idx_y] * mktcap[crit2, 0, 0]) / np.sum(mktcap[crit2, 0, 0])

        for v_ in value_.keys():
            data = pd.DataFrame(np.cumprod(1. + np.concatenate([true_y, true_y_mw, pred_arr[v_], pred_arr_mw[v_]], axis=-1), axis=0),
                                columns=['true_y', 'true_y_mw', 'pred1', 'pred2', 'pred1_mw', 'pred2_mw'])

            data['diff1'] = np.cumprod(1. + pred_arr[v_][:, :1] - true_y)
            data['diff2'] = np.cumprod(1. + pred_arr[v_][:, -1:] - true_y)
            data['diff1_mw'] = np.cumprod(1. + pred_arr_mw[v_][:, :1] - true_y_mw)
            data['diff2_mw'] = np.cumprod(1. + pred_arr_mw[v_][:, -1:] - true_y_mw)


            # ################################ figure 1
            # equal fig
            fig = plt.figure()
            fig.suptitle('{} ~ {}'.format(start_date, end_date))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            ax1.plot(data[['true_y', 'pred1', 'pred2', 'diff1', 'diff2']])
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax1.legend(['true_y', 'pred1', 'pred2', 'diff1', 'diff2'], loc='center left', bbox_to_anchor=(1, 0.5))
            if ylog:
                ax1.set_yscale('log', basey=2)

            ax2.plot(data[['true_y_mw', 'pred1_mw', 'pred2_mw', 'diff1_mw', 'diff2_mw']])
            box = ax2.get_position()
            ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax2.legend(['true_y_mw', 'pred1_mw', 'pred2_mw', 'diff1_mw', 'diff2_mw'], loc='center left', bbox_to_anchor=(1, 0.5))
            if ylog:
                ax2.set_yscale('log', basey=2)

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



