
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


class Feature:
    def __init__(self, label_feature='logy_5d', pred_feature='pos_5d'):
        self._init_features(label_feature, pred_feature)

    def _init_features(self, label_feature, pred_feature):
        # dict: classification
        self.structure = {
            'logy': ['5d', '20d', '60d', '120d', '250d'],
            'pos_5d': '5d',
            'pos_20d': '20d',
            'pos_60d': '60d',
            'std': ['20d', '60d', '120d'],
            'mdd': ['20d', '60d', '120d'],
            'fft': ['3com', '100com']
        }
        self.model_predictor_list = ['logy', 'pos_5d', 'pos_20d', 'std', 'mdd', 'fft']
        self.label_feature = label_feature
        self.pred_feature = pred_feature

    def labels_for_mtl(self, features_list, labels, size_value):
        labels_mtl = dict()
        for key in self.structure.keys():
            if isinstance(self.structure[key], str):    # classification
                labels_mtl[key] = np.stack([labels[:, :, features_list.index(key)] > 0, labels[:, :, features_list.index(key)] <= 0],
                                           axis=-1) * 1.
            else:
                labels_mtl[key] = np.stack([labels[:, :, features_list.index(key + '_' + item)] for item in self.structure[key]],
                                           axis=-1)
        labels_mtl['size_value'] = size_value

        return labels_mtl

    def processing_split_new(self, df_not_null, m_days, k_days, sampling_days, calc_length=0, label_type=None, additional_dict=None):
        # if type(df.columns) == pd.MultiIndex:
        #     df.columns = df.columns.droplevel(0)
        features_data_dict = dict()
        features_label_dict = dict()
        log_p = np.log(df_not_null.values, dtype=np.float32)

        main_class, sub_class = self.label_feature.split('_')
        n_days = int(sub_class[:(-1)])

        if label_type is None:
            assert len(log_p) == ((calc_length + m_days) + 1)

            log_p_wo_calc = log_p[calc_length:]
            assert len(log_p_wo_calc) == (m_days + 1)

            log_p = log_p - log_p[0, :]
            log_p_wo_calc = log_p_wo_calc - log_p_wo_calc[0, :]

            features_data_dict['logy'] = {'5d': log_y_nd(log_p, 5)[calc_length:],
                                          '20d': log_y_nd(log_p, 20)[calc_length:],
                                          '60d': log_y_nd(log_p, 60)[calc_length:],
                                          '120d': log_y_nd(log_p, 120)[calc_length:],
                                          '250d': log_y_nd(log_p, 250)[calc_length:]}

            features_data_dict['std'] = {'20d': std_nd(log_p, 20)[calc_length:],
                                         '60d': std_nd(log_p, 60)[calc_length:],
                                         '120d': std_nd(log_p, 120)[calc_length:]}

            features_data_dict['pos'] = {'5d': np.sign(features_data_dict['logy']['5d']),
                                         '20d': np.sign(features_data_dict['logy']['20d']),
                                         '60d': np.sign(features_data_dict['logy']['60d'])}

            features_data_dict['mdd'] = {'20d': mdd_nd(log_p_wo_calc, 20),
                                         '60d': mdd_nd(log_p_wo_calc, 60),
                                         '120d': mdd_nd(log_p_wo_calc, 120)}

            features_data_dict['fft'] = {'3com': fft(log_p_wo_calc, 3, m_days, k_days),
                                         '6com': fft(log_p_wo_calc, 6, m_days, k_days),
                                         '100com': fft(log_p_wo_calc, 100, m_days, k_days)}

        else:
            log_p_wo_calc = log_p[calc_length:][:(k_days + m_days + 1)]
            assert len(log_p_wo_calc) == (k_days + m_days + 1)

            log_p = log_p - log_p[0, :]
            log_p_wo_calc = log_p_wo_calc - log_p_wo_calc[0, :]

            # data part
            features_data_dict['logy'] = {'5d': log_y_nd(log_p, 5)[calc_length:][:(m_days + 1)],
                                    '20d': log_y_nd(log_p, 20)[calc_length:][:(m_days + 1)],
                                    '60d': log_y_nd(log_p, 60)[calc_length:][:(m_days + 1)],
                                    '120d': log_y_nd(log_p, 120)[calc_length:][:(m_days + 1)],
                                    '250d': log_y_nd(log_p, 250)[calc_length:][:(m_days + 1)]}

            features_data_dict['std'] = {'20d': std_nd(log_p, 20)[calc_length:][:(m_days + 1)],
                                    '60d': std_nd(log_p, 60)[calc_length:][:(m_days + 1)],
                                    '120d': std_nd(log_p, 120)[calc_length:][:(m_days + 1)]}

            features_data_dict['pos'] = {'5d': np.sign(features_data_dict['logy']['5d']),
                                    '20d': np.sign(features_data_dict['logy']['20d']),
                                    '60d': np.sign(features_data_dict['logy']['60d'])}

            features_data_dict['mdd'] = {'20d': mdd_nd(log_p_wo_calc, 20)[:(m_days + 1)],
                                    '60d': mdd_nd(log_p_wo_calc, 60)[:(m_days + 1)],
                                    '120d': mdd_nd(log_p_wo_calc, 120)[:(m_days + 1)]}

            features_data_dict['fft'] = {'3com': fft(log_p_wo_calc, 3, m_days, k_days)[:(m_days + 1)],
                                    '6com': fft(log_p_wo_calc, 6, m_days, k_days)[:(m_days + 1)],
                                    '100com': fft(log_p_wo_calc, 100, m_days, k_days)[:(m_days + 1)]}


            if label_type == 'trainable_label':
                assert len(log_p) == ((calc_length + m_days) + (calc_length + k_days) + 1)
                # label part
                if n_days == 5:
                    features_label_dict['logy'] = {'5d': log_y_nd(log_p, 5)[(calc_length + m_days):][:(n_days + 1)][::n_days],
                                                   '20d': log_y_nd(log_p, 20)[(calc_length + m_days):][:(n_days + 1)][::n_days],
                                                   '60d': log_y_nd(log_p, 60)[(calc_length + m_days):][:(n_days + 1)][::n_days],
                                                   '120d': log_y_nd(log_p, 120)[(calc_length + m_days):][:(n_days + 1)][::n_days],
                                                   '250d': log_y_nd(log_p, 250)[(calc_length + m_days):][:(n_days + 1)][::n_days]}

                    features_label_dict['std'] = {'5d': std_nd(log_p, 5)[(calc_length + m_days):][:(n_days + 1)][::n_days],
                                                  '20d': std_nd(log_p, 20)[(calc_length + m_days):][:(n_days + 1)][::n_days],
                                                  '60d': std_nd(log_p, 60)[(calc_length + m_days):][:(n_days + 1)][::n_days],
                                                  '120d': std_nd(log_p, 120)[(calc_length + m_days):][:(n_days + 1)][::n_days]}

                else:
                    features_label_dict['logy'] = {'5d': log_y_nd(log_p, 5)[(calc_length+m_days):][:(5 + 1)][::5],
                                            '20d': log_y_nd(log_p, 20)[(calc_length+m_days):][:(20 + 1)][::20],
                                            '60d': log_y_nd(log_p, 60)[(calc_length+m_days):][:(60 + 1)][::60],
                                            '120d': log_y_nd(log_p, 120)[(calc_length+m_days):][:(120 + 1)][::120],
                                            '250d': log_y_nd(log_p, 250)[(calc_length+m_days):][:(250 + 1)][::250]}

                    features_label_dict['std'] = {'5d': std_nd(log_p, 5)[(calc_length+m_days):][:(5 + 1)][::5],
                                            '20d': std_nd(log_p, 20)[(calc_length+m_days):][:(20 + 1)][::20],
                                            '60d': std_nd(log_p, 60)[(calc_length+m_days):][:(60 + 1)][::60],
                                            '120d': std_nd(log_p, 120)[(calc_length+m_days):][:(120 + 1)][::120]}

                features_label_dict['pos'] = {'5d': np.sign(features_label_dict['logy']['5d']),
                                        '20d': np.sign(features_label_dict['logy']['20d']),
                                        '60d': np.sign(features_label_dict['logy']['60d'])}

                features_label_dict['mdd'] = {'20d': mdd_nd(log_p_wo_calc, 20)[m_days:][::k_days],
                                        '60d':  mdd_nd(log_p_wo_calc, 60)[m_days:][::k_days],
                                        '120d':  mdd_nd(log_p_wo_calc, 120)[m_days:][::k_days]}

                features_label_dict['fft'] = {'3com': fft(log_p_wo_calc, 3, m_days, k_days)[m_days:][::k_days],
                                        '6com': fft(log_p_wo_calc, 6, m_days, k_days)[m_days:][::k_days],
                                        '100com': fft(log_p_wo_calc, 100, m_days, k_days)[m_days:][::k_days]}

            elif label_type == 'test_label':
                assert len(log_p) == ((calc_length + m_days) + k_days + 1)

                if main_class == 'logy':
                    features_label_list = log_y_nd(log_p, n_days)[(calc_length+m_days):][:(n_days + 1)][::n_days]
                else:
                    print('[Feature class]label_type: {} Not Implemented for {}'.format(label_type, main_class))
                    raise NotImplementedError

        features_list = list()
        features_data = list()
        features_label = list()
        for key in self.structure.keys():
            if isinstance(self.structure[key], list):
                f_list_temp = list()
                for item in self.structure[key]:
                    f_list_temp.append(key + '_' + item)
                    features_data.append(features_data_dict[key][item])
                    if label_type == 'trainable_label':
                        features_label.append(features_label_dict[key][item])
                features_list = features_list + f_list_temp
            elif isinstance(self.structure[key], str):
                k, v = key.split('_')
                features_data.append(features_data_dict[k][v])
                if label_type == 'trainable_label':
                    features_label.append(features_label_dict[k][v])
                features_list = features_list + [key]

        if additional_dict is not None:
            for key in additional_dict.keys():
                data_raw = np.array(additional_dict[key])[calc_length:][:(m_days + 1)]
                label_raw = np.array(additional_dict[key])[(calc_length+m_days):][:(n_days + 1)][::n_days]

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

    def processing_split(self, df_not_null, m_days, k_days, calc_length=0):
        # if type(df.columns) == pd.MultiIndex:
        #     df.columns = df.columns.droplevel(0)
        features_dict = dict()
        log_p = np.log(df_not_null.values, dtype=np.float32)
        log_p_wo_calc = log_p[calc_length:]

        log_p = log_p - log_p[0, :]
        log_p_wo_calc = log_p_wo_calc - log_p_wo_calc[0, :]

        features_dict['logy'] = {'5d': log_y_nd(log_p, 5)[calc_length:],
                                '20d': log_y_nd(log_p, 20)[calc_length:],
                                '60d': log_y_nd(log_p, 60)[calc_length:],
                                '120d': log_y_nd(log_p, 120)[calc_length:]}

        features_dict['std'] = {'20d': std_nd(log_p, 20)[calc_length:],
                                '60d': std_nd(log_p, 60)[calc_length:],
                                '120d': std_nd(log_p, 120)[calc_length:]}

        features_dict['pos'] = {'5d': np.sign(features_dict['logy']['5d']),
                                '20d': np.sign(features_dict['logy']['20d']),
                                '60d': np.sign(features_dict['logy']['60d'])}

        features_dict['mdd'] = {'20d': mdd_nd(log_p_wo_calc, 20),
                                '60d': mdd_nd(log_p_wo_calc, 60),
                                '120d': mdd_nd(log_p_wo_calc, 120)}

        features_dict['fft'] = {'3com': fft(log_p_wo_calc, 3, m_days, k_days),
                                '6com': fft(log_p_wo_calc, 6, m_days, k_days),
                                '100com': fft(log_p_wo_calc, 100, m_days, k_days)}
        # features_dict['fft'] = {'3com': fft(log_p, 3, m_days, k_days, calc_length),
        #                         '6com': fft(log_p, 6, m_days, k_days, calc_length),
        #                         '100com': fft(log_p, 100, m_days, k_days, calc_length)}

        # features_dict['cum_log'] = {'5d': np.cumsum(features_dict['logy']['5d'], axis=0)}

        features_list = list()
        features_data = list()
        for key in self.structure.keys():
            if isinstance(self.structure[key], list):
                f_list_temp = list()
                for item in self.structure[key]:
                    f_list_temp.append(key + '_' + item)
                    features_data.append(features_dict[key][item])
                features_list = features_list + f_list_temp
            elif isinstance(self.structure[key], str):
                k, v = key.split('_')
                features_data.append(features_dict[k][v])
                features_list = features_list + [key]

        features_data = np.stack(features_data, axis=-1)

        assert len(features_list) == features_data.shape[-1]
        # feature_df = pd.DataFrame(np.transpose(features_data[:, :, 0]), columns=features_list)
        return features_list, features_data

    def predict_plot_mtl_cross_section_test(self, model, dataset_list, save_dir='out.png', ylog=False, time_step=1):
        if dataset_list is False:
            return False
        else:
            input_enc_list, output_dec_list, target_dec_list, features_list, additional_infos, start_date, end_date = dataset_list

        # if self.label_feature[:3] == 'pos':
        #     idx_y_nm = 'logy_' + self.structure[self.label_feature]
        # else:
        #     idx_y_nm = self.label_feature
        #
        # idx_y = features_list.index(idx_y_nm)
        idx_y = features_list.index(self.label_feature)

        true_y = np.zeros(int(np.ceil(len(input_enc_list) / time_step)) + 1)
        true_y_mw = np.zeros(int(np.ceil(len(input_enc_list) / time_step)) + 1)

        pred_q1 = np.zeros_like(true_y)
        pred_q2 = np.zeros_like(true_y)
        pred_q3 = np.zeros_like(true_y)
        pred_q4 = np.zeros_like(true_y)
        pred_q5 = np.zeros_like(true_y)
        pos_wgt = np.zeros_like(true_y)

        pred_q1_mw = np.zeros_like(true_y)
        pred_q2_mw = np.zeros_like(true_y)
        pred_q3_mw = np.zeros_like(true_y)
        pred_q4_mw = np.zeros_like(true_y)
        pred_q5_mw = np.zeros_like(true_y)

        size_value_list = [add_info['mktcap'] for add_info in additional_infos]
        for i, (input_enc_t, output_dec_t, target_dec_t, size_value) in enumerate(zip(input_enc_list, output_dec_list, target_dec_list, size_value_list)):
            if i % time_step != 0:
                continue
            t = i // time_step + 1
            assert np.sum(input_enc_t[:, -1, idx_y] - output_dec_t[:, 0, idx_y]) == 0
            new_output_t = np.zeros_like(output_dec_t)
            new_output_t[:, 0, :] = output_dec_t[:, 0, :] + size_value[:, 0, :]

            features = {'input': input_enc_t, 'output': new_output_t}
            labels = target_dec_t

            predictions = model.predict_mtl(features)
            # p_ret, p_pos, p_vol, p_mdd = predictions

            true_y[t] = np.mean(labels[:, 0, idx_y])
            value_ = predictions[self.pred_feature][:, 0, 0]

            q1_crit, q2_crit, q3_crit, q4_crit = np.percentile(value_, q=[80, 60, 40, 20])
            crit1 = (value_ >= q1_crit)
            crit2 = ((value_ >= q2_crit) & (value_ < q1_crit))
            crit3 = ((value_ >= q3_crit) & (value_ < q2_crit))
            crit4 = ((value_ >= q4_crit) & (value_ < q3_crit))
            crit5 = (value_ < q4_crit)
            pred_q1[t] = np.mean(labels[crit1, 0, idx_y])
            pred_q2[t] = np.mean(labels[crit2, 0, idx_y])
            pred_q3[t] = np.mean(labels[crit3, 0, idx_y])
            pred_q4[t] = np.mean(labels[crit4, 0, idx_y])
            pred_q5[t] = np.mean(labels[crit5, 0, idx_y])
            pos_wgt[t] = np.sum(value_ > 0.5) / len(value_)

            true_y_mw[t] = np.sum(labels[:, 0, idx_y] * size_value[:, 0, 0]) / np.sum(size_value[:, 0, 0])
            pred_q1_mw[t] = np.sum(labels[crit1, 0, idx_y] * size_value[crit1, 0, 0]) / np.sum(size_value[crit1, 0, 0])
            pred_q2_mw[t] = np.sum(labels[crit2, 0, idx_y] * size_value[crit2, 0, 0]) / np.sum(size_value[crit2, 0, 0])
            pred_q3_mw[t] = np.sum(labels[crit3, 0, idx_y] * size_value[crit3, 0, 0]) / np.sum(size_value[crit3, 0, 0])
            pred_q4_mw[t] = np.sum(labels[crit4, 0, idx_y] * size_value[crit4, 0, 0]) / np.sum(size_value[crit4, 0, 0])
            pred_q5_mw[t] = np.sum(labels[crit5, 0, idx_y] * size_value[crit5, 0, 0]) / np.sum(size_value[crit5, 0, 0])

        data = pd.DataFrame({'true_y': np.cumprod(1. + true_y),
                             'pred_ls': np.cumprod(1. + pred_q1 - pred_q5),
                             'pred_q1': np.cumprod(1. + pred_q1),
                             'pred_q2': np.cumprod(1. + pred_q2),
                             'pred_q3': np.cumprod(1. + pred_q3),
                             'pred_q4': np.cumprod(1. + pred_q4),
                             'pred_q5': np.cumprod(1. + pred_q5),
                             'pos_wgt': pos_wgt,
                             'true_y_mw': np.cumprod(1. + true_y_mw),
                             'pred_ls_mw': np.cumprod(1. + pred_q1_mw - pred_q5_mw),
                             'pred_q1_mw': np.cumprod(1. + pred_q1_mw),
                             'pred_q2_mw': np.cumprod(1. + pred_q2_mw),
                             'pred_q3_mw': np.cumprod(1. + pred_q3_mw),
                             'pred_q4_mw': np.cumprod(1. + pred_q4_mw),
                             'pred_q5_mw': np.cumprod(1. + pred_q5_mw),
        })

        # equal fig
        fig = plt.figure()
        fig.suptitle('{} ~ {}'.format(start_date, end_date))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        ax1.plot(data[['true_y', 'pred_ls', 'pred_q1', 'pred_q5']])
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax1.legend(['true_y', 'long-short', 'long', 'short'], loc='center left', bbox_to_anchor=(1, 0.5))
        if ylog:
            ax1.set_yscale('log', basey=2)

        ax2.plot(data[['true_y', 'pred_q1', 'pred_q2', 'pred_q3', 'pred_q4', 'pred_q5']])
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax2.legend(['true_y', 'q1', 'q2', 'q3', 'q4', 'q5'], loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.set_yscale('log', basey=2)

        # ax3.plot(data[['pos_wgt']])
        # box = ax3.get_position()
        # ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # ax3.legend(['positive wgt'], loc='center left', bbox_to_anchor=(1, 0.5))

        # value fig
        ax3.plot(data[['true_y_mw', 'pred_ls_mw', 'pred_q1_mw', 'pred_q5_mw']])
        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax3.legend(['true_y(mw)', 'long-short', 'long', 'short'], loc='center left', bbox_to_anchor=(1, 0.5))
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

    features_data = np.stack([log_5y, log_20y, log_60y, log_120y,
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




