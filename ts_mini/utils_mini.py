
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def predict_plot_mtl_cross_section_test(model, dataset_list, save_dir='out.png', ylog=False, eval_type='pos'):
    if dataset_list is False:
        return False
    else:
        input_enc_list, output_dec_list, target_dec_list, features_list, additional_infos, start_date, end_date = dataset_list

    idx_y = features_list.index('log_y')

    true_y = np.zeros(len(input_enc_list) + 1)
    pred_ls = np.zeros_like(true_y)

    pred_q1 = np.zeros_like(true_y)
    pred_q2 = np.zeros_like(true_y)
    pred_q3 = np.zeros_like(true_y)
    pred_q4 = np.zeros_like(true_y)
    pred_q5 = np.zeros_like(true_y)

    for i, (input_enc_t, output_dec_t, target_dec_t) in enumerate(zip(input_enc_list, output_dec_list, target_dec_list)):
        t = i + 1
        assert np.sum(input_enc_t[:, -1, :] - output_dec_t[:, 0, :]) == 0
        new_output_t = np.zeros_like(output_dec_t)
        new_output_t[:, 0, :] = output_dec_t[:, 0, :]

        features = {'input': input_enc_t, 'output': new_output_t}
        labels = target_dec_t

        predictions = model.predict_mtl(features)
        # p_ret, p_pos, p_vol, p_mdd = predictions

        true_y[t] = np.mean(labels[:, 0, idx_y])
        if eval_type == 'pos':
            value_ = predictions['pos'][:, 0, 0]
        elif eval_type == 'pos20':
            value_ = predictions['pos20'][:, 0, 0]
        elif eval_type == 'ret':
            value_ = predictions['ret'][:, 0, 0]
        elif eval_type == 'ir20':
            value_ = predictions['ret'][:, 0, 1] / np.abs(predictions['std'][:, 0, 0])
        elif eval_type == 'ir60':
            value_ = predictions['ret'][:, 0, 2] / np.abs(predictions['std'][:, 0, 1])
        elif eval_type == 'ir120':
            value_ = predictions['ret'][:, 0, 3] / np.abs(predictions['std'][:, 0, 2])

        q1_crit, q2_crit, q3_crit, q4_crit = np.percentile(value_, q=[80, 60, 40, 20])
        pred_q1[t] = np.mean(labels[value_ >= q1_crit, 0, idx_y])
        pred_q2[t] = np.mean(labels[(value_ >= q2_crit) & (value_ < q1_crit), 0, idx_y])
        pred_q3[t] = np.mean(labels[(value_ >= q3_crit) & (value_ < q2_crit), 0, idx_y])
        pred_q4[t] = np.mean(labels[(value_ >= q4_crit) & (value_ < q3_crit), 0, idx_y])
        pred_q5[t] = np.mean(labels[(value_ < q4_crit), 0, idx_y])

    data = pd.DataFrame({'true_y': np.cumprod(1. + true_y),
                         'pred_ls': np.cumprod(1. + pred_q1 - pred_q5),
                         'pred_q1': np.cumprod(1. + pred_q1),
                         'pred_q2': np.cumprod(1. + pred_q2),
                         'pred_q3': np.cumprod(1. + pred_q3),
                         'pred_q4': np.cumprod(1. + pred_q4),
                         'pred_q5': np.cumprod(1. + pred_q5)
    })

    fig = plt.figure()
    fig.suptitle('{} ~ {}'.format(start_date, end_date))
    ax1, ax2 = fig.subplots(2, 1)
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

    fig.savefig(save_dir)
    print("figure saved. (dir: {})".format(save_dir))
    plt.close(fig)

