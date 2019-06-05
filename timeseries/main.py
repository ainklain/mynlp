# https://github.com/NLP-kr/tensorflow-ml-nlp


from timeseries.config import Config
from timeseries.model import TSModel

import data_process
from timeseries.data_process import dataset_process, load_data, \
    enc_processing, dec_output_processing, dec_target_processing, pred_next_string


import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


def predict_plot(model, dataset, columns_list, size=250, save_dir='out.jpg'):
    cost_rate = 0.002
    idx_y = columns_list.index('y')
    idx_pos = columns_list.index('positive')

    true_y = np.zeros(size)
    pred_both = np.zeros_like(true_y)
    pred_pos = np.zeros_like(true_y)
    pred_y = np.zeros_like(true_y)
    pred_avg = np.zeros_like(true_y)

    for j, (features, labels) in enumerate(dataset.take(size)):
        prev_w_both = 0
        prev_w_pos = 0
        prev_w_y = 0
        prev_w_avg = 0
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


def main():
    configs = Config()

    name = 'gpa'
    data_out_path = os.path.join(os.getcwd(), './out/{}'.format(name))
    os.makedirs(data_out_path, exist_ok=True)


    # 훈련 데이터와 테스트 데이터를 가져온다.
    train_input, train_label, eval_input, eval_label, columns_list = load_data(configs.data_path, name=name, token_length=5)

    # 훈련셋 인코딩 만드는 부분이다.
    train_input_enc = train_input[:]
    # 훈련셋 디코딩 입력 부분 만드는 부분이다.
    train_output_dec = [label[:-1] for label in train_label]
    # 훈련셋 디코딩 출력 부분 만드는 부분이다.
    train_target_dec = [label[1:] for label in train_label]

    # 훈련셋 인코딩 만드는 부분이다.
    eval_input_enc = eval_input[:]
    # 훈련셋 디코딩 입력 부분 만드는 부분이다.
    eval_output_dec = [label[:-1] for label in eval_label]
    # 훈련셋 디코딩 출력 부분 만드는 부분이다.
    eval_target_dec = [label[1:] for label in eval_label]

    dataset_train = dataset_process(train_input_enc, train_output_dec, train_target_dec, configs.batch_size)
    dataset_insample_test = dataset_process(train_input_enc, train_output_dec, train_target_dec, 1, mode='test')
    dataset_eval = dataset_process(eval_input_enc, eval_output_dec, eval_target_dec, 1, mode='test')

    model = TSModel(configs)
    if os.path.exists(configs.f_name):
        model.load_model(configs.f_name)

    for i, (features, labels) in enumerate(dataset_train.take(configs.train_steps)):
        model.train(features, labels)
        if i % 100 == 0:
            # model.encoder.show(i, save=True)
            model.save_model(configs.f_name)



            predict_plot(model, dataset_insample_test, columns_list, len(train_input_enc), save_dir='out/{}/train_{}.jpg'.format(name, i))
            predict_plot(model, dataset_eval, columns_list, len(eval_input_enc), save_dir='out/{}/eval_{}.jpg'.format(name, i))






