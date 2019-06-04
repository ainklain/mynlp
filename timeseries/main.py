# https://github.com/NLP-kr/tensorflow-ml-nlp


from timeseries.config import Config
from timeseries.model import TSModel

import data_process
from timeseries.data_process import dataset_process, load_data, \
    enc_processing, dec_output_processing, dec_target_processing, pred_next_string


import numpy as np
import os
import sys
import tensorflow as tf



def main():
    configs = Config()

    data_out_path = os.path.join(os.getcwd(), './out')
    os.makedirs(data_out_path, exist_ok=True)


    # 훈련 데이터와 테스트 데이터를 가져온다.
    train_input, train_label, eval_input, eval_label = load_data(configs.data_path, embedding_length=configs.embedding_per_feature)

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
    dataset_eval = dataset_process(eval_input_enc, eval_output_dec, eval_target_dec, configs.batch_size)


    model = TSModel(configs)
    if os.path.exists(configs.f_name):
        model.load_model(configs.f_name)

    for i, (features, labels) in enumerate(dataset_train.take(configs.train_steps)):
        model.train(features, labels)
        if i % 200 == 0:
            model.save_model(configs.f_name)

            predict_input_enc, predic_input_enc_length = enc_processing(["가끔 궁금해"], char2idx, configs.max_sequence_length, configs.tokenize_as_morph)
            # 학습 과정이 아니므로 디코딩 입력은
            # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
            predict_output_dec, predic_output_decLength = dec_output_processing([""], char2idx, configs.max_sequence_length, configs.tokenize_as_morph)
            # 학습 과정이 아니므로 디코딩 출력 부분도
            # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
            predict_target_dec = dec_target_processing([""], char2idx, configs.max_sequence_length, configs.tokenize_as_morph)

            for i in range(configs.max_sequence_length):
                if i > 0:
                    predict_output_dec, _ = dec_output_processing([answer], char2idx, configs.max_sequence_length, configs.tokenize_as_morph)
                    predict_target_dec = dec_target_processing([answer], char2idx, configs.max_sequence_length, configs.tokenize_as_morph)
                # 예측을 하는 부분이다.

                dataset_test = dataset_process(predict_input_enc, predict_output_dec, predict_target_dec, 1)
                for (feature, _) in dataset_test.take(1):
                    predictions = model.predict(feature)

                answer, finished = pred_next_string(predictions.numpy(), idx2char)

                if finished:
                    break

            # 예측한 값을 인지 할 수 있도록
            # 텍스트로 변경하는 부분이다.
            print("answer: ", answer)





