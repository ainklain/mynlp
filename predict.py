

from config import Config
from model import NLPModel
import data_process
from data_process import dataset_process, load_vocabulary, load_data, \
    enc_processing, dec_output_processing, dec_target_processing, pred_next_string



import numpy as np
import os
import sys
import tensorflow as tf

if __name__ == '__main__':
    configs = Config()
    arg_length = len(sys.argv)

    assert arg_length >= 2

    char2idx, idx2char, configs.vocabulary_length = load_vocabulary(configs.vocabulary_path, configs.data_path, configs.tokenize_as_morph)

    input = " ".join(sys.argv[1:])
    print(input)
    predict_input_enc, predict_input_enc_length = enc_processing([input], char2idx, configs.max_sequence_length, configs.tokenize_as_morph)
    predict_output_dec, predict_output_dec_length = dec_output_processing([""], char2idx, configs.max_sequence_length, configs.tokenize_as_morph)
    predict_target_dec = dec_target_processing([""], char2idx, configs.max_sequence_length, configs.tokenize_as_morph)

    model = NLPModel(configs)
    if os.path.exists(configs.f_name):
        model.load_model(configs.f_name)

    for i in range(configs.max_sequence_length):
        if i > 0:
            predict_output_dec, predict_output_decLength = dec_output_processing([answer], char2idx, configs.max_sequence_length, configs.tokenize_as_morph)
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
