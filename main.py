# https://github.com/NLP-kr/tensorflow-ml-nlp


from model import NLPModel
import data_process
from data_process import dataset_process, load_vocabulary, load_data, \
    enc_processing, dec_output_processing, dec_target_processing, pred_next_string

import numpy as np
import os
import sys
import tensorflow as tf

class Config:
    def __init__(self):
        self.batch_size = 64
        self.train_steps = 20
        self.dropout_width = 0.5
        self.embedding_size = 128
        self.learning_rate = 1e-3
        self.shuffle_seek = 1000
        self.max_sequence_length = 25
        self.model_hidden_size = 128
        self.ffn_hidden_size = 512
        self.attention_head_size = 4
        self.layer_size = 2
        self.data_path = './data/ChatBotData.csv'
        self.vocabulary_path = './data/vocabularyData.txt'
        self.checkpoint_path = './ckpt/'
        self.tokenize_as_morph = False
        self.xavier_initializer = True



def main():
    configs = Config()

    data_out_path = os.path.join(os.getcwd(), './out')
    os.makedirs(data_out_path, exist_ok=True)

    # 데이터를 통한 사전 구성 한다.
    char2idx, idx2char, configs.vocabulary_length = load_vocabulary(configs.vocabulary_path, configs.data_path, configs.tokenize_as_morph)
    # 훈련 데이터와 테스트 데이터를 가져온다.
    train_input, train_label, eval_input, eval_label = load_data(configs.data_path)

    # 훈련셋 인코딩 만드는 부분이다.
    train_input_enc, train_input_enc_length = enc_processing(train_input, char2idx, configs.max_sequence_length, configs.tokenize_as_morph)
    # 훈련셋 디코딩 입력 부분 만드는 부분이다.
    train_output_dec, train_output_dec_length = dec_output_processing(train_label, char2idx, configs.max_sequence_length, configs.tokenize_as_morph)
    # 훈련셋 디코딩 출력 부분 만드는 부분이다.
    train_target_dec = dec_target_processing(train_label, char2idx, configs.max_sequence_length, configs.tokenize_as_morph)

    # 평가셋 인코딩 만드는 부분이다.
    eval_input_enc, eval_input_enc_length = enc_processing(eval_input, char2idx, configs.max_sequence_length, configs.tokenize_as_morph)
    # 평가셋 인코딩 만드는 부분이다.
    eval_output_dec, eval_output_dec_length = dec_output_processing(eval_label, char2idx, configs.max_sequence_length, configs.tokenize_as_morph)
    # 평가셋 인코딩 만드는 부분이다.
    eval_target_dec = dec_target_processing(eval_label, char2idx, configs.max_sequence_length, configs.tokenize_as_morph)

    # # 현재 경로'./'에 현재 경로 하부에
    # # 체크 포인트를 저장한 디렉토리를 설정한다.
    # check_point_path = os.path.join(os.getcwd(), configs.check_point_path)
    # # 디렉토리를 만드는 함수이며 두번째 인자 exist_ok가
    # # True이면 디렉토리가 이미 존재해도 OSError가
    # # 발생하지 않는다.
    # # exist_ok가 False이면 이미 존재하면
    # # OSError가 발생한다.
    # os.makedirs(check_point_path, exist_ok=True)

    dataset_train = dataset_process(train_input_enc, train_output_dec, train_target_dec, configs.batch_size)
    dataset_eval = dataset_process(eval_input_enc, eval_output_dec, eval_target_dec, configs.batch_size)


    model = NLPModel(configs)
    for i, (features, labels) in enumerate(dataset_train.take(configs.train_steps)):
        model.train(features, labels)


    predict_input_enc, predic_input_enc_length = enc_processing(["가끔 궁금해"], char2idx, configs.max_sequence_length, configs.tokenize_as_morph)
    # 학습 과정이 아니므로 디코딩 입력은
    # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
    predict_output_dec, predic_output_decLength = dec_output_processing([""], char2idx, configs.max_sequence_length, configs.tokenize_as_morph)
    # 학습 과정이 아니므로 디코딩 출력 부분도
    # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
    predict_target_dec = dec_target_processing([""], char2idx, configs.max_sequence_length, configs.tokenize_as_morph)

    dataset_test = dataset_process(predict_input_enc, predict_output_dec, predict_target_dec, 1)
    for i in range(configs.max_sequence_length):
        if i > 0:
            predict_output_dec, _ = dec_output_processing([answer], char2idx)
            predict_target_dec = dec_target_processing([answer], char2idx)
        # 예측을 하는 부분이다.
        for (feature, _) in dataset_test.take(1):
            predictions = model.predict(feature)

        answer, finished = pred_next_string(predictions, idx2char)

        if finished:
            break

    # 예측한 값을 인지 할 수 있도록
    # 텍스트로 변경하는 부분이다.
    print("answer: ", answer)

















    # 에스티메이터 구성한다.
    classifier = tf.estimator.Estimator(
        model_fn=ml.Model,  # 모델 등록한다.
        model_dir=DEFINES.check_point_path,  # 체크포인트 위치 등록한다.
        params={  # 모델 쪽으로 파라메터 전달한다.
            'embedding_size': DEFINES.embedding_size,
            'model_hidden_size': DEFINES.model_hidden_size,  # 가중치 크기 설정한다.
            'ffn_hidden_size': DEFINES.ffn_hidden_size,
            'attention_head_size': DEFINES.attention_head_size,
            'learning_rate': DEFINES.learning_rate,  # 학습율 설정한다.
            'vocabulary_length': vocabulary_length,  # 딕셔너리 크기를 설정한다.
            'embedding_size': DEFINES.embedding_size,  # 임베딩 크기를 설정한다.
            'layer_size': DEFINES.layer_size,
            'max_sequence_length': DEFINES.max_sequence_length,
            'xavier_initializer': DEFINES.xavier_initializer
        })

    # 학습 실행
    classifier.train(input_fn=lambda: data.train_input_fn(
        train_input_enc, train_output_dec, train_target_dec, DEFINES.batch_size), steps=DEFINES.train_steps)

    eval_result = classifier.evaluate(input_fn=lambda: data.eval_input_fn(
        eval_input_enc, eval_output_dec, eval_target_dec, DEFINES.batch_size))
    print('\nEVAL set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # 테스트용 데이터 만드는 부분이다.
    # 인코딩 부분 만든다.
    predict_input_enc, _ = enc_processing(["가끔 궁금해"], char2idx)
    # 학습 과정이 아니므로 디코딩 입력은
    # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
    predict_output_dec, _ = dec_output_processing([""], char2idx)
    # 학습 과정이 아니므로 디코딩 출력 부분도
    # 존재하지 않는다.(구조를 맞추기 위해 넣는다.)
    predict_target_dec = dec_target_processing([""], char2idx)

    dataset_test = dataset_process(predict_input_enc, predict_output_dec, predict_target_dec, 1)
    for i in range(configs.max_sequence_length):
        if i > 0:
            predict_output_dec, _ = dec_output_processing([answer], char2idx)
            predict_target_dec = dec_target_processing([answer], char2idx)
        # 예측을 하는 부분이다.
        for (feature, _) in dataset_test.take(1):
            predictions = model.predict(feature)

        answer, finished = pred_next_string(predictions, idx2char)

        if finished:
            break

    # 예측한 값을 인지 할 수 있도록
    # 텍스트로 변경하는 부분이다.
    print("answer: ", answer)








