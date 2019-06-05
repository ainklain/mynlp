
import pandas as pd
import tensorflow as tf
import tqdm
import os
import re
from sklearn.model_selection import train_test_split
import numpy as np


def normalize(arr_x, eps=1e-6):
    return (arr_x - np.mean(arr_x, axis=0)) / (np.std(arr_x, axis=0) + eps)


def std_arr(arr_x, n):
    stdarr = np.zeros_like(arr_x)
    for t in range(1, len(arr_x)):
        stdarr[t] = np.std(arr_x[max(0, t-n):(t+1)])

    return stdarr


def mdd_arr(logcumarr_x, n):
    mddarr = np.zeros_like(logcumarr_x)
    for t in range(len(logcumarr_x)):
        mddarr[t] = logcumarr_x[t] - np.max(logcumarr_x[max(0, t-n):(t+1)])

    return mddarr


def dict_to_list(dict, key_list=None):
    arr = list()
    if key_list is None:
        key_list = dict.keys()

    for key in key_list:
        arr.append(dict[key])

    return np.stack(arr, axis=1), list(key_list)



def load_data(data_path, name='kospi', token_length=5):
    # 판다스를 통해서 데이터를 불러온다.
    data_df = pd.read_csv(data_path, header=0)
    # 질문과 답변 열을 가져와 question과 answer에 넣는다.

    y_1d = np.array(data_df[name], dtype=np.float32)
    log_cum_y = np.cumsum(np.log(1. + y_1d))


    features = dict()
    # daily returns
    features['y'] = np.concatenate([log_cum_y[:token_length],
                                    log_cum_y[token_length:] - log_cum_y[:-token_length]])[::token_length]

    # cumulative returns
    features['log_cum_y'] = np.cumsum(np.log(1. + features['y']))

    # positive
    features['positive'] = (features['y'] >= 0) * np.array(1., dtype=np.float32) - (features['y'] < 0) * np.array(1., dtype=np.float32)

    # moving average
    for n in [5, 20, 60, 120]:
        if n == token_length:
            continue
        features['y_{}d'.format(n)] = np.concatenate([log_cum_y[:n], log_cum_y[n:] - log_cum_y[:-n]])[::token_length]

    # std
    for n in [20, 60, 120]:
        features['std{}d'.format(n)] = std_arr(y_1d, n)[::token_length]
        features['mdd{}d'.format(n)] = mdd_arr(log_cum_y, n)[::token_length]

    features_list_normalize, columns_list_normal = dict_to_list(features, ['log_cum_y'])
    features_list_linear, columns_list_linear = dict_to_list(features, features.keys() - ['log_cum_y'])
    question = list()
    answer = list()

    M = 60 // token_length
    K = 20 // token_length
    for i in range(M, len(features['y']) - K):
        sub_features_linear = features_list_linear[(i - M): (i + K)]
        sub_features_normalize = normalize(features_list_normalize[(i - M): (i + K)])

        sub_features = np.concatenate([sub_features_linear, sub_features_normalize], axis=1)
        columns_list = columns_list_linear + columns_list_normal
        question.append(sub_features[:M])
        answer.append(sub_features[M-1:])

    train_input, eval_input, train_label, eval_label = train_test_split(question, answer, test_size=0.33, random_state=123)
    # 그 값을 리턴한다.
    return train_input, train_label, eval_input, eval_label, columns_list


class FeatureCalculator:
    def y_1d(self, prc_arr):
        return prc_arr[1:] / prc_arr[:-1] - 1

    def log_y_1d(self, prc_arr):
        return np.log(prc_arr[1:] / prc_arr[:-1])

    def log_cum_y(self, y):
        return np.cumsum(np.log(1. + y))

class DataGenerator:
    def __init__(self, data_path):
        # data_path ='./timeseries/asset_data.csv'
        data_df = pd.read_csv(data_path, index_col=0)
        self.df_pivoted = data_df.pivot(index='eval_d', columns='bbticker')

    def generate_dataset(self, start_d, end_d, token_length=5):
        df_selected = self.df_pivoted[(self.df_pivoted.index >= start_d) & (self.df_pivoted.index <= end_d)]
        full_data_selected = df_selected.ix[:, np.sum(df_selected.isna(), axis=0) == 0]

        train_rate = 0.6
        train_dataset = full_data_selected.iloc[:int(len(full_data_selected) * train_rate)]
        eval_dataset = full_data_selected.iloc[:int(len(full_data_selected) * train_rate)]

        factor_data_train = self.data_to_factor(train_dataset, token_length)
        factor_data_eval = self.data_to_factor(eval_dataset, token_length)

    def data_to_factor(self, dataset_df, token_length=5):
        prc = np.array(dataset_df, dtype=np.float32)

        fc = FeatureCalculator()
        y_1d = fc.y_1d(prc)
        log_cum_y = fc.log_cum_y(y_1d)

        features = dict()
        # daily returns
        features['y'] = np.concatenate([log_cum_y[:token_length],
                                        log_cum_y[token_length:] - log_cum_y[:-token_length]])[::token_length]

        # cumulative returns
        features['log_cum_y'] = np.cumsum(np.log(1. + features['y']))

        # positive
        features['positive'] = (features['y'] >= 0) * np.array(1., dtype=np.float32) - (features['y'] < 0) * np.array(
            1., dtype=np.float32)

        # moving average
        for n in [5, 20, 60, 120]:
            if n == token_length:
                continue
            features['y_{}d'.format(n)] = np.concatenate([log_cum_y[:n], log_cum_y[n:] - log_cum_y[:-n]])[
                                          ::token_length]

        # std
        for n in [20, 60, 120]:
            features['std{}d'.format(n)] = std_arr(y_1d, n)[::token_length]
            features['mdd{}d'.format(n)] = mdd_arr(log_cum_y, n)[::token_length]

        features_list_normalize, columns_list_normal = dict_to_list(features, ['log_cum_y'])
        features_list_linear, columns_list_linear = dict_to_list(features, features.keys() - ['log_cum_y'])
        question = list()
        answer = list()

        M = 60 // token_length
        K = 20 // token_length
        for i in range(M, len(features['y']) - K):
            sub_features_linear = features_list_linear[(i - M): (i + K)]
            sub_features_normalize = normalize(features_list_normalize[(i - M): (i + K)])

            sub_features = np.concatenate([sub_features_linear, sub_features_normalize], axis=1)
            columns_list = columns_list_linear + columns_list_normal
            question.append(sub_features[:M])
            answer.append(sub_features[M - 1:])


# 인덱스화 할 value와 키가 워드이고
# 값이 인덱스인 딕셔너리를 받는다.
def enc_processing(value):


    return np.asarray(sequences_input_index)


# 인덱스화 할 value 키가 워드 이고 값이
# 인덱스인 딕셔너리를 받는다.
def dec_output_processing(value, dictionary, max_sequence_length, tokenize_as_morph=False):
    # 인덱스 값들을 가지고 있는
    # 배열이다.(누적된다)
    sequences_output_index = []
    # 하나의 디코딩 입력 되는 문장의
    # 길이를 가지고 있다.(누적된다)
    sequences_length = []
    # 한줄씩 불어온다.
    for sequence in value:
        # FILTERS = "([~.,!?\"':;)(])"
        # 정규화를 사용하여 필터에 들어 있는
        # 값들을 "" 으로 치환 한다.
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        # 하나의 문장을 디코딩 할때 가지고
        # 있기 위한 배열이다.
        sequence_index = []
        # 디코딩 입력의 처음에는 START가 와야 하므로
        # 그 값을 넣어 주고 시작한다.
        # 문장에서 스페이스 단위별로 단어를 가져와서 딕셔너리의
        # 값인 인덱스를 넣어 준다.
        sequence_index = [dictionary[STD]] + [dictionary[word] for word in sequence.split()]
        # 문장 제한 길이보다 길어질 경우 뒤에 토큰을 자르고 있다.
        if len(sequence_index) > max_sequence_length:
            sequence_index = sequence_index[:max_sequence_length]
        # 하나의 문장에 길이를 넣어주고 있다.
        sequences_length.append(len(sequence_index))
        # max_sequence_length보다 문장 길이가
        # 작다면 빈 부분에 PAD(0)를 넣어준다.
        sequence_index += (max_sequence_length - len(sequence_index)) * [dictionary[PAD]]
        # 인덱스화 되어 있는 값을
        # sequences_output_index 넣어 준다.
        sequences_output_index.append(sequence_index)
    # 인덱스화된 일반 배열을 넘파이 배열로 변경한다.
    # 이유는 텐서플로우 dataset에 넣어 주기 위한
    # 사전 작업이다.
    # 넘파이 배열에 인덱스화된 배열과 그 길이를 넘겨준다.
    return np.asarray(sequences_output_index), sequences_length


# 인덱스화 할 value와 키가 워드 이고
# 값이 인덱스인 딕셔너리를 받는다.
def dec_target_processing(value, dictionary, max_sequence_length, tokenize_as_morph=False):
    # 인덱스 값들을 가지고 있는
    # 배열이다.(누적된다)
    sequences_target_index = []
    # 한줄씩 불어온다.
    for sequence in value:
        # FILTERS = "([~.,!?\"':;)(])"
        # 정규화를 사용하여 필터에 들어 있는
        # 값들을 "" 으로 치환 한다.
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        # 문장에서 스페이스 단위별로 단어를 가져와서
        # 딕셔너리의 값인 인덱스를 넣어 준다.
        # 디코딩 출력의 마지막에 END를 넣어 준다.
        sequence_index = [dictionary[word] for word in sequence.split()]
        # 문장 제한 길이보다 길어질 경우 뒤에 토큰을 자르고 있다.
        # 그리고 END 토큰을 넣어 준다
        if len(sequence_index) >= max_sequence_length:
            sequence_index = sequence_index[:max_sequence_length - 1] + [dictionary[END]]
        else:
            sequence_index += [dictionary[END]]
        # max_sequence_length보다 문장 길이가
        # 작다면 빈 부분에 PAD(0)를 넣어준다.
        sequence_index += (max_sequence_length - len(sequence_index)) * [dictionary[PAD]]
        # 인덱스화 되어 있는 값을
        # sequences_target_index에 넣어 준다.
        sequences_target_index.append(sequence_index)
    # 인덱스화된 일반 배열을 넘파이 배열로 변경한다.
    # 이유는 텐서플로우 dataset에 넣어 주기 위한 사전 작업이다.
    # 넘파이 배열에 인덱스화된 배열과 그 길이를 넘겨준다.
    return np.asarray(sequences_target_index)


def rearrange(input, output, target):
    features = {"input": input, "output": output}
    return features, target


# 학습에 들어가 배치 데이터를 만드는 함수이다.
def dataset_process(train_input_enc, train_output_dec, train_target_dec, batch_size, mode='train'):
    # Dataset을 생성하는 부분으로써 from_tensor_slices부분은
    # 각각 한 문장으로 자른다고 보면 된다.
    # train_input_enc, train_output_dec, train_target_dec
    # 3개를 각각 한문장으로 나눈다.
    dataset = tf.data.Dataset.from_tensor_slices((train_input_enc, train_output_dec, train_target_dec))
    # 전체 데이터를 섞는다.
    dataset = dataset.shuffle(buffer_size=len(train_input_enc))
    # 배치 인자 값이 없다면  에러를 발생 시킨다.
    assert batch_size is not None, "train batchSize must not be None"
    # from_tensor_slices를 통해 나눈것을
    # 배치크기 만큼 묶어 준다.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # 데이터 각 요소에 대해서 rearrange 함수를
    # 통해서 요소를 변환하여 맵으로 구성한다.
    dataset = dataset.map(rearrange)
    # repeat()함수에 원하는 에포크 수를 넣을수 있으면
    # 아무 인자도 없다면 무한으로 이터레이터 된다.
    if mode == 'train':
        dataset = dataset.repeat()
    elif mode == 'test':
        dataset = dataset.repeat(1)
    else:
        dataset = dataset.repeat()
    # make_one_shot_iterator를 통해 이터레이터를
    # 만들어 준다.
    # 이터레이터를 통해 다음 항목의 텐서
    # 개체를 넘겨준다.
    return dataset


def pred_next_string(value, dictionary):
    # 텍스트 문장을 보관할 배열을 선언한다.
    sentence_string = []
    is_finished = False

    # 인덱스 배열 하나를 꺼내서 v에 넘겨준다.
    for v in value:
        # 딕셔너리에 있는 단어로 변경해서 배열에 담는다.
        sentence_string = [dictionary[index] for index in v]

    answer = ""
    # 패딩값도 담겨 있으므로 패딩은 모두 스페이스 처리 한다.
    for word in sentence_string:
        if word == END:
            is_finished = True
            break

        if word != PAD and word != END:
            answer += word
            answer += " "

    # 결과를 출력한다.
    return answer, is_finished







