# https://github.com/NLP-kr/tensorflow-ml-nlp

from timeseries.config import Config
from timeseries.model import TSModel
from timeseries.data_process import dataset_process, load_data, DataGenerator, DataScheduler

from timeseries.rl import MyEnv, PPO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


EP_MAX = 1000
BATCH = 2048
GAMMA = 0.99
LAMBDA = 0.95


def main_all_asset_dataprocess_modified():
        configs = Config()

        # initiate and load model
        # model = TSModel(configs)
        # if os.path.exists(configs.f_name):
        #     model.load_model(configs.f_name)

        # get data for all assets and dates
        ds = DataScheduler(configs)
        # ds.test_end_idx = ds.base_idx + 1000
        ds.set_idx(4000)

        ii = 0
        while not ds.done:
            model = TSModel(configs)
            configs.f_name = 'ts_model_test1.4'
            if os.path.exists(configs.f_name):
                model.load_model(configs.f_name)

            # ds.set_idx(3000)
            ds.train(model,
                   train_steps=configs.train_steps,
                   eval_steps=10,
                   save_steps=50,
                   early_stopping_count=5,
                   model_name=configs.f_name)

            # env = MyEnv(model, data_scheduler=ds, configs=configs, trading_costs=0.001)
            #
            # ppo = PPO(env)
            # f_name = './{}.pkl'.format('actor_v1.0')
            # if os.path.exists(f_name):
            #     ppo.load_model(f_name)


            test_dataset_list, features_list = ds.test(model)
            # test_dataset_list, features_list = ds.test(model, ppo)

            ds.next()

            # ds.set_idx(4005)
            # ds.train_tickers(model,
            #                  ['aex index', 'spx index', 'kospi index', 'krw krwt curncy'],
            #                train_steps=configs.train_steps,
            #                eval_steps=10,
            #                save_steps=50,
            #                early_stopping_count=2,
            #                model_name='ts_model_test')

            # input_enc, output_dec, target_dec, features_list = ds.test_bbticker(model, 'kospi index')
            # print(input_enc[0, 0, :], '\n', output_dec[0, 0, :], '\n', target_dec[0, 0, :])
            # print(features_list)
            test_dataset = dataset_process(input_enc, output_dec, target_dec, batch_size=1)  # , mode='test')
            columns_list = features_list
            predict_plot(model, test_dataset, features_list,  size=100)

            ii += 1
            if ii > 10000:
                break


def main_all_asset():
    configs = Config()

    data_out_path = os.path.join(os.getcwd(), configs.data_out_path)
    os.makedirs(data_out_path, exist_ok=True)

    # initiate and load model
    model = TSModel(configs)
    if os.path.exists(configs.f_name):
        model.load_model(configs.f_name)

    # get data for all assets and dates
    dg = DataGenerator(configs.data_path)

    t = 2500
    for t in range(2500, len(dg.date_)-250, 250):
        print(t)
        # make directories for graph results (both train and test one)
        os.makedirs(os.path.join(data_out_path, '/{}/'.format(t)), exist_ok=True)
        os.makedirs(os.path.join(data_out_path, '/{}/test/'.format(t)), exist_ok=True)
        # 10 years
        start_t = dg.date_[t - 2500]
        end_t = dg.date_[t]
        train_input, train_label, eval_input, eval_label, features_list = \
            dg.generate_dataset(start_t, end_t, m_days=60, k_days=20)

        # 훈련셋 인코딩 만드는 부분이다.
        train_input_enc = train_input[:]
        # 훈련셋 디코딩 입력 부분 만드는 부분이다.
        train_output_dec = train_label[:, :-1, :]
        # 훈련셋 디코딩 출력 부분 만드는 부분이다.
        train_target_dec = train_label[:, 1:, :]

        # 훈련셋 인코딩 만드는 부분이다.
        eval_input_enc = eval_input[:]
        # 훈련셋 디코딩 입력 부분 만드는 부분이다.
        eval_output_dec = eval_label[:, :-1, :]
        # 훈련셋 디코딩 출력 부분 만드는 부분이다.
        eval_target_dec = eval_label[:, 1:, :]

        dataset_train = dataset_process(train_input_enc, train_output_dec, train_target_dec, configs.batch_size)
        dataset_insample_test = dataset_process(train_input_enc, train_output_dec, train_target_dec, 1, mode='test')
        dataset_eval = dataset_process(eval_input_enc, eval_output_dec, eval_target_dec, 1, mode='test')

        for i, (features, labels) in enumerate(dataset_train.take(configs.train_steps)):
            print_loss = False
            if i % 50 == 0:
                # model.encoder.show(i, save=True)
                model.save_model(configs.f_name)

                predict_plot(model, dataset_insample_test, features_list, 250, save_dir='out/{}/train_{}.png'.format(t, i))
                predict_plot(model, dataset_eval, features_list, 250, save_dir='out/{}/eval_{}.png'.format(t, i))

            if i % 10 == 0:
                print_loss = True
                model.evaluate(dataset_eval, steps=20)
                print("[t: {} / i: {}] min_eval_loss:{} / count:{}".format(t, i, model.eval_loss, model.eval_count))
                if model.eval_count >= 10:
                    print("[t: {} / i: {}] train finished.".format(t, i))
                    model.weight_to_optim()
                    break

            model.train(features, labels, print_loss=print_loss)


        test_dataset_temp = dg.get_full_dataset(dg.date_[t-60], dg.date_[t+250])
        bbticker = list(test_dataset_temp.columns)
        test_dataset_arr = test_dataset_temp.values

        for j in range(len(bbticker)):
            test_input, test_label, _ = dg.data_to_factor(test_dataset_arr[:, j:j+1], token_length=5, m_days=60, k_days=20, seed=-1)
            dataset_test = dataset_process(test_input[:], test_label[:, :-1, :], test_label[:, 1:, :], 1, mode='test')

            predict_plot(model, dataset_test, features_list, size=len(test_input), save_dir='./out/{}/test/{}.png'.format(t, bbticker[j]))


def main_single_asset():
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



            predict_plot(model, dataset_insample_test, columns_list, len(train_input_enc), save_dir='out/{}/train_{}.png'.format(name, i))
            predict_plot(model, dataset_eval, columns_list, len(eval_input_enc), save_dir='out/{}/eval_{}.png'.format(name, i))






