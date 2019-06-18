
from timeseries.config import Config
from timeseries.model import TSModel
from timeseries.data_process import dataset_process, load_data, DataGenerator, DataScheduler
from timeseries.rl import MyEnv, MyActor, PPO

import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import os
import tensorflow as tf
import time


EP_MAX = 1000
BATCH = 1024
GAMMA = 0.99
LAMBDA = 0.95

def discount(x, gamma, terminal_array=None):
    if terminal_array is None:
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    else:
        y, adv = 0, []
        terminals_reversed = terminal_array[1:][::-1]
        for step, dt in enumerate(reversed(x)):
            y = dt + gamma * y * (1 - terminals_reversed[step])
            adv.append(y)
        return np.array(adv)[::-1]


class RunningStats:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.std = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.std = np.maximum(np.sqrt(self.var), 1e-6)
        self.count = batch_count + self.count



def main():
    configs = Config()

    # get data for all assets and dates
    ds = DataScheduler(configs, is_infocode=True)
    ds.test_end_idx = ds.base_idx + 1000

    ii = 0
    # while not ds.done:
    for _ in range(1):
        model = TSModel(configs)
        configs.f_name = 'ts_model_test1.0'
        if os.path.exists(configs.f_name):
            model.load_model(configs.f_name)

        # ds.set_idx(8000)
        ds.train(model,
                 train_steps=configs.train_steps,
                 eval_steps=10,
                 save_steps=200,
                 early_stopping_count=20,
                 model_name=configs.f_name)

        env = MyEnv(model, data_scheduler=ds, configs=configs, trading_costs=0.001)
        test_env = MyEnv(model, data_scheduler=ds, configs=configs, trading_costs=0.001)
        # actor = MyActor(1, configs.max_sequence_length_out, configs.embedding_size)

        t, done = 0, False
        buffer_s, buffer_a, buffer_r, buffer_v, buffer_done = [], [], [], [], []

        rolling_r = RunningStats()

        ppo = PPO(env)
        f_name = './{}.pkl'.format('actor_v1.0_5')
        if os.path.exists(f_name):
            ppo.load_model(f_name)

        EP_MAX = 100000
        for episode in range(EP_MAX + 1):
            print("episode:{}".format(episode))
            s_t = time.time()
            if episode == 0:
                new_data = True
            else:
                new_data = False

            s = env.reset(length=201, n_tasks=5, new_data=new_data)
            while True:
                a, v = ppo.evaluate_state(s, stochastic=True)

                if t == BATCH:
                    rewards = np.array(buffer_r)
                    rolling_r.update(rewards)
                    rewards = np.clip(rewards / rolling_r.std, -10, 10)
                    v_final = [v * (1 - done)]
                    values = np.array(buffer_v + v_final).squeeze()
                    dones = np.array(buffer_done + [done])

                    delta = rewards + GAMMA * values[1:] * (1 - dones[1:]) - values[:-1]
                    advantage = discount(delta, GAMMA * LAMBDA, dones)
                    returns = advantage + np.array(buffer_v).squeeze()
                    advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)

                    bs, ba, br, badv = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        returns), np.vstack(advantage)
                    s_batch = bs
                    a_batch = ba
                    r_batch = br
                    adv_batch = badv
                    graph_summary = ppo.update(bs, ba, br, badv)
                    buffer_s, buffer_a, buffer_r, buffer_v, buffer_done = [], [], [], [], []
                    t = 0
                    # break

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_v.append(v)
                buffer_done.append(done)

                a = tf.clip_by_value(a, env.action_space.low, env.action_space.high)
                s, r, done, _ = env.step(np.squeeze(a))
                buffer_r.append(r)
                t += 1

                if done:
                    print("global step: {}".format(ppo.global_step))
                    if ppo.global_step % 100 == 0:
                        env.render(save_filename='./out/env_{}_{}.png'.format(ppo.global_step, episode))

                    ppo.save_model(f_name)
                    break
            e_t = time.time()
            print('episode time: {}'.format(e_t - s_t))


            if episode % 200 == 0:
                for ep_i in range(5):
                    s = env.reset(length=201, n_tasks=5, new_data=False, task_i=ep_i)
                    while True:
                        a, v = ppo.evaluate_state(s, stochastic=False)
                        a = tf.clip_by_value(a, env.action_space.low, env.action_space.high)
                        s, r, done, _ = env.step(np.squeeze(a))

                        if done:
                            print("global step: {}".format(ppo.global_step))
                            env.render(save_filename='./out/rltest/env_{}_{}.png'.format(ep_i, ppo.global_step))
                            break

            if episode % 500 == 0:
                if episode == 0:
                    test_new_data = True
                else:
                    test_new_data = False

                for ep_it in range(5):
                    s_t = test_env.reset(length=201, n_tasks=5, new_data=test_new_data, task_i=ep_it)
                    if test_new_data is True:
                        test_new_data = False
                    while True:
                        a_t, v_t = ppo.evaluate_state(s_t, stochastic=False)
                        a_t = tf.clip_by_value(a_t, test_env.action_space.low, test_env.action_space.high)
                        s_t, r_t, done_t, _ = test_env.step(np.squeeze(a_t))

                        if done_t:
                            print("global step: {}".format(ppo.global_step))
                            test_env.render(save_filename='./out/rltest/new/env_{}_{}.png'.format(ep_it, ppo.global_step))
                            break



            test_i = 100
            s = env.reset_test(start_idx=[8000])
            while True:
                a, v = ppo.evaluate_state(s, stochastic=False)
                a = tf.clip_by_value(a, env.action_space.low, env.action_space.high)
                s, r, done, _ = env.step(np.squeeze(a))

                if done:
                    print("global step: {}".format(ppo.global_step))
                    env.render(save_filename='./out/rltest/new_env_{}.png'.format(test_i))
                    break



        env.close()


        test_dataset_list, features_list = ds.test(model)

        ds.next()
