
from timeseries.data_process import dataset_process
from timeseries.distribution import DiagonalGaussian


import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import gym
import tensorflow as tf
from time import time
import pandas as pd
import pickle
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.regularizers import l2


LR = 1e-4
L2_REG = 0.001
EPOCHS = 10
MINIBATCH = 32
ENTROPY_BETA = 0.01
VF_COEFF = 1.0


class MyActor(Model):
    def __init__(self, n_action, n_timesteps, n_features, n_hidden=3):
        super().__init__()
        self.flatten = Flatten()
        self.conv_v = Conv2D(8, (n_timesteps, 1), dtype=tf.float32)
        self.conv_h = Conv2D(8, (1, n_features), dtype=tf.float32)

        self.hidden_layers = dict()
        for i in range(n_hidden):
            self.hidden_layers[i] = Dense(64, activation='relu', kernel_regularizer=l2(L2_REG))

        self.mu_layer = Dense(n_action, activation='sigmoid', kernel_regularizer=l2(L2_REG))
        self.critic_layer = Dense(1, activation='linear', kernel_regularizer=l2(L2_REG))

    def call(self, x):
        x = tf.cast(x, tf.float32)
        if len(x.shape) == 3:
            x = tf.expand_dims(x, -1)
        elif len(x.shape) == 2:
            x = tf.expand_dims(x, 0)
            x = tf.expand_dims(x, -1)
        else:
            raise NotImplementedError

        x_h = self.flatten(self.conv_h(x))
        x_v = self.flatten(self.conv_v(x))

        x = tf.concat([x_v, x_h], axis=-1)

        for key in self.hidden_layers.keys():
            x = self.hidden_layers[key](x)

        return self.mu_layer(x), self.critic_layer(x)



class PPO:
    def __init__(self, env):
        self.s_dim = env.observation_space.shape
        self.a_dim = env.action_space.shape[0]
        self.a_bound = (env.action_space.high[0] - env.action_space.low[0]) / 2
        n_timesteps, n_features = self.s_dim[-2:]

        self.dist = DiagonalGaussian(self.a_dim)

        self.actor = MyActor(self.a_dim, n_timesteps, n_features)
        self.log_sigma = tf.Variable(tf.zeros(self.a_dim))

        self.old_actor = MyActor(self.a_dim, n_timesteps, n_features)
        self.old_log_sigma = tf.Variable(tf.zeros(self.a_dim))

        self.optimizer = tf.optimizers.Adam(LR)

        self.global_step = 0

        self._initialize()

    def _initialize(self):
        s = np.zeros(self.s_dim)
        _ = self.actor(s)
        _ = self.old_actor(s)

    def assign_old_network(self):
        self.old_actor.set_weights(self.actor.get_weights())
        self.old_log_sigma.assign(self.log_sigma.numpy())

    def evaluate_state(self, state, stochastic=True):
        mu, value_ = self.actor(state)
        if stochastic:
            action = self.dist.sample({'mean': mu, 'log_std': self.log_sigma})
        else:
            action = mu
        return action, value_

    def polynomial_epsilon_decay(self, learning_rate, global_step, decay_steps, end_learning_rate, power):
        global_step_ = min(global_step, decay_steps)
        decayed_learning_rate = (learning_rate - end_learning_rate) * (1 - global_step_ / decay_steps) ** (power) \
                                + end_learning_rate

        return decayed_learning_rate

    def update(self, s_batch, a_batch, r_batch, adv_batch):
        start = time()
        e_time = []

        self.assign_old_network()

        for epoch in range(EPOCHS):
            idx = np.arange(len(s_batch))
            np.random.shuffle(idx)

            loss_per_epoch = 0
            for i in range(len(s_batch) // MINIBATCH):
                epsilon_decay = self.polynomial_epsilon_decay(0.1, self.global_step, 1e5, 0.01, power=1.0)
                s_mini = s_batch[idx[i * MINIBATCH: (i + 1) * MINIBATCH]]
                a_mini = a_batch[idx[i * MINIBATCH: (i + 1) * MINIBATCH]]
                r_mini = r_batch[idx[i * MINIBATCH: (i + 1) * MINIBATCH]]
                adv_mini = adv_batch[idx[i * MINIBATCH: (i + 1) * MINIBATCH]]
                with tf.GradientTape() as tape:
                    mu_old, v_old = self.old_actor(s_mini)
                    mu, v = self.actor(s_mini)
                    ratio = self.dist.likelihood_ratio_sym(
                        a_mini,
                        {'mean': mu_old * self.a_bound, 'log_std': self.old_log_sigma},
                        {'mean': mu * self.a_bound, 'log_std': self.log_sigma})
                    # ratio = tf.maximum(logli, 1e-6) / tf.maximum(old_logli, 1e-6)
                    ratio = tf.clip_by_value(ratio, 0, 10)
                    surr1 = adv_mini.squeeze() * ratio
                    surr2 = adv_mini.squeeze() * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
                    loss_pi = - tf.reduce_mean(tf.minimum(surr1, surr2))

                    clipped_value_estimate = v_old + tf.clip_by_value(v - v_old, -epsilon_decay, epsilon_decay)
                    loss_v1 = tf.math.squared_difference(clipped_value_estimate, r_mini)
                    loss_v2 = tf.math.squared_difference(v, r_mini)
                    loss_v = tf.reduce_mean(tf.maximum(loss_v1, loss_v2)) * 0.5

                    entropy = self.dist.entropy({'mean': mu, 'log_std': self.log_sigma})
                    pol_entpen = -ENTROPY_BETA * tf.reduce_mean(entropy)

                    loss = loss_pi + loss_v * VF_COEFF + pol_entpen

                grad = tape.gradient(loss, self.actor.trainable_variables + [self.log_sigma])
                self.optimizer.apply_gradients(zip(grad, self.actor.trainable_variables + [self.log_sigma]))

                loss_per_epoch = loss_per_epoch + loss
                # print("epoch: {} - {}/{} ({:.3f}%),  loss: {:.8f}".format(epoch, i, len(s_batch) // MINIBATCH,
                #                                                           i / (len(s_batch) // MINIBATCH) * 100., loss))
                # if i % 10 == 0:
                #     print(grad[-1])

            print("epoch: {} - loss: {}".format(epoch, loss_per_epoch / (len(s_batch) // MINIBATCH) * 100))

        self.global_step += 1

    def save_model(self, f_name):
        w_dict = {}
        w_dict['actor'] = self.actor.get_weights()
        w_dict['log_sigma'] = self.log_sigma.numpy()
        w_dict['global_step'] = self.global_step

        # f_name = os.path.join(model_path, model_name)
        with open(f_name, 'wb') as f:
            pickle.dump(w_dict, f)

        print("model saved. (path: {})".format(f_name))

    def load_model(self, f_name):
        # f_name = os.path.join(model_path, model_name)
        with open(f_name, 'rb') as f:
            w_dict = pickle.load(f)
        self.actor.set_weights(w_dict['actor'])
        self.log_sigma.assign(w_dict['log_sigma'])
        self.global_step = w_dict['global_step']

        print("model loaded. (path: {})".format(f_name))



class MyEnv(gym.Env):
    def __init__(self, model, data_scheduler, configs, trading_costs=0.001):
        super().__init__()
        self.model = model
        self.data_scheduler = data_scheduler
        self.trading_costs = trading_costs

        self.n_timesteps = configs.max_sequence_length_out
        self.n_features = configs.embedding_size
        self.action_space = gym.spaces.Box(0, 1, shape=(1, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                # shape=(1, self.n_timesteps, self.n_features),
                                                shape=(1, 1, self.n_features),
                                                dtype=np.float32)

    def reset(self, start_idx=None, length=200, step_size=5, n_tasks=1, new_data=True):
        self.render_call = 0

        self.i_step = 0
        self.step_size = step_size
        self.prev_position = 0

        self.a_history = np.zeros(length)
        self.r_history = np.zeros([length, 4])
        self.nav_history = np.ones(length)
        self.cost_history = np.zeros(length)
        self.cum_y_history = np.ones(length)

        self.n_tasks = n_tasks
        task_i = np.random.random_integers(0, n_tasks - 1)
        if new_data is True:
            self.datasets, self.features = self.get_datasets(start_idx, length, step_size, n_tasks)
            self.datasets_to_env(self.datasets, self.features, length)

        self.env_data = self.env_data_list[task_i]
        obs = self.env_data[self.i_step]['obs'].numpy()
        return obs

    def step(self, action):
        log_y = self.env_data[self.i_step]['log_y'].numpy()

        cost = self.trading_costs * np.abs(action - self.prev_position)
        r_instant = (np.exp(log_y) - 1.) * action - cost

        self.prev_position = deepcopy(action)
        prev_step = np.max([0, self.i_step - 1])
        self.a_history[self.i_step] = deepcopy(action)
        self.cost_history[self.i_step] = self.cost_history[prev_step] + cost
        self.nav_history[self.i_step] = self.nav_history[prev_step] * (1. + r_instant)
        self.cum_y_history[self.i_step] = self.cum_y_history[prev_step] * np.exp(log_y)

        # r_relative = 0
        # if r_instant > np.exp(log_y) - 1:
        r_relative = r_instant - (np.exp(log_y)-1.)

        r_delayed = 0.
        if self.i_step > 0 and self.i_step * self.step_size % 250 == 0:
            base_idx = self.i_step - 250 // self.step_size
            if self.nav_history[self.i_step] >= 1.07 * self.nav_history[base_idx]:
                # 7% outperform than prev year-end
                r_delayed = 0.2 + (self.nav_history[self.i_step] - 1.07 * self.nav_history[base_idx])
            elif self.nav_history[self.i_step] < np.max(self.nav_history[base_idx:(self.i_step+1)]) * 0.9:
                r_delayed = -0.5

            # cum_nav = self.nav_history[self.i_step] / self.nav_history[base_idx] - 1
            # cum_y = self.cum_y_history[self.i_step] / self.cum_y_history[base_idx] - 1
            # if cum_nav >= cum_y + 0.05:
            #     r_relative += 0.1
            # elif cum_nav < cum_y - 0.1:
            #     r_relative -= 0.1

        r_delayed = 0       # 조건 초기화

        if self.i_step == len(self.env_data) - 1:
            done = True
            obs_ = None
            if self.nav_history[self.i_step] > (1.07) ** (self.i_step / (250 // self.step_size)):
                r_delayed = 0.5
            else:
                r_delayed = -0.000
        else:
            obs_ = self.env_data[self.i_step + 1]['obs'].numpy()
            if self.nav_history[self.i_step] < np.max(self.nav_history[:(self.i_step+1)]) * 0.8:
                r_delayed = -0.5
                done =True
            else:
                done = False
        info = None

        r_total = r_instant + r_delayed + r_relative
        self.r_history[self.i_step, :] = np.array([r_total, r_instant, r_delayed, r_relative])

        self.i_step += 1

        return obs_, r_total, done, info

    def render(self, mode='human', statistics=False, save_filename=None):
        return self._render(mode=mode, statistics=statistics, save_filename=save_filename)

    def _render(self, mode='human', statistics=False, save_filename=None):
        if mode == 'human':
            if self.render_call == -1:
                print("n_envs > 1. no rendering")
                return None

            if self.render_call == 0:
                self.fig = plt.figure()
                self.ax1, self.ax2, self.ax3, self.ax4 = self.fig.subplots(4, 1)
                self.render_call += 1

            self._get_image(statistics)

            if self.render_call == 0:
                self.ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fancybox=True, shadow=True)
                self.render_call += 1
                self.ims = []

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            if save_filename is not None:
                self.fig.savefig(save_filename)
                print("fig saved. ({})".format(save_filename))
                plt.close(self.fig)

    def _get_image(self, statistics=False):
        last_step = self.i_step
        x_ = np.arange(last_step)
        nav = self.nav_history[:last_step]
        cum_y = self.cum_y_history[:last_step]
        self.ax1.plot(x_, nav, label='nav', color='k')
        self.ax1.plot(x_, cum_y, label='original', color='b')
        self.ax1.legend()
        self.ax1.set_yscale('log', basey=2)

        actions = self.a_history[:last_step]
        pal = sns.color_palette("hls", 19)
        self.ax2.stackplot(x_, actions, colors=pal)
        self.ax2.set_ylim([0., 1.])

        self.ax3.plot(x_, self.cost_history[:last_step])

        # df = pd.DataFrame(self.r_history[:last_step, 1:], columns=['instant', 'delayed', 'relative'])
        # self.ax4_1 = self.ax4.twinx()
        # df.plot(kind='bar', stacked=False, ax=self.ax4, grid=False)
        # self.ax4_1.plot(self.ax4.get_xticks(), self.r_history[:last_step, 0], linestyle='-', color='k')
        #
        # lines, labels = self.ax4.get_legend_handles_labels()
        # lines2, labels2 = self.ax4_1.get_legend_handles_labels()
        # self.ax4.legend(lines + lines2, labels + labels2, loc='best')
        # self.ax4.yaxis.set_ticks_position=('right')
        # self.ax4_1.yaxis.set_ticks_position = ('left')

        self.ax4.plot(x_, np.cumsum(self.r_history[:last_step, 0]), label='total', color='k')
        self.ax4.plot(x_, self.r_history[:last_step, 1], label='instant', color='b')
        self.ax4.plot(x_, self.r_history[:last_step, 2], label='delayed', color='g')
        self.ax4.plot(x_, self.r_history[:last_step, 3], label='relative', color='r')
        self.ax4.legend()

        if statistics:
            print('not implemented')
            # mean_return = np.mean(render_data['nav_returns']) * 250
            # std_return = np.std(render_data['nav_returns'], ddof=1) * np.sqrt(250)
            # cum_return = nav_squeeze[-1] - 1
            # total_cost = np.sum(render_data['costs'])
            #
            # ew_mean_return = np.mean(render_data['ew_returns']) * 250
            # ew_std_return = np.std(render_data['ew_returns'], ddof=1) * np.sqrt(250)
            # ew_cum_return = ew_nav_squeeze[-1] - 1
            #
            # max_nav = 1.
            # max_nav_i = 0
            # mdd_i = 0.
            # mdd = list()
            # for i in range(last_step):
            #     if nav_squeeze[i] >= max_nav:
            #         max_nav = nav_squeeze[i]
            #         max_nav_i = i
            #     else:
            #         mdd_i = np.min(nav_squeeze[max_nav_i:(i+1)]) / max_nav - 1.
            #     mdd.append(mdd_i)
            # max_mdd = np.min(mdd)
            #
            # print('model == ret:{} / std:{} / cum_return:{} / max_mdd:{} / cost:{}'.format(
            #     mean_return, std_return, cum_return, max_mdd, total_cost))
            # print('ew_model == ret:{} / std:{} / cum_return:{}'.format(
            #     ew_mean_return, ew_std_return, ew_cum_return))


    def reset_test(self, start_idx=8000, length=2000, step_size=5, n_tasks=1, new_data=True, bbtickers=['kospi index']):
        self.render_call = 0

        self.i_step = 0
        self.step_size = step_size
        self.prev_position = 0

        self.a_history = np.zeros(length)
        self.r_history = np.zeros([length, 4])
        self.nav_history = np.ones(length)
        self.cost_history = np.zeros(length)
        self.cum_y_history = np.ones(length)

        self.n_tasks = n_tasks
        task_i = np.random.random_integers(0, n_tasks - 1)
        if new_data is True:
            self.datasets, self.features = self.get_testdatasets(start_idx, length, step_size, n_tasks, bbtickers=bbtickers)
            self.datasets_to_env(self.datasets, self.features, length)

        self.env_data = self.env_data_list[task_i]
        obs = self.env_data[self.i_step]['obs'].numpy()
        return obs

    def datasets_to_env(self, datasets, features, length):
        s_t = time()
        self.env_data_list = list()
        idx_y = features.index('log_y')
        for dataset in datasets:
            env_data = list()
            for j, (features, labels) in enumerate(dataset.take(length)):
                obs = self.model.predict(features)
                true_log_y = labels[0, 0, idx_y]
                env_data.append({'obs': obs[:, :1, :], 'log_y': true_log_y})
            self.env_data_list.append(env_data)

        e_t = time()
        print('datasets_to_env time: {}'.format(e_t - s_t))


    def dataset_to_env(self, dataset, features, length=-1):
        s_t = time()
        self.env_data = list()
        idx_y = features.index('log_y')
        for j, (features, labels) in enumerate(dataset.take(length)):
            obs = self.model.predict(features)
            true_log_y = labels[0, 0, idx_y]
            self.env_data.append({'obs': obs, 'log_y': true_log_y})

        e_t = time()
        print('dataset_to_env time: {}'.format(e_t - s_t))

    def get_datasets(self, start_idx, length, step_size, n_tasks=1, no_future=True):
        s_t = time()
        ds = self.data_scheduler

        if start_idx is None:
            s_idx = ds.train_begin_idx + ds.m_days
            e_idx = ds.eval_begin_idx - length
            start_idx = np.random.random_integers(s_idx, e_idx, n_tasks)

        if step_size is None:
            step_size = self.data_scheduler.sampling_days

        datasets = list()
        for idx in start_idx:
            input_enc, output_dec, target_dec, features_list = self.data_scheduler._dataset_custom(start_idx=idx, end_idx=idx + length, step_size=step_size)

            if no_future:
                new_output = np.zeros_like(output_dec)
                new_output[:, 0, :] = output_dec[:, 0, :]
                for t in range(self.n_timesteps):
                    if t > 0:
                        new_output[:, t, :] = obs[:, t - 1, :]
                    features_pred = {'input': input_enc, 'output': new_output}
                    obs = self.model.predict(features_pred)
            else:
                new_output = output_dec

            datasets.append(dataset_process(input_enc, new_output, target_dec, batch_size=1))

        e_t = time()
        print('get_datasets time: {}'.format(e_t - s_t))
        return datasets, features_list

    def get_testdatasets(self, start_idx, length, step_size, n_tasks=1, bbtickers=None, no_future=True):
        s_t = time()
        ds = self.data_scheduler

        if start_idx is None:
            s_idx = ds.train_begin_idx + ds.m_days
            e_idx = ds.eval_begin_idx - length
            start_idx = np.random.random_integers(s_idx, e_idx, n_tasks)

        if step_size is None:
            step_size = self.data_scheduler.sampling_days

        datasets = list()
        for idx in start_idx:
            input_enc, output_dec, target_dec, features_list = self.data_scheduler._dataset_custom(start_idx=idx, end_idx=idx + length, step_size=step_size, bbtickers=bbtickers)

            if no_future:
                new_output = np.zeros_like(output_dec)
                new_output[:, 0, :] = output_dec[:, 0, :]
                for t in range(self.n_timesteps):
                    if t > 0:
                        new_output[:, t, :] = obs[:, t - 1, :]
                    features_pred = {'input': input_enc, 'output': new_output}
                    obs = self.model.predict(features_pred)
            else:
                new_output = output_dec

            datasets.append(dataset_process(input_enc, new_output, target_dec, batch_size=1, mode='test'))

        e_t = time()
        print('get_testdatasets time: {}'.format(e_t - s_t))
        return datasets, features_list



