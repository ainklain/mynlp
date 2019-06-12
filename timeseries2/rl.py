
from timeseries.data_process import dataset_process

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten


class MyActor(Model):
    def __init__(self, n_action, n_timesteps, n_features, n_hidden=3):
        super().__init__()
        self.flatten = Flatten()
        self.conv_v = Conv2D(8, (n_timesteps, 1), dtype=tf.float32)
        self.conv_h = Conv2D(8, (1, n_features), dtype=tf.float32)

        self.hidden_layers = dict()
        for i in range(n_hidden):
            self.hidden_layers[i] = Dense(64, activation='relu')

        self.out_layer = Dense(1, activation='sigmoid')

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

        return self.out_layer(x)



class PPO:
    def __init__(self, env):
        self.s_dim = env.observation_space.shape
        n_timesteps, n_features = np.squeeze(self.s_dim).shape


        self.actor = MyActor(n_timesteps, n_features, env.step_size)




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
                                                shape=(self.n_timesteps, self.n_features),
                                                dtype=np.float32)

    def reset(self, start_idx=None, length=200, step_size=5, n_tasks=1):
        self.render_call = 0

        self.i_step = 0
        self.step_size = step_size
        self.prev_position = 0

        self.a_history = np.zeros(length)
        self.r_history = np.zeros(length)
        self.nav_history = np.zeros(length)
        self.cost_history = np.zeros(length)
        self.cum_y_history = np.zeros(length)

        self.n_tasks = n_tasks
        if n_tasks == 1:
            datasets, features = self.get_datasets(start_idx, length, step_size, n_tasks)
        else:
            raise NotImplementedError

        self.dataset_to_env(datasets[0], features, length)
        obs = self.env_data[self.i_step]['obs'].numpy()
        return obs

    def step(self, action):
        log_y = self.env_data[self.i_step]['log_y'].numpy()

        cost = self.trading_costs * np.abs(action - self.prev_position)
        r_instant = (np.exp(log_y) - 1.) * action - cost

        if self.i_step == 0:
            prev_nav = 1.
            prev_cum_y = 1.
        else:
            prev_nav = self.nav_history[self.i_step - 1]
            prev_cum_y = self.cum_y_history[self.i_step - 1]

        self.cost_history[self.i_step] = cost
        self.a_history[self.i_step] = action
        self.nav_history[self.i_step] = prev_nav * (1. + r_instant)
        self.cum_y_history[self.i_step] = prev_cum_y * np.exp(log_y)

        r_delayed = 0.
        if self.i_step > 0 and self.i_step * self.step_size % 250 == 0:
            base_idx = self.i_step - 250 // self.step_size
            if self.nav_history[self.i_step] >= 1.07 * self.nav_history[base_idx]:
                # 7% outperform than prev year-end
                r_delayed = 1.
            elif self.nav_history[self.i_step] >= np.max([1., np.max(self.nav_history[base_idx:(self.i_step+1)]) * 0.95]):
                r_delayed = 0.5
            elif self.nav_history[self.i_step] < np.max(self.nav_history[base_idx:(self.i_step+1)]) * 0.9:
                r_delayed = -1.

        r_total = r_instant + r_delayed
        self.r_history[self.i_step] = r_total

        self.i_step += 1
        if self.i_step == len(self.env_data):
            done = True
            obs_ = None
        else:
            obs_ = self.env_data[self.i_step]['obs'].numpy()
            done = False
        info = None
        return obs_, r_total, info, done

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
                self.ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fancybox=True, shadow=True)
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
        self.ax1.plot(x_, nav, color='k')
        self.ax1.plot(x_, cum_y, color='b')

        actions = self.a_history
        pal = sns.color_palette("hls", 19)
        self.ax2.stackplot(x_, actions, colors=pal)

        self.ax3.plot(x_, self.cost_history[:last_step])

        self.ax4.plot(x_, self.r_history[:last_step])

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





    def dataset_to_env(self, dataset, features, length):
        self.env_data = list()
        idx_y = features.index('log_y')
        for j, (features, labels) in enumerate(dataset.take(length)):
            obs = self.model.predict(features)
            true_log_y = labels[0, 0, idx_y]
            self.env_data.append({'obs': obs, 'log_y': true_log_y})

    def get_datasets(self, start_idx, length, step_size, n_tasks=1):
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
            datasets.append(dataset_process(input_enc, output_dec, target_dec, batch_size=1))

        return datasets, features_list


