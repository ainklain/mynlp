
from copy import deepcopy
import numpy as np
from gym import spaces
from gym import Env
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

from timeseries.data_process import dataset_process

from . import register_env


@register_env('korea-stock')
class MyEnv(Env):
    def __init__(self, model, data_scheduler, configs,
                 length=200,
                 trading_costs=0.001,
                 n_tasks_dict=None):
        super().__init__()
        self.model = model
        self.data_scheduler = data_scheduler
        self.trading_costs = trading_costs

        if n_tasks_dict is not None:
            self.n_tasks_dict = n_tasks_dict
        else:
            self.n_tasks_dict = dict({'train': 10, 'eval': 2, 'test': 2})

        self.n_timesteps = configs.max_sequence_length_out
        self.n_features = configs.embedding_size
        self.action_space = spaces.Box(0, 1, shape=(1, ), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.n_features,),
                                            dtype=np.float32)

        self._sample_tasks(length)

    def get_all_task_idx(self):
        return range(len(self.env_data_list))

    def _sample_tasks(self, length=200):
        self.length = length
        self.get_datasets(length)

    def get_datasets(self, length):
        # n_tasks = n_train_tasks + n_eval_tasks + n_test_tasks
        ds = self.data_scheduler

        self.env_data_list = list()
        for mode in (['train', 'eval', 'test']):
            s_t = time()
            print("mode:{}".format(mode))
            input_enc, output_dec, target_dec, features_list = ds._dataset(mode)
            new_output = np.zeros_like(output_dec)
            new_output[:, 0, :] = output_dec[:, 0, :]
            dataset = dataset_process(input_enc, new_output, target_dec, batch_size=1)

            e_t = time()
            print('get_datasets time: {}'.format(e_t - s_t))

            s_t = time()
            env_data = list()
            idx_y = features_list.index('log_y')
            for j, (features, labels) in enumerate(dataset.take(length * self.n_tasks_dict[mode])):
                if (j > 0) and (j % self.n_tasks_dict[mode] == 0):
                    self.env_data_list.append(env_data)
                    env_data = list()

                prediction = self.model.predict(features)
                true_log_y = labels[0, 0, idx_y]
                env_data.append({'obs': np.squeeze(prediction[:, :1, :]), 'log_y': true_log_y})

            e_t = time()
            print('env_data time: {}'.format(e_t - s_t))


    def reset_task(self, task_i):
        self.env_data = self.env_data_list[task_i]

    def reset(self):
        self.render_call = 0

        self.i_step = 0
        self.prev_position = 0

        self.a_history = np.zeros(self.length)
        self.r_history = np.zeros([self.length, 4])
        self.nav_history = np.ones(self.length)
        self.cost_history = np.zeros(self.length)
        self.cum_y_history = np.ones(self.length)

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

        r_relative = r_instant - (np.exp(log_y)-1.)

        r_delayed = 0       # 조건 초기화

        if self.i_step == len(self.env_data) - 1:
            done = True
            obs_ = None
            if self.nav_history[self.i_step] > 1.07 ** (self.i_step / (250 // self.step_size)):
                r_delayed = 0.5
            else:
                r_delayed = -0.000
        else:
            obs_ = self.env_data[self.i_step + 1]['obs'].numpy()
            if self.nav_history[self.i_step] < np.max(self.nav_history[:(self.i_step+1)]) * 0.8:
                r_delayed = -0.05
                done = False
            else:
                done = False
        info = dict()

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

        self.ax4.plot(x_, np.cumsum(self.r_history[:last_step, 0]), label='total', color='k')
        self.ax4.plot(x_, self.r_history[:last_step, 1], label='instant', color='b')
        self.ax4.plot(x_, self.r_history[:last_step, 2], label='delayed', color='g')
        self.ax4.plot(x_, self.r_history[:last_step, 3], label='relative', color='r')
        self.ax4.legend()

        if statistics:
            print('not implemented')
