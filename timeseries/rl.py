
import numpy as np
import gym

class MyEnv(gym.Env):
    def __init__(self, model, data_scheduler):
        super().__init__()
        self.model = model
        self.data_scheduler = data_scheduler

    def reset(self, start_idx=None, length=None, step_size=None):
        self.get_dataset(start_idx, length, step_size)

    def step(self):
        pass

    def render(self):
        pass

    def get_dataset(self, start_idx, length, step_size):
        ds = self.data_scheduler
        if start_idx is None:
            s_idx = ds.train_begin_idx + ds.m_days
            e_idx = ds.eval_begin_idx - length
            start_idx = np.random.

        if length is None:
            pass

        if step_size is None:
            step_size = self.data_scheduler.sampling_days
        self.data_scheduler._dataset_custom()



