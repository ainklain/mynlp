import pandas as pd
from pandas.core.series import Series
from matplotlib import pyplot as plt
import numpy as np


class TimeSeries:
    def __init__(self
                 , batch_size
                 , max_num_context
                 , window_length=250
                 , predict_length=20):
        """
        :param batch_size:
        :param max_num_context:
        :param window_length:
        """
        self.batch_size = batch_size
        self.max_num_context = max_num_context
        self.window_length = window_length
        self.predict_length = predict_length

    def get_timeseries(self):
        data = pd.read_csv('./data/data_for_metarl.csv', index_col=0)
        base_y = data['usdkrw']

    def generate_set(self, base_y: Series):

        base_logp = np.log(1 + base_y).cumsum() - np.log(1 + base_y.iloc[0])

        context_set = []

        context_x = np.arange(1 + self.window_length)[::-1] / self.window_length * (-1)
        target_x = np.arange(1 + self.predict_length)[1:] / self.predict_length
        for t in range(self.window_length, len(base_y), 20):
            context_dict = dict()
            context_t = base_logp.iloc[(t-self.window_length):(t+1)]
            mu = context_t.mean()
            sig = context_t.std()
            context_y = (context_t - mu) / sig
            if t + self.predict_length < len(base_y):
                target_y = (base_logp.iloc[(t+1):(t+self.predict_length+1)] - mu) / sig
            else:
                target_y = None

            context_dict['context_x'] = context_x
            context_dict['context_y'] = context_y
            context_dict['target_x'] = target_x
            context_dict['target_y'] = target_y
            context_dict['context_mu'] = mu
            context_dict['context_sig'] = sig

            context_set.append(context_dict)

        self.context_set = context_set

    def generate(self, base_i, seq_len=10, is_train=True):
        assert base_i > seq_len
        if is_train:
            context_i = np.random.randint(seq_len, base_i)
        else:
            context_i = base_i

        num_context = self.max_num_context

        context_selected = self.context_set[(context_i-seq_len+1):(context_i+1)]
        for t in range(seq_len):
            context_dict = context_selected[t]
            context_len = len(context_dict['context_x'])








