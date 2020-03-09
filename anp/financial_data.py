import collections
import pandas as pd
from pandas.core.series import Series
from matplotlib import pyplot as plt
import numpy as np
import torch


ContextSet = collections.namedtuple(
    "ContextSet",
    ("query", "target_y", "num_total_points", "num_context_points",
     "hyperparams"))


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

    def get_timeseries(self, ts_nm='mkt_rf'):
        data = pd.read_csv('./data/data_for_metarl.csv', index_col=0)
        base_y = data[ts_nm]
        return base_y

    def generate_set(self, base_y: Series):

        base_logp = np.log(1 + base_y).cumsum() - np.log(1 + base_y.iloc[0])

        context_set = []

        context_x = torch.from_numpy(np.arange(1 + self.window_length)[::-1] / self.window_length * (-1)).float()
        target_x = torch.from_numpy(np.arange(1 + self.predict_length)[1:] / self.predict_length).float()
        for t in range(self.window_length, len(base_y), 20):
            context_t = torch.from_numpy(base_logp.iloc[(t-self.window_length):(t+1)].to_numpy(np.float32))
            mu = context_t.mean()
            sig = context_t.std()
            context_y = (context_t - mu) / sig
            if t + self.predict_length < len(base_y):
                target_y = (torch.from_numpy(base_logp.iloc[(t+1):(t+self.predict_length+1)].to_numpy(np.float32)) - mu) / sig
            else:
                target_y = None

            context_set.append(ContextSet(query=((context_x, context_y), target_x),
                                          target_y=target_y,
                                          num_total_points=len(context_x) + len(target_x),
                                          num_context_points=len(context_x),
                                          hyperparams={'context_mu': mu, 'context_sig': sig}))

        self.context_set = context_set

    def generate(self, base_i, seq_len=10, is_train=True):
        assert base_i > seq_len
        if is_train:
            ctx_i = np.random.randint(seq_len, base_i)
        else:
            ctx_i = base_i

        num_context = self.max_num_context

        ctx_x_list, ctx_y_list = [], []
        tgt_x_list, tgt_y_list = [], []
        hp_list = []
        ctx_selected = self.context_set[(ctx_i-seq_len+1):(ctx_i+1)]
        for t in range(seq_len):
            ((c_x, c_y), t_x), t_y, len_total, len_ctx, hyperparams = ctx_selected[t]

            idx = np.random.choice(np.arange(len_ctx), num_context, replace=False)
            ctx_x_list.append(c_x[idx])
            ctx_y_list.append(c_y[idx])
            tgt_x_list.append(t_x)
            tgt_y_list.append(t_y)
            hp_list.append(hyperparams)

        return ContextSet(query=((ctx_x_list, ctx_y_list), tgt_x_list),
                          target_y=tgt_y_list,
                          num_total_points=len_total,
                          num_context_points=len_ctx,
                          hyperparams=hp_list)










