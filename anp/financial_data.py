import collections
import pandas as pd
from pandas.core.series import Series
from matplotlib import pyplot as plt
import numpy as np
import torch
from ts_torch import torch_util_mini as tu

# # #### profiler start ####
import builtins

try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile
# # #### profiler end ####


# ContextSet = collections.namedtuple(
#     "ContextSet",
#     ("query", "target_y", "num_total_points", "num_context_points",
#      "hyperparams"))


class ContextSet:
    def __init__(self, query, target_y, num_total_points, num_context_points, hyperparams=None):
        self.query = query
        self.target_y = target_y
        self.num_total_points = num_total_points
        self.num_context_points = num_context_points
        self.hyperparams = hyperparams

    def to(self, device):
        ((context_x, context_y), target_x) = self.query
        self.query = ((tu.to_device(device, context_x), tu.to_device(device, context_y)), tu.to_device(device, target_x))
        self.target_y = tu.to_device(device, self.target_y)
        self.hyperparams = tu.to_device(device, self.hyperparams)

    @property
    def data(self):
        return (self.query, self.target_y, self.num_total_points, self.num_context_points, self.hyperparams)


class TimeSeries:
    def __init__(self
                 , batch_size
                 , max_num_context=250
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
        self.x_dim = 1
        self.y_dim = 1
        self.context_set = None

    def get_timeseries(self, ts_nm='mkt_rf'):
        data = pd.read_csv('./data/data_for_metarl.csv', index_col=0)
        base_y = data[ts_nm]
        return base_y

    @profile
    def prepare_entire_dataset(self, base_y: Series):

        base_logp = np.log(1 + base_y).cumsum() - np.log(1 + base_y.iloc[0])

        context_set = []
        target_x = (torch.arange(1 + self.window_length + self.predict_length).float() / self.window_length - 1).reshape([-1, self.x_dim])
        context_x = target_x[:(1+self.window_length)]
        for t in range(self.window_length, len(base_y), 20):
            context_t = torch.from_numpy(base_logp.iloc[(t - self.window_length):(t + 1)].to_numpy(np.float32)).reshape([-1, self.y_dim])
            mu = context_t.mean()
            sig = context_t.std()
            context_y = context_t
            # context_y = (context_t - mu) / sig
            if t + self.predict_length < len(base_y):
                target_y = torch.from_numpy(
                    base_logp.iloc[(t - self.window_length):(t + self.predict_length + 1)].to_numpy(np.float32))
                # target_y = (torch.from_numpy(
                #     base_logp.iloc[(t - self.window_length):(t + self.predict_length + 1)].to_numpy(np.float32)) - mu) / sig
                target_y = target_y.reshape([-1, self.y_dim])
            else:
                target_y = None

            context_set.append(ContextSet(query=((context_x, context_y), target_x),
                                          target_y=target_y,
                                          num_total_points=len(target_x),
                                          num_context_points=len(context_x),
                                          hyperparams={'context_mu': mu, 'context_sig': sig}))

        self.context_set = context_set

    @property
    def max_len(self):
        return len(self.context_set) if self.context_set is not None else 0

    # @profile
    def generate(self, base_i, seq_len=10, is_train=True):
        # base_i = 50; seq_len = 10; is_train=True; t=0
        assert base_i >= seq_len + self.predict_length // 20  # lookahead 방지
        if is_train:
            ctx_i = np.random.randint(seq_len, base_i - self.predict_length // 20)
            batch_size = self.batch_size
            num_context = 100
        else:
            ctx_i = base_i
            batch_size = 1
            num_context = self.max_num_context

        assert ctx_i < len(self.context_set)

        ctx_x_list, ctx_y_list = [], []
        tgt_x_list, tgt_y_list = [], []
        hp_list = []
        ctx_selected = self.context_set[(ctx_i - seq_len + 1):(ctx_i + 1)]

        c_x_batch = torch.zeros(batch_size, num_context, self.x_dim)
        c_y_batch = torch.zeros(batch_size, num_context, self.y_dim)
        t_x_batch = torch.zeros(batch_size, num_context + self.predict_length, self.x_dim)
        t_y_batch = torch.zeros(batch_size, num_context + self.predict_length, self.y_dim)

        for t in range(seq_len):
            ((c_x, c_y), t_x), t_y, len_total, len_ctx, hyperparams = ctx_selected[t].data
            if t == 0:
                mu, sig = hyperparams['context_mu'], hyperparams['context_sig']

            # 0번째 moment로 normalize (process dynamics 기억)
            c_y = (c_y - mu) / sig
            if t_y is not None:
                t_y = (t_y - mu) / sig
            else:
                t_y_batch = None

            for i in range(batch_size):
                c_idx = np.sort(np.random.choice(np.arange(len_ctx), num_context, replace=False))
                t_idx = np.concatenate([c_idx, np.arange(len_ctx, len_total)])

                np.random.shuffle(c_idx)
                c_x_batch[i] = c_x[c_idx]
                c_y_batch[i] = c_y[c_idx]
                if is_train:
                    np.random.shuffle(t_idx)

                t_x_batch[i] = t_x[t_idx]
                if t_y_batch is not None:
                    t_y_batch[i] = t_y[t_idx]

            ctx_x_list.append(c_x_batch)
            ctx_y_list.append(c_y_batch)
            tgt_x_list.append(t_x_batch)
            tgt_y_list.append(t_y_batch)
            hp_list.append(hyperparams)

        return ContextSet(query=((ctx_x_list, ctx_y_list), tgt_x_list),
                          target_y=tgt_y_list,
                          num_total_points=len_total,
                          num_context_points=len_ctx,
                          hyperparams=hp_list)










