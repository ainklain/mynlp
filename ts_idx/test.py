import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import pandas as pd

from tensorflow.keras.applications.resnet50 import ResNet50

from pyts.image import GramianAngularField

# from pyts.datasets import load_gunpoint
# X, _, _, _ = load_gunpoint(return_X_y=True)

data_path = './data/data_for_metarl.csv'
data_df = pd.read_csv(data_path)
data_df.set_index('eval_d', inplace=True)
date_ = list(data_df.index)

X = data_df.values[:250].transpose()
# Transform the time series into Gramian Angular Fields
gasf = GramianAngularField(image_size=64, method='summation')
X_gasf = np.array(gasf.fit_transform(X), dtype=np.float32)
gadf = GramianAngularField(image_size=24, method='difference')
X_gadf = gadf.fit_transform(X)

base_model = ResNet50(input_shape=(64, 64, 3), include_top=False)
X_test = np.stack([X_gasf[0], X_gasf[1], X_gasf[2]], axis=-1)
base_model(X_test)
base_model.trainable = False



# Show the images for the first time series
fig = plt.figure(figsize=(12, 7))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(1, 2),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.3,
                 )

images = [X_gasf[0], X_gadf[0]]
titles = ['Gramian Angular Summation Field',
          'Gramian Angular Difference Field']
for image, title, ax in zip(images, titles, grid):
    im = ax.imshow(image, cmap='rainbow', origin='lower')
    ax.set_title(title, fontdict={'fontsize': 16})
ax.cax.colorbar(im)
ax.cax.toggle_label(True)

plt.suptitle('Gramian Angular Fields', y=0.92, fontsize=20)
plt.show()

class DataGeneratorIndex:
    def __init__(self):
        data_path = './data/data_for_metarl.csv'
        data_df = pd.read_csv(data_path)
        data_df.set_index('eval_d', inplace=True)
        date_ = list(data_df.index)

        feature = dict()
        logy = np.log(1. + data_df.values)
        logp = np.cumsum(logy, axis=0) - logy[0]

        for n in [5, 20, 60]:
            feature['logy_{}'.format(n)] = log_y_nd(logp, n)
            feature['std_{}'.format(n)] = std_nd(logp, n)
            feature['mdd_{}'.format(n)] = mdd_nd(logp, n)
        feature['stdnew_{}'.format(5)] = std_nd_new(logp, n)
        feature['pos_{}'.format(5)] = np.sign(feature['logy_{}'.format(5)])


        min_d = 100
        time_steps = 5
        lookback = 60

        dataset = {}
        for i in range(min_d, len(date_) - time_steps, time_steps):
            dataset[date_[i]] = {'data': [], 'label': None}
            dataset[date_[i]]['label'] = [int(feature['pos_5'][i+time_steps][0] > 0), int(feature['pos_5'][i+time_steps][0] < 0)]
            for j in range(0, lookback, time_steps):
                tmp = logy[(i - j - time_steps):(i - j)]
                for key in feature.keys():
                    tmp = np.r_[tmp, feature[key][(i - j - 1):(i - j), :]]

                if j == 0:
                    tmp_all = tmp.flatten()
                else:
                    tmp_all = np.c_[tmp_all, tmp.flatten()]
            dataset[date_[i]]['data'] = tmp_all.transpose()

        seq_size, dim = dataset[date_[min_d]]['data'].shape
        pos_encoding = positional_encoding(seq_size, dim)

        train_start_d = date_[min_d]
        train_end_d = date_[1000]
        eval_start_d = date_[1000]
        eval_end_d = date_[1200]


        train_set = list()
        n_task = 10
        support_data = [[] for _ in range(n_task)]
        target_data = [[] for _ in range(n_task)]

        tasks = np.random.randint(min_d + 120, 1000 - 20, n_task)
        for i, task in enumerate(tasks):
            support_start_d = task - 120
            support_end_d = task
            target_end_d = task + 20
            for d in dataset.keys():
                if d >= date_[support_start_d] and d < date_[support_end_d]:
                    support_data[i].append(dataset[d])
                elif d < date_[target_end_d]:
                    target_data[i].append(dataset[d])





import tensorflow as tf
import sys

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Embedding


class BNN(object):
    def __init__(self,
                 dim_input,
                 dim_output,
                 dim_hidden,
                 num_layers,
                 is_bnn=True):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        self.is_bnn = is_bnn

    def log_likelihood_data(self, predict_y, target_y, log_gamma):
        if not self.is_bnn:
            raise NotImplementedError()

        err_y = predict_y - target_y

        log_like_data = 0.5 * log_gamma - 0.5 * tf.exp(log_gamma) * tf.square(err_y)
        return log_like_data

    def log_prior_weight(self, ):
        if not self.is_bnn:
            raise NotImplementedError()

        pass

    def mse_data(self, predict_y, target_y):
        return tf.reduce_sum(tf.square(predict_y, target_y), axis=1)

    def forward

    def list2vec(self, list_in):
        return tf.concat([tf.reshape(ww, [-1]) for ww in list_in], axis=0)

    def vec2dic(self, ):







class FeedForward(Model):
    def __init__(self, dim_out, num_units, out_activation='linear'):
        super().__init__()
        self.in_layer = Dense(num_units, activation=tf.nn.relu)
        self.out_layer = Dense(dim_out, activation=out_activation)

    def call(self, inputs):
        return self.out_layer(self.in_layer(inputs))







