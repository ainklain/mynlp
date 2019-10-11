import numpy as np
from collections import OrderedDict
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from absl import flags
FLAGS = flags.FLAGS



class MyNet(Model):
    def __init__(self, dim_out, dim_hiddens, out_activation='linear'):
        super().__init__()

        self.hidden_layer = list()
        for i, dim_h in enumerate(dim_hiddens):
            self.hidden_layer.append(Dense(dim_h,
                                           activation=tf.nn.relu,
                                           kernel_initializer='glorot_uniform',
                                           bias_initializer=tf.initializers.RandomNormal(0.0, 0.01)))
        self.out_layer = Dense(dim_out,
                               activation=out_activation,
                               kernel_initializer='glorot_uniform',
                               bias_initializer=tf.initializers.RandomNormal(0.0, 0.01))

    def call(self, inputs):
        x = inputs
        for h_layer in self.hidden_layer:
            x = h_layer(x)

        return self.out_layer(x)


net = MyNet(1, [10, 5])
net(np.array([[1, 2, 3], [2, 3, 4]]))
with tf.GradientTape(persistent=True) as tape:
    var_lists = net.trainable_variables
    y = net(np.array([[1, 2, 3], [2, 3, 4]]))
    loss = tf.reduce_mean(y - np.array([[2], [3]]))

    grad = tape.gradient(loss, var_lists)
grad2 = tape.gradient(loss, var_lists)

del tape



# network
class BNN(object):
    def __init__(self,
                 dim_input,
                 dim_output,
                 dim_hidden,
                 num_layers,
                 is_bnn=True):
        # set model size
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        self.net = MyNet(dim_out=dim_output, dim_hiddens=[dim_hidden for _ in range(len(num_layers))])
        self.memory = dict()

        # for bayesian
        self.is_bnn = is_bnn

        self._initialize()

    def _initialize(self):
        x = tf.zeros([1, self.dim_input])
        self.net(x)

        if self.is_bnn:
            init_val = np.random.normal(-np.log(FLAGS.m_l),  0.001, [1])
            self.log_lambda = tf.Variable(initial_value=init_val, dtype=tf.float32)

            print('log_lambda: ', init_val)

            init_val = np.random.normal(-np.log(FLAGS.m_g),  0.001, [1])
            self.log_gamma = tf.Variable(initial_value=init_val, dtype=tf.float32)

            print('log_gamma: ', init_val)

    def memory_wgt(self, name):
        self.memory[name] = {'net': self.net.trainable_variables,
                             'log_lambda': self.log_lambda,
                             'log_gamma': self.log_gamma}

    def forward_network(self, inputs):
        return self.net(inputs)

    # data log-likelihood
    def log_likelihood_data(self, predict_y, target_y):
        # only for bnn
        if not self.is_bnn:
            NotImplementedError()

        log_gamma = self.log_gamma

        # error
        error_y = predict_y - target_y

        # compute log-prob
        log_lik_data = 0.5 * log_gamma - 0.5 * tf.exp(log_gamma) * tf.square(error_y)
        return log_lik_data

    # weight log-prior
    def log_prior_weight(self):
        # only for bnn
        if not self.is_bnn:
            NotImplementedError()

        # get lambda, gamma
        log_lambda = self.log_lambda
        log_gamma = self.log_gamma

        # get only weights
        W_vec = self.list2vec(self.net.trainable_variables)
        num_params = tf.cast(W_vec.shape[0], tf.float32)

        # get data log-prior
        log_prior_gamma = (FLAGS.a_g - 1) * log_gamma - FLAGS.b_g * tf.exp(log_gamma) + log_gamma

        # get weight log-prior
        W_diff = W_vec
        log_prior_w = 0.5 * num_params * log_lambda - 0.5 * tf.exp(log_lambda) * tf.reduce_sum(W_diff ** 2)
        log_prior_lambda = (FLAGS.a_l - 1) * log_lambda - FLAGS.b_l * tf.exp(log_lambda) + log_lambda

        return log_prior_w, log_prior_gamma, log_prior_lambda

    # mse data
    def mse_data(self, predict_y, target_y):
        return tf.reduce_sum(tf.square(predict_y - target_y), axis=1)

    # list of params to vector
    def list2vec(self, wgt_list):
        return tf.concat([tf.reshape(val, [-1]) for val in wgt_list], axis=0)
