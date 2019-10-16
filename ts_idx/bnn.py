import numpy as np
from collections import OrderedDict
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from absl import flags
FLAGS = flags.FLAGS


class MyNet:
    def forward(self, inputs, W):
        return W['out_layer'](W['hidden_layer'](inputs))

# net = MyNet()
#
# W = {'hidden_layer': Dense(3), 'out_layer': Dense(1)}
# W2 = {'hidden_layer': Dense(3), 'out_layer': Dense(1)}
#
# _ = net.forward(np.array([[2, 3]]), W)
# _ = net.forward(np.array([[2, 3]]), W2)
#
# var_lists = []
# var_lists2 = []
# with tf.GradientTape() as tape2:
#     with tf.GradientTape() as tape:
#         for key in W.keys():
#             var_lists += W[key].trainable_variables
#         for var in var_lists:
#             var_lists2.append(tf.add(var, 0.))
#
#         x = net.forward(np.array([[2, 3]]), W)
#
#         loss = (x - 3) ** 2
#     grad = tape.gradient(loss, var_lists)
# grad2 = tape2.gradient(loss, var_lists2)
#
#     W2['hidden_layer'].weights[0] = W['hidden_layer'].weights[0] + grad[0]
#     W2['hidden_layer'].weights[1] = W['hidden_layer'].weights[1] + grad[1]
#
#     W2['out_layer'].weights[0] = W['out_layer'].weights[0] + grad[2]
#     W2['out_layer'].weights[1] = W['out_layer'].weights[1] + grad[3]
#     x2 = net.forward(np.array([[2, 3]]), W2)
#
#     loss2 = loss + tf.stop_gradient((x2 - 3) ** 2)
#
# grad2 = tape2.gradient(loss2, var_lists2)


# net = MyNet()
# net(np.array([[1, 2]]))
>>>>>>> b6ddca8178a46dea091585aeff100b401f5201d5
#
# net2 = MyNet()
# net2(np.array([[1, 2]]))
#
#
# y_true = np.array([[3]])
# with tf.GradientTape(persistent=True) as tape2:
#     with tf.GradientTape() as tape:
#         var_lists = net.trainable_variables
#         y = net(np.array([[1, 2]]))
#         loss = tf.square(y - y_true)
#     grad = tape.gradient(loss, var_lists)
#     var_lists_ = [tf.add(var_lists[i], grad[i]) for i in range(len(var_lists))]
#     for i, v in enumerate(var_lists):
#         net2.trainable_variables[i] = v
#
# grad2 = tape2.gradient(var_lists_, var_lists)
# grad_ = tape2.gradient(net.trainable_variables, var_lists)
# grad3 = tape2.gradient(net2.trainable_variables, var_lists)
# del tape2
#
#
#
#
# class A:
#     def __init__(self):
#         self.x = tf.Variable(1.0)
#         self.y = tf.Variable(2.0)
#
#     def get_weights(self):
#         return [self.x, self.y]
#
#     def f(self):
#         x = self.x
#         y = self.y
#         z = 2 * x * x * x + 3 * y * y
#         return z
#
# class B:
#     def __init__(self):
#         self.A = A()
#
#     def f(self, C):
#         with tf.GradientTape(persistent=True) as tape:
#             var = self.A.get_weights()
#             print(var == C)
#             for i, v in enumerate(var):
#
#                 print(v, C[i])
#                 print(id(v), id(C[i]))
#
#         del tape
#
#     def g(self):
#         with tf.GradientTape(persistent=True) as tape:
#             C = self.A.get_weights()
#             self.f(C)
#
#         del tape
#
# b = B()
# b.g()


# net = MyNet(1, [10, 5])
# net(np.array([[1, 2, 3], [2, 3, 4]]))
# with tf.GradientTape() as tape:
#     var_lists = net.trainable_variables
#     y = net(np.array([[1, 2, 3], [2, 3, 4]]))
#     loss = tf.reduce_mean(y - np.array([[2], [3]]))
#
#     grad = tape.gradient(loss, var_lists)
# # grad2 = tape.gradient(loss, var_lists)
#
# del tape


class BNN:
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

        # for bayesian
        self.is_bnn = is_bnn

    def construct_network_weights(self, scope='network'):
        # init

        params = OrderedDict()

        # initializer
        fc_initializer = tf.initializers.glorot_uniform()
        b_initializer = tf.initializers.RandomNormal(0, 0.01)

        # layer 1
        params['w1'] = tf.Variable(fc_initializer([self.dim_input, self.dim_hidden]), name=scope + '_w1')
        params['b1'] = tf.Variable(b_initializer([self.dim_hidden]), name=scope + '_b1')

        # layer l
        for l in range(self.num_layers):
            if l < (self.num_layers - 1):
                dim_output = self.dim_hidden
            else:
                dim_output = self.dim_output
            params['w{}'.format(l + 2)] = tf.Variable(fc_initializer([self.dim_hidden, dim_output])
                                                      , name=scope + '_w{}'.format(l + 2))
            params['b{}'.format(l + 2)] = tf.Variable(b_initializer([dim_output])
                                                      , name=scope + '_b{}'.format(l + 2))

        if self.is_bnn:
            init_val = np.random.normal(-np.log(FLAGS.m_l),  0.001, [1])
            params['log_lambda'] = tf.Variable(name=scope + "_log_lambda",
                                               initial_value=init_val,
                                               dtype=tf.float32)

            print('log_lambda: ', init_val)

            init_val = np.random.normal(-np.log(FLAGS.m_g),  0.001, [1])
            params['log_gamma'] = tf.Variable(name=scope + "_log_gamma",
                                              initial_value=init_val,
                                              dtype=tf.float32)

            print('log_gamma: ', init_val)

        return params

    # data log-likelihood
    def log_likelihood_data(self, predict_y, target_y, log_gamma):
        # only for bnn
        if not self.is_bnn:
            NotImplementedError()

        # error
        error_y = predict_y - target_y

        # compute log-prob
        log_lik_data = 0.5 * log_gamma - 0.5 * tf.exp(log_gamma) * tf.square(error_y)
        return log_lik_data

    # weight log-prior
    def log_prior_weight(self, W_dict):
        # only for bnn
        if not self.is_bnn:
            NotImplementedError()

        # convert into vector list
        W_vec = self.dicval2vec(W_dict)

        # get lambda, gamma
        log_lambda = tf.reshape(W_vec[-2], (1,))
        log_gamma = tf.reshape(W_vec[-1], (1,))

        # get only weights
        W_vec = W_vec[:-2]
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

    # forward
    def forward_network(self, x, W_dict):
        hid = tf.keras.activations.relu(tf.matmul(x, W_dict['w1']) + W_dict['b1'])
        for l in range(self.num_layers):
            hid = tf.matmul(hid, W_dict['w{}'.format(l + 2)]) + W_dict['b{}'.format(l + 2)]
            if l < (self.num_layers - 1):
                hid = tf.keras.activations.relu(hid)
        return hid

    # list of params into vec
    def list2vec(self, list_in):
        return tf.concat([tf.reshape(ww, [-1]) for ww in list_in], axis=0)

    # vec param into dic
    def vec2dic(self, W_vec):
        if self.is_bnn:
            log_lambda = tf.reshape(W_vec[-2], (1,))
            log_gamma = tf.reshape(W_vec[-1], (1,))
            W_vec = W_vec[:-2]

            W_dic = self.network_weight_vec2dict(W_vec)

            W_dic['log_lambda'] = log_lambda
            W_dic['log_gamma'] = log_gamma
        else:
            W_dic = self.network_weight_vec2dict(W_vec)
        return W_dic

    # vec param into dic
    def network_weight_vec2dict(self, W_vec):
        W_dic = OrderedDict()
        dim_list = [self.dim_input] + [self.dim_hidden] * self.num_layers + [self.dim_output]

        # for each layer
        for l in range(len(dim_list) - 1):
            # input / output dim
            dim_input, dim_output = dim_list[l], dim_list[l + 1]

            # reshape into param
            W_dic['w{}'.format(l + 1)] = tf.reshape(W_vec[:(dim_input * dim_output)], [dim_input, dim_output])
            W_dic['b{}'.format(l + 1)] = W_vec[(dim_input * dim_output):((dim_input * dim_output) + dim_output)]

            # remove loaded param
            if l < (len(dim_list) - 2):
                W_vec = W_vec[((dim_input * dim_output) + dim_output):]
        return W_dic

    # dic to vec
    def dicval2vec(self, dic):
        return tf.concat([tf.reshape(val, [-1]) for val in dic.values()], axis=0)


# class MyNet(Model):
#     def __init__(self, dim_out, dim_hiddens, out_activation='linear'):
#         super().__init__()
#
#         self.hidden_layer = list()
#         for i, dim_h in enumerate(dim_hiddens):
#             self.hidden_layer.append(Dense(dim_h,
#                                            activation=tf.nn.relu,
#                                            kernel_initializer='glorot_uniform',
#                                            bias_initializer=tf.initializers.RandomNormal(0.0, 0.01)))
#         self.out_layer = Dense(dim_out,
#                                activation=out_activation,
#                                kernel_initializer='glorot_uniform',
#                                bias_initializer=tf.initializers.RandomNormal(0.0, 0.01))
#
#     def call(self, inputs):
#         x = inputs
#         for h_layer in self.hidden_layer:
#             x = h_layer(x)
#
#         return self.out_layer(x)
#
#
# # network
# class BNN(object):
#     def __init__(self,
#                  dim_input,
#                  dim_output,
#                  dim_hidden,
#                  num_layers,
#                  is_bnn=True):
#         # set model size
#         self.dim_input = dim_input
#         self.dim_output = dim_output
#         self.dim_hidden = dim_hidden
#         self.num_layers = num_layers
#
#         self.net = MyNet(dim_out=dim_output, dim_hiddens=[dim_hidden for _ in range(num_layers)])
#         self.memory = dict()
#
#         # for bayesian
#         self.is_bnn = is_bnn
#
#         self._initialize()
#
#     def _initialize(self):
#         x = tf.zeros([1, self.dim_input])
#         self.net(x)
#
#         if self.is_bnn:
#             init_val = np.random.normal(-np.log(FLAGS.m_l),  0.001, [1])
#             self.log_lambda = tf.Variable(initial_value=init_val, dtype=tf.float32, name='log_lambda')
#
#             print('log_lambda: ', init_val)
#
#             init_val = np.random.normal(-np.log(FLAGS.m_g),  0.001, [1])
#             self.log_gamma = tf.Variable(initial_value=init_val, dtype=tf.float32, name='log_gamma')
#
#             print('log_gamma: ', init_val)
#
#     def memory_wgt(self, name):
#         self.memory[name] = {'net': self.net.trainable_variables,
#                              'log_lambda': self.log_lambda,
#                              'log_gamma': self.log_gamma}
#
#     def forward_network(self, inputs):
#         return self.net(inputs)
#
#     # data log-likelihood
#     def log_likelihood_data(self, predict_y, target_y):
#         # only for bnn
#         if not self.is_bnn:
#             NotImplementedError()
#
#         log_gamma = self.log_gamma
#
#         # error
#         error_y = predict_y - target_y
#
#         # compute log-prob
#         log_lik_data = 0.5 * log_gamma - 0.5 * tf.exp(log_gamma) * tf.square(error_y)
#         return log_lik_data
#
#     # weight log-prior
#     def log_prior_weight(self):
#         # only for bnn
#         if not self.is_bnn:
#             NotImplementedError()
#
#         # get lambda, gamma
#         log_lambda = self.log_lambda
#         log_gamma = self.log_gamma
#
#         # get only weights
#         W_vec = self.list2vec(self.net.trainable_variables)
#         num_params = tf.cast(W_vec.shape[0], tf.float32)
#
#         # get data log-prior
#         log_prior_gamma = (FLAGS.a_g - 1) * log_gamma - FLAGS.b_g * tf.exp(log_gamma) + log_gamma
#
#         # get weight log-prior
#         W_diff = W_vec
#         log_prior_w = 0.5 * num_params * log_lambda - 0.5 * tf.exp(log_lambda) * tf.reduce_sum(W_diff ** 2)
#         log_prior_lambda = (FLAGS.a_l - 1) * log_lambda - FLAGS.b_l * tf.exp(log_lambda) + log_lambda
#
#         return log_prior_w, log_prior_gamma, log_prior_lambda
#
#     # mse data
#     def mse_data(self, predict_y, target_y):
#         return tf.reduce_sum(tf.square(predict_y - target_y), axis=1)
#
#     def get_trainable_variables(self):
#         return self.net.trainable_variables + [self.log_lambda, self.log_gamma]
#
#     def assign_values(self, wgt_list):
#         tr_var = self.get_trainable_variables()
#         assert len(tr_var) == len(wgt_list)
#         for i, var in enumerate(tr_var):
#             var.assign(wgt_list)
#
#     def convert(self, value_, from_='vector', to_='list'):
#         w_list = list()
#         if from_ == 'vector':
#             # net_weight + lambda + gamma
#             i = 0
#             for var in self.get_trainable_variables():
#                 var_shape = var.shape
#                 w_list.append(var.assign(tf.reshape(value_[i:(i + np.prod(var_shape))], var_shape)))
#                 i += np.prod(var_shape)
#             assert i == len(value_)
#         elif from_ == 'list':
#             w_list = value_[:]
#         else:
#             raise NotImplementedError
#
#         if to_ == 'list':
#             return w_list
#         elif to_ == 'dict':
#             return {'net': w_list[:-2], 'lambda': w_list[-2], 'gamma': w_list[-1]}
#         elif to_ == 'vector':
#             return tf.concat([tf.reshape(val, [-1]) for val in w_list], axis=0)
#         else:
#             raise NotImplementedError
#
#     # list of params to vector
#     def list2vec(self, wgt_list):
#         return self.convert(wgt_list, from_='list', to_='vector')
#
#     def vec2list(self, wgt_vec):
#         return self.convert(wgt_vec, from_='vector', to_='list')


