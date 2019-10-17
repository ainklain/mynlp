import numpy as np
from collections import OrderedDict
import re
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM
from absl import flags
FLAGS = flags.FLAGS



# class MyNet_test(Model):
#     def __init__(self, ):
#         super().__init__()
#
#         self.h = Dense(3, name='h', kernel_initializer='ones')
#         self.o = Dense(1, name='o', kernel_initializer='ones')
#
#     def call(self, inputs):
#         x = inputs
#
#         return self.o(self.h(x))
#
# net = MyNet_test()
# net(tf.zeros([1, 3]))
#
# layer_list = []
# attr_list = []
# for var in net.trainable_weights:
#     _, layer_nm, attr_nm = list(filter(lambda x: x != '', re.split('/|:\d', var.name)))
#     layer_list.append(layer_nm)
#     attr_list.append(attr_nm)
#
# add_value = []
# with tf.GradientTape(persistent=True) as tape2:
#     with tf.GradientTape() as tape:
#         var_lists = net.trainable_variables
#         y = net(tf.convert_to_tensor(np.array([[1, 0, 0]]), dtype=tf.float32))
#         loss = tf.square(y - 2)
#     grad = tape.gradient(loss, var_lists)
#     print('var:{}'.format(var_lists[0]))
#     print('grad:{}'.format(grad[0]))
#     print('add:{}'.format(var_lists[0] + grad[0]))
#     new_var_lists = []
#     for i in range(len(attr_list)):
#         obj = net.get_layer(layer_list[i])
#         print('before:{}'.format(getattr(obj, attr_list[i])))
#         setattr(obj, attr_list[i], getattr(obj, attr_list[i]) + grad[i])
#         print('after:{}'.format(getattr(obj, attr_list[i])))
#         new_var_lists.append(getattr(obj, attr_list[i]))
#         if i == 0:
#             print('var:{}\nreal:{}'.format(var_lists[i], net.h.kernel))
#
#     y2 = net(tf.convert_to_tensor(np.array([[1, 0, 0]]), dtype=tf.float32))
#     grad2 = tape2.gradient(y2, var_lists)
#     grad3 = tape2.gradient(y2, new_var_lists)
#     print('var:{}'.format(var_lists[0]))
#     print('grad2:{}'.format(grad2[0]))
#     print('grad3:{}'.format(grad3[0]))
#     print('add:{}'.format(var_lists[0] + grad2[0]))
#
#     for i in range(len(attr_list)):
#         obj = net.get_layer(layer_list[i])
#         print('before:{}'.format(getattr(obj, attr_list[i])))
#         setattr(obj, attr_list[i], getattr(obj, attr_list[i]) + grad2[i])
#         print('after:{}'.format(getattr(obj, attr_list[i])))
#
#         if i == 0:
#             print('var:{}\nreal:{}'.format(var_lists[i], net.h.kernel))


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

        self.net = MyNet(dim_out=dim_output, dim_hiddens=[dim_hidden for _ in range(num_layers)])
        self.memory = dict()

        # for bayesian
        self.is_bnn = is_bnn

        self._initialize()

    def _initialize(self):
        x = tf.zeros([1, self.dim_input])
        self.net(x)

        # 업데이트된 layers를 찾아내기 위한 리스트 (덮어쓰면 trainable_weights에서 사라지기 때문)
        self._parse_attrs()

        if self.is_bnn:
            init_val = np.random.normal(-np.log(FLAGS.m_l),  0.001, [1])
            self.log_lambda = tf.Variable(initial_value=init_val, dtype=tf.float32, name='log_lambda')

            print('log_lambda: ', init_val)

            init_val = np.random.normal(-np.log(FLAGS.m_g),  0.001, [1])
            self.log_gamma = tf.Variable(initial_value=init_val, dtype=tf.float32, name='log_gamma')

            print('log_gamma: ', init_val)

    def _parse_attrs(self):
        if hasattr(self, 'layer_list'):
            print('already parsed.')
            return None

        self.layer_list = []
        self.attr_list = []

        for var in self.net.trainable_weights:
            _, layer_nm, attr_nm = list(filter(lambda x: x != '', re.split('/|:\d', var.name)))
            self.layer_list.append(layer_nm)
            self.attr_list.append(attr_nm)

    def get_var_lists(self):
        var_lists = []
        for i in range(len(self.attr_list)):
            obj = self.net.get_layer(self.layer_list[i])
            var_lists.append(getattr(obj, self.attr_list[i]))

        var_lists += [self.log_lambda, self.log_gamma]
        return var_lists

    def set_var_lists(self, var_lists):
        assert len(var_lists) == len(self.attr_list) + 2
        for i in range(len(self.attr_list)):
            obj = self.net.get_layer(self.layer_list[i])
            setattr(obj,  self.attr_list[i], var_lists[i])

        self.log_lambda = var_lists[-2]
        self.log_gamma = var_lists[-1]

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

        W_list = self.get_var_lists()
        # get lambda, gamma
        log_lambda = tf.reshape(W_list[-2], (1,))
        log_gamma = tf.reshape(W_list[-1], (1,))

        # get only weights
        W_vec = self.list2vec(W_list)[:-2]
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

    # def get_trainable_variables(self):
    #     return self.net.trainable_variables + [self.log_lambda, self.log_gamma]

    def convert(self, value_, from_='vector', to_='list'):
        w_list = list()
        if from_ == 'vector':
            # net_weight + lambda + gamma
            j = 0
            for i in range(len(self.attr_list)):
                obj = self.net.get_layer(self.layer_list[i])
                var_shape = getattr(obj, self.attr_list[i]).shape

                w_list.append(tf.reshape(value_[j:(j + np.prod(var_shape))], var_shape))
                j += np.prod(var_shape)

            assert j == len(value_) - 2
            w_list += [tf.reshape(value_[-2], (1,)), tf.reshape(value_[-1], (1,))]  # lambda and gamma

        elif from_ == 'list':
            w_list = value_[:]
        else:
            raise NotImplementedError

        if to_ == 'list':
            return w_list
        elif to_ == 'dict':
            return {'net': w_list[:-2], 'lambda': w_list[-2], 'gamma': w_list[-1]}
        elif to_ == 'vector':
            return tf.concat([tf.reshape(val, [-1]) for val in w_list], axis=0)
        else:
            raise NotImplementedError

    # list of params to vector
    def list2vec(self, wgt_list):
        return self.convert(wgt_list, from_='list', to_='vector')

    def vec2list(self, wgt_vec):
        return self.convert(wgt_vec, from_='vector', to_='list')
