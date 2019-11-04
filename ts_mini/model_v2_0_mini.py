
# from ts_mini.features_mini import labels_for_mtl

import pickle
import numpy as np
import tensorflow as tf
import sys

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D


def positional_encoding(dim, sentence_length):
    encoded_vec = np.array([pos / np.power(10000, 2 * i / dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.constant(encoded_vec.reshape([sentence_length, dim]), dtype=tf.float32)

def layer_norm(inputs, eps=1e-6):
    # LayerNorm(x + Sublayer(x))
    feature_shape = inputs.get_shape()[-1:]
    #  평균과 표준편차을 넘겨 준다.
    mean = tf.math.reduce_mean(inputs, [-1], keepdims=True)
    std = tf.math.reduce_std(inputs, [-1], keepdims=True)
    beta = tf.Variable(tf.zeros(feature_shape), trainable=False)
    gamma = tf.Variable(tf.ones(feature_shape), trainable=False)

    return gamma * (inputs - mean) / (std + eps) + beta


def sublayer_connection(inputs, sublayer, dropout=0.2):
    outputs = layer_norm(inputs + Dropout(dropout)(sublayer))
    return outputs


class FeedForward(Model):
    def __init__(self, num_units, dim_out, out_activation='linear'):
        super().__init__()
        self.in_layer = Dense(num_units, activation=tf.nn.relu)
        self.out_layer = Dense(dim_out, activation=out_activation)

    def call(self, inputs):
        return self.out_layer(self.in_layer(inputs))


class MultiHeadAttention(Model):
    def __init__(self, num_units, heads):
        super().__init__()
        self.heads = heads

        self.q_layer = Dense(num_units, activation=tf.nn.relu)
        self.k_layer = Dense(num_units, activation=tf.nn.relu)
        self.v_layer = Dense(num_units, activation=tf.nn.relu)

        self.output_layer = Dense(num_units, activation=tf.nn.relu)

    def scaled_dot_product_attention(self, query, key, value, masked):
        key_dim_size = float(key.get_shape().as_list()[-1])
        key = tf.transpose(key, perm=[0, 2, 1])
        outputs = tf.matmul(query, key) / tf.sqrt(key_dim_size)
        # print("MHA: matmul_size {} (q: {}, k: {})".format(tf.matmul(query, key).shape, query.shape, key.shape))
        if masked:
            diag_vals = tf.ones_like(outputs[0, :, :])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        self.attention_map = tf.nn.softmax(outputs)

        return tf.matmul(self.attention_map, value)

    def show(self, ep, save=True):
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig = plt.figure()
        # plt.imshow(self.attention_map[0], cmap='hot', interpolation='nearest')
        sns.heatmap(self.attention_map[0], cmap='Greens')
        if save is not None:
            fig.savefig('./out/{}.png'.format(ep))
            plt.close(fig)
        else:
            plt.show()

    def call(self, query, key, value, masked=False):

        # print("before: [q: {}, k: {}, v:{}]".format(query.shape, key.shape, value.shape))
        query = tf.concat(tf.split(self.q_layer(query), self.heads, axis=-1), axis=0)
        key = tf.concat(tf.split(self.k_layer(key), self.heads, axis=-1), axis=0)
        value = tf.concat(tf.split(self.v_layer(value), self.heads, axis=-1), axis=0)

        # print("after: [q: {}, k: {}, v:{}]".format(query.shape, key.shape, value.shape))
        attention_map = self.scaled_dot_product_attention(query, key, value, masked=masked)

        attn_outputs = tf.concat(tf.split(attention_map, self.heads, axis=0), axis=-1)
        attn_outputs = self.output_layer(attn_outputs)

        return attn_outputs


class Encoder(Model):
    def __init__(self, dim_input, model_hidden_size, ffn_hidden_size, heads, num_layers):
        # dim_input = embedding_size
        super().__init__()
        dim_output = dim_input
        self.num_layers = num_layers

        self.enc_layers = dict()
        for i in range(num_layers):
            self.enc_layers['multihead_attn_' + str(i)] = MultiHeadAttention(model_hidden_size, heads)
            self.enc_layers['ff_' + str(i)] = FeedForward(ffn_hidden_size, dim_output)

    def show(self, ep, save=True):
        self.enc_layers['multihead_attn_' + str(self.num_layers - 1)].show(ep, save=save)

    def call(self, inputs, dropout=0.2):
        x = inputs
        for i in range(self.num_layers):
            x = sublayer_connection(x, self.enc_layers['multihead_attn_' + str(i)](x, x, x), dropout=dropout)
            x = sublayer_connection(x, self.enc_layers['ff_' + str(i)](x), dropout=dropout)

        return x


class Decoder(Model):
    def __init__(self, dim_input, dim_output, model_hidden_size, ffn_hidden_size, heads, num_layers):
        super().__init__()
        dim_output = dim_input

        self.num_layers = num_layers

        self.dec_layers = dict()
        for i in range(num_layers):
            self.dec_layers['masked_multihead_attn_' + str(i)] = MultiHeadAttention(model_hidden_size, heads)
            self.dec_layers['multihead_attn_' + str(i)] = MultiHeadAttention(model_hidden_size, heads)
            self.dec_layers['ff_' + str(i)] = FeedForward(ffn_hidden_size, dim_output)

        self.logit_layer = Dense(dim_output)

    def call(self, inputs, encoder_outputs, dropout=0.2):
        x = inputs
        for i in range(self.num_layers):
            x = sublayer_connection(x, self.dec_layers['masked_multihead_attn_' + str(i)](x, x, x, masked=True), dropout=dropout)
            x = sublayer_connection(x, self.dec_layers['multihead_attn_' + str(i)](x, encoder_outputs, encoder_outputs), dropout=dropout)
            x = sublayer_connection(x, self.dec_layers['ff_' + str(i)](x), dropout=dropout)

        return self.logit_layer(x)


class ConvModel(Model):
    def __init__(self, embedding_size):
        super().__init__()
        self.conv1 = Conv1D(embedding_size, 1, activation='relu')

    def call(self, inputs):
        # (None, T, n_features) -> (None, T, embedding_size)
        return self.conv1(inputs)


class MarketModel:
    """omit embedding time series. just 1-D data used"""

    def __init__(self, configs, feature_cls, weight_scheme='ew'):
        self.weight_scheme = weight_scheme

        self.input_seq_size = configs.m_days // configs.sampling_days + 1
        # self.output_seq_size = configs.k_days // configs.sampling_days
        self.output_seq_size = 1
        self.position_encode_in = positional_encoding(configs.d_model_mm, self.input_seq_size)
        self.position_encode_out = positional_encoding(configs.d_model_mm, self.output_seq_size)

        self.conv_embedding = ConvModel(embedding_size=configs.d_model_mm)
        self.encoder = Encoder(dim_input=configs.d_model_mm,
                               model_hidden_size=configs.model_hidden_size_mm,
                               ffn_hidden_size=configs.ffn_hidden_size_mm,
                               heads=configs.attention_head_size_mm,
                               num_layers=configs.layer_size_mm)

        self.decoder = Decoder(dim_input=configs.d_model_mm,
                               dim_output=configs.d_model_mm,
                               model_hidden_size=configs.model_hidden_size_mm,
                               ffn_hidden_size=configs.ffn_hidden_size_mm,
                               heads=configs.attention_head_size_mm,
                               num_layers=configs.layer_size_mm)

        self.predictor = FeedForward(16, 1, out_activation='linear')
        self.feature_cls = feature_cls

        self.optim_encoder_w = None
        self.optim_decoder_w = None
        self.optim_predictor_w = None
        self.dropout_train = configs.dropout_mm

        self.accuracy = tf.metrics.Accuracy()

        self.optimizer = tf.optimizers.Adam(configs.learning_rate_mm)

        self._initialize(configs)

    def _initialize(self, configs):
        feature_temp = tf.zeros([1, self.input_seq_size, configs.embedding_size_mm], dtype=tf.float32)
        # embed_temp = self.embedding(feature_temp)
        embed_temp = self.conv_embedding(feature_temp)
        enc_temp = self.encoder(embed_temp)
        dec_temp = self.decoder(embed_temp, enc_temp)

        _ = self.predictor(dec_temp)
        self.optim_predictor_w = self.predictor.get_weights()
        self.optim_conv_embed_w = self.conv_embedding.get_weights()
        self.optim_encoder_w = self.encoder.get_weights()
        self.optim_decoder_w = self.decoder.get_weights()

        self._reset_eval_param()

    def weight_to_optim(self):
        self.conv_embedding.set_weights(self.optim_conv_embed_w)
        self.encoder.set_weights(self.optim_encoder_w)
        self.decoder.set_weights(self.optim_decoder_w)
        self.predictor.set_weights(self.optim_predictor_w)

        self._reset_eval_param()

    def _reset_eval_param(self):
        self.eval_loss = 100000
        self.eval_count = 0

    def train(self, features, labels, print_loss=False):
        with tf.GradientTape(persistent=True) as tape:
            x_embed = self.conv_embedding(features['input']) + self.position_encode_in
            y_embed = self.conv_embedding(features['output']) + self.position_encode_out

            encoder_output = self.encoder(x_embed, dropout=self.dropout_train)
            predict = self.decoder(y_embed, encoder_output, dropout=self.dropout_train)

            var_lists = self.conv_embedding.trainable_variables \
                        + self.encoder.trainable_variables \
                        + self.decoder.trainable_variables \
                        + self.predictor.trainable_variables

            pred_ = self.predictor(predict)

            loss_ = tf.losses.categorical_crossentropy(labels, pred_)

        grad = tape.gradient(loss_, var_lists)
        self.optimizer.apply_gradients(zip(grad, var_lists))

        del tape

        if print_loss:
            print_str = "loss_: {:.6f}".format(np.mean(loss_.numpy()))
            print(print_str)

    def evaluate(self, datasets, steps=-1):
        loss_avg = 0
        for i, (features, labels) in enumerate(datasets.take(steps)):
            x_embed = self.conv_embedding(features['input']) + self.position_encode_in
            y_embed = self.conv_embedding(features['output']) + self.position_encode_out

            encoder_output = self.encoder(x_embed, dropout=0.)
            predict = self.decoder(y_embed, encoder_output, dropout=0.)
            pred_ = self.predictor(predict)

            loss_ = tf.losses.categorical_crossentropy(labels, pred_)
            loss_avg += np.mean(loss_.numpy())

        loss_avg = loss_avg / i
        print("eval loss:{} (steps:{})".format(loss_avg, i))
        if loss_avg < self.eval_loss:
            self.eval_loss = loss_avg
            self.eval_count = 0
            self.optim_encoder_w = self.encoder.get_weights()
            self.optim_decoder_w = self.decoder.get_weights()
            self.optim_predictor_w = self.predictor.get_weights()
        else:
            self.eval_count += 1

    def predict(self, feature):

        x_embed = self.conv_embedding(feature['input']) + self.position_encode_in
        y_embed = self.conv_embedding(feature['output']) + self.position_encode_out

        encoder_output = self.encoder(x_embed, dropout=0.)
        predict = self.decoder(y_embed, encoder_output, dropout=0.)

        pred_ = self.predictor(predict)

        return pred_
        # return pred_ret, pred_pos, pred_vol, pred_mdd

    def save_model(self, f_name):
        if f_name[-4:] != '.pkl':
            f_name = f_name + ".pkl"

        w_dict = {}
        w_dict['conv_embedding'] = self.optim_conv_embed_w
        w_dict['encoder'] = self.optim_encoder_w
        w_dict['decoder'] = self.optim_decoder_w
        w_dict['predictor'] = self.optim_predictor_w

        # f_name = os.path.join(model_path, model_name)
        with open(f_name, 'wb') as f:
            pickle.dump(w_dict, f)

        print("model saved. (path: {})".format(f_name))

    def load_model(self, f_name):
        if f_name[-4:] != '.pkl':
            f_name = f_name + ".pkl"

        # f_name = os.path.join(model_path, model_name)
        with open(f_name, 'rb') as f:
            w_dict = pickle.load(f)

        self.optim_conv_embed_w = w_dict['conv_embedding']
        self.optim_encoder_w = w_dict['encoder']
        self.optim_decoder_w = w_dict['decoder']
        self.optim_predictor_w = w_dict['predictor']

        self.conv_embedding.set_weights(self.optim_conv_embed_w)
        self.encoder.set_weights(self.optim_encoder_w)
        self.decoder.set_weights(self.optim_decoder_w)
        self.predictor.set_weights(self.optim_predictor_w)

        print("model loaded. (path: {})".format(f_name))


class TSModel:
    """omit embedding time series. just 1-D data used"""
    def __init__(self, configs, feature_cls, weight_scheme='ew'):
        self.weight_scheme = weight_scheme

        self.input_seq_size = configs.m_days // configs.sampling_days + 1
        # self.output_seq_size = configs.k_days // configs.sampling_days
        self.output_seq_size = 1
        self.position_encode_in = positional_encoding(configs.d_model, self.input_seq_size)
        self.position_encode_out = positional_encoding(configs.d_model, self.output_seq_size)

        self.conv_embedding = ConvModel(embedding_size=configs.d_model)
        self.encoder = Encoder(dim_input=configs.d_model,
                               model_hidden_size=configs.model_hidden_size,
                               ffn_hidden_size=configs.ffn_hidden_size,
                               heads=configs.attention_head_size,
                               num_layers=configs.layer_size)

        self.decoder = Decoder(dim_input=configs.d_model,
                               dim_output=configs.d_model,
                               model_hidden_size=configs.model_hidden_size,
                               ffn_hidden_size=configs.ffn_hidden_size,
                               heads=configs.attention_head_size,
                               num_layers=configs.layer_size)

        self.predictor = dict()
        self.predictor_helper = dict()
        n_size = 16
        for key in configs.model_predictor_list:
            tags = key.split('_')
            if tags[0] in configs.features_structure['regression'].keys():
                if key in ['cslogy', 'csstd']:
                    self.predictor[key] = FeedForward(n_size, len(configs.features_structure['regression'][key]), out_activation='sigmoid')
                else:
                    self.predictor[key] = FeedForward(n_size, len(configs.features_structure['regression'][key]))
            elif tags[0] in configs.features_structure['classification'].keys():
                self.predictor[key] = FeedForward(n_size, 2, out_activation='softmax')
                self.predictor_helper[key] = configs.features_structure['regression']['logy'].index(int(tags[1]))
            # elif tags[0] in configs.features_structure['crosssection'].keys():
            #     self.predictor[key] = FeedForward(64, len(configs.features_structure['regression'][key]))

        self.feature_cls = feature_cls

        self.optim_encoder_w = None
        self.optim_decoder_w = None
        self.optim_predictor_w = dict()
        self.dropout_train = configs.dropout

        self.accuracy = tf.metrics.Accuracy()

        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        # lr = tf.optimizers.schedules.PolynomialDecay(1e-3, 2000, 5e-4)
        # lr = tf.optimizers.schedules.PiecewiseConstantDecay([50, 150, 300], [1e-2, 1e-3, 1e-4, 1e-5])
        self.optimizer = tf.optimizers.Adam(configs.learning_rate)

        self._initialize(configs)

    def _initialize(self, configs):
        feature_temp = tf.zeros([1, self.input_seq_size, configs.embedding_size], dtype=tf.float32)
        # embed_temp = self.embedding(feature_temp)
        embed_temp = self.conv_embedding(feature_temp)
        enc_temp = self.encoder(embed_temp)
        dec_temp = self.decoder(embed_temp, enc_temp)

        for key in self.predictor.keys():
            _ = self.predictor[key](dec_temp)
            self.optim_predictor_w[key] = self.predictor[key].get_weights()

        self.optim_conv_embed_w = self.conv_embedding.get_weights()
        self.optim_encoder_w = self.encoder.get_weights()
        self.optim_decoder_w = self.decoder.get_weights()

        self._reset_eval_param()

    def weight_to_optim(self):
        self.conv_embedding.set_weights(self.optim_conv_embed_w)
        self.encoder.set_weights(self.optim_encoder_w)
        self.decoder.set_weights(self.optim_decoder_w)
        for key in self.predictor.keys():
            self.predictor[key].set_weights(self.optim_predictor_w[key])

        self._reset_eval_param()

    def _reset_eval_param(self):
        self.eval_loss = 100000
        self.eval_count = 0

    def train_mtl(self, features, labels_mtl, print_loss=False):
        with tf.GradientTape(persistent=True) as tape:
            x_embed = self.conv_embedding(features['input']) + self.position_encode_in
            y_embed = self.conv_embedding(features['output']) + self.position_encode_out

            encoder_output = self.encoder(x_embed, dropout=self.dropout_train)
            predict = self.decoder(y_embed, encoder_output, dropout=self.dropout_train)

            var_lists = self.conv_embedding.trainable_variables + self.encoder.trainable_variables + self.decoder.trainable_variables

            pred_each = dict()
            loss_each = dict()
            loss = None
            for key in self.predictor.keys():
                var_lists += self.predictor[key].trainable_variables
                pred_each[key] = self.predictor[key](predict)

                if self.weight_scheme == 'mw':
                    adj_weight = labels_mtl['size_factor'][:, :, 0] * 2.  # size value 평균이 0.5 이므로 기존이랑 스케일 맞추기 위해 2 곱

                else:
                    adj_weight = 1.
                adj_importance = labels_mtl['importance_wgt']

                if key[:3] == 'pos':
                    loss_each[key] = tf.losses.categorical_crossentropy(labels_mtl[key], pred_each[key]) \
                                     * tf.abs(labels_mtl['logy'][:, :, self.predictor_helper[key]]) \
                                     * adj_weight * adj_importance

                else:
                    loss_each[key] = tf.losses.MSE(labels_mtl[key], pred_each[key]) * adj_weight * adj_importance

                if loss is None:
                    loss = loss_each[key]
                else:
                    loss += loss_each[key]

        grad = tape.gradient(loss, var_lists)
        self.optimizer.apply_gradients(zip(grad, var_lists))

        del tape

        if print_loss:
            print_str = ""
            for key in loss_each.keys():
                print_str += "loss_{}: {:.6f} / ".format(key, np.mean(loss_each[key].numpy()))

            print(print_str)

    def evaluate_mtl(self, datasets, features_list, steps=-1):
        loss_avg = 0
        for i, (features, labels, size_factors, importance_wgt) in enumerate(datasets.take(steps)):
            labels_mtl = self.feature_cls.labels_for_mtl(features_list, labels, size_factors, importance_wgt)

            x_embed = self.conv_embedding(features['input']) + self.position_encode_in
            y_embed = self.conv_embedding(features['output']) + self.position_encode_out

            encoder_output = self.encoder(x_embed, dropout=0.)
            predict = self.decoder(y_embed, encoder_output, dropout=0.)

            pred_each = dict()
            loss_each = dict()
            loss = None
            for key in self.predictor.keys():
                pred_each[key] = self.predictor[key](predict)

                if self.weight_scheme == 'mw':
                    adj_weight = labels_mtl['size_factor'][:, :, 0] * 2.
                else:
                    adj_weight = 1.

                # adj_importance = labels_mtl['importance_wgt']
                if key[:3] == 'pos':
                    loss_each[key] = tf.losses.categorical_crossentropy(labels_mtl[key], pred_each[key]) \
                                     * tf.abs(labels_mtl['logy'][:, :, self.predictor_helper[key]]) \
                                     * adj_weight # * adj_importance
                else:
                    loss_each[key] = tf.losses.MSE(labels_mtl[key], pred_each[key]) * adj_weight #* adj_importance

                if loss is None:
                    loss = loss_each[key]
                else:
                    loss += loss_each[key]

            loss_avg += np.mean(loss.numpy())

        loss_avg = loss_avg / i
        print("eval loss:{} (steps:{})".format(loss_avg, i))
        if loss_avg < self.eval_loss:
            self.eval_loss = loss_avg
            self.eval_count = 0
            self.optim_encoder_w = self.encoder.get_weights()
            self.optim_decoder_w = self.decoder.get_weights()
            for key in self.predictor.keys():
                self.optim_predictor_w[key] = self.predictor[key].get_weights()

        else:
            self.eval_count += 1

    def predict_mtl(self, feature):

        x_embed = self.conv_embedding(feature['input']) + self.position_encode_in
        y_embed = self.conv_embedding(feature['output']) + self.position_encode_out

        encoder_output = self.encoder(x_embed, dropout=0.)
        predict = self.decoder(y_embed, encoder_output, dropout=0.)

        pred_each = dict()
        for key in self.predictor.keys():
            pred_each[key] = self.predictor[key](predict)

        return pred_each
        # return pred_ret, pred_pos, pred_vol, pred_mdd

    def save_model(self, f_name):
        if f_name[-4:] != '.pkl':
            f_name = f_name + ".pkl"

        w_dict = {}
        w_dict['conv_embedding'] = self.optim_conv_embed_w
        w_dict['encoder'] = self.optim_encoder_w
        w_dict['decoder'] = self.optim_decoder_w
        w_dict['predictor'] = self.optim_predictor_w

        # f_name = os.path.join(model_path, model_name)
        with open(f_name, 'wb') as f:
            pickle.dump(w_dict, f)

        print("model saved. (path: {})".format(f_name))

    def load_model(self, f_name):
        if f_name[-4:] != '.pkl':
            f_name = f_name + ".pkl"

        # f_name = os.path.join(model_path, model_name)
        with open(f_name, 'rb') as f:
            w_dict = pickle.load(f)

        self.optim_conv_embed_w = w_dict['conv_embedding']
        self.optim_encoder_w = w_dict['encoder']
        self.optim_decoder_w = w_dict['decoder']
        self.optim_predictor_w = w_dict['predictor']

        self.conv_embedding.set_weights(self.optim_conv_embed_w)
        self.encoder.set_weights(self.optim_encoder_w)
        self.decoder.set_weights(self.optim_decoder_w)
        for key in self.optim_predictor_w.keys():
            self.predictor[key].set_weights(self.optim_predictor_w[key])

        print("model loaded. (path: {})".format(f_name))
