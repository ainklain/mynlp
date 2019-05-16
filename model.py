
import pickle
import numpy as np
import tensorflow as tf
import sys

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Embedding


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
    def __init__(self, dim_out, num_units):
        super().__init__()
        self.in_layer = Dense(num_units, activation=tf.nn.relu)
        self.out_layer = Dense(dim_out)

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

        if masked:
            diag_vals = tf.ones_like(outputs[0, :, :])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        attention_map = tf.nn.softmax(outputs)

        return tf.matmul(attention_map, value)

    def call(self, query, key, value, masked=False):
        query = tf.concat(tf.split(self.q_layer(query), self.heads, axis=-1), axis=0)
        key = tf.concat(tf.split(self.k_layer(key), self.heads, axis=-1), axis=0)
        value = tf.concat(tf.split(self.v_layer(value), self.heads, axis=-1), axis=0)

        attention_map = self.scaled_dot_product_attention(query, key, value, masked=masked)

        attn_outputs = tf.concat(tf.split(attention_map, self.heads, axis=0), axis=-1)
        attn_outputs = self.output_layer(attn_outputs)

        return attn_outputs


class Encoder(Model):
    def __init__(self, dim_input, model_hidden_size, ffn_hidden_size, heads, num_layers):
        # dim_input = embedding_size
        super().__init__()

        self.num_layers = num_layers

        self.enc_layers = dict()
        for i in range(num_layers):
            self.enc_layers['multihead_attn_' + str(i)] = MultiHeadAttention(model_hidden_size, heads)
            self.enc_layers['ff_' + str(i)] = FeedForward(dim_input, ffn_hidden_size)

    def call(self, inputs):
        x = inputs
        for i in range(self.num_layers):
            x = sublayer_connection(x, self.enc_layers['multihead_attn_' + str(i)](x, x, x))
            x = sublayer_connection(x, self.enc_layers['ff_' + str(i)](x))

        return x


class Decoder(Model):
    def __init__(self, dim_input, dim_output, model_hidden_size, ffn_hidden_size, heads, num_layers):
        super().__init__()

        self.num_layers = num_layers

        self.dec_layers = dict()
        for i in range(num_layers):
            self.dec_layers['masked_multihead_attn_' + str(i)] = MultiHeadAttention(model_hidden_size, heads)
            self.dec_layers['multihead_attn_' + str(i)] = MultiHeadAttention(model_hidden_size, heads)
            self.dec_layers['ff_' + str(i)] = FeedForward(dim_input, ffn_hidden_size)

        self.logit_layer = Dense(dim_output)

    def call(self, inputs, encoder_outputs):
        x = inputs
        for i in range(self.num_layers):
            x = sublayer_connection(x, self.dec_layers['masked_multihead_attn_' + str(i)](x, x, x, masked=True))
            x = sublayer_connection(x, self.dec_layers['multihead_attn_' + str(i)](x, encoder_outputs, encoder_outputs))
            x = sublayer_connection(x, self.dec_layers['ff_' + str(i)](x))

        return self.logit_layer(x)


class NLPModel:
    def __init__(self, configs):
        if configs.xavier_initializer:
            embeddings_initializer = 'glorot_normal'
        else:
            embeddings_initializer = 'uniform'

        self.vocabulary_length = configs.vocabulary_length

        self.position_encode = positional_encoding(configs.embedding_size, configs.max_sequence_length)

        self.embedding = Embedding(configs.vocabulary_length, configs.embedding_size,
                              embeddings_initializer=embeddings_initializer)

        self.encoder = Encoder(dim_input=configs.embedding_size,
                               model_hidden_size=configs.model_hidden_size,
                               ffn_hidden_size=configs.ffn_hidden_size,
                               heads=configs.attention_head_size,
                               num_layers=configs.layer_size)

        self.decoder = Decoder(dim_input=configs.embedding_size,
                               dim_output=configs.vocabulary_length,
                               model_hidden_size=configs.model_hidden_size,
                               ffn_hidden_size=configs.ffn_hidden_size,
                               heads=configs.attention_head_size,
                               num_layers=configs.layer_size)

        self.accuracy = tf.metrics.Accuracy()

        self.optimizer = tf.optimizers.Adam(configs.learning_rate)

        self._initialize(configs)

    def _initialize(self, configs):
        embed_temp = tf.zeros([1, configs.max_sequence_length, configs.embedding_size], dtype=tf.float32)
        enc_temp = self.encoder(embed_temp)
        _ = self.decoder(embed_temp, enc_temp)

    def train(self, features, labels):
        with tf.GradientTape() as tape:
            x_embed = self.embedding(features['input']) + self.position_encode
            y_embed = self.embedding(features['output']) + self.position_encode

            encoder_output = self.encoder(x_embed)
            logits = self.decoder(y_embed, encoder_output)

            predict = tf.argmax(logits, 2)

            labels_ = tf.one_hot(labels, self.vocabulary_length)

            # var_lists = self.encoder.trainable_variables + self.decoder.trainable_variables
            var_lists = self.embedding.trainable_variables + self.encoder.trainable_variables + self.decoder.trainable_variables
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
        grad = tape.gradient(loss, var_lists)
        self.optimizer.apply_gradients(zip(grad, var_lists))

        self.accuracy.update_state(labels, predict)

    def predict(self, feature):

        x_embed = self.embedding(feature['input']) + self.position_encode
        y_embed = self.embedding(feature['output']) + self.position_encode

        encoder_output = self.encoder(x_embed)
        logits = self.decoder(y_embed, encoder_output)

        predict = tf.argmax(logits, 2)

        return predict

    def save_model(self, f_name):
        w_dict = {}
        w_dict['embedding'] = self.embedding.get_weights()
        w_dict['encoder'] = self.encoder.get_weights()
        w_dict['decoder'] = self.decoder.get_weights()

        # f_name = os.path.join(model_path, model_name)
        with open(f_name, 'wb') as f:
            pickle.dump(w_dict, f)

        print("model saved. (path: {})".format(f_name))

    def load_model(self, f_name):
        # f_name = os.path.join(model_path, model_name)
        with open(f_name, 'rb') as f:
            w_dict = pickle.load(f)

        self.embedding.set_weights(w_dict['embedding'])
        self.encoder.set_weights(w_dict['encoder'])
        self.decoder.set_weights(w_dict['decoder'])

        print("model loaded. (path: {})".format(f_name))
