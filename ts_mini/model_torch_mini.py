# https://github.com/JayParks/transformer
# from ts_mini.features_mini import labels_for_mtl

import math
import numpy as np
import pickle
import sys
import torch

from torch import nn
from torch.nn import functional as F, init
from torch.autograd import Variable


# from tensorflow.keras import Model
# from tensorflow.keras.layers import Dense, Dropout, Embedding

# ####################### Module ##########################

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x n_heads x len_q x d_k]
        # k: [b_size x n_heads x len_k x d_k]
        # v: [b_size x n_heads x len_v x d_v] note: (len_k == len_v)

        # attn: [b_size x n_heads x len_q x len_k]
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)
        attn = self.dropout(self.softmax(scores))

        # outputs: [b_size x n_heads x len_q x d_v]
        context = torch.matmul(attn, v)

        return context, attn


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True,)
        std = z.std(dim=-1, keepdim=True,)
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out


class PosEncoding(nn.Module):
    def __init__(self, max_seq_len, d_word_vec):
        super(PosEncoding, self).__init__()
        pos_enc = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / d_word_vec) for j in range(d_word_vec)]
            for pos in range(max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pad_row = np.zeros([1, d_word_vec])
        pos_enc = np.concatenate([pad_row, pos_enc]).astype(np.float32)

        # additional single row for PAD idx
        self.pos_enc = nn.Embedding(max_seq_len + 1, d_word_vec)
        # fix positional encoding: exclude weight from grad computation
        self.pos_enc.weight = nn.Parameter(torch.from_numpy(pos_enc), requires_grad=False)

    def forward(self, input_len):
        # max_len = torch.max(input_len)
        # tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        input_pos = torch.LongTensor([list(range(1, input_len+1))])

        return self.pos_enc(input_pos)


# ####################### Sublayers ##########################
class _MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = Linear([d_model, d_k * n_heads])
        self.w_k = Linear([d_model, d_k * n_heads])
        self.w_v = Linear([d_model, d_v * n_heads])

        self.attention = ScaledDotProductAttention(d_k, dropout)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_k x d_model]
        b_size = q.size(0)

        # q_s: [b_size x n_heads x len_q x d_k]
        # k_s: [b_size x n_heads x len_k x d_k]
        # v_s: [b_size x n_heads x len_k x d_v]
        q_s = self.w_q(q).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.w_k(k).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.w_v(v).view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask:  # attn_mask: [b_size x len_q x len_k]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # context: [b_size x n_heads x len_q x d_v], attn: [b_size x n_heads x len_q x len_k]
        context, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)
        # context: [b_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)

        # return the context and attention weights
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.multihead_attn = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.proj = Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q
        # context: a tensor of shape [b_size x len_q x n_heads * d_v]
        context, attn = self.multihead_attn(q, k, v, attn_mask=attn_mask)

        # project back to the residual size, outputs: [b_size x len_q x d_model]
        output = self.dropout(self.proj(context))
        return self.layer_norm(residual + output), attn


class MultiBranchAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout):
        super(MultiBranchAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_branches = n_branches

        self.multihead_attn = _MultiHeadAttention(d_k, d_v, d_model, n_branches, dropout)
        # additional parameters for BranchedAttention
        self.w_o = nn.ModuleList([Linear(d_v, d_model) for _ in range(n_branches)])
        self.w_kp = torch.rand(n_branches)
        self.w_kp = nn.Parameter(self.w_kp/self.w_kp.sum())
        self.w_a = torch.rand(n_branches)
        self.w_a = nn.Parameter(self.w_a/self.w_a.sum())

        self.pos_ffn = nn.ModuleList([
            PoswiseFeedForwardNet(d_model, d_ff//n_branches, dropout) for _ in range(n_branches)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

        init.xavier_normal(self.w_o)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q

        # context: a tensor of shape [b_size x len_q x n_branches * d_v]
        context, attn = self.multih_attn(q, k, v, attn_mask=attn_mask)

        # context: a list of tensors of shape [b_size x len_q x d_v] len: n_branches
        context = context.split(self.d_v, dim=-1)

        # outputs: a list of tensors of shape [b_size x len_q x d_model] len: n_branches
        outputs = [self.w_o[i](context[i]) for i in range(self.n_branches)]
        outputs = [kappa * output for kappa, output in zip(self.w_kp, outputs)]
        outputs = [pos_ffn(output) for pos_ffn, output in zip(self.pos_ffn, outputs)]
        outputs = [alpha * output for alpha, output in zip(self.w_a, outputs)]

        # output: [b_size x len_q x d_model]
        output = self.dropout(torch.stack(outputs).sum(dim=0))
        return self.layer_norm(residual + output), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))

        # outputs: [b_size x len_q x d_model]
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        return self.layer_norm(residual + output)


# ####################### Layers ##########################
class ConvEmbeddingLayer(nn.Module):
    # input features의 수를 d_model만큼의 1d conv로 재생산

    def __init__(self, n_features, d_model):
        super(ConvEmbeddingLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=d_model, kernel_size=1)

    def forward(self, inputs):
        # input shape: (b_size, T, n_features)
        inputs = inputs.contiguous().transpose(-2, -1)
        # (b_size, n_features, T) -> (b_size, d_model, T)
        outputs = self.conv1(inputs)
        return outputs.contiguous().transpose(-2, -1)


class EncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_inputs, self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs,
                                               enc_inputs, attn_mask=self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


class WeightedEncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout=0.1):
        super(WeightedEncoderLayer, self).__init__()
        self.enc_self_attn = MultiBranchAttention(d_k, d_v, d_model, d_ff, n_branches, dropout)

    def forward(self, enc_inputs, self_attn_mask):
        return self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, attn_mask=self_attn_mask)


class DecoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.dec_enc_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs, attn_mask=self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs, attn_mask=enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attn, dec_enc_attn


class WeightedDecoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout=0.1):
        super(WeightedDecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_branches, dropout)
        self.dec_enc_attn = MultiBranchAttention(d_k, d_v, d_model, d_ff, n_branches, dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs, attn_mask=self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs, attn_mask=enc_attn_mask)

        return dec_outputs, dec_self_attn, dec_enc_attn


# ####################### Models ##########################
def proj_prob_simplex(inputs):
    # project updated weights onto a probability simplex
    # see https://arxiv.org/pdf/1101.6081.pdf
    sorted_inputs, sorted_idx = torch.sort(inputs.view(-1), descending=True)
    dim = len(sorted_inputs)
    for i in reversed(range(dim)):
        t = (sorted_inputs[:i+1].sum() - 1) / (i+1)
        if sorted_inputs[i] > t:
            break
    return torch.clamp(inputs-t, min=0.0)


def get_attn_pad_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(data_utils.PAD).unsqueeze(1)  # b_size x 1 x len_k
    return pad_attn_mask.expand(b_size, len_q, len_k)  # b_size x len_q x len_k


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask



class TSModel(nn.Module):
    def __init__(self, configs, features_cls, weight_scheme='mw'):

        super(TSModel, self).__init__()
        self.cnn_embedding = ConvEmbeddingLayer(n_features=configs.n_features, d_model=configs.d_model)
        self.encoder = Encoder(configs.n_layers, configs.d_k, configs.d_v, configs.d_model, configs.d_ff, configs.n_heads,
                               configs.max_input_seq_len, configs.src_vocab_size, configs.dropout, configs.weighted_model)
        self.decoder = Decoder(configs.n_layers, configs.d_k, configs.d_v, configs.d_model, configs.d_ff, configs.n_heads,
                               configs.max_output_seq_len, configs.tgt_vocab_size, configs.dropout, configs.weighted_model)
        self.tgt_proj = Linear(configs.d_model, configs.tgt_vocab_size, bias=False)
        self.weighted_model = configs.weighted_model









class Encoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, dropout=0.1, weighted=False):
        super(Encoder, self).__init__()
        self.d_model = d_model
        # self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=data_utils.PAD,)
        self.pos_emb = PosEncoding(max_seq_len * 10, d_model) # TODO: *10 fix
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = EncoderLayer if not weighted else WeightedEncoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, enc_inputs, enc_inputs_len, return_attn=False):
        # enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = enc_inputs + self.pos_emb(enc_inputs_len) # Adding positional encoding TODO: note
        enc_outputs = self.dropout_emb(enc_outputs)

        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            if return_attn:
                enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, dropout=0.1, weighted=False):
        super(Decoder, self).__init__()
        self.d_model = d_model
        # self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=data_utils.PAD, )
        self.pos_emb = PosEncoding(max_seq_len * 10, d_model) # TODO: *10 fix
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = DecoderLayer if not weighted else WeightedDecoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, return_attn=False):
        # dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = dec_inputs + self.pos_emb(dec_inputs_len) # Adding positional encoding # TODO: note
        dec_outputs = self.dropout_emb(dec_outputs)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_pad_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs,
                                                             self_attn_mask=dec_self_attn_mask,
                                                             enc_attn_mask=dec_enc_attn_pad_mask)
            if return_attn:
                dec_self_attns.append(dec_self_attn)
                dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, opt):
        super(Transformer, self).__init__()
        self.encoder = Encoder(opt.n_layers, opt.d_k, opt.d_v, opt.d_model, opt.d_ff, opt.n_heads,
                               opt.max_src_seq_len, opt.src_vocab_size, opt.dropout, opt.weighted_model)
        self.decoder = Decoder(opt.n_layers, opt.d_k, opt.d_v, opt.d_model, opt.d_ff, opt.n_heads,
                               opt.max_tgt_seq_len, opt.tgt_vocab_size, opt.dropout, opt.weighted_model)
        self.tgt_proj = Linear(opt.d_model, opt.tgt_vocab_size, bias=False)
        self.weighted_model = opt.weighted_model

        if opt.share_proj_weight:
            print('Sharing target embedding and projection..')
            self.tgt_proj.weight = self.decoder.tgt_emb.weight

        if opt.share_embs_weight:
            print('Sharing source and target embedding..')
            assert opt.src_vocab_size == opt.tgt_vocab_size, \
                'To share word embeddings, the vocabulary size of src/tgt should be the same'
            self.encoder.src_emb.weight = self.decoder.tgt_emb.weight

    def trainable_params(self):
        # Avoid updating the position encoding
        params = filter(lambda p: p[1].requires_grad, self.named_parameters())
        # Add a separate parameter group for the weighted_model
        param_groups = []
        base_params = {'params': [], 'type': 'base'}
        weighted_params = {'params': [], 'type': 'weighted'}
        for name, param in params:
            if 'w_kp' in name or 'w_a' in name:
                weighted_params['params'].append(param)
            else:
                base_params['params'].append(param)
        param_groups.append(base_params)
        param_groups.append(weighted_params)

        return param_groups

    def encode(self, enc_inputs, enc_inputs_len, return_attn=False):
        return self.encoder(enc_inputs, enc_inputs_len, return_attn)

    def decode(self, dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, return_attn=False):
        return self.decoder(dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, return_attn)

    def forward(self, enc_inputs, enc_inputs_len, dec_inputs, dec_inputs_len, return_attn=False):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, enc_inputs_len, return_attn)
        dec_outputs, dec_self_attns, dec_enc_attns = \
            self.decoder(dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, return_attn)
        dec_logits = self.tgt_proj(dec_outputs)

        return dec_logits.view(-1, dec_logits.size(-1)), \
               enc_self_attns, dec_self_attns, dec_enc_attns

    def proj_grad(self):
        if self.weighted_model:
            for name, param in self.named_parameters():
                if 'w_kp' in name or 'w_a' in name:
                    param.data = proj_prob_simplex(param.data)
        else:
            pass



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
    def __init__(self, dim_out, num_units, out_activation='linear'):
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

        self.num_layers = num_layers

        self.enc_layers = dict()
        for i in range(num_layers):
            self.enc_layers['multihead_attn_' + str(i)] = MultiHeadAttention(model_hidden_size, heads)
            self.enc_layers['ff_' + str(i)] = FeedForward(dim_input, ffn_hidden_size)

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

        self.num_layers = num_layers

        self.dec_layers = dict()
        for i in range(num_layers):
            self.dec_layers['masked_multihead_attn_' + str(i)] = MultiHeadAttention(model_hidden_size, heads)
            self.dec_layers['multihead_attn_' + str(i)] = MultiHeadAttention(model_hidden_size, heads)
            self.dec_layers['ff_' + str(i)] = FeedForward(dim_input, ffn_hidden_size)

        self.logit_layer = Dense(dim_output)

    def call(self, inputs, encoder_outputs, dropout=0.2):
        x = inputs
        for i in range(self.num_layers):
            x = sublayer_connection(x, self.dec_layers['masked_multihead_attn_' + str(i)](x, x, x, masked=True), dropout=dropout)
            x = sublayer_connection(x, self.dec_layers['multihead_attn_' + str(i)](x, encoder_outputs, encoder_outputs), dropout=dropout)
            x = sublayer_connection(x, self.dec_layers['ff_' + str(i)](x), dropout=dropout)

        return self.logit_layer(x)


class TSModel:
    """omit embedding time series. just 1-D data used"""
    def __init__(self, configs, feature_cls, weight_scheme='ew'):
        self.weight_scheme = weight_scheme

        self.input_seq_size = configs.m_days // configs.sampling_days
        # self.output_seq_size = configs.k_days // configs.sampling_days
        self.output_seq_size = 1
        self.position_encode_in = positional_encoding(configs.embedding_size, self.input_seq_size)
        self.position_encode_out = positional_encoding(configs.embedding_size, self.output_seq_size)

        self.encoder = Encoder(dim_input=configs.embedding_size,
                               model_hidden_size=configs.model_hidden_size,
                               ffn_hidden_size=configs.ffn_hidden_size,
                               heads=configs.attention_head_size,
                               num_layers=configs.layer_size)

        self.decoder = Decoder(dim_input=configs.embedding_size,
                               dim_output=configs.embedding_size,
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
                    self.predictor[key] = FeedForward(len(configs.features_structure['regression'][key]), n_size, out_activation='sigmoid')
                else:
                    self.predictor[key] = FeedForward(len(configs.features_structure['regression'][key]), n_size)
            elif tags[0] in configs.features_structure['classification'].keys():
                self.predictor[key] = FeedForward(2, n_size, out_activation='softmax')
                self.predictor_helper[key] = configs.features_structure['regression']['logy'].index(int(tags[1]))
            # elif tags[0] in configs.features_structure['crosssection'].keys():
            #     self.predictor[key] = FeedForward(len(configs.features_structure['regression'][key]), 64)

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
        enc_temp = self.encoder(feature_temp)
        dec_temp = self.decoder(feature_temp, enc_temp)

        for key in self.predictor.keys():
            _ = self.predictor[key](dec_temp)
            self.optim_predictor_w[key] = self.predictor[key].get_weights()

        self.optim_encoder_w = self.encoder.get_weights()
        self.optim_decoder_w = self.decoder.get_weights()

        self._reset_eval_param()

    def weight_to_optim(self):
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
            x_embed = features['input'] + self.position_encode_in
            y_embed = features['output'] + self.position_encode_out

            encoder_output = self.encoder(x_embed, dropout=self.dropout_train)
            predict = self.decoder(y_embed, encoder_output, dropout=self.dropout_train)

            var_lists = self.encoder.trainable_variables + self.decoder.trainable_variables

            pred_each = dict()
            loss_each = dict()
            loss = None
            for key in self.predictor.keys():
                pred_each[key] = self.predictor[key](predict)
                var_lists += self.predictor[key].trainable_variables

                if self.weight_scheme == 'mw':
                    adj_weight = labels_mtl['size_value'][:, :, 0] * 2.  # size value 평균이 0.5 이므로 기존이랑 스케일 맞추기 위해 2 곱

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

            # if 'cslogy' in labels_mtl.keys():
            #     cs_loc = np.stack([features['output'][:, :, idx] for idx in labels_mtl['cslogy_idx']], axis=-1)
            #     loss_each['cs_loc'] = tf.losses.MSE(cs_loc, pred_each['cslogy']) * adj_weight * 0.1
            #     loss += loss_each['cs_loc']

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
        for i, (features, labels, size_values, importance_wgt) in enumerate(datasets.take(steps)):
            labels_mtl = self.feature_cls.labels_for_mtl(features_list, labels, size_values, importance_wgt)

            x_embed = features['input'] + self.position_encode_in
            y_embed = features['output'] + self.position_encode_out

            encoder_output = self.encoder(x_embed, dropout=0.)
            predict = self.decoder(y_embed, encoder_output, dropout=0.)

            var_lists = self.encoder.trainable_variables + self.decoder.trainable_variables
            pred_each = dict()
            loss_each = dict()
            loss = None
            for key in self.predictor.keys():
                pred_each[key] = self.predictor[key](predict)
                var_lists += self.predictor[key].trainable_variables

                if self.weight_scheme == 'mw':
                    adj_weight = labels_mtl['size_value'][:, :, 0] * 2.
                else:
                    adj_weight = 1.

                # adj_importance = labels_mtl['importance_wgt']
                if key[:3] == 'pos':
                    loss_each[key] = tf.losses.categorical_crossentropy(labels_mtl[key], pred_each[key]) \
                                     * tf.abs(labels_mtl['logy'][:, :, self.predictor_helper[key]]) \
                                     * adj_weight # * adj_importance
                else:
                    loss_each[key] = tf.losses.MSE(labels_mtl[key], pred_each[key]) * adj_weight #* adj_importance

                # if key == 'pos':
                #     loss_each[key] = tf.losses.categorical_crossentropy(labels_mtl[key], pred_each[key]) * tf.abs(labels_mtl['ret'][:, :, 0])
                # elif key == 'pos20':
                #     loss_each[key] = tf.losses.categorical_crossentropy(labels_mtl[key], pred_each[key]) * tf.abs(
                #         labels_mtl['ret'][:, :, 1])
                # else:
                #     loss_each[key] = tf.losses.MSE(labels_mtl[key], pred_each[key])

                if loss is None:
                    loss = loss_each[key]
                else:
                    loss += loss_each[key]

            # if 'cslogy_idx' in labels_mtl.keys():
            #     cs_loc = np.stack([features['output'][:, :, idx] for idx in labels_mtl['cslogy_idx']], axis=-1)
            #     loss_each['cs_loc'] = tf.losses.MSE(cs_loc, pred_each['cslogy']) * adj_weight * 0.1
            #     loss += loss_each['cs_loc']

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

        x_embed = feature['input'] + self.position_encode_in
        y_embed = feature['output'] + self.position_encode_out

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

        self.optim_encoder_w = w_dict['encoder']
        self.optim_decoder_w = w_dict['decoder']
        self.optim_predictor_w = w_dict['predictor']

        self.encoder.set_weights(self.optim_encoder_w)
        self.decoder.set_weights(self.optim_decoder_w)
        for key in self.optim_predictor_w.keys():
            self.predictor[key].set_weights(self.optim_predictor_w[key])

        print("model loaded. (path: {})".format(f_name))
