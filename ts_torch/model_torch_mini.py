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

# # #### profiler start ####
import builtins

try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile
# # #### profiler end ####


# ####################### Module ##########################
class Constant:
    def __init__(self):
        self.PAD = -9999

const = Constant()


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()

    def params2vec(self, requires_grad_only=True):
        if requires_grad_only:
            paramsvec = torch.nn.ParameterList()
            for p in self.parameters():
                if p.requires_grad:
                    paramsvec.append(p)
        else:
            paramsvec = torch.nn.ParameterList(self.parameters())

        return paramsvec

    def load_from_vecs(self, paramsvec, requires_grad_only=True):
        dic = self.state_dict()
        i = 0
        for key in dic.keys():
            if requires_grad_only and not dic[key].requires_grad:
                continue

            dic[key] = paramsvec[i]
            i += 1

        self.load_state_dict(dic)

    def split_wgt(self, weights_list, pos, len):
        next_pos = pos + len
        return weights_list[pos:next_pos], next_pos

    def get_children_dict(self, weights_list, requires_grad_only=True):
        self.debug(weights_list, requires_grad_only)

        children_dict = dict()
        params_pos = 0
        for c in self.named_children():
            # c: Tuple(name, module)
            if type(c[1]) == nn.ModuleList:
                wgt = []
                for l in c[1]:
                    if not hasattr(l, 'params2vec'):
                        continue
                    params_len = len(l.params2vec(requires_grad_only))
                    wgt_i, params_pos = self.split_wgt(weights_list, params_pos, params_len)
                    wgt.append(wgt_i)
            elif type(c[1]) == nn.ModuleDict:
                wgt = dict()
                for key in c[1].keys():
                    if not hasattr(c[1][key], 'params2vec'):
                        continue
                    params_len = len(c[1][key].params2vec(requires_grad_only))
                    wgt_i, params_pos = self.split_wgt(weights_list, params_pos, params_len)
                    wgt[key] = wgt_i
            else:
                if not hasattr(c[1], 'params2vec'):
                    continue
                params_len = len(c[1].params2vec(requires_grad_only))
                wgt, params_pos = self.split_wgt(weights_list, params_pos, params_len)

            # model, weights_list 순서
            children_dict[c[0]] = dict(m=c[1], w=wgt)

        return children_dict

    def debug(self, weights_list, requires_grad_only=True):
        assert len(weights_list) == len(self.params2vec(requires_grad_only)), "size of weights_list not matched. ({}/{})".format(len(weights_list), len(self.params2vec(requires_grad_only)))

    def compute_graph(self, inputs, weights_list=[]):
        if len(weights_list) == 0:
            return self.forward(inputs)


class Conv2d(nn.Conv2d, Base):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(*args, **kwargs)

    def compute_graph(self, inputs, weights_list):
        self.debug(weights_list)

        return F.conv2d(inputs, weight=weights_list[0], bias=weights_list[1], stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)


class Conv1d(nn.Conv1d, Base):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)

    def compute_graph(self, inputs, weights_list):
        self.debug(weights_list)

        return F.conv1d(inputs, weight=weights_list[0], bias=weights_list[1], stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)


class Linear(Base):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)

    def compute_graph(self, inputs, weights_list):
        self.debug(weights_list)
        return F.linear(inputs, weight=weights_list[0], bias=weights_list[1])


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


class LayerNormalization(Base):
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

    def compute_graph(self, z, weights_list):
        self.debug(weights_list)

        gamma, beta = weights_list[0], weights_list[1]
        mean = z.mean(dim=-1, keepdim=True,)
        std = z.std(dim=-1, keepdim=True,)
        ln_out = (z - mean) / (std + self.eps)
        ln_out = gamma * ln_out + beta

        return ln_out


class PosEncoding(Base):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PosEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #.transpose(0, 1)
        self.register_buffer('pe', pe)

    # @profile
    def forward(self, x):
        # x shape: [batch, len, d_model]
        x = x + self.pe[:, x.size(1), :]
        return self.dropout(x)


# class PosEncoding(Base):
#     def __init__(self, d_word_vec, max_len):
#         super(PosEncoding, self).__init__()
#         pos_enc = np.array(
#             [[pos / np.power(10000, 2.0 * (j // 2) / d_word_vec) for j in range(d_word_vec)]
#             for pos in range(max_len)])
#         pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
#         pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
#         pad_row = np.zeros([1, d_word_vec])
#         pos_enc = np.concatenate([pad_row, pos_enc]).astype(np.float32)
#
#         # additional single row for PAD idx
#         self.pos_enc = nn.Embedding(max_len + 1, d_word_vec, sparse=True)
#         # fix positional encoding: exclude weight from grad computation
#         self.pos_enc.weight = nn.Parameter(torch.from_numpy(pos_enc), requires_grad=False)
#
#     def forward(self, input_len):
#         max_len = torch.max(input_len)
#         tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
#         # print("is cuda:{}".format(input_len.is_cuda))
#         # input_pos = torch.LongTensor([list(range(1, input_len+1))])
#         input_pos = tensor([list(range(1, int(len)+1)) + [0]*int(max_len-len) for len in input_len])
#         if input_len.is_cuda:
#             self.pos_enc.cuda()
#
#         return self.pos_enc(input_pos)
#
#     # def compute_graph(self, input_len, weights_list):
#     #     self.debug(weights_list)
#     #
#     #     max_len = torch.max(input_len)
#     #     tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
#     #     # print("is cuda:{}".format(input_len.is_cuda))
#     #     # input_pos = torch.LongTensor([list(range(1, input_len+1))])
#     #     input_pos = tensor([list(range(1, int(len)+1)) + [0]*int(max_len-len) for len in input_len])
#     #
#     #     return F.embedding(input_pos, weight=weights_list[0])


# ####################### Sublayers ##########################

class _MultiHeadAttention(Base):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = Linear(d_model, d_k * n_heads)
        self.w_k = Linear(d_model, d_k * n_heads)
        self.w_v = Linear(d_model, d_v * n_heads)

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

        if attn_mask is not None:  # attn_mask: [b_size x len_q x len_k]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # context: [b_size x n_heads x len_q x d_v], attn: [b_size x n_heads x len_q x len_k]
        context, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)
        # context: [b_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)

        # return the context and attention weights
        return context, attn

    def compute_graph(self, q, k, v, attn_mask, weights_list):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_k x d_model]
        b_size = q.size(0)

        c_dict = self.get_children_dict(weights_list)
        w_q = c_dict['w_q']
        w_k = c_dict['w_k']
        w_v = c_dict['w_v']
        # q_s: [b_size x n_heads x len_q x d_k]
        # k_s: [b_size x n_heads x len_k x d_k]
        # v_s: [b_size x n_heads x len_k x d_v]
        q_s = w_q['m'].compute_graph(q, w_q['w']).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = w_k['m'].compute_graph(k, w_k['w']).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = w_v['m'].compute_graph(v, w_v['w']).view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask is not None:  # attn_mask: [b_size x len_q x len_k]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # context: [b_size x n_heads x len_q x d_v], attn: [b_size x n_heads x len_q x len_k]
        context, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)
        # context: [b_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)

        # return the context and attention weights
        return context, attn


class MultiHeadAttention(Base):
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

    def compute_graph(self, q, k, v, attn_mask, weights_list):
        c_dict = self.get_children_dict(weights_list)
        multihead_attn = c_dict['multihead_attn']
        proj = c_dict['proj']
        layer_norm = c_dict['layer_norm']
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q
        # context: a tensor of shape [b_size x len_q x n_heads * d_v]
        context, attn = multihead_attn['m'].compute_graph(q, k, v, attn_mask=attn_mask, weights_list=multihead_attn['w'])

        # project back to the residual size, outputs: [b_size x len_q x d_model]
        output = self.dropout(proj['m'].compute_graph(context, weights_list=proj['w']))
        return layer_norm['m'].compute_graph(residual + output, weights_list=layer_norm['w']), attn


# NOTE: error ?!
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

    def compute_graph(self, q, k, v, attn_mask, weights_list):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q

        c_dict = self.get_children_dict(weights_list)
        multihead_attn, = c_dict['multihead_attn']
        proj = c_dict['proj']

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


class FeedForward(Base):
    def __init__(self, d_in, d_ff, d_out, out_activation=None):
        super(FeedForward, self).__init__()
        self.relu = nn.ReLU()
        self.in_layer = Linear(d_in, d_ff)
        self.out_layer = Linear(d_ff, d_out)

        if out_activation == 'sigmoid':
            self.out_a = nn.Sigmoid()
        elif out_activation == 'softmax':
            self.out_a = nn.Softmax()
        elif out_activation == 'positive':
            self.out_a = (lambda x: torch.log(1 + torch.exp(x)) + 1e-06)
        else:
            self.out_a = None

    def forward(self, inputs):
        # inputs: [b_size x d_model]
        output = self.relu(self.in_layer(inputs))
        output = self.out_layer(output)

        if self.out_a is not None:
            output = self.out_a(output)

        return output

    def compute_graph(self, inputs, weights_list):
        c_dict = self.get_children_dict(weights_list)
        in_layer = c_dict['in_layer']
        out_layer = c_dict['out_layer']
        # inputs: [b_size x d_model]
        output = self.relu(in_layer['m'].compute_graph(inputs, weights_list=in_layer['w']))
        output = out_layer['m'].compute_graph(output, weights_list=out_layer['w'])

        if self.out_a is not None:
            output = self.out_a(output)

        return output


class PoswiseFeedForwardNet(Base):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
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

    def compute_graph(self, inputs, weights_list):
        c_dict = self.get_children_dict(weights_list)
        conv1 = c_dict['conv1']
        conv2 = c_dict['conv2']
        layer_norm = c_dict['layer_norm']

        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.relu(conv1['m'].compute_graph(inputs.transpose(1, 2), weights_list=conv1['w']))

        # outputs: [b_size x len_q x d_model]
        output = conv2['m'].compute_graph(output, weights_list=conv2['w']).transpose(1, 2)
        output = self.dropout(output)

        return layer_norm['m'].compute_graph(residual + output, weights_list=layer_norm['w'])


# ####################### Layers ##########################
class ConvEmbeddingLayer(Base):
    # input features의 수를 d_model만큼의 1d conv로 재생산

    def __init__(self, n_features, d_model):
        super(ConvEmbeddingLayer, self).__init__()
        self.conv1 = Conv1d(in_channels=n_features, out_channels=d_model-n_features, kernel_size=1)

    def forward(self, inputs):
        # input shape: (b_size, T, n_features)
        inputs_t = inputs.contiguous().transpose(-2, -1)
        # (b_size, n_features, T) -> (b_size, T, d_model)
        outputs = self.conv1(inputs_t).contiguous().transpose(-2, -1)
        # (b_size, d_model, T) -> (b_size, T, d_model)
        outputs = torch.cat((inputs, outputs), axis=-1)
        return outputs

    def compute_graph(self, inputs, weights_list):
        c_dict = self.get_children_dict(weights_list)
        conv1 = c_dict['conv1']

        # input shape: (b_size, T, n_features)
        inputs_t = inputs.contiguous().transpose(-2, -1)
        # (b_size, n_features, T) -> (b_size, T, d_model)
        outputs = conv1['m'].compute_graph(inputs_t, conv1['w']).contiguous().transpose(-2, -1)
        # (b_size, d_model, T) -> (b_size, T, d_model)
        outputs = torch.cat((inputs, outputs), axis=-1)
        return outputs


class EncoderLayer(Base):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_inputs, self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs,
                                               enc_inputs, attn_mask=self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn

    def compute_graph(self, enc_inputs, self_attn_mask, weights_list):
        c_dict = self.get_children_dict(weights_list)
        enc_self_attn = c_dict['enc_self_attn']
        pos_ffn = c_dict['pos_ffn']

        enc_outputs, attn = enc_self_attn['m'].compute_graph(enc_inputs, enc_inputs, enc_inputs
                                                             , attn_mask=self_attn_mask
                                                             , weights_list=enc_self_attn['w'])
        enc_outputs = pos_ffn['m'].compute_graph(enc_outputs, weights_list=pos_ffn['w'])

        return enc_outputs, attn


class WeightedEncoderLayer(Base):
    def __init__(self, d_k, d_v, d_model, d_ff, n_branches, dropout=0.1):
        super(WeightedEncoderLayer, self).__init__()
        self.enc_self_attn = MultiBranchAttention(d_k, d_v, d_model, d_ff, n_branches, dropout)

    def forward(self, enc_inputs, self_attn_mask):
        return self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, attn_mask=self_attn_mask)

    def compute_graph(self, enc_inputs, self_attn_mask, weights_list):
        c_dict = self.get_children_dict(weights_list)
        enc_self_attn = c_dict['enc_self_attn']
        return enc_self_attn['m'].compute_graph(enc_inputs, enc_inputs, enc_inputs
                                                , attn_mask=self_attn_mask
                                                , weights_list=enc_self_attn['w'])


class DecoderLayer(Base):
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

    def compute_graph(self, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask, weights_list):
        c_dict = self.get_children_dict(weights_list)
        dec_self_attn = c_dict['dec_self_attn']
        dec_enc_attn = c_dict['dec_enc_attn']
        pos_ffn = c_dict['pos_ffn']

        dec_outputs, dec_self_attn = dec_self_attn['m'].compute_graph(dec_inputs, dec_inputs, dec_inputs
                                                                      , attn_mask=self_attn_mask, weights_list=dec_self_attn['w'])
        dec_outputs, dec_enc_attn = dec_enc_attn['m'].compute_graph(dec_outputs, enc_outputs, enc_outputs
                                                                    , attn_mask=enc_attn_mask, weights_list=dec_enc_attn['w'])
        dec_outputs = pos_ffn['m'].compute_graph(dec_outputs, weights_list=pos_ffn['w'])

        return dec_outputs, dec_self_attn, dec_enc_attn


class WeightedDecoderLayer(Base):
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

    def compute_graph(self, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask, weights_list):
        c_dict = self.get_children_dict(weights_list)
        dec_self_attn = c_dict['dec_self_attn']
        dec_enc_attn = c_dict['dec_enc_attn']

        dec_outputs, dec_self_attn = dec_self_attn['m'].compute_graph(dec_inputs, dec_inputs, dec_inputs
                                                                      , attn_mask=self_attn_mask
                                                                      , weights_list=dec_self_attn['w'])
        dec_outputs, dec_enc_attn = dec_enc_attn['m'].compute_graph(dec_outputs, enc_outputs, enc_outputs
                                                                    , attn_mask=enc_attn_mask
                                                                    , weights_list=dec_enc_attn['w'])

        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(Base):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, dropout=0.1, weighted=False):
        # n_layers, d_k, d_v, d_model, d_ff, n_heads, max_seq_len, dropout, weighted = configs.n_layers, configs.d_k, configs.d_v, configs.d_model, configs.d_ff, configs.n_heads, configs.max_input_seq_len, configs.dropout, configs.weighted_model
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.pos_emb = PosEncoding(d_model, dropout=dropout, max_len=max_seq_len * 2)  # TODO: *10 fix
        # self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = EncoderLayer if not weighted else WeightedEncoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    # @profile
    def forward(self, enc_inputs, enc_inputs_len, return_attn=False):
        # enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_inputs)  # Adding positional encoding TODO: note
        # enc_outputs = enc_inputs + self.pos_emb(enc_inputs_len)  # Adding positional encoding TODO: note
        # enc_outputs = self.dropout_emb(enc_outputs)

        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            if return_attn:
                enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns

    def compute_graph(self, enc_inputs, enc_inputs_len, weights_list, return_attn=False):
        c_dict = self.get_children_dict(weights_list)
        pos_emb = c_dict['pos_emb']
        layers = c_dict['layers']

        # enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = enc_inputs + pos_emb['m'].compute_graph(enc_inputs_len, weights_list=pos_emb['w'])  # Adding positional encoding TODO: note
        enc_outputs = self.dropout_emb(enc_outputs)

        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer_m, layer_w in zip(layers['m'], layers['w']):
            enc_outputs, enc_self_attn = layer_m.compute_graph(enc_outputs, enc_self_attn_mask, weights_list=layer_w)
            if return_attn:
                enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class Decoder(Base):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, dropout=0.1, weighted=False):
        super(Decoder, self).__init__()
        self.d_model = d_model
        # self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=data_utils.PAD, )
        self.pos_emb = PosEncoding(d_model, dropout=dropout, max_len=max_seq_len * 2)  # TODO: *10 fix
        # self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = DecoderLayer if not weighted else WeightedDecoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    # @profile
    def forward(self, dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, return_attn=False):
        # dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_inputs)  # Adding positional encoding TODO: note

        # dec_outputs = dec_inputs + self.pos_emb(dec_inputs_len)  # Adding positional encoding # TODO: note
        # dec_outputs = self.dropout_emb(dec_outputs)

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

    def compute_graph(self, dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, weights_list, return_attn=False):
        c_dict = self.get_children_dict(weights_list)
        pos_emb = c_dict['pos_emb']
        layers = c_dict['layers']

        # dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = dec_inputs + pos_emb['m'].compute_graph(dec_inputs_len, weights_list=pos_emb['w'])  # Adding positional encoding # TODO: note
        dec_outputs = self.dropout_emb(dec_outputs)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_pad_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer_m, layer_w in zip(layers['m'], layers['w']):
            dec_outputs, dec_self_attn, dec_enc_attn = layer_m.compute_graph(dec_outputs, enc_outputs,
                                                             self_attn_mask=dec_self_attn_mask,
                                                             enc_attn_mask=dec_enc_attn_pad_mask, weights_list=layer_w)
            if return_attn:
                dec_self_attns.append(dec_self_attn)
                dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


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
    assert seq_q.dim() == 3 and seq_k.dim() == 3
    b_size, len_q = seq_q.size()[:2]
    b_size, len_k = seq_k.size()[:2]
    # 모든 features 값의 합이 PAD보다 작으면 PAD로 가정
    pad_attn_mask = torch.sum(seq_k.data, axis=-1).le(const.PAD).unsqueeze(1)  # b_size x 1 x len_k
    return pad_attn_mask.expand(b_size, len_q, len_k)  # b_size x len_q x len_k


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 3
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).bool()
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask


# d_k, d_v, d_model, d_ff, n_heads, dropout
class TSModel(Base):
    def __init__(self, configs, features_cls, weight_scheme='mw'):
        super(TSModel, self).__init__()
        c = configs

        self.use_uncertainty = c.use_uncertainty
        self.weight_scheme = weight_scheme

        self.input_seq_size = c.m_days // c.sampling_days + 1
        # self.output_seq_size = configs.k_days // configs.sampling_days
        self.output_seq_size = 1

        self.conv_embedding = ConvEmbeddingLayer(n_features=c.embedding_size, d_model=c.d_model)
        self.encoder = Encoder(c.n_layers, c.d_k, c.d_v, c.d_model, c.d_ff,
                               c.n_heads, c.max_input_seq_len, c.dropout, c.weighted_model)
        self.decoder = Decoder(c.n_layers, c.d_k, c.d_v, c.d_model, c.d_ff,
                               c.n_heads, c.max_output_seq_len, c.dropout, c.weighted_model)
        self.weighted_model = c.weighted_model

        self.predictor = nn.ModuleDict()
        if self.use_uncertainty:
            self.suffix_var = '-var'
            self.predictor_var = nn.ModuleDict()
        self.predictor_helper = dict()
        n_size = 64
        for key in c.model_predictor_list:
            tags = key.split('_')
            for cls in c.features_structure.keys():
                for arr_base in c.features_structure[cls].keys():
                    if tags[0] in c.features_structure[cls][arr_base].keys():
                        if cls == 'regression':
                            if key in ['cslogy', 'csstd']:
                                self.predictor[key] = FeedForward(c.d_model, n_size, 1, out_activation='sigmoid')
                                if c.use_uncertainty:
                                    self.predictor_var[key] = FeedForward(c.d_model, n_size, 1, out_activation='positive')
                                # self.predictor[key] = FeedForward(c.d_model, n_size, len(c.features_structure['regression'][key]), out_activation='sigmoid')
                            else:
                                self.predictor[key] = FeedForward(c.d_model, n_size, 1, out_activation='linear')
                                if c.use_uncertainty:
                                    self.predictor_var[key] = FeedForward(c.d_model, n_size, 1, out_activation='positive')
                                # self.predictor[key] = FeedForward(c.d_model, n_size, len(c.features_structure['regression'][key]))
                        else:
                            self.predictor[key] = FeedForward(c.d_model, n_size, 2, out_activation='linear')
                            if c.use_uncertainty:
                                self.predictor_var[key] = FeedForward(c.d_model, n_size, 2, out_activation='positive')
                            self.predictor_helper[key] = c.features_structure['regression']['logp_base']['logy'].index(c.k_days)
                        # elif tags[0] in configs.features_structure['crosssection'].keys():
                        #     self.predictor[key] = FeedForward(64, len(configs.features_structure['regression'][key]))

        self.features_cls = features_cls

        self.optim_state_dict = self.state_dict()
        self.dropout_train = c.dropout
        if c.use_maml:
            self.inner_lr = c.inner_lr

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

    def encode(self, enc_inputs, return_attn=False):
        return self.encoder(enc_inputs, self.input_seq_size, return_attn)

    def decode(self, dec_inputs, enc_inputs, enc_outputs, return_attn=False):
        return self.decoder(dec_inputs, self.output_seq_size, enc_inputs, enc_outputs, return_attn)

    def forward(self, features, return_attn=False):
        # features = {'input': torch.zeros(2, 25, 23), 'output': torch.zeros(2, 1, 23)}
        device = features['input'].device
        # self.to(device)

        enc_in = self.conv_embedding(features['input'])
        dec_in = self.conv_embedding(features['output'])

        input_seq_size = torch.Tensor([enc_in.shape[1] for _ in range(enc_in.shape[0])]).to(device)
        output_seq_size = torch.Tensor([dec_in.shape[1] for _ in range(dec_in.shape[0])]).to(device)

        enc_out, enc_self_attns = self.encoder(enc_in, input_seq_size, return_attn)
        predict, dec_self_attns, dec_enc_attns = self.decoder(dec_in, output_seq_size, enc_in, enc_out, return_attn)

        pred_each = dict()
        for key in self.predictor.keys():
            pred_each[key] = self.predictor[key](predict)
            if self.use_uncertainty:
                pred_each[key+self.suffix_var] = self.predictor_var[key](predict)

        return pred_each, enc_self_attns, dec_self_attns, dec_enc_attns

    def compute_graph(self, features, weights_list, return_attn=False):
        # features = {'input': torch.zeros(2, 25, 23), 'output': torch.zeros(2, 1, 23)}
        device = features['input'].device
        self.to(device)

        c_dict = self.get_children_dict(weights_list)
        conv_embedding = c_dict['conv_embedding']
        encoder = c_dict['encoder']
        decoder = c_dict['decoder']
        predictor = c_dict['predictor']
        if self.use_uncertainty:
            predictor_var = c_dict['predictor_var']

        enc_in = conv_embedding['m'].compute_graph(features['input'], weights_list=conv_embedding['w'])
        dec_in = conv_embedding['m'].compute_graph(features['output'], weights_list=conv_embedding['w'])

        input_seq_size = torch.Tensor([enc_in.shape[1] for _ in range(enc_in.shape[0])]).to(device)
        output_seq_size = torch.Tensor([dec_in.shape[1] for _ in range(dec_in.shape[0])]).to(device)

        enc_out, enc_self_attns = encoder['m'].compute_graph(enc_in, input_seq_size, return_attn=return_attn, weights_list=encoder['w'])
        predict, dec_self_attns, dec_enc_attns = decoder['m'].compute_graph(dec_in, output_seq_size, enc_in, enc_out, return_attn=return_attn, weights_list=decoder['w'])

        pred_each = dict()
        for key in predictor['m'].keys():
            pred_each[key] = predictor['m'][key].compute_graph(predict, weights_list=predictor['w'][key])
            if self.use_uncertainty:
                pred_each[key+self.suffix_var] = predictor_var['m'][key].compute_graph(predict, weights_list=predictor_var['w'][key])

        return pred_each #, enc_self_attns, dec_self_attns, dec_enc_attns

    def predict_mtl(self, features):
        ret = self.forward(features)
        return ret[0]

    # @profile
    def forward_with_loss(self, features, labels_mtl, return_attn=False):
        # features = {'input': torch.zeros(2, 25, 23), 'output': torch.zeros(2, 1, 23)}
        device = features['input'].device
        self.to(device)

        enc_in = self.conv_embedding(features['input'])
        dec_in = self.conv_embedding(features['output'])

        input_seq_size = torch.Tensor([enc_in.shape[1] for _ in range(enc_in.shape[0])]).to(device)
        output_seq_size = torch.Tensor([dec_in.shape[1] for _ in range(dec_in.shape[0])]).to(device)

        enc_out, enc_self_attns = self.encoder(enc_in, input_seq_size, return_attn)
        predict, dec_self_attns, dec_enc_attns = self.decoder(dec_in, output_seq_size, enc_in, enc_out, return_attn)

        if self.weight_scheme == 'mw':
            # adj_weight = torch.exp(labels_mtl['size_rnk'] * 2.) / torch.sum(torch.exp(labels_mtl['size_rnk'] * 2.))   # size_rnk: 0~1사이 랭크
            adj_weight = labels_mtl['size_rnk'] * 2.
        else:
            adj_weight = torch.ones_like(labels_mtl['size_rnk'])

        if labels_mtl.get('importance_wgt') is not None:  # TODO: MAML에서 정의 안됨
            adj_importance = labels_mtl['importance_wgt']
        else:
            adj_importance = 1.

        pred_each = dict()
        loss_each = dict()
        for key in self.predictor.keys():
            pred_each[key] = self.predictor[key](predict)
            if self.use_uncertainty:
                pred_each[key+self.suffix_var] = self.predictor_var[key](predict)

            if key[:3] == 'pos':
                if self.use_uncertainty:  # TODO : uncertainty for classification
                    criterion = torch.nn.CrossEntropyLoss(reduction='none')
                else:
                    criterion = torch.nn.CrossEntropyLoss(reduction='none')

                pred_each[key] = pred_each[key].contiguous().transpose(1, -1)
                loss_each[key] = criterion(pred_each[key], labels_mtl[key])
                adj_logy = torch.abs(labels_mtl['logy'][:, :, self.predictor_helper[key]])
                assert loss_each[key].shape == adj_logy.shape
                loss_each[key] = loss_each[key] * adj_logy
            else:
                if self.use_uncertainty:
                    criterion = 0.5 * torch.log(pred_each[key+self.suffix_var]) + torch.pow(labels_mtl[key]-pred_each[key], 2) / (2 * pred_each[key+self.suffix_var]) + 10
                else:
                    criterion = torch.nn.MSELoss(reduction='none')

                loss_each[key] = criterion(pred_each[key], labels_mtl[key]).mean(axis=-1)

            loss_shape = loss_each[key].shape

            if labels_mtl.get('importance_wgt') is not None:  # TODO: MAML에서 정의 안됨
                loss_each[key] = loss_each[key] * adj_weight.reshape(loss_shape) * adj_importance.reshape(loss_shape)
            else:
                loss_each[key] = loss_each[key] * adj_weight.reshape(loss_shape)

        return pred_each, loss_each #, enc_self_attns, dec_self_attns, dec_enc_attns,

    def compute_graph_with_loss(self, features, labels_mtl, weights_list, return_attn=False):
        # features = {'input': torch.zeros(2, 25, 23), 'output': torch.zeros(2, 1, 23)}
        device = features['input'].device
        self.to(device)

        c_dict = self.get_children_dict(weights_list)
        conv_embedding = c_dict['conv_embedding']
        encoder = c_dict['encoder']
        decoder = c_dict['decoder']
        predictor = c_dict['predictor']

        enc_in = conv_embedding['m'].compute_graph(features['input'], weights_list=conv_embedding['w'])
        dec_in = conv_embedding['m'].compute_graph(features['output'], weights_list=conv_embedding['w'])

        input_seq_size = torch.Tensor([enc_in.shape[1] for _ in range(enc_in.shape[0])]).to(device)
        output_seq_size = torch.Tensor([dec_in.shape[1] for _ in range(dec_in.shape[0])]).to(device)

        enc_out, enc_self_attns = encoder['m'].compute_graph(enc_in, input_seq_size, return_attn=return_attn, weights_list=encoder['w'])
        predict, dec_self_attns, dec_enc_attns = decoder['m'].compute_graph(dec_in, output_seq_size, enc_in, enc_out, return_attn=return_attn, weights_list=decoder['w'])

        if self.weight_scheme == 'mw':
            adj_weight = labels_mtl['size_rnk'] * 2.  # size_rnk: 0~1사이 랭크
        else:
            adj_weight = 1.

        # adj_importance = labels_mtl['importance_wgt']   # TODO: maml 시 importance 현재 정의 안됨

        pred_each = dict()
        loss_each = dict()
        for key in predictor['m'].keys():
            pred_each[key] = predictor['m'][key].compute_graph(predict, weights_list=predictor['w'][key])

            if key[:3] == 'pos':
                criterion = torch.nn.CrossEntropyLoss(reduction='none')
                pred_each[key] = pred_each[key].contiguous().transpose(1, -1)
                loss_each[key] = criterion(pred_each[key], labels_mtl[key])
                adj_logy = torch.abs(labels_mtl['logy'][:, :, self.predictor_helper[key]])
                assert loss_each[key].shape == adj_logy.shape
                loss_each[key] = loss_each[key] * adj_logy
            else:
                criterion = torch.nn.MSELoss(reduction='none')
                loss_each[key] = criterion(pred_each[key], labels_mtl[key]).mean(axis=-1)

            loss_shape = loss_each[key].shape
            loss_each[key] = loss_each[key] * adj_weight.reshape(loss_shape)  # * adj_importance.reshape(loss_shape) # TODO: maml 시 importance 현재 정의 안됨

        return pred_each, loss_each #, enc_self_attns, dec_self_attns, dec_enc_attns,

    def fast_predict(self, features_s, labels_s, features_t):
        weights_list = self.params2vec(requires_grad_only=True)
        pred_s, loss_each_s = self.compute_graph_with_loss(features_s, labels_s, weights_list=weights_list)

        train_losses = 0
        for key in loss_each_s.keys():
            train_losses += loss_each_s[key].mean()

        grad = torch.autograd.grad(train_losses, weights_list, retain_graph=True, create_graph=True)
        fast_weights = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, weights_list)))

        pred_t = self.compute_graph(features_t, weights_list=fast_weights)

        return pred_t

    def proj_grad(self):
        if self.weighted_model:
            for name, param in self.named_parameters():
                if 'w_kp' in name or 'w_a' in name:
                    param.data = proj_prob_simplex(param.data)
        else:
            pass

    def save_to_optim(self):
        self.optim_state_dict = self.state_dict()

    def load_from_optim(self):
        self.load_state_dict(self.optim_state_dict)

    def save_model(self, save_path):
        torch.save(self.optim_state_dict, save_path)

    def load_model(self, load_path):
        self.optim_state_dict = torch.load(load_path)
        self.load_state_dict(self.optim_state_dict)
        self.eval()





