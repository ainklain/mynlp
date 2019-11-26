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
class Constant:
    def __init__(self):
        self.PAD = -9999

const = Constant()


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
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # print("is cuda:{}".format(input_len.is_cuda))
        # input_pos = torch.LongTensor([list(range(1, input_len+1))])
        input_pos = tensor([list(range(1, int(len)+1)) + [0]*int(max_len-len) for len in input_len])
        if input_len.is_cuda:
            self.pos_enc.cuda()

        return self.pos_enc(input_pos)


# ####################### Sublayers ##########################
class _MultiHeadAttention(nn.Module):
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


class FeedForward(nn.Module):
    def __init__(self, d_in, d_ff, d_out, out_activation=None):
        super(FeedForward, self).__init__()
        self.relu = nn.ReLU()
        self.in_layer = Linear(d_in, d_ff)
        self.out_layer = Linear(d_ff, d_out)

        if out_activation == 'sigmoid':
            self.out_a = nn.Sigmoid()
        elif out_activation == 'softmax':
            self.out_a = nn.Softmax()
        else:
            self.out_a = None

    def forward(self, inputs):
        # inputs: [b_size x d_model]
        output = self.relu(self.in_layer(inputs))
        output = self.out_layer(output)

        if self.out_a is not None:
            output = self.out_a(output)

        return output


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
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=d_model-n_features, kernel_size=1)

    def forward(self, inputs):
        # input shape: (b_size, T, n_features)
        inputs_t = inputs.contiguous().transpose(-2, -1)
        # (b_size, n_features, T) -> (b_size, T, d_model)
        outputs = self.conv1(inputs_t).contiguous().transpose(-2, -1)
        # (b_size, d_model, T) -> (b_size, T, d_model)
        outputs = torch.cat((inputs, outputs), axis=-1)
        return outputs


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


class Encoder(nn.Module):
    def __init__(self, n_layers, d_k, d_v, d_model, d_ff, n_heads,
                 max_seq_len, dropout=0.1, weighted=False):
        # n_layers, d_k, d_v, d_model, d_ff, n_heads, max_seq_len, dropout, weighted = configs.n_layers, configs.d_k, configs.d_v, configs.d_model, configs.d_ff, configs.n_heads, configs.max_input_seq_len, configs.dropout, configs.weighted_model
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.pos_emb = PosEncoding(max_seq_len * 10, d_model)  # TODO: *10 fix
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = EncoderLayer if not weighted else WeightedEncoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, enc_inputs, enc_inputs_len, return_attn=False):
        # enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = enc_inputs + self.pos_emb(enc_inputs_len)  # Adding positional encoding TODO: note
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
        self.pos_emb = PosEncoding(max_seq_len * 10, d_model)  # TODO: *10 fix
        self.dropout_emb = nn.Dropout(dropout)
        self.layer_type = DecoderLayer if not weighted else WeightedDecoderLayer
        self.layers = nn.ModuleList(
            [self.layer_type(d_k, d_v, d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, dec_inputs, dec_inputs_len, enc_inputs, enc_outputs, return_attn=False):
        # dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = dec_inputs + self.pos_emb(dec_inputs_len)  # Adding positional encoding # TODO: note
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
class TSModel(nn.Module):
    def __init__(self, configs, features_cls, weight_scheme='mw', device='cpu'):
        super(TSModel, self).__init__()

        self.weight_scheme = weight_scheme
        self.device = device

        self.input_seq_size = configs.m_days // configs.sampling_days + 1
        # self.output_seq_size = configs.k_days // configs.sampling_days
        self.output_seq_size = 1

        self.conv_embedding = ConvEmbeddingLayer(n_features=configs.embedding_size, d_model=configs.d_model)
        self.encoder = Encoder(configs.n_layers, configs.d_k, configs.d_v, configs.d_model, configs.d_ff,
                               configs.n_heads, configs.max_input_seq_len, configs.dropout, configs.weighted_model)
        self.decoder = Decoder(configs.n_layers, configs.d_k, configs.d_v, configs.d_model, configs.d_ff,
                               configs.n_heads, configs.max_output_seq_len, configs.dropout, configs.weighted_model)
        self.weighted_model = configs.weighted_model

        self.predictor = nn.ModuleDict()
        self.predictor_helper = dict()
        n_size = 32
        for key in configs.model_predictor_list:
            tags = key.split('_')
            if tags[0] in configs.features_structure['regression'].keys():
                if key in ['cslogy', 'csstd']:
                    self.predictor[key] = FeedForward(configs.d_model, n_size, len(configs.features_structure['regression'][key]), out_activation='sigmoid')
                else:
                    self.predictor[key] = FeedForward(configs.d_model, n_size, len(configs.features_structure['regression'][key]))
            elif tags[0] in configs.features_structure['classification'].keys():
                self.predictor[key] = FeedForward(configs.d_model, n_size, 2, out_activation='linear')
                self.predictor_helper[key] = configs.features_structure['regression']['logy'].index(int(tags[1]))
            # elif tags[0] in configs.features_structure['crosssection'].keys():
            #     self.predictor[key] = FeedForward(64, len(configs.features_structure['regression'][key]))

        self.features_cls = features_cls

        self.optim_encoder_w = None
        self.optim_decoder_w = None
        self.optim_predictor_w = dict()
        self.dropout_train = configs.dropout

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

        enc_in = self.conv_embedding(features['input'])
        dec_in = self.conv_embedding(features['output'])

        input_seq_size = torch.Tensor([enc_in.shape[1] for _ in range(enc_in.shape[0])]).to(self.device)
        output_seq_size = torch.Tensor([dec_in.shape[1] for _ in range(dec_in.shape[0])]).to(self.device)

        enc_out, enc_self_attns = self.encoder(enc_in, input_seq_size, return_attn)
        predict, dec_self_attns, dec_enc_attns = self.decoder(dec_in, output_seq_size, enc_in, enc_out, return_attn)

        pred_each = dict()
        for key in self.predictor.keys():
            pred_each[key] = self.predictor[key](predict)

        return pred_each, enc_self_attns, dec_self_attns, dec_enc_attns

    def predict_mtl(self, features):
        ret = self.forward(features)
        return ret[0]

    def forward_with_loss(self, features, labels_mtl, return_attn=False):
        # features = {'input': torch.zeros(2, 25, 23), 'output': torch.zeros(2, 1, 23)}

        enc_in = self.conv_embedding(features['input'])
        dec_in = self.conv_embedding(features['output'])

        input_seq_size = torch.Tensor([enc_in.shape[1] for _ in range(enc_in.shape[0])]).to(self.device)
        output_seq_size = torch.Tensor([dec_in.shape[1] for _ in range(dec_in.shape[0])]).to(self.device)

        enc_out, enc_self_attns = self.encoder(enc_in, input_seq_size, return_attn)
        predict, dec_self_attns, dec_enc_attns = self.decoder(dec_in, output_seq_size, enc_in, enc_out, return_attn)

        pred_each = dict()
        loss_each = dict()
        loss = None
        for key in self.predictor.keys():
            pred_each[key] = self.predictor[key](predict)

            if self.weight_scheme == 'mw':
                adj_weight = labels_mtl['size_factor'][:, :, 0] * 2.
            else:
                adj_weight = 1.

            adj_importance = labels_mtl['importance_wgt'][:, :, 0]

            if key[:3] == 'pos':
                criterion = torch.nn.CrossEntropyLoss(reduction='none')
                pred_each[key] = pred_each[key].contiguous().transpose(1, -1)
                loss_each[key] = criterion(pred_each[key], labels_mtl[key]) \
                                 * torch.abs(labels_mtl['logy'][:, :, self.predictor_helper[key]]) \
                                 * adj_weight * adj_importance
            else:
                criterion = torch.nn.MSELoss(reduction='none')
                loss_each[key] = criterion(pred_each[key], labels_mtl[key]) * adj_weight * adj_importance

            if loss is None:
                loss = loss_each[key].mean()
            else:
                loss += loss_each[key].mean()

        return pred_each, loss #, enc_self_attns, dec_self_attns, dec_enc_attns,

    def proj_grad(self):
        if self.weighted_model:
            for name, param in self.named_parameters():
                if 'w_kp' in name or 'w_a' in name:
                    param.data = proj_prob_simplex(param.data)
        else:
            pass








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

