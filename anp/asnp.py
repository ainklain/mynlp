from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
import collections
import math
# from tqdm import tqdm
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from ts_torch import torch_util_mini as tu
from anp.gp import NPRegressionDescription, GPCurvesReader
from anp.financial_data import ContextSet, TimeSeries


# # #### profiler start ####
import builtins

try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile
# # #### profiler end ####


class configs:
    def DEFINE_integer(self, attr_nm, value_, desc_):
        setattr(self, attr_nm, value_)

    def DEFINE_float(self, attr_nm, value_, desc_):
        setattr(self, attr_nm, value_)

    def DEFINE_string(self, attr_nm, value_, desc_):
        setattr(self, attr_nm, value_)


flags = configs()
FLAGS = flags

# models
flags.DEFINE_integer('HIDDEN_SIZE', 128, 'hidden unit size of network')
flags.DEFINE_string('MODEL_TYPE', 'NP', "{NP|SNP}")
flags.DEFINE_float('beta', 1.0, 'weight to kl loss term')
# dataset
flags.DEFINE_string('dataset', 'gp', '{gp}')
flags.DEFINE_integer('case', 1, '{1|2|3}')
flags.DEFINE_integer('MAX_CONTEXT_POINTS', 500,
                     'max context size at each time-steps')
flags.DEFINE_integer('LEN_SEQ', 20, 'sequence length')
flags.DEFINE_integer('LEN_GIVEN', 10, 'given context length')
flags.DEFINE_integer('LEN_GEN', 10, 'generalization test sequence length')
# gp dataset
flags.DEFINE_float('l1_min', 0.7, 'l1 initial boundary')
flags.DEFINE_float('l1_max', 1.2, 'l1 initial boundary')
flags.DEFINE_float('l1_vel', 0.03, 'l1 kernel parameter dynamics')
flags.DEFINE_float('sigma_min', 1.0, 'sigma initial boundary')
flags.DEFINE_float('sigma_max', 1.6, 'sigma initial boundary')
flags.DEFINE_float('sigma_vel', 0.05, 'sigma kernel parameter dynamics')
# training
flags.DEFINE_integer('TRAINING_ITERATIONS', 1000000, 'training iteration')
flags.DEFINE_integer('PLOT_AFTER', 1000, 'plot iteration')
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_string('log_folder', 'logs', 'log folder')


def done_decorator(f):
    def decorated(*args, **kwargs):
        print("{} ...ing".format(f.__name__))
        result = f(*args, **kwargs)
        print("{} ...done".format(f.__name__))
        return result
    return decorated


def reordering(whole_query, target_y, pred_y, std_y, temporal=False):

    (context_x, context_y), target_x = whole_query

    if not temporal:
        for i in range(len(context_x)):
            context_x[i] = context_x[i][:,:,:-1]
        target_x = np.array(target_x)[:,:,:,:-1]

    context_x_list = context_x
    context_y_list = context_y
    target_x_list = target_x
    target_y_list = target_y
    pred_y_list = pred_y
    std_y_list = std_y

    return (target_x_list, target_y_list, context_x_list, context_y_list,
            pred_y_list, std_y_list)


def bmm(x: torch.Tensor, y:torch.Tensor):
    # x shape: (B1, B2, ..., A, B)
    # y shape: (B1, B2, ..., B, C)
    # out shape: (B1, B2, ..., A, C)
    assert len(x.shape) >= 3
    assert x.shape[:-2] == y.shape[:-2]
    out_shape = list(x.shape[:-1]) + list(y.shape[-1:])
    out = torch.bmm(x.view(-1, x.shape[-2], x.shape[-1]), y.view(-1, y.shape[-2], y.shape[-1]))
    return out.reshape(out_shape)


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Define the LSTM layer
        self.lstm_cell = nn.LSTMCell(self.input_dim, self.hidden_dim)

    def init_hidden(self, batch_sizes:list):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(batch_sizes + [self.hidden_dim]),
                torch.zeros(batch_sizes + [self.hidden_dim]))

    def forward(self, input, hidden=None):
        # Forward pass through LSTM layer
        # input shape: [batch, input_size]
        # shape of hidden: (a, b), where a and b both
        # have shape (batch_size, hidden_dim).

        # input shape: (B1, B2, ..., A) -> (B1 * B2 * ..., A)
        input_shape = list(input.shape)
        input = input.reshape([-1, input_shape[-1]])

        if hidden is None:
            h, c = self.init_hidden(input.shape[0])
        else:
            h = hidden[0].reshape([-1, self.hidden_dim])
            c = hidden[1].reshape([-1, self.hidden_dim])

        h, c = self.lstm_cell(input, (h, c))
        # hidden = self.lstm_cell(input.view(self.batch_size, -1), hidden)
        hidden_shape = input_shape[:-1]+[-1]
        return h.reshape(hidden_shape), c.reshape(hidden_shape)


class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """

    def __init__(self, num_hidden_k):
        """
        :param num_hidden_k: dimension of hidden
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query):
        # Get attention score
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)

        attn = torch.softmax(attn, dim=-1)

        # Dropout
        attn = self.attn_dropout(attn)

        # Get Context Vector
        result = torch.bmm(attn, value)

        return result, attn


class Attention(nn.Module):
    """
    Attention Network
    """

    def __init__(self, d_hidden, d_model, h=4):
        """
        :param d_hidden: dimension of hidden
        :param h: num of heads
        """
        super(Attention, self).__init__()

        self.d_hidden = d_hidden
        self.d_hidden_per_attn = d_hidden // h
        self.h = h

        self.key = Linear(d_hidden, d_hidden, bias=False)
        self.value = Linear(d_model, d_hidden, bias=False)
        self.query = Linear(d_hidden, d_hidden, bias=False)

        self.multihead = MultiheadAttention(self.d_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(d_hidden * 2, d_hidden)

        self.layer_norm = nn.LayerNorm(d_hidden)

    def forward(self, key, value, query):
        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        residual = query

        # Make multihead
        key = self.key(key).view(batch_size, seq_k, self.h, self.d_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_k, self.h, self.d_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.d_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.d_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.d_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.d_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(key, value, query)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q, self.d_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)

        result = torch.cat([residual, result], dim=-1)

        # Concatenate context vector with input (most important)
        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + residual

        # Layer normalization
        result = self.layer_norm(result)

        return result, attns


class LatentModel(nn.Module):
    """
    Latent Model (Attentive Neural Process)
    """

    def __init__(self, num_hidden, d_x=1, d_y=1):
        super(LatentModel, self).__init__()
        self.latent_encoder = LatentEncoder(num_hidden, num_hidden, input_dim=d_x+d_y)
        self.deterministic_encoder = DeterministicEncoder(num_hidden, input_dim=d_x+d_y)
        self.decoder = Decoder(num_hidden)
        self.BCELoss = nn.BCELoss()

    def forward(self, query, target_y=None):

        (context_x, context_y), target_x = query
        num_targets = target_x.size(1)

        # prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y)
        prior = self.latent_encoder(context_x, context_y)

        # For training
        if target_y is not None:
            # posterior_mu, posterior_var, posterior = self.latent_encoder(target_x, target_y)
            # z = posterior
            posterior = self.latent_encoder(target_x, target_y)
            z = posterior.sample()

        # For Generation
        else:
            # z = prior
            z = prior.sample()


        z = z.unsqueeze(1).repeat(1, num_targets, 1)  # [B, T_target, H]
        r = self.deterministic_encoder(context_x, context_y, target_x)  # [B, T_target, H]

        # mu should be the prediction of target y
        dist, mu, sigma = self.decoder(r, z, target_x)

        # For Training
        if target_y is not None:
            log_p = dist.log_prob(target_y).squeeze()

            posterior = self.latent_encoder(target_x, target_y)
            kl = torch.distributions.kl_divergence(posterior, prior).sum(dim=-1, keepdims=True)
            kl = kl.repeat([1, num_targets])
            loss = - (log_p - kl / torch.tensor(num_targets).float()).mean()

            # # get log probability
            # bce_loss = self.BCELoss(torch.sigmoid(mu), target_y)
            #
            # # get KL divergence between prior and posterior
            # kl = self.kl_div(prior_mu, prior_var, posterior_mu, posterior_var)
            #
            # # maximize prob and minimize KL divergence
            # loss = bce_loss + kl

        # For Generation
        else:
            log_p = None
            kl = None
            loss = None

        return mu, sigma, log_p, kl, loss

    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (torch.exp(posterior_var) + (posterior_mu - prior_mu) ** 2) / torch.exp(prior_var) - 1. + (
                    prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div


class ImaginaryContext(nn.Module):
    def __init__(self, d_hidden, d_model, n_head=2, k_slot=25):
        super(ImaginaryContext, self).__init__()
        self.k_slot = k_slot    # number of imaginary context
        self.d_hidden = d_hidden
        self.d_model = d_model

        # key_inference
        self.ikey_lstm = LSTMCell(d_hidden+d_model, d_hidden)
        self.ikey_hidden_layer = Linear(d_hidden, d_hidden, w_init='relu')
        self.ikey_infer_mu = Linear(d_hidden, d_hidden)
        self.ikey_infer_logsigma = Linear(d_hidden, d_hidden)

        # imagination tracker
        self.itracker_lstm = LSTMCell(d_hidden+d_model, d_model)
        self.itracker_cross_attentions = nn.ModuleList([Attention(d_hidden, d_model, n_head) for _ in range(2)])
        self.itracker_hidden_layer = Linear(d_hidden, d_hidden, w_init='relu')
        self.itracker_mu = Linear(d_hidden, d_model)
        self.itracker_logsigma = Linear(d_hidden, d_model)

        self.variables = dict()

    def reset_variables(self, type_):
        self.variables[type_] = dict(x_im=None, v_im=None, ikey_hidden=None, itracker_hidden=None)

    def _initialize_variables(self, batch_size, type_):
        self.variables[type_] = dict(
            x_im=torch.zeros([batch_size, self.k_slot, self.d_hidden]),
            v_im=torch.zeros([batch_size, self.k_slot, self.d_model]),
            ikey_hidden=self.ikey_lstm.init_hidden([batch_size, self.k_slot]),
            itracker_hidden=self.itracker_lstm.init_hidden([batch_size, self.k_slot])
        )

    def get_variables(self, type_, device='cpu'):
        assert type_ in self.variables.keys(), 'should initialize variables first'

        var_dict = self.variables[type_]
        for key in var_dict.keys():
            if isinstance(var_dict[key], tuple):
                var_dict[key] = (var_dict[key][0].to(device), var_dict[key][1].to(device))
            else:
                var_dict[key] = var_dict[key].to(device)
        return var_dict['x_im'], var_dict['v_im'], var_dict['ikey_hidden'], var_dict['itracker_hidden']

    def forward(self, x_re_t, v_re_t, type_='prior', device='cpu'):
        """
            context_x : context_real x (shape: [batch, num_contexts, d_hidden])
            context_r : "representation of context" = f_orderinv(context_x, context_y)
              (shape: [batch, num_contexts, d_model])
        """
        batch, num_contexts, _ = x_re_t.shape

        # get t-1 values
        if self.variables[type_]['x_im'] is None:
            self._initialize_variables(batch, type_)  # batch_lstm = batch * num_contexts

        x_im_prev, u_im_prev, ikey_hidden_prev, itracker_hidden_prev = self.get_variables(type_, device)

        # imaginary key inference
        if num_contexts > 0:
            ikey_v_re_t = v_re_t.mean(dim=1, keepdim=True).repeat([1, self.k_slot, 1])
        else:
            ikey_v_re_t = torch.zeros(batch, self.k_slot, self.d_model)

        ikey_input = torch.cat([x_im_prev, ikey_v_re_t], dim=-1)
        ikey_hidden = self.ikey_lstm(ikey_input, ikey_hidden_prev)  # ikey_hidden_t: (h_t, c_t)
        ikey_hidden_t = torch.relu(self.ikey_hidden_layer(ikey_hidden[0]))

        # x_t ~ N(f_mu(h_t), f_sigma(h_t))
        ikey_mu = self.ikey_infer_mu(ikey_hidden_t)
        ikey_log_sigma = self.ikey_infer_logsigma(ikey_hidden_t)
        ikey_sigma = torch.exp(0.5 * ikey_log_sigma)
        x_im_dist = torch.distributions.Normal(loc=ikey_mu, scale=ikey_sigma)

        # (x_im_t shape:[batch_lstm, d_hidden]: projection된 input으로 보자. mu, sigma layer에 반영)
        x_im_t = x_im_dist.sample()

        # imagination tracker
        itracker_input = torch.cat([x_im_t, u_im_prev], dim=-1)
        itracker_hidden = self.itracker_lstm(itracker_input, itracker_hidden_prev)  # itracker_hidden: (h_t, c_t)

        # query: x_im only (shape: [batch_lstm, d_hidden] => [batch, n_context, d_hidden])
        # key: x_im + x_re (shape: [batch_lstm, d_hidden] => [batch, n_context*2, d_hidden])
        # value: h_im + v_re (shape: [batch_lstm, d_model] => [batch, n_context*2, d_model])

        # if num_contexts > 0:
        key_ = torch.cat([x_re_t, x_im_t], dim=1)
        value_ = torch.cat([v_re_t, itracker_hidden[0]], dim=1)
        query_ = x_im_t
        # else:
        #     key_ = x_im_t
        #     value_ = v_re_t
        #     query_ = x_im_t

        for attn in self.itracker_cross_attentions:
            query_, _ = attn(key_, value_, query_)  # key / value / query

        # x_t ~ N(f_mu(a_t), f_sigma(a_t))
        itracker_mu = self.itracker_mu(query_)
        itracker_log_sigma = self.itracker_logsigma(query_)
        itracker_sigma = torch.exp(0.5 * itracker_log_sigma)
        v_im_dist = torch.distributions.Normal(loc=itracker_mu, scale=itracker_sigma)

        # (u_im_t shape:[batch, d_model]
        v_im_t = v_im_dist.sample()

        # set prev variables
        self.variables[type_]['x_im'] = x_im_t
        self.variables[type_]['u_im'] = v_im_t
        self.variables[type_]['ikey_hidden'] = ikey_hidden
        self.variables[type_]['itracker_hidden'] = itracker_hidden

        return x_im_dist, v_im_dist, x_im_t, v_im_t


class CommonLatentEncoder(nn.Module):
    def __init__(self, d_x, d_y, d_hidden, d_model):
        super(CommonLatentEncoder, self).__init__()
        self.input_projection = Linear(d_x + d_y, d_hidden)
        self.hidden_layer = nn.ModuleList([Linear(d_hidden, d_hidden) for _ in range(2)])
        self.output_projection = Linear(d_hidden, d_model)

    def forward(self, x_re_t, y_re_t):
        input_re_t = torch.cat([x_re_t, y_re_t], dim=-1)
        context = self.input_projection(input_re_t)
        for h_layer in self.hidden_layer:
            context = torch.relu(h_layer(context))

        return context, self.output_projection(context)


class GlobalLatentEncoder(nn.Module):
    def __init__(self, d_hidden, d_model):
        super(GlobalLatentEncoder, self).__init__()
        self.hidden_layer = nn.ModuleList([Linear(d_hidden, d_hidden) for _ in range(2)])
        self.input_projection = Linear(d_model, d_hidden)
        self.output_projection = Linear(d_hidden, d_model)

        self.mu = Linear(d_model, d_hidden)
        self.log_sigma = Linear(d_model, d_hidden)

    def forward(self, v_re_t, v_im_t):

        v_re_t = self.input_projection(v_re_t)
        for h_layer in self.hidden_layer:
            v_re_t = torch.relu(h_layer(v_re_t))

        v_re_t = self.output_projection(v_re_t)
        v_t = torch.cat([v_re_t, v_im_t], dim=1).mean(dim=1)
        mu_ = self.mu(v_t)
        log_sigma_ = self.log_sigma(v_t)

        # reparameterization trick
        sigma_ = torch.exp(0.5 * log_sigma_)
        eps = torch.randn_like(sigma_)
        z_t = eps.mul(sigma_).add_(mu_)

        dist = torch.distributions.Normal(loc=mu_, scale=sigma_)
        # z_t = dist.sample()

        return dist, z_t


class QueryDepLatentEncoder(nn.Module):
    def __init__(self, d_x, d_hidden, d_model, n_head=2):
        super(QueryDepLatentEncoder, self).__init__()
        self.hidden_layer = nn.ModuleList([Linear(d_hidden, d_hidden) for _ in range(2)])
        self.input_projection = Linear(d_model, d_hidden)
        self.output_projection = Linear(d_hidden, d_model)
        self.target_projection = Linear(d_x, d_hidden)
        self.cross_attention = nn.ModuleList([Attention(d_hidden, d_model, n_head) for _ in range(2)])

    def forward(self, context_re_t, context_im_t, target_x):
        x_re_t, v_re_t = context_re_t
        x_im_t, v_im_t = context_im_t

        v_re_t = self.input_projection(v_re_t)
        for h_layer in self.hidden_layer:
            v_re_t = torch.relu(h_layer(v_re_t))

        v_re_t = self.output_projection(v_re_t)

        key_ = torch.cat([x_re_t, x_im_t], dim=1)       # shape: [batch, n_re + n_im, n_hidden]
        value_ = torch.cat([v_re_t, v_im_t], dim=1)     # shape: [batch, n_re + n_im, n_model]
        query_ = self.target_projection(target_x)   # shape: [batch, n_re, n_hidden]

        for attn in self.cross_attention:
            query_, _ = attn(key_, value_, query_)  # key / value / query

        return query_


class Decoder(nn.Module):
    """
    Decoder for generation
    """
    def __init__(self, d_x, d_hidden):
        super(Decoder, self).__init__()
        self.target_projection = Linear(d_x, d_hidden)
        self.linears = nn.ModuleList([Linear(d_hidden * 3, d_hidden * 3, w_init='relu') for _ in range(3)])
        self.final_projection = Linear(d_hidden * 3, 2)

    def forward(self, a_t, z_t, target_x):
        batch_size, num_targets, _ = target_x.size()
        # project vector with dimension 2 --> num_hidden
        target_x = self.target_projection(target_x)

        # concat all vectors (r,z,target_x)
        hidden = torch.cat([a_t, z_t, target_x], dim=-1)

        # mlp layers
        for linear in self.linears:
            hidden = torch.relu(linear(hidden))

        # get mu and sigma
        y_pred = self.final_projection(hidden)

        # Get the mean an the variance
        mu, log_sigma = torch.chunk(y_pred, 2, dim=-1)

        # Bound the variance
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        dist = torch.distributions.Normal(loc=mu, scale=sigma)

        return dist, mu, sigma


class ASNP(nn.Module):
    def __init__(self, d_x, d_y, d_hidden, d_model, n_head=2):
        super(ASNP, self).__init__()
        self.common_latent_encoder = CommonLatentEncoder(d_x, d_y, d_hidden, d_model)
        self.global_latent_encoder = GlobalLatentEncoder(d_hidden, d_model)
        self.query_latent_encoder = QueryDepLatentEncoder(d_x, d_hidden, d_model, n_head)
        self.decoder = Decoder(d_x, d_hidden)
        self.imaginary_context = ImaginaryContext(d_hidden, d_model, n_head)

    def reset_variables(self):
        self.imaginary_context.reset_variables('prior')
        self.imaginary_context.reset_variables('posterior')

    def forward(self, context, target_y_list=None):
        # i, context, target_y_list = 0, query, target_y
        (context_x_seq, context_y_seq), target_x_seq = context

        self.reset_variables()

        total_loss = torch.tensor(0.).to(tu.device)
        mu_list, sigma_list = [], []
        log_p_list, kl_list = [], []
        log_p_seen, log_p_unseen = [], []
        log_p_wo_con, log_p_w_con = 0, 0
        mse_list, mse_wo_con, mse_w_con = [], 0, 0
        cnt_wo, cnt_w = torch.tensor(0.0), torch.tensor(0.0)

        for i in range(len(context_x_seq)):
            context_x, context_y, target_x = context_x_seq[i].to(tu.device), context_y_seq[i].to(tu.device), target_x_seq[i].to(tu.device)
            num_targets = target_x.size(1)

            x_re_t, v_re_t = self.common_latent_encoder(context_x, context_y)

            # imaginary context update
            x_im_dist, v_im_dist, x_im_t, v_im_t = self.imaginary_context(x_re_t, v_re_t, 'prior', tu.device)

            # global latent encoder (prior)
            z_dist_prior, z_t_prior = self.global_latent_encoder(v_re_t, v_im_t)

            # encoding value for real context
            if target_y_list is not None:
                target_y = target_y_list[i].to(tu.device)
                x_re_t_posterior, v_re_t_posterior = self.common_latent_encoder(target_x, target_y)

                # imaginary context update
                x_im_dist, v_im_dist, x_im_t, v_im_t = self.imaginary_context(x_re_t_posterior, v_re_t_posterior, 'posterior', tu.device)

                # global latent encoder (posterior)
                z_dist_posterior, z_t = self.global_latent_encoder(v_re_t_posterior, v_im_t)
            else:
                z_t = z_t_prior

            z_t = z_t.unsqueeze(1).repeat(1, num_targets, 1)  # [B, T_target, H]

            # query dependent latent encoder
            context_re_t = (x_re_t, v_re_t)
            context_im_t = (x_im_t, v_im_t)
            a_t = self.query_latent_encoder(context_re_t, context_im_t, target_x)

            # decode latent values and get target_y
            target_dist, target_mu, target_sigma = self.decoder(a_t, z_t, target_x)
            mu_list.append(target_mu)
            sigma_list.append(target_sigma)

            if target_y_list is not None:
                log_p = target_dist.log_prob(target_y).squeeze()
                kl = torch.distributions.kl_divergence(z_dist_posterior, z_dist_prior).sum(dim=-1, keepdims=True)
                kl = kl.repeat([1, num_targets])
                loss = -(log_p - kl / torch.tensor(num_targets).float()).mean()
                total_loss += loss
            else:
                log_p = None
                kl = None
                loss = None

        return total_loss, mu_list, sigma_list


def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = 0.001 * warmup_step ** 0.5 * min(step_num * warmup_step ** -1.5, step_num ** -0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def plot_functions(ep, target_x, target_y, context_x, context_y, pred_y, std):
    """Plots the predicted mean and variance and the context points.

    Args:
        target_x: An array of shape [B,num_targets,1] that contains the
            x values of the target points.
        target_y: An array of shape [B,num_targets,1] that contains the
            y values of the target points.
        context_x: An array of shape [B,num_contexts,1] that contains
            the x values of the context points.
        context_y: An array of shape [B,num_contexts,1] that contains
            the y values of the context points.
        pred_y: An array of shape [B,num_targets,1] that contains the
            predicted means of the y values at the target points in target_x.
        std: An array of shape [B,num_targets,1] that contains the
            predicted std dev of the y values at the target points in target_x.
    """
    fig = plt.figure()
    # Plot everything
    plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
    plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
    plt.plot(context_x[0], context_y[0], 'ko', markersize=10)
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - std[0, :, 0],
        pred_y[0, :, 0] + std[0, :, 0],
        alpha=0.2,
        facecolor='#65c9f7',
        interpolate=True)

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    plt.grid('off')
    ax = plt.gca()
    fig.savefig('./anp/out/test_{}.png'.format(ep))
    plt.close(fig)


def plot_functions_1d(ep, len_seq, len_gen, plot_data, h_x_list=None):
    """Plots the predicted mean and variance and the context points.
        Args:
        target_x: An array of shape [B,num_targets,1] that contains the
            x values of the target points.
        target_y: An array of shape [B,num_targets,1] that contains the
            y values of the target points.
        context_x: An array of shape [B,num_contexts,1] that contains
            the x values of the context points.
        context_y: An array of shape [B,num_contexts,1] that contains
            the y values of the context points.
        pred_y: An array of shape [B,num_targets,1] that contains the
            predicted means of the y values at the target points in target_x.
        std: An array of shape [B,num_targets,1] that contains the
            predicted std dev of the y values at the target points in target_x.
    """
    target_x, target_y, context_x, context_y, pred_y, std = plot_data
    plt.figure(figsize=(6.4, 4.8*(len_seq+len_gen)))
    for t in range(len_seq+len_gen):
        plt.subplot(len_seq+len_gen,1,t+1)
        # Plot everything
        plt.plot(target_x[t][0], target_y[t][0], 'k:', linewidth=2)
        plt.plot(target_x[t][0], pred_y[t][0], 'b', linewidth=2)
        if len(context_x[t]) != 0:
            plt.plot(context_x[t][0], context_y[t][0], 'ko', markersize=10)
        if h_x_list is not None:
            h_y_list = []
            for h_x in h_x_list[t][0]:
                min_val = 10000
                idx = 0
                for i, t_x in enumerate(target_x[t][0]):
                    if abs(h_x-t_x) < min_val:
                        min_val = abs(h_x-t_x)
                        idx = i
                h_y_list.append(target_y[t][0][idx])
            plt.plot(h_x_list[t][0],h_y_list, 'ro', markersize=10)
        plt.fill_between(
            target_x[t][0, :, 0],
            pred_y[t][0, :, 0] - std[t][0, :, 0],
            pred_y[t][0, :, 0] + std[t][0, :, 0],
            alpha=0.2,
            facecolor='#65c9f7',
            interpolate=True)

        # Make the plot pretty
        plt.yticks([-4, -2, 0, 2, 4], fontsize=16)
        plt.xticks([-4, -2, 0, 2, 4], fontsize=16)
        #plt.ylim([-2, 2])
        plt.grid('off')
        ax = plt.gca()

    plt.savefig('./anp/out/test_{}.png'.format(ep))
    # fig.savefig('./anp/out/test_{}.png'.format(ep))
    plt.close()
    # image = misc.imread('./anp/out/test_{}.png'.format(ep), mode='RGB')
    #
    # return image


def plot_functions_(ep, plot_data, plot_seq_num=-1):
    """Plots the predicted mean and variance and the context points.
        Args:
        target_x: An array of shape [B,num_targets,1] that contains the
            x values of the target points.
        target_y: An array of shape [B,num_targets,1] that contains the
            y values of the target points.
        context_x: An array of shape [B,num_contexts,1] that contains
            the x values of the context points.
        context_y: An array of shape [B,num_contexts,1] that contains
            the y values of the context points.
        pred_y: An array of shape [B,num_targets,1] that contains the
            predicted means of the y values at the target points in target_x.
        std: An array of shape [B,num_targets,1] that contains the
            predicted std dev of the y values at the target points in target_x.
    """
    t_x_list, t_y_list, c_x_list, c_y_list, pred_y_list, std_list = plot_data
    target_x, target_y = tu.np_ify(t_x_list[plot_seq_num]), tu.np_ify(t_y_list[plot_seq_num])
    context_x, context_y = tu.np_ify(c_x_list[plot_seq_num]), tu.np_ify(c_y_list[plot_seq_num])
    pred_y, std = tu.np_ify(pred_y_list[plot_seq_num]), tu.np_ify(std_list[plot_seq_num])

    fig = plt.figure()
    # Plot everything
    plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
    plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
    plt.plot(context_x[0], context_y[0], 'ko', markersize=5)
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - std[0, :, 0],
        pred_y[0, :, 0] + std[0, :, 0],
        alpha=0.2,
        facecolor='#65c9f7',
        interpolate=True)

    # Make the plot pretty
    # plt.yticks([-2, 0, 2], fontsize=16)
    # plt.xticks([-2, 0, 2], fontsize=16)
    # plt.ylim([-2, 2])
    plt.grid('off')
    ax = plt.gca()
    fig.savefig('./anp/out/test_{}.png'.format(ep))
    plt.close(fig)


def main3():
    TRAINING_ITERATIONS = 500000 #@param {type:"number"}
    MAX_CONTEXT_POINTS = 250 #@param {type:"number"}
    PLOT_AFTER = 500 #@param {type:"number"}

    base_i = 100
    # Train dataset
    dataset = TimeSeries(batch_size=FLAGS.batch_size, max_num_context=MAX_CONTEXT_POINTS, predict_length=120)
    base_y = dataset.get_timeseries('mkt_rf')
    dataset.generate_set(base_y)

    # dataset_train = GPCurvesReader(
    #     batch_size=FLAGS.batch_size, max_num_context=FLAGS.MAX_CONTEXT_POINTS,
    #     len_seq=FLAGS.LEN_SEQ, len_given=FLAGS.LEN_GIVEN,
    #     len_gen=FLAGS.LEN_GEN,
    #     l1_min=FLAGS.l1_min, l1_max=FLAGS.l1_max, l1_vel=FLAGS.l1_vel,
    #     sigma_min=FLAGS.sigma_min, sigma_max=FLAGS.sigma_max,
    #     sigma_vel=FLAGS.sigma_vel, temporal=True,
    #     case=FLAGS.case)
    #
    #
    # # Test dataset
    # dataset_test = GPCurvesReader(
    #     batch_size=FLAGS.batch_size, max_num_context=FLAGS.MAX_CONTEXT_POINTS,
    #     testing=True,
    #     len_seq=FLAGS.LEN_SEQ, len_given=FLAGS.LEN_GIVEN,
    #     len_gen=FLAGS.LEN_GEN,
    #     l1_min=FLAGS.l1_min, l1_max=FLAGS.l1_max, l1_vel=FLAGS.l1_vel,
    #     sigma_min=FLAGS.sigma_min, sigma_max=FLAGS.sigma_max,
    #     sigma_vel=FLAGS.sigma_vel, temporal=True,
    #     case=FLAGS.case)

    # data_train = dataset_train.generate_temporal_curves(seed=None)
    # data_test = dataset_test.generate_temporal_curves(seed=123)


    # Sizes of the layers of the MLPs for the encoders and decoder
    # The final output layer of the decoder outputs two values, one for the mean and
    # one for the variance of the prediction at the target location

    model = ASNP(d_x=1, d_y=1, d_hidden=128, d_model=128, n_head=4).to(tu.device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    it = 0
    for it in range(TRAINING_ITERATIONS):

        # data_train = dataset_train.generate_temporal_curves(seed=None)
        data_train = dataset.generate(base_i, seq_len=FLAGS.LEN_SEQ, is_train=True)
        # data_train = to_device(data_train, 'cuda:0')
        model.train()
        # Define the loss
        query, target_y = data_train.query, data_train.target_y
        loss,  mu_list, sigma_list = model(query,  target_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Plot the predictions in `PLOT_AFTER` intervals
        if it % PLOT_AFTER == 0:
            # data_test = dataset_test.generate_temporal_curves(seed=123)
            for ii, date_i in enumerate([base_i - 20, base_i, base_i + 20]):
                data_test = dataset.generate(date_i, seq_len=FLAGS.LEN_SEQ, is_train=False)
                # data_test = dataset_test.generate_curves()
                # data_test = to_device(data_test, 'cuda:0')
                model.eval()
                with torch.set_grad_enabled(False):
                    loss, _, _ = model(data_test.query, data_test.target_y)

                    # Get the predicted mean and variance at the target points for the testing set
                    _, mu_list, sigma_list = model(data_test.query)
                loss_value, pred_y, std_y, target_y, whole_query = loss, mu_list, sigma_list, data_test.target_y, data_test.query

                plot_data = reordering(whole_query, target_y, pred_y, std_y, temporal=True)
                if FLAGS.dataset == 'gp':
                    # plot_functions_1d(it, FLAGS.LEN_SEQ, FLAGS.LEN_GEN, plot_data)
                    plot_functions_(it + ii, plot_data, plot_seq_num=-1)

                # (context_x, context_y), target_x = whole_query
                print('Iteration: {} [date_i: {}], loss: {}'.format(it, date_i, tu.np_ify(loss_value)))
            #
            # # Plot the prediction and the context
            # plot_functions(it, tu.np_ify(target_x), tu.np_ify(target_y), tu.np_ify(context_x), tu.np_ify(context_y), tu.np_ify(pred_y), tu.np_ify(std_y))


def main2():
    TRAINING_ITERATIONS = 100000 #@param {type:"number"}
    MAX_CONTEXT_POINTS = 250 #@param {type:"number"}
    PLOT_AFTER = 500 #@param {type:"number"}

    # Train dataset
    # dataset = TimeSeries(batch_size=FLAGS.batch_size, max_num_context=MAX_CONTEXT_POINTS)
    # base_y = dataset.get_timeseries('mkt_rf')
    # dataset.generate_set(base_y)

    dataset_train = GPCurvesReader(
        batch_size=FLAGS.batch_size, max_num_context=FLAGS.MAX_CONTEXT_POINTS,
        len_seq=FLAGS.LEN_SEQ, len_given=FLAGS.LEN_GIVEN,
        len_gen=FLAGS.LEN_GEN,
        l1_min=FLAGS.l1_min, l1_max=FLAGS.l1_max, l1_vel=FLAGS.l1_vel,
        sigma_min=FLAGS.sigma_min, sigma_max=FLAGS.sigma_max,
        sigma_vel=FLAGS.sigma_vel, temporal=True,
        case=FLAGS.case)

    # Test dataset
    dataset_test = GPCurvesReader(
        batch_size=FLAGS.batch_size, max_num_context=FLAGS.MAX_CONTEXT_POINTS,
        testing=True,
        len_seq=FLAGS.LEN_SEQ, len_given=FLAGS.LEN_GIVEN,
        len_gen=FLAGS.LEN_GEN,
        l1_min=FLAGS.l1_min, l1_max=FLAGS.l1_max, l1_vel=FLAGS.l1_vel,
        sigma_min=FLAGS.sigma_min, sigma_max=FLAGS.sigma_max,
        sigma_vel=FLAGS.sigma_vel, temporal=True,
        case=FLAGS.case)

    # data_train = dataset_train.generate_temporal_curves(seed=None)
    # data_test = dataset_test.generate_temporal_curves(seed=123)

    # Sizes of the layers of the MLPs for the encoders and decoder
    # The final output layer of the decoder outputs two values, one for the mean and
    # one for the variance of the prediction at the target location

    model = ASNP(d_x=1, d_y=1, d_hidden=128, d_model=128, n_head=4).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    it = 0
    for it in range(TRAINING_ITERATIONS):
        data_train = dataset_train.generate_temporal_curves(seed=None)
        # data_train = dataset.generate(50, seq_len=FLAGS.LEN_SEQ, is_train=True)
        # data_train = to_device(data_train, 'cuda:0')
        model.train()
        # Define the loss
        query, target_y = data_train.query, data_train.target_y
        loss,  mu_list, sigma_list = model(query,  target_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Plot the predictions in `PLOT_AFTER` intervals
        if it % PLOT_AFTER == 0:
            data_test = dataset_test.generate_temporal_curves(seed=123)
            # data_test = dataset.generate(50, seq_len=FLAGS.LEN_SEQ, is_train=False)
            # data_test = dataset_test.generate_curves()
            # data_test = to_device(data_test, 'cuda:0')
            model.eval()
            with torch.set_grad_enabled(False):
                loss, _, _ = model(data_test.query, data_test.target_y)

                # Get the predicted mean and variance at the target points for the testing set
                _, mu_list, sigma_list = model(data_test.query)
            loss_value, pred_y, std_y, target_y, whole_query = loss, mu_list, sigma_list, data_test.target_y, data_test.query

            plot_data = reordering(whole_query, target_y, pred_y, std_y, temporal=True)
            if FLAGS.dataset == 'gp':
                # plot_functions_1d(it, FLAGS.LEN_SEQ, FLAGS.LEN_GEN, plot_data)
                plot_functions_(it, plot_data, plot_seq_num=0)
                plot_functions_(it + 5, plot_data, plot_seq_num=5)
                plot_functions_(it + 10, plot_data, plot_seq_num=10)
                plot_functions_(it + 20, plot_data, plot_seq_num=-1)


            # (context_x, context_y), target_x = whole_query
            print('Iteration: {}, loss: {}'.format(it, tu.np_ify(loss_value)))
            #
            # # Plot the prediction and the context
            # plot_functions(it, tu.np_ify(target_x), tu.np_ify(target_y), tu.np_ify(context_x), tu.np_ify(context_y), tu.np_ify(pred_y), tu.np_ify(std_y))


