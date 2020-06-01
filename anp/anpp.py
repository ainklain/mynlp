

from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch.nn import functional as F
import collections
import math
# from tqdm import tqdm
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from ts_torch import torch_util_mini as tu
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
        # Linear
        return self.linear_layer(x)


class LatentContext(nn.Module):
    """
    Latent Encoder [For prior, posterior]
    """

    def __init__(self, num_hidden, num_latent, input_dim=3):
        super(LatentContext, self).__init__()
        self.input_projection = Linear(input_dim, num_hidden)
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.penultimate_layer = Linear(num_hidden, num_hidden, w_init='relu')
        self.mu = Linear(num_hidden, num_latent)
        self.log_sigma = Linear(num_hidden, num_latent)

        self.local_penultimate_layer = Linear(num_hidden, num_hidden, w_init='relu')
        self.local_mu = Linear(num_hidden, num_latent)
        self.local_log_sigma = Linear(num_hidden, num_latent)

    @profile
    def forward(self, x, y):
        # LatentContext
        # concat location (x) and value (y)
        encoder_input = torch.cat([x, y], dim=-1)

        # project vector with dimension 3 --> num_hidden
        encoder_input = self.input_projection(encoder_input)

        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        # global dist
        hidden = encoder_input.mean(dim=1)
        hidden = torch.relu(self.penultimate_layer(hidden))

        global_mu = self.mu(hidden)
        global_log_sigma = self.log_sigma(hidden)
        global_sigma = torch.exp(0.5 * global_log_sigma)
        # global_dist = torch.distributions.Normal(loc=mu, scale=sigma)

        # # reparameterization trick
        # sigma = torch.exp(0.5 * log_sigma)
        # eps = torch.randn_like(sigma)
        # z = eps.mul(sigma).add_(mu)

        # local dist
        local_hidden = torch.relu(self.local_penultimate_layer(encoder_input))
        local_mu = self.local_mu(local_hidden)
        local_log_sigma = self.local_log_sigma(local_hidden)
        local_sigma = torch.exp(0.5 * local_log_sigma)
        # local_dist = torch.distributions.Normal(loc=local_mu, scale=local_sigma)

        # return distribution
        # return mu, log_sigma, z
        return global_mu, global_sigma, local_mu, local_sigma


class LatentEncoder(nn.Module):
    def __init__(self, num_hidden, num_latent, input_dim=2):
        super(LatentEncoder, self).__init__()
        self.latent_context = LatentContext(num_hidden, num_latent, input_dim)
        self.context_projection = Linear(1, num_hidden)
        self.target_projection = Linear(1, num_hidden)
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.out_layer = Linear(num_latent * 2, num_latent, w_init='relu')

    @profile
    def forward(self, context_x, context_y, target_x):
        # LatentEncoder
        num_targets = target_x.size(1)

        global_mu, global_sigma, local_mu, local_sigma = self.latent_context(context_x, context_y)
        # local
        local_eps = torch.randn_like(local_sigma)
        local_z = local_eps.mul(local_sigma).add_(local_mu)

        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        # cross attention layer
        for attention in self.cross_attentions:
            query, _ = attention(keys, local_z, query)

        # global
        global_eps = torch.randn_like(global_sigma)
        global_z = global_eps.mul(global_sigma).add_(global_mu)
        global_z = global_z.unsqueeze(1).repeat(1, num_targets, 1)  # [B, T_target, H]

        c_latent = torch.cat([query, global_z], dim=-1)
        c = torch.relu(self.out_layer(c_latent))
        return c, global_z


class DeterministicEncoder(nn.Module):
    """
    Deterministic Encoder [r]
    """

    def __init__(self, num_hidden, input_dim=3):
        super(DeterministicEncoder, self).__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.input_projection = Linear(input_dim, num_hidden)
        self.context_projection = Linear(1, num_hidden)
        self.target_projection = Linear(1, num_hidden)

    @profile
    def forward(self, context_x, context_y, target_x):
        # DeterministicEncoder
        # concat context location (x), context value (y)
        encoder_input = torch.cat([context_x, context_y], dim=-1)

        # project vector with dimension 3 --> num_hidden
        encoder_input = self.input_projection(encoder_input)

        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        # query: target_x, key: context_x, value: representation
        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        # cross attention layer
        for attention in self.cross_attentions:
            query, _ = attention(keys, encoder_input, query)

        return query


class Decoder(nn.Module):
    """
    Decoder for generation
    """

    def __init__(self, num_hidden):
        super(Decoder, self).__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.target_projection = Linear(1, num_hidden)
        self.linears = nn.ModuleList([Linear(num_hidden * 4, num_hidden * 4, w_init='relu') for _ in range(3)])
        self.penultimate_layer = Linear(num_hidden * 4, num_hidden, w_init='relu')
        self.final_projection = Linear(num_hidden, 2)

    @profile
    def forward(self, r, z, c, target_x):
        # Decoder
        batch_size, num_targets, _ = target_x.size()
        # project vector with dimension 2 --> num_hidden
        target_x = self.target_projection(target_x)

        # concat all vectors (r,z,c, target_x)
        hidden = torch.cat([r, z, c, target_x], dim=-1)

        # mlp layers
        for linear in self.linears:
            hidden = torch.relu(linear(hidden))

        hidden = self.penultimate_layer(hidden)

        for attention in self.self_attentions:
            hidden, _ = attention(hidden, hidden, hidden)

        # get mu and sigma
        y_pred = self.final_projection(hidden)

        # Get the mean an the variance
        mu, log_sigma = torch.chunk(y_pred, 2, dim=-1)

        # Bound the variance
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        dist = torch.distributions.Normal(loc=mu, scale=sigma)

        return dist, mu, sigma


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
        # MHA
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

    def __init__(self, num_hidden, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        """
        super(Attention, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm = nn.LayerNorm(num_hidden)

    @profile
    def forward(self, key, value, query):
        # Attention

        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        residual = query

        # Make multihead
        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(key, value, query)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)

        # Concatenate context vector with input (most important)
        result = torch.cat([residual, result], dim=-1)

        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + residual

        # Layer normalization
        result = self.layer_norm(result)

        return result, attns


class LatentModel(nn.Module):
    def __init__(self, num_hidden, d_x=1, d_y=1):
        super(LatentModel, self).__init__()
        self.latent_encoder = LatentEncoder(num_hidden, num_hidden, input_dim=d_x+d_y)
        self.deterministic_encoder = DeterministicEncoder(num_hidden, input_dim=d_x+d_y)
        self.decoder = Decoder(num_hidden)
        self.BCELoss = nn.BCELoss()
        self.optim_state_dict = self.state_dict()

    @profile
    def forward(self, query, target_y=None):
        # LatentModel
        (context_x, context_y), target_x = query
        num_targets = target_x.size(1)

        n_context = context_x.shape[1]
        c, z = self.latent_encoder(context_x, context_y, target_x)
        r = self.deterministic_encoder(context_x, context_y, target_x)  # [B, T_target, H]

        # mu should be the prediction of target y
        dist, mu, sigma = self.decoder(r, z, c, target_x)

        # For Training
        if target_y is not None:
            log_p = dist.log_prob(target_y).squeeze()

            prior_g_mu, prior_g_sigma, prior_l_mu, prior_l_sigma = self.latent_encoder.latent_context(context_x, context_y)
            posterior_g_mu, posterior_g_sigma, posterior_l_mu, posterior_l_sigma = self.latent_encoder.latent_context(target_x, target_y)

            # global
            global_posterior = torch.distributions.Normal(loc=posterior_g_mu, scale=posterior_g_sigma)
            global_prior = torch.distributions.Normal(loc=prior_g_mu, scale=prior_g_sigma)
            global_kl = torch.distributions.kl_divergence(global_posterior, global_prior).sum(dim=-1, keepdims=True)
            global_kl = global_kl.repeat([1, num_targets])


            # local
            posterior_l_mu = posterior_l_mu.mean(dim=1, keepdims=True).repeat([1, n_context, 1])
            posterior_l_sigma = posterior_l_sigma.mean(dim=1, keepdims=True).repeat([1, n_context, 1])
            local_posterior = torch.distributions.Normal(loc=posterior_l_mu, scale=posterior_l_sigma)

            local_prior = torch.distributions.Normal(loc=prior_l_mu, scale=prior_l_sigma)

            local_kl = torch.distributions.kl_divergence(local_posterior, local_prior).sum(dim=[1, 2])
            local_kl = local_kl.unsqueeze(1).repeat([1, num_targets])

            loss = - (log_p - global_kl / torch.tensor(num_targets).float() - local_kl / torch.tensor(num_targets).float()).mean()

        # For Generation
        else:
            log_p = None
            global_kl = None
            local_kl = None
            loss = None

        return mu, sigma, log_p, global_kl, local_kl, loss

    def save_to_optim(self):
        self.optim_state_dict = self.state_dict()

    def load_from_optim(self):
        self.load_state_dict(self.optim_state_dict)


def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = 0.001 * warmup_step ** 0.5 * min(step_num * warmup_step ** -1.5, step_num ** -0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# @profile
def plot_functions(path, ep, target_x, target_y, context_x, context_y, pred_y, std):
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
    file_path = os.path.join(path, 'test_{}.png'.format(ep))
    if not os.path.exists(file_path):
        fig.savefig(os.path.join(path, 'test_{}.png'.format(ep)))
    else:
        fig.savefig(os.path.join(path, 'test_{}_1.png'.format(ep)))
    plt.close(fig)


def save_model(path, ep, model, optimizer):
    save_path = os.path.join(path, "saved_model.pt")
    torch.save({
        'ep': ep,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)


def load_model(path, model, optimizer):
    load_path = os.path.join(path, "saved_model.pt")
    if not os.path.exists(load_path):
        return False

    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(tu.device)
    model.eval()

    return checkpoint['ep']


class Configs:
    def __init__(self, ts_nm):
        # data setting
        self.ts_nm = ts_nm
        self.max_context_points = 100
        self.batch_size = 128
        self.predict_length = 60
        self.seq_len = 1    # anpp: 1 / asnp: >= 1

        # train setting
        self.train_iter = 1000
        self.eval_iter = 1
        self.plot_after = 1000
        self.max_epoch = 100

        # model setting
        self.num_hidden = 64
        self.learning_rate = 1e-3

        # output setting
        self._base_out_path = './anp/out/anpp/'
        self.set_ts_nm(self.ts_nm)

    def get_folder_nm(self, date_=None, nickname=''):
        if date_ is None:
            date_ = datetime.today().strftime('%Y%m%d')

        return '{}_{}_{}'.format(date_, self.ts_nm, nickname)

    def make_path(self, out_folder):
        self.out_path = os.path.join(self._base_out_path, out_folder)
        os.makedirs(self.out_path, exist_ok=False)

    def set_ts_nm(self, ts_nm):
        assert ts_nm in ['mkt_rf', 'smb', 'hml', 'rmw', 'wml', 'call_rate', 'kospi', 'mom', 'beme', 'gpa', 'usdkrw']
        self.ts_nm = ts_nm

        date_ = datetime.today().strftime('%Y%m%d')
        dir_count = len([x for x in os.listdir(self._base_out_path)
                         if os.path.isdir(os.path.join(self._base_out_path, x))])
        out_folder = self.get_folder_nm(date_, str(dir_count+1))
        self.make_path(out_folder)


def train(base_i, configs, dataset, model, optimizer, is_train=True):
    if is_train:
        iter = configs.train_iter
        model.train()
    else:
        iter = configs.eval_iter
        model.eval()

    losses = 0
    for it in range(iter):
        data_train = dataset.generate(base_i, seq_len=configs.seq_len, is_train=is_train)
        # data_train = dataset_train.generate_curves()
        data_train.to(tu.device)
        # Define the loss
        ((c_x, c_y), t_x), t_y = data_train.query,  data_train.target_y
        train_query = ((c_x[0], c_y[0]), t_x[0])
        train_target_y = t_y[0]
        with torch.set_grad_enabled(is_train):
            _, _, log_prob, _, _, loss = model(train_query, train_target_y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses += tu.np_ify(loss)

    losses = losses / iter
    return losses


def plot(date_i, plot_i, path, dataset, model):
    model.eval()

    try:
        data_test = dataset.generate(date_i, seq_len=1, is_train=False)
        data_test.to(tu.device)
    except:
        print('data generation error: {}'.format(date_i))
        return False

    with torch.set_grad_enabled(False):
        ((c_x, c_y), t_x), t_y = data_test.query, data_test.target_y
        test_query = ((c_x[0], c_y[0]), t_x[0])
        test_target_y = t_y[0]
        _, _, log_prob, _, _, loss = model(test_query, test_target_y)

        # Get the predicted mean and variance at the target points for the testing set
        mu, sigma, _, _, _, _ = model(test_query)

    loss_value, pred_y, std_y, target_y, ((context_x, context_y), target_x) = tu.np_ify(loss), tu.np_ify(
        mu), tu.np_ify(sigma), tu.np_ify(test_target_y), test_query
    context_x, context_y, target_x = tu.np_ify(context_x), tu.np_ify(context_y), tu.np_ify(target_x)
    print('date: {}, loss: {}'.format(date_i, loss_value))

    # Plot the prediction and the context
    plot_functions(path, plot_i, target_x, target_y, context_x, context_y, pred_y, std_y)


def main():
    configs = Configs('kospi')
    dataset = TimeSeries(batch_size=configs.batch_size,
                         max_num_context=configs.max_context_points,
                         predict_length=configs.predict_length)

    base_y = dataset.get_timeseries(configs.ts_nm)
    dataset.prepare_entire_dataset(base_y)

    model = LatentModel(configs.num_hidden)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)

    ep = load_model(configs.out_path, model, optimizer)
    if ep is False:
        ep = 0

    model.to(tu.device)

    min_eval_loss = 99999
    earlystop_count = 0
    pred_point = configs.predict_length // 20
    for ii, base_i in enumerate(range(10 + configs.seq_len + pred_point, dataset.max_len - 1)):
        # if ii == 0:
        #     for _ in range(20):
        #         train(base_i - pred_point, configs, dataset, model, optimizer, is_train=True)

        ep = 0
        print("[base_i: {}, ep: {}]".format(base_i, ep))
        while ep < configs.max_epoch:
            eval_loss = train(base_i - pred_point, configs, dataset, model, optimizer, is_train=False)
            if ep > 10 and min_eval_loss > eval_loss:
                model.save_to_optim()
                min_eval_loss = eval_loss
                earlystop_count = 0
            else:
                earlystop_count += 1

            print("[base_i: {}, ep: {}] eval_loss: {} / count: {}".format(base_i, ep, eval_loss, earlystop_count))
            if earlystop_count >= 20:
                model.load_from_optim()
                plot(base_i - pred_point, base_i, configs.out_path, dataset, model)
                plot(base_i, base_i, configs.out_path, dataset, model)

                min_eval_loss = 99999
                earlystop_count = 0
                break
            if ep % 5:
                plot(0, base_i, configs.out_path, dataset, model)
                plot(base_i - pred_point, base_i, configs.out_path, dataset, model)

            train_loss = train(base_i - pred_point, configs, dataset, model, optimizer, is_train=True)
            ep += 1









@profile
def main2():
    ts_nm = 'kospi'
    path = './anp/out/{}/'.format(ts_nm)
    os.makedirs(path, exist_ok=True)

    TRAINING_ITERATIONS = 100000 #@param {type:"number"}
    MAX_CONTEXT_POINTS = 250 #@param {type:"number"}
    PLOT_AFTER = 1000 #@param {type:"number"}
    batch_size = 128
    base_i = 100
    dataset = TimeSeries(batch_size=batch_size, max_num_context=MAX_CONTEXT_POINTS, predict_length=120)
    base_y = dataset.get_timeseries(ts_nm)
    dataset.prepare_entire_dataset(base_y)

    model = LatentModel(128)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    ep = load_model(path, model, optimizer)
    if ep is False:
        ep = 0

    model.to(tu.device)
    for it in range(ep, TRAINING_ITERATIONS + 1):
        data_train = dataset.generate(base_i, seq_len=1, is_train=True)
        # data_train = dataset_train.generate_curves()
        data_train.to(tu.device)
        model.train()
        # Define the loss
        ((c_x, c_y), t_x), t_y = data_train.query,  data_train.target_y
        train_query = ((c_x[0], c_y[0]), t_x[0])
        train_target_y = t_y[0]
        _, _, log_prob, _, _, loss = model(train_query, train_target_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Plot the predictions in `PLOT_AFTER` intervals
        if it % PLOT_AFTER == 0:
            for ii, date_i in enumerate([base_i - 20, base_i, base_i + 20]):
                # ii = 0; date_i = base_i
                data_test = dataset.generate(date_i, seq_len=1, is_train=False)
                # data_test = dataset_test.generate_curves()
                data_test.to(tu.device)
                model.eval()
                with torch.set_grad_enabled(False):
                    ((c_x, c_y), t_x), t_y = data_test.query,  data_test.target_y
                    test_query = ((c_x[0], c_y[0]), t_x[0])
                    test_target_y = t_y[0]
                    _, _, log_prob, _, _, loss = model(test_query, test_target_y)
                    # _, _, log_prob, _, loss = model(data_train.query, data_train.target_y)

                    # Get the predicted mean and variance at the target points for the testing set
                    mu, sigma, _, _, _, _ = model(test_query)
                    # mu, sigma, _, _, _ = model(data_test.query)
                loss_value, pred_y, std_y, target_y, ((context_x, context_y), target_x) = tu.np_ify(loss), tu.np_ify(mu), tu.np_ify(sigma), tu.np_ify(test_target_y), test_query
                context_x, context_y, target_x = tu.np_ify(context_x), tu.np_ify(context_y), tu.np_ify(target_x)
                print('Iteration: {}, loss: {}'.format(it, loss_value))

                # Plot the prediction and the context
                plot_functions(path, it + ii - 1, target_x, target_y, context_x, context_y, pred_y, std_y)

        # Plot the predictions in `PLOT_AFTER` intervals
        if it > 0 and it % 10000 == 0:

            path_all = './anp/out/{}/{}/'.format(ts_nm, it)
            os.makedirs(path_all, exist_ok=True)
            for ii in range(-base_i, dataset.max_len - base_i - 1):
                try:
                    data_test = dataset.generate(date_i + ii, seq_len=1, is_train=False)
                    data_test.to(tu.device)
                except:
                    print('data generation error: {}'.format(ii))
                    continue
                # data_test = dataset_test.generate_curves()
                # data_test = to_device(data_test, 'cuda:0')
                model.eval()
                with torch.set_grad_enabled(False):
                    ((c_x, c_y), t_x), t_y = data_test.query,  data_test.target_y
                    test_query = ((c_x[0], c_y[0]), t_x[0])
                    test_target_y = t_y[0]
                    _, _, log_prob, _, _, loss = model(test_query, test_target_y)
                    # _, _, log_prob, _, loss = model(data_train.query, data_train.target_y)

                    # Get the predicted mean and variance at the target points for the testing set
                    mu, sigma, _, _, _, _ = model(test_query)
                    # mu, sigma, _, _, _ = model(data_test.query)
                loss_value, pred_y, std_y, target_y, ((context_x, context_y), target_x) = tu.np_ify(loss), tu.np_ify(mu), tu.np_ify(sigma), tu.np_ify(test_target_y), test_query
                context_x, context_y, target_x = tu.np_ify(context_x), tu.np_ify(context_y), tu.np_ify(target_x)
                print('Iteration: {}, loss: {}'.format(it, loss_value))

                # Plot the prediction and the context
                plot_functions(path_all, it + ii, target_x, target_y, context_x, context_y, pred_y, std_y)








