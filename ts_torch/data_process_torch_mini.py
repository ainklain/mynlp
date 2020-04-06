
import pandas as pd
import pickle
import time
import torch
import torch.cuda.nvtx as nvtx
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from ts_torch import torch_util_mini as tu
from ts_torch.logger_torch import logger
import numpy as np
import os


# SWA start
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.to(param1.device).data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def schedule(epoch, configs):
    t = (epoch) / (configs.swa_start if configs.use_swa else configs.epochs)
    lr_ratio = configs.swa_lr / configs.lr_init if configs.use_swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return configs.lr_init * factor

# SWA end
def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


def normalize(x):
    return (x - np.mean(x)) / np.std(x, ddof=1)


def cleansing_missing_value(df_selected, n_allow_missing_value=5, to_log=True, reset_first_value=True):
    mask = np.sum(df_selected.isna(), axis=0) <= n_allow_missing_value
    df = df_selected.loc[:, mask].ffill().bfill()
    if reset_first_value:
        df = df / df.iloc[0]

    if to_log:
        df = np.log(df)

    return df


def done_decorator(f):
    def decorated(*args, **kwargs):
        print("{} ...ing".format(f.__name__))
        result = f(*args, **kwargs)
        print("{} ...done".format(f.__name__))
        return result
    return decorated


def done_decorator_with_logger(f):
    def decorated(*args, **kwargs):     # args[0] : self
        args[0].logger.info("[%s] start...", f.__name__)
        result = f(*args, **kwargs)
        args[0].logger.info("[%s] done...", f.__name__)
        return result
    return decorated


class DataScheduler:
    def __init__(self, configs, features_cls):
        self.data_generator = DataGeneratorDynamic(features_cls)
        self.data_generator._initialize(configs) # TODO: 임시
        # self.data_market = DataGeneratorMarket(features_cls, c.data_type_mm)
        self.configs = configs
        self.features_cls = features_cls
        self.retrain_days = configs.retrain_days

        self._initialize(configs)

    def _initialize(self, configs):
        self.base_idx = configs.train_set_length
        self.train_begin_idx = 0
        self.eval_begin_idx = int(configs.train_set_length * configs.trainset_rate)
        self.test_begin_idx = self.base_idx - configs.m_days
        self.test_end_idx = self.base_idx + configs.retrain_days

        self._make_dir(configs)

        self.logger = logger(self.__class__.__name__, configs)
        self.logger_train = logger(self.__class__.__name__ + '_train',
                                   configs,
                                   filename='train_log',
                                   use_stream_handler=True)

        self.dataloader = {'train': False, 'eval': False}

        if configs.univ_type == 'selected':
            self.set_idx(5100)

    def _make_dir(self, configs):
        # data path for fetching data
        self.data_path = os.path.join(os.getcwd(), 'data', '{}_{}_{}_{}'.format(configs.univ_type, configs.sampling_days, configs.m_days, configs.data_type))
        os.makedirs(self.data_path, exist_ok=True)
        # make a directory for outputs
        self.data_out_path = os.path.join(os.getcwd(), configs.data_out_path, self.configs.f_name)
        os.makedirs(self.data_out_path, exist_ok=True)

    @done_decorator_with_logger
    def set_idx(self, base_idx):
        c = self.configs

        self.base_idx = base_idx
        self.train_begin_idx = np.max([0, base_idx - c.train_set_length])
        # self.train_begin_idx = 4500
        self.eval_begin_idx = int(c.train_set_length * c.trainset_rate) + np.max([0, base_idx - c.train_set_length])
        self.test_begin_idx = base_idx - c.m_days
        # self.test_end_idx = base_idx + c.retrain_days
        self.test_end_idx = min(base_idx + c.retrain_days, self.data_generator.max_length - c.k_days - 1)
        self.logger.info('[set_idx] base_idx: %d / train_begin_idx: %d / eval_begin_idx: %d / test_begin_idx: %d / test_end_idx: %d'
                         , self.base_idx, self.train_begin_idx, self.eval_begin_idx, self.test_begin_idx, self.test_end_idx)

    def get_data_params(self, mode='train'):
        c = self.configs
        dg = self.data_generator
        data_params = dict()

        if mode == 'train':
            start_idx = self.train_begin_idx + c.m_days
            end_idx = self.eval_begin_idx - c.k_days
            data_params['balance_class'] = True
            data_params['label_type'] = 'trainable_label'   # trainable: calc_length 반영
            decaying_factor = c.train_decaying_factor   # 기간별 샘플 중요도
        elif mode == 'eval':
            start_idx = self.eval_begin_idx + c.m_days
            end_idx = self.test_begin_idx - c.k_days
            data_params['balance_class'] = True
            data_params['label_type'] = 'trainable_label'   # trainable: calc_length 반영
            decaying_factor = 1.   # 기간별 샘플 중요도
        elif mode == 'test':
            start_idx = self.test_begin_idx + c.m_days
            # start_idx = self.test_begin_idx
            end_idx = self.test_end_idx
            data_params['balance_class'] = False
            data_params['label_type'] = 'test_label'        # test: 예측하고자 하는 것만 반영 (k_days)
            decaying_factor = 1.   # 기간별 샘플 중요도
        elif mode == 'test_insample':
            start_idx = self.train_begin_idx + c.m_days
            end_idx = self.test_begin_idx - c.k_days
            data_params['balance_class'] = False
            data_params['label_type'] = 'test_label'        # test: 예측하고자 하는 것만 반영 (k_days)
            decaying_factor = 1.   # 기간별 샘플 중요도
        elif mode == 'test_insample2':      # train/eval/test 모두 포함 # TODO 원상복구 필요!!
            start_idx = self.train_begin_idx + c.retrain_days
            # end_idx = self.eval_begin_idx
            # start_idx = self.eval_begin_idx - c.retrain_days
            end_idx = self.test_end_idx
            data_params['balance_class'] = False
            data_params['label_type'] = 'test_label'        # test: 예측하고자 하는 것만 반영 (k_days)
            data_params['lengths'] = [(self.eval_begin_idx - start_idx) // c.k_days, (self.test_begin_idx - start_idx) // c.k_days]
            decaying_factor = 1.   # 기간별 샘플 중요도
        elif mode == 'predict':
            start_idx = self.test_begin_idx + c.m_days
            # start_idx = self.test_begin_idx
            end_idx = self.test_end_idx
            data_params['balance_class'] = False
            data_params['label_type'] = None            # label 없이 과거데이터만으로 스코어 산출
            decaying_factor = 1.   # 기간별 샘플 중요도
        else:
            raise NotImplementedError

        if max(start_idx, end_idx) < len(dg.date_):
            print("start idx:{} ({}) / end idx: {} ({})".format(start_idx, dg.date_[start_idx], end_idx, dg.date_[end_idx]))

        return start_idx, end_idx, data_params, decaying_factor

    def _fetch_data(self, date_i, force_calc=False):
        dg = self.data_generator
        data_path = self.data_path
        key_list = self.configs.key_list
        c = self.configs

        file_nm = os.path.join(data_path, '{}.pkl'.format(date_i))
        if force_calc or not os.path.exists(file_nm):
            self.logger.debug('[_fetch_data]: %d', date_i)
            result = dg.sample_data(date_i, c)
            if result is False:
                self.logger.debug('[_fetch_data]: %d False returned', date_i)
                return None

            pickle.dump(result, open(file_nm, 'wb'))
            self.logger.debug('[_fetch_data]: %d saved to %s', date_i, file_nm)

        else:
            result = pickle.load(open(file_nm, 'rb'))
            self.logger.debug('[_fetch_data]: %d loaded from %s', date_i, file_nm)

        features_dict, labels_dict, spot_dict = result

        n_assets = len(spot_dict['asset_list'])
        n_features = len(key_list)
        M = c.m_days // c.sampling_days + 1

        question = np.stack([features_dict[key] for key in key_list], axis=-1).astype(np.float32)
        question = np.transpose(question, axes=(1, 0, 2))
        assert question.shape == (n_assets, M, n_features)

        answer = np.zeros([n_assets, 2, n_features], dtype=np.float32)

        answer[:, 0, :] = question[:, -1, :]  # temporary
        answer[:, 1, :] = np.stack([labels_dict[key] if labels_dict[key] is not None
                                    else np.zeros(n_assets) for key in key_list], axis=-1)

        if c.use_macro:
            features_macro, labels_macro = [], []
            for base_key in c.add_features.keys():  # base_key : [returns / values]
                calc_macro_list, calc_feature_list = c.add_features[base_key]
                m_idx = []
                for m in calc_macro_list:
                    m_idx.append(spot_dict['macro_list'][base_key].index(m))

                # features_dict['macro_dict'][base_key][key] shape: [M, macro_list]
                features_macro += [features_dict['macro_dict'][base_key][key][:, m_idx] for key in calc_feature_list]
                # labels_dict['macro_dict'][base_key][key] shape: [macro_list, ]
                labels_macro += [labels_dict['macro_dict'][base_key][key][m_idx] if labels_dict['macro_dict'][base_key][key] is not None
                                 else np.zeros([len(m_idx),]) for key in calc_feature_list]

            features_macro = np.tile(np.concatenate(features_macro, axis=-1)[np.newaxis, :], (n_assets, 1, 1))
            labels_macro = np.tile(np.concatenate(labels_macro, axis=-1)[np.newaxis, :], (n_assets, 1, 1))
            labels_macro = np.concatenate([features_macro[:, -1:, :], labels_macro], axis=1)

            question = np.concatenate([question, features_macro], axis=-1)
            answer = np.concatenate([answer, labels_macro], axis=-1)

        return question[:], answer[:, :-1, :], answer[:, 1:, :], spot_dict

    def nearest_d_from_m_end(self, m_end_date_list):
        date_arr = np.array(self.date_)
        nearest_d_list = [date_arr[date_arr <= d_][-1] for d_ in m_end_date_list]
        nearest_idx = np.array([self.date_.index(d_) for d_ in nearest_d_list])
        return nearest_d_list, nearest_idx

    def next_d_from_m_end(self, m_end_date_list):
        date_arr = np.array(self.date_ + ['9999-12-31'])  # 에러 방지용
        next_d_list = [date_arr[date_arr > d_][0] for d_ in m_end_date_list]
        next_idx = np.array([list(date_arr).index(d_) for d_ in next_d_list])
        return next_d_list, next_idx

    def _dataset_t(self, base_d, **kwargs):
        univ = self.data_generator.univ

        features_list = self.configs.key_list_with_macro

        recent_d, recent_idx = self.nearest_d_from_m_end([base_d])
        recent_d, recent_idx = recent_d[0], recent_idx[0]

        fetch_data = self._fetch_data(recent_idx, **kwargs)
        if fetch_data is None:
            return False

        enc_in, dec_in, dec_out, add_info = fetch_data

        assert enc_in.shape[-1] == len(features_list)   # macro 있든 없든 동일해야

        add_info['factor_d'] = base_d
        add_info['model_d'] = recent_d
        add_info['univ'] = univ[univ.eval_m == base_d]
        add_info['importance_wgt'] = np.array([1 for _ in range(len(enc_in))], dtype=np.float32)

        return enc_in, dec_in, dec_out, features_list, add_info

    def dataloader_t(self, recent_month_end, **kwargs):
        c = self.configs
        _dataset_t = self._dataset_t(recent_month_end, **kwargs)

        enc_in, dec_in, dec_out, features_list, add_infos = _dataset_t

        enc_in[np.isnan(enc_in)] = 0
        dec_in[np.isnan(dec_in)] = 0

        new_dec_in = np.zeros_like(enc_in)
        new_dec_in[:, 0, :] = enc_in[:, 0, :]

        if c.size_encoding:
            new_dec_in[:] += np.array(add_infos['size_rnk']).reshape(-1, 1, 1)

        features = {'input': torch.from_numpy(enc_in), 'output': torch.from_numpy(new_dec_in)}
        dataloader = [features, add_infos]
        return dataloader, features_list, add_infos['asset_list'], None, None

    @done_decorator_with_logger
    def _dataset_monthly(self, mode='test'):
        assert mode in ['test', 'test_insample', 'test_insample2', 'predict']
        c = self.configs
        dg = self.data_generator
        prc_df = dg.df_pivoted_all
        univ = dg.univ

        # parameter setting
        enc_in, dec_in, dec_out = [], [], []
        additional_infos = []  # test/predict 인경우 list, train/eval인 경우 dict
        start_idx, end_idx, data_params, decaying_factor = self.get_data_params(mode)
        features_list = c.key_list_with_macro

        idx_balance = c.key_list.index(c.balancing_key)

        # month end data setting
        factor_d_list = list(univ.eval_m.unique())
        nearest_d_list, nearest_idx = self.nearest_d_from_m_end(factor_d_list)
        selected = (nearest_idx >= start_idx) & (nearest_idx <= end_idx)
        model_idx_arr = nearest_idx[selected]
        # factor_d_arr = np.array(factor_d_list)[selected]
        # model_d_arr = np.array(nearest_d_list)[selected]

        # 수익률 산출용 (매월 마지막일 기준 스코어 산출-> 그 다음날 종가기준매매)
        next_d_list, next_idx = self.next_d_from_m_end(factor_d_list)
        # next_d_arr = np.array(next_d_list)[selected]

        n_loop = np.ceil((end_idx - start_idx) / c.sampling_days)
        for idx in model_idx_arr:
            fetch_data = self._fetch_data(idx)
            if fetch_data is None:
                continue

            i = list(nearest_idx).index(idx)
            tmp_ein, tmp_din, tmp_dout, add_info = fetch_data
            assert tmp_ein.shape[-1] == len(features_list)  # macro 있든 없든 동일해야

            # next y
            assets = add_info['asset_list']

            if next_d_list[i+1] == '9999-12-31':
                next_y = prc_df.loc[next_d_list[i], assets]
                next_y[:] = 0.
            else:
                prc_df_selected = prc_df.loc[(prc_df.index >=next_d_list[i-1]) & (prc_df.index <= next_d_list[i+1]), assets]
                prc_df_selected = prc_df_selected.ffill()
                next_y = prc_df_selected.loc[next_d_list[i+1], assets] / prc_df_selected.loc[next_d_list[i], assets] - 1

            add_info['next_y'] = next_y
            add_info['factor_d'] = factor_d_list[i]
            add_info['model_d'] = nearest_d_list[i]
            add_info['inv_d'] = next_d_list[i]
            add_info['univ'] = univ[univ.eval_m == factor_d_list[i]]
            add_info['importance_wgt'] = np.array([decaying_factor ** (n_loop - i - 1)
                                                          for _ in range(len(tmp_ein))], dtype=np.float32)

            balancing_list = ['mktcap', 'size_rnk', 'importance_wgt']
            for nm_ in balancing_list:
                add_info[nm_] = np.array(add_info[nm_], dtype=np.float32).squeeze()

            enc_in.append(tmp_ein)
            dec_in.append(tmp_din)
            dec_out.append(tmp_dout)
            additional_infos.append(add_info)

        if len(enc_in) == 0:
            return False

        start_date = self.date_[start_idx]
        end_date = self.date_[end_idx]

        return enc_in, dec_in, dec_out, features_list, additional_infos, start_date, end_date

    @done_decorator_with_logger
    def _dataset(self, mode='train'):
        c = self.configs

        enc_in, dec_in, dec_out = [], [], []
        add_infos_list = []  # test/predict 인경우 list, train/eval인 경우 dict
        start_idx, end_idx, data_params, decaying_factor = self.get_data_params(mode)
        features_list = c.key_list_with_macro

        self.logger.info('[_dataset]: mode:%s / start_i:%d / end_i:%d', mode, start_idx, end_idx)
        idx_balance = c.key_list.index(c.balancing_key)

        balancing_list = ['mktcap', 'size_rnk', 'importance_wgt']   # TODO: configs로 옮겨야됨
        n_loop = np.ceil((end_idx - start_idx) / c.sampling_days)
        for i, d in enumerate(range(start_idx, end_idx, c.sampling_days)):
            # try:
            fetch_data = self._fetch_data(d)
            # except Exception as e:
            #     self.logger.error("[_dataset] self._fetch_data error (d=%d)", d)
            #     self.logger.error("[_dataset] features_list: %s / idx_balance %d / decaying_factor %d", features_list, idx_balance, decaying_factor)

            if fetch_data is None:
                continue

            tmp_ein, tmp_din, tmp_dout, add_info = fetch_data
            assert tmp_ein.shape[-1] == len(features_list)  # macro 있든 없든 동일해야

            add_info['importance_wgt'] = np.array([decaying_factor ** (n_loop - i - 1)
                                                          for _ in range(len(tmp_ein))], dtype=np.float32)
            if data_params['balance_class'] is True and c.balancing_method == 'each':
                self.logger.info("[_dataset] 'each' balancing_method applied.")

                idx_bal = self.balanced_index(tmp_dout[:, 0, idx_balance])
                tmp_ein, tmp_din, tmp_dout = tmp_ein[idx_bal], tmp_din[idx_bal], tmp_dout[idx_bal]
                for nm_ in balancing_list:
                    add_info[nm_] = add_info[nm_].iloc[idx_bal]

            enc_in.append(tmp_ein)
            dec_in.append(tmp_din)
            dec_out.append(tmp_dout)
            add_infos_list.append(add_info)

        if len(enc_in) == 0:
            return False

        if mode in ['train', 'eval']:
            add_infos = dict()
            enc_in = np.concatenate(enc_in, axis=0)
            dec_in = np.concatenate(dec_in, axis=0)
            dec_out = np.concatenate(dec_out, axis=0)

            if data_params['balance_class'] is True and c.balancing_method == 'once':
                self.logger.info("[_dataset] 'once' balancing_method applied.")
                idx_bal = self.balanced_index(dec_out[:, 0, idx_balance])
                enc_in, dec_in, dec_out = enc_in[idx_bal], dec_in[idx_bal], dec_out[idx_bal]

                for nm_ in balancing_list:
                    val_ = np.concatenate([np.squeeze(add_info[nm_]) for add_info in add_infos_list], axis=0)
                    add_infos[nm_] = val_[idx_bal]
            else:
                for nm_ in balancing_list:
                    val_ = np.concatenate([np.squeeze(add_info[nm_]) for add_info in add_infos_list], axis=0)
                    add_infos[nm_] = val_[:]
        else:
            add_infos = []
            for add_info in add_infos_list:
                add_info_temp = add_info.copy()
                for nm_ in balancing_list:
                    add_info_temp[nm_] = np.array(add_info[nm_], dtype=np.float32).squeeze()

                add_infos.append(add_info_temp)

        start_date = self.date_[start_idx]
        end_date = self.date_[end_idx]

        return enc_in, dec_in, dec_out, features_list, add_infos, start_date, end_date

    @done_decorator_with_logger
    def _dataset_maml(self, mode='train'):
        c = self.configs

        spt_list, tgt_list, importance_wgt = [], [], []
        start_idx, end_idx, data_params, decaying_factor = self.get_data_params(mode)
        features_list = c.key_list_with_macro

        idx_balance = c.key_list.index(c.balancing_key)

        balancing_list = ['mktcap', 'size_rnk']     # TODO: configs로 옮겨야됨
        # for i, d in enumerate(reversed(range(start_idx + c.k_days, end_idx, c.k_days))):
        # for i, d in enumerate(range(start_idx + c.k_days, end_idx, c.k_days)):
        if mode in ['test', 'test_insample', 'test_insample2', 'predict']:
            n_loop = np.ceil((end_idx - start_idx - c.k_days) / c.k_days)
            tasks = np.arange(start_idx + c.k_days, end_idx, c.k_days)
        else:
            n_loop = c.n_tasks+1
            tasks = np.random.choice(np.arange(start_idx + c.k_days, end_idx, c.k_days), c.n_tasks+1, replace=False)

        for i, d in enumerate(tasks):
            support_data = self._fetch_data(d - c.k_days)
            target_data = self._fetch_data(d)

            if support_data is None or target_data is None:
                continue
            spt_ein, spt_din, spt_dout, spt_add_info = support_data
            tgt_ein, tgt_din, tgt_dout, tgt_add_info = target_data

            assert spt_ein.shape[-1] == len(features_list)  # macro 있든 없든 동일해야
            assert tgt_ein.shape[-1] == len(features_list)  # macro 있든 없든 동일해야

            importance_wgt.append(decaying_factor ** (n_loop - i - 1))

            if mode in ['train','eval'] and c.pred_feature.split('_')[0] in c.features_structure['classification'].keys():
                spt_idx_bal = self.balanced_index(spt_dout[:, 0, idx_balance])
                spt_ein, spt_din, spt_dout = spt_ein[spt_idx_bal], spt_din[spt_idx_bal], spt_dout[spt_idx_bal]

                tgt_idx_bal = self.balanced_index(tgt_dout[:, 0, idx_balance])
                tgt_ein, tgt_din, tgt_dout = tgt_ein[tgt_idx_bal], tgt_din[tgt_idx_bal], tgt_dout[tgt_idx_bal]
                for nm_ in balancing_list:
                    spt_add_info[nm_] = spt_add_info[nm_].iloc[spt_idx_bal]
                    tgt_add_info[nm_] = tgt_add_info[nm_].iloc[tgt_idx_bal]

            for nm_ in balancing_list:
                spt_add_info[nm_] = np.array(spt_add_info[nm_], dtype=np.float32).squeeze()
                tgt_add_info[nm_] = np.array(tgt_add_info[nm_], dtype=np.float32).squeeze()

            # TODO: 임시 처리 (nmsize nan값 0 처리)
            spt_ein[np.isnan(spt_ein)] = 0
            spt_din[np.isnan(spt_din)] = 0
            spt_dout[np.isnan(spt_dout)] = 0
            tgt_ein[np.isnan(tgt_ein)] = 0
            tgt_din[np.isnan(tgt_din)] = 0
            tgt_dout[np.isnan(tgt_dout)] = 0

            assert np.nanmax(np.abs(spt_ein[:, -1, :] - spt_din[:, 0, :])) == 0
            assert np.nanmax(np.abs(tgt_ein[:, -1, :] - tgt_din[:, 0, :])) == 0
            # 미래데이터 안 땡겨쓰게
            spt_new_dec_in = np.zeros_like(spt_din)
            spt_new_dec_in[:, 0, :] = spt_din[:, 0, :]

            tgt_new_dec_in = np.zeros_like(tgt_din)
            tgt_new_dec_in[:, 0, :] = tgt_din[:, 0, :]

            if c.size_encoding:
                spt_new_dec_in[:] += spt_add_info['size_rnk'].reshape(-1, 1, 1)
                tgt_new_dec_in[:] += tgt_add_info['size_rnk'].reshape(-1, 1, 1)

            # plot용
            idx_y = features_list.index(c.label_feature)
            tgt_add_info['next_y'] = np.exp(tgt_dout[:, 0, idx_y]) - 1.
            # tgt_add_info['next_y'] = tgt_dout[:, 0, idx_y]

            spt_list.append([spt_ein, spt_din, spt_dout, spt_add_info])
            tgt_list.append([tgt_ein, tgt_din, tgt_dout, tgt_add_info])

        if len(spt_list) == 0:
            return False

        start_date = self.date_[start_idx]
        end_date = self.date_[end_idx]

        return spt_list, tgt_list, features_list, importance_wgt, start_date, end_date

    def balanced_index(self, balance_arr):
        where_p = (balance_arr > 0)
        where_n = (balance_arr < 0)
        if np.min([np.sum(where_p), np.sum(where_n)]) == 0:
            return np.arange(len(balance_arr))
            # return np.array(np.ones_like(balance_arr), dtype=bool)

        n_max = np.max([np.sum(where_p), np.sum(where_n)])
        idx_pos = np.concatenate([np.random.choice(np.where(where_p)[0], np.sum(where_p), replace=False),
                                  np.random.choice(np.where(where_p)[0], n_max - np.sum(where_p), replace=True)])
        idx_neg = np.concatenate([np.random.choice(np.where(where_n)[0], np.sum(where_n), replace=False),
                                  np.random.choice(np.where(where_n)[0], n_max - np.sum(where_n), replace=True)])

        idx_bal = np.concatenate([idx_pos, idx_neg])
        return idx_bal

    @done_decorator_with_logger
    def _dataloader(self, mode, is_monthly=False):
        # self = ds; mode = 'test'; is_monthly=False
        c = self.configs
        batch_size = dict(train=c.train_batch_size, eval=c.eval_batch_size, test=1)

        self.logger.info("[_dataloader] mode: %s / is_monthly: %s", mode, is_monthly)
        if is_monthly:
            assert mode in ['test', 'test_insample', 'test_insample2', 'predict']
            _dataset = self._dataset_monthly(mode)
        else:
            _dataset = self._dataset(mode)

        if _dataset is False:
            self.logger.info("[_dataloader] [mode: %s] no data (_dataset is False)", mode)
            return False

        enc_in, dec_in, dec_out, features_list, add_infos, start_d, end_d = _dataset

        if mode in ['train', 'eval']:
            # TODO: 임시 처리 (nmsize nan값 0 처리)
            enc_in[np.isnan(enc_in)] = 0
            dec_in[np.isnan(dec_in)] = 0
            dec_out[np.isnan(dec_out)] = 0
            assert np.nanmax(np.abs(enc_in[:, -1, :] - dec_in[:, 0, :])) == 0
            # 미래데이터 안 땡겨쓰게
            new_dec_in = np.zeros_like(dec_in)
            new_dec_in[:, 0, :] = dec_in[:, 0, :]

            if c.size_encoding:
                new_dec_in[:] += add_infos['size_rnk'].reshape(-1, 1, 1)

            dataloader = data_loader(enc_in, new_dec_in, dec_out, add_infos, batch_size=batch_size[mode], num_workers=c.num_workers)
            self.logger.info("[_dataloader] mode: %s / batchsize: %s", mode, batch_size)
            return dataloader, features_list

        elif mode in ['test', 'predict', 'test_insample', 'test_insample2']:
            idx_y = features_list.index(c.label_feature)
            idx_pred = features_list.index(c.get_main_feature(c.pred_feature))
            all_assets_list = list()
            features = list()
            for ein_t, din_t, dout_t, add_info in zip(enc_in, dec_in, dec_out, add_infos):
                # ein_t, din_t, dout_t, add_info = next(iter(zip(enc_in, dec_in, dec_out, add_infos)))
                all_assets_list = sorted(list(set(all_assets_list + add_info['asset_list'])))
                # TODO: 임시 처리 (nmsize nan값 0 처리)
                ein_t[np.isnan(ein_t)] = 0
                din_t[np.isnan(din_t)] = 0
                dout_t[np.isnan(dout_t)] = 0
                # data format
                assert np.nanmax(np.abs(ein_t[:, -1, :] - din_t[:, 0, :])) == 0
                # 미래데이터 안 땡겨쓰게
                new_din_t = np.zeros_like(din_t)
                new_din_t[:, 0, :] = din_t[:, 0, :]

                # label 값 (t+1수익률)
                # add_info_temp['next_y'] = np.exp(tmp_dout[:, 0, idx_y]) - 1.

                # if is_monthly:
                #     print("TEST", np.sum(add_info['next_y'] - (np.exp(dout_t[:, 0, idx_y]) - 1)))
                # add_info['next_y'] = dout_t[:, 0, idx_y]
                if not is_monthly:
                    add_info['next_y'] = np.exp(dout_t[:, 0, idx_y]) - 1.

                add_info['next_label'] = dout_t[:, 0, idx_pred]

                if c.size_encoding:
                    new_din_t[:] += np.array(add_info['size_rnk']).reshape(-1, 1, 1)

                # torch로 변환
                features.append({'input': torch.from_numpy(ein_t), 'output': torch.from_numpy(new_din_t)})

            dataloader = [features, add_infos]
            return dataloader, features_list, all_assets_list, start_d, end_d
        else:
            raise NotImplementedError

    @done_decorator_with_logger
    def _dataloader_maml(self, mode):
        # self = ds; mode = 'train'
        _dataset = self._dataset_maml(mode)

        if _dataset is False:
            self.logger.info("[_dataloader_maml] [mode: %s] no data (_dataset is False)", mode)
            return False

        spt_list, tgt_list, features_list, importance_wgt, start_date, end_date = _dataset
        if mode in ['train', 'eval']:
            n_tasks = self.configs.n_tasks
            if n_tasks < 0:
                dataloader = data_loader_maml(spt_list, tgt_list, sampler=None, shuffle=True)
            else:
                sampler = WeightedRandomSampler(importance_wgt, self.configs.n_tasks, replacement=False)
                dataloader = data_loader_maml(spt_list, tgt_list, sampler=sampler)
            all_assets_list = []

        elif mode in ['test', 'predict', 'test_insample', 'test_insample2']:
            all_assets_list = []
            for spt_, tgt_ in zip(spt_list, tgt_list):
                # spt_, tgt_ = spt_list[0], tgt_list[0]
                all_assets_list = sorted(list(set(all_assets_list + list(spt_[-1]['asset_list']) + list(tgt_[-1]['asset_list']))))

            dataloader = data_loader_maml(spt_list, tgt_list, sampler=None)
            # dataloader = [spt_list, tgt_list]

        return dataloader, features_list, all_assets_list, start_date, end_date

    @done_decorator_with_logger
    def train(self, model, optimizer, scheduler, performer, num_epochs):
        early_stopping_count = self.configs.early_stopping_count
        
        min_eval_loss = 99999
        stop_count = 0
        for ep in range(num_epochs):
            # if ep % 50 == 0:
            #     lr = 0.01 / (ep / 50 + 1)
            # else:
            #     lr = lr * (1 - (ep % 50) / 50)
            #
            # adjust_learning_rate(optimizer, lr)
            if ep % 5 == 0:
                self.logger.info("[train] [Ep %d] plot", ep)
                # self.test_plot(performer, model, ep, is_monthly=False)
                # self.test_plot(performer, model, ep, is_monthly=True)
                self.test_plot(performer, model, ep, is_monthly=False, is_insample=True)
                self.test_plot(performer, model, ep, is_monthly=True, is_insample=True)

            self.logger.info("[train] [Ep %d] model evaluation ...", ep)
            eval_loss = self.step_epoch(ep, model, optimizer, scheduler=scheduler, is_train=False)
            if eval_loss is False:
                return False

            if eval_loss > min_eval_loss:
                stop_count += 1
            else:
                model.save_to_optim()
                min_eval_loss = eval_loss
                stop_count = 0

            self.logger.info("[train] [Ep %d] count: %d/%d", ep, stop_count, early_stopping_count)
            self.logger_train.info("[train] [Ep %d] count: %d/%d", ep, stop_count, early_stopping_count)
            if stop_count >= early_stopping_count:
                self.logger.info("[train] [Ep %d] Early Stopped", ep)
                self.logger_train.info("[train] [Ep %d] Early Stopped", ep)
                model.load_from_optim()
                self.test_plot(performer, model, ep + 100, is_monthly=False)
                self.test_plot(performer, model, ep + 100, is_monthly=True, is_insample=True)

                break

            with torch.autograd.profiler.profile() as prof:
                train_loss = self.step_epoch(ep, model, optimizer, scheduler=scheduler, is_train=True)

            print(prof.key_averages().table(sort_by="self_cpu_time_total"))

            if train_loss is False:
                return False

    @done_decorator_with_logger
    def train_swa(self, model, model_swa, optimizer, scheduler, performer, num_epochs):
        c = self.configs

        swa_start = c.swa_start
        eval_freq = c.eval_freq
        swa_c_epochs = c.swa_c_epochs
        early_stopping_count = c.early_stopping_count

        min_eval_loss = 99999
        stop_count = 0
        swa_n = 0
        for ep in range(num_epochs):
            # lr = schedule(epoch=ep, configs=c)
            # adjust_learning_rate(optimizer, lr)

            if ep == 0 or ep % eval_freq == 0:
                self.logger.info("[train] [Ep %d] plot", ep)
                # self.test_plot(performer, model, ep, is_monthly=False)
                # self.test_plot(performer, model, ep, is_monthly=True)
                self.test_plot(performer, model_swa, ep, is_monthly=True, nickname='_swa', is_insample=True)
                self.test_plot(performer, model, ep, is_monthly=True, is_insample=True)

            self.logger.info("[train] [Ep %d] model evaluation ...", ep)
            if ep % eval_freq == 0 or ep == num_epochs - 1:
                eval_loss = self.step_epoch(ep, model, optimizer, scheduler=scheduler, is_train=False)
                if eval_loss is False:
                    return False

            if (ep + 1) >= swa_start and (ep + 1 - swa_start) % swa_c_epochs == 0:
                moving_average(model_swa, model, 1.0 / (swa_n + 1))
                swa_n += 1
                if ep % eval_freq == 0 or ep == num_epochs - 1:
                    swa_eval_loss = self.step_epoch(ep, model_swa, optimizer, scheduler=scheduler, is_train=False, use_swa=True)
                    print('eval: {} swa: {}'.format(eval_loss, swa_eval_loss))
                else:
                    swa_eval_loss = None


            if ep % eval_freq == 0 or ep == num_epochs - 1:
                if eval_loss > min_eval_loss:
                    stop_count += 1
                else:
                    model.save_to_optim()
                    min_eval_loss = eval_loss
                    stop_count = 0

            self.logger.info("[train] [Ep %d] count: %d/%d", ep, stop_count, early_stopping_count)
            self.logger_train.info("[train] [Ep %d] count: %d/%d", ep, stop_count, early_stopping_count)
            if stop_count >= early_stopping_count:
                self.logger.info("[train] [Ep %d] Early Stopped", ep)
                self.logger_train.info("[train] [Ep %d] Early Stopped", ep)
                model.load_from_optim()
                self.test_plot(performer, model, ep + 100, is_monthly=False)
                self.test_plot(performer, model, ep + 100, is_monthly=True, is_insample=True)

                break

            train_loss = self.step_epoch(ep, model, optimizer, scheduler=scheduler, is_train=True)
            if train_loss is False:
                return False

    @done_decorator_with_logger
    def train_maml(self, model, optimizer, performer, num_epochs, early_stopping_count=2):
        min_eval_loss = 99999
        stop_count = 0
        for ep in range(num_epochs):
            if ep % 5 == 0:
                self.logger.info("[train_maml] [Ep %d] plot", ep)
                self.logger_train.info("[train_maml] [Ep %d] plot", ep)
                self.test_plot_maml(performer, model, ep, is_monthly=False)
                # self.test_plot_maml(performer, model, ep, is_monthly=False, is_insample=True)
                self.test_plot(performer, model, ep, is_monthly=False)
                # self.test_plot(performer, model, ep, is_monthly=False, is_insample=True)

            self.logger.info("[train_maml] [Ep %d] model evaluation ...", ep)
            eval_loss = self.step_epoch_maml(ep, model, optimizer, is_train=False)
            if eval_loss is False:
                return False

            if eval_loss > min_eval_loss:
                model.load_from_optim()
                stop_count += 1
            else:
                model.save_to_optim()
                min_eval_loss = eval_loss
                stop_count = 0

            self.logger.info("[train_maml] [Ep %d] count: {}/{}", ep, stop_count, early_stopping_count)
            if stop_count >= early_stopping_count:
                self.logger.info("[train] [Ep %d] Early Stopped", ep)
                model.load_from_optim()
                self.test_plot_maml(performer, model, ep, is_monthly=False)

                break

            # print('[Ep {}] model train ...'.format(ep))
            train_loss = self.step_epoch_maml(ep, model, optimizer, is_train=True)
            if train_loss is False:
                return False

    # @profile
    def step_epoch(self, ep, model, optimizer, scheduler=None, is_train=True, use_swa=False):
        if is_train:
            mode = 'train'
            model.train()
        else:
            mode = 'eval' # TODO 원상복구 필요!!
            model.eval()
            # mode = 'train'
            # model.train()

        if ep == 0:
            self.dataloader[mode] = self._dataloader(mode)
            if self.dataloader[mode] is False:
                return False

        dataloader, features_list = self.dataloader[mode]
        if ep == 0:
            self.logger.info("[step_epoch][Ep %d][%s] f_list: %s", ep, mode, features_list)

        if scheduler is not None:
            scheduler.step()
            if is_train:
                self.logger.info("[step_epoch][Ep %d] learning rate: %d", ep, optimizer.param_groups[0]['lr'])

        if use_swa:
            bn_update(dataloader, model)
        total_loss = 0
        i = 0

        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            for features, labels, add_infos in dataloader:
                #  features, labels, add_infos = next(iter(dataloader))
                # nvtx.range_push('batch start')
                # nvtx.range_push('copy to device')
                features, labels, add_infos = to_device(tu.device, [features, labels, add_infos])
                # nvtx.range_pop()
                with torch.set_grad_enabled(is_train):
                    # nvtx.range_push('add noise')
                    features_with_noise = {'input': features['input'], 'output': features['output']}
                    labels_with_noise = labels
                    if is_train:
                        if self.configs.adversarial_training is True:
                            labels_mtl_noise = self.labels_torch(features_list, labels_with_noise, add_infos)
                            features_with_noise = Noise.adversarial_noise(features_with_noise, labels_mtl_noise, model)
                        else:
                            # add random noise for features
                            features_with_noise['input'] = Noise.random_noise(features_with_noise['input'], p=0.5)
                            features_with_noise['input'] = Noise.random_mask(features_with_noise['input'], p=0.9, mask_p=0.2)

                            # add random noise for labels
                            labels_with_noise = Noise.random_noise(labels, p=0.2)
                            labels_with_noise = Noise.random_flip(labels_with_noise, p=0.1, flip_p=0.2)
                    # nvtx.range_pop()

                    labels_mtl = self.labels_torch(features_list, labels_with_noise, add_infos)
                    # pred, _, _, _ = model.forward(features_with_noise, labels_mtl)
                    # nvtx.range_push('forward pass')
                    pred, loss_each = model.forward_with_loss(features_with_noise, labels_mtl)

                    losses = 0
                    for key in loss_each.keys():
                        losses += loss_each[key].mean()
                    # nvtx.range_pop()

                    if is_train:
                        # nvtx.range_push('backward pass')
                        optimizer.zero_grad()
                        losses.backward()
                        optimizer.step()
                        # nvtx.range_pop()

                    total_loss += losses
                    if i % 10 == 0:
                        self.logger.debug("i:%d loss:%f total_loss:%f", i, float(tu.np_ify(losses)), float(tu.np_ify(total_loss)))
                    i += 1

                # nvtx.range_pop()

        print(prof.key_averages().table(sort_by="self_cpu_time_total"))

        total_loss = tu.np_ify(total_loss) / i
        if is_train:
            print_str = "".format(ep, mode)
            size_str = "".format(ep, mode)
            for key in loss_each.keys():
                print_str += "{}- {:.4f} / ".format(key, tu.np_ify(loss_each[key].mean()))
                size_str += "{} - {} / ".format(key, loss_each[key].shape)
            self.logger.info("[step_epoch][Ep %d][%s] %s", ep, mode, print_str)
            self.logger.info("[step_epoch][Ep %d][%s][size] %s", ep, mode, size_str)
            self.logger_train.info("[step_epoch][Ep %d][%s] %s", ep, mode, print_str)

            return total_loss
        else:
            self.logger.info("[step_epoch][Ep %d][%s] total loss: %.6f", ep, mode, total_loss)
            self.logger_train.info("[step_epoch][Ep %d][%s] total loss: %.6f", ep, mode, total_loss)
            return total_loss

    def step_epoch_maml(self, ep, model, optimizer, is_train=True):
        # ep=0;is_train = True;
        c = self.configs

        if is_train:
            mode = 'train'
            model.train()
        else:
            mode = 'eval'
            # model.train()
            model.eval()

        if ep == 0:
            self.dataloader[mode] = self._dataloader_maml(mode)
            if self.dataloader[mode] is False:
                return False

        taskloader, features_list, _, _, _ = self.dataloader[mode]
        if ep == 0:
            self.logger.info("[step_epoch_maml][Ep %d][%s] f_list: %s", ep, mode, features_list)

        total_losses = 0
        n_task = 0
        for spt_ds, tgt_ds in taskloader:
            #  spt_ds, tgt_ds = next(iter(taskloader))
            features_s, labels_s, add_infos_s = spt_ds
            f_with_noise_s = {'input': features_s['input'].squeeze(0), 'output': features_s['output'].squeeze(0)}
            labels_with_noise_s = labels_s.squeeze(0)

            features_t, labels_t, add_infos_t = tgt_ds
            f_with_noise_t = {'input': features_t['input'].squeeze(0), 'output': features_t['output'].squeeze(0)}
            labels_with_noise_t = labels_t.squeeze(0)
            if is_train:
                if self.configs.adversarial_training is True:
                    labels_mtl_noise_s = self.labels_torch(features_list, labels_with_noise_s, add_infos_s)
                    labels_mtl_noise_t = self.labels_torch(features_list, labels_with_noise_t, add_infos_t)
                    f_with_noise_s = Noise.adversarial_noise(f_with_noise_s, labels_mtl_noise_s, model)
                    f_with_noise_t = Noise.adversarial_noise(f_with_noise_t, labels_mtl_noise_t, model)
                else:
                    # add random noise for features
                    f_with_noise_s['input'] = Noise.random_noise(f_with_noise_s['input'], p=0.5)
                    f_with_noise_s['input'] = Noise.random_mask(f_with_noise_s['input'], p=0.9, mask_p=0.2)

                    # add random noise for labels
                    labels_with_noise_s = Noise.random_noise(labels_with_noise_s, p=0.2)
                    labels_with_noise_s = Noise.random_flip(labels_with_noise_s, p=0.9, flip_p=0.2)

                    # add random noise for features
                    f_with_noise_t['input'] = Noise.random_noise(f_with_noise_t['input'], p=0.5)
                    f_with_noise_t['input'] = Noise.random_mask(f_with_noise_t['input'], p=0.9, mask_p=0.2)

                    # add random noise for labels
                    labels_with_noise_t = Noise.random_noise(labels_with_noise_t, p=0.2)
                    labels_with_noise_t = Noise.random_flip(labels_with_noise_t, p=0.9, flip_p=0.2)

            # TODO: maml시에 importance_wgt 사용 불가 (임시로 labels_torch에서 maml 받아서 없앰)  dataloader_maml도 수정해야
            labels_mtl_s = self.labels_torch(features_list, labels_with_noise_s, add_infos_s)
            to_device(tu.device, [f_with_noise_s, labels_mtl_s])
            # pred, _, _, _ = model.forward(features_with_noise, labels_mtl)
            weights_list = model.params2vec(requires_grad_only=True)
            pred_s, loss_each_s = model.compute_graph_with_loss(f_with_noise_s, labels_mtl_s, weights_list=weights_list)
            to_device('cpu', [f_with_noise_s, labels_mtl_s])

            train_losses = 0
            for key in loss_each_s.keys():
                train_losses += loss_each_s[key].mean()

            # train_losses.backward()
            grad = torch.autograd.grad(train_losses, weights_list, retain_graph=True, create_graph=True)
            fast_weights = list(map(lambda p: p[1] - c.inner_lr * p[0], zip(grad, weights_list)))

            with torch.set_grad_enabled(is_train):
                labels_mtl_t = self.labels_torch(features_list, labels_with_noise_t, add_infos_t)
                to_device(tu.device, [f_with_noise_t, labels_mtl_t])
                pred_t, loss_each_t = model.compute_graph_with_loss(f_with_noise_t, labels_mtl_t, weights_list=fast_weights)
                to_device('cpu', [f_with_noise_t, labels_mtl_t])

                task_losses = 0
                for key in loss_each_t.keys():
                    task_losses += loss_each_t[key].mean()

                total_losses += task_losses
            n_task += 1

        total_losses = total_losses / n_task
        if is_train:
            optimizer.zero_grad()
            total_losses.backward()
            optimizer.step()

        self.logger.info("[step_epoch_maml][Ep %d][%s] total loss: %.6f (n tasks: %d)", ep, mode, total_losses, n_task)
        self.logger_train.info("[step_epoch_maml][Ep %d][%s] total loss: %.6f (n tasks: %d)", ep, mode, total_losses, n_task)
        # print('[Ep {}][{}] total - {:.4f} (n tasks: {})'.format(ep, mode, total_losses, n_task))
        return total_losses

    @done_decorator_with_logger
    def test_plot(self, performer, model, ep, is_monthly, is_insample=False, nickname=""):
        # self=ds; ep=0; is_monthly = False; is_insample=False
        model.eval()

        if is_insample:
            mode = 'test_insample2'
            start_idx, end_idx, data_params, decaying_factor = self.get_data_params(mode)
            timeseries_lengths = data_params['lengths']
        else:
            mode = 'test'
            timeseries_lengths = None
        self_mode = 'single_' + mode

        if is_monthly:
            # mode = 'test_monthly'
            self_mode = self_mode + '_monthly'
            performer_func = performer.predict_plot_monthly
        else:
            performer_func = performer.predict_plot_mtl

        self.logger.info("[test_plot][Ep %d][%s]", ep, self_mode)
        self.logger_train.info("[test_plot][Ep %d][%s]", ep, self_mode)
        if (ep == 0) or (self.dataloader.get(self_mode) is None):
            self.dataloader[self_mode] = self._dataloader(mode, is_monthly=is_monthly)

        if self.dataloader[self_mode] is False:
            return False

        dataloader_set = self.dataloader[self_mode]
        test_out_path = os.path.join(self.data_out_path, '{}/{}'.format(self.base_idx, self_mode))
        os.makedirs(test_out_path, exist_ok=True)

        performer_func(model, dataloader_set, save_dir=test_out_path, file_nm='test_{}{}.png'.format(ep, nickname)
                       , ylog=False, ls_method='ls_5_20', plot_all_features=True, logy=False, timeseries_lengths=timeseries_lengths)
        # performer_func(model, dataloader_set, save_dir=test_out_path, file_nm='test_{}-log{}.png'.format(ep, nickname)
        #                , ylog=False, ls_method='ls_5_20', plot_all_features=True, logy=True)
        # performer_func(model, dataloader_set, save_dir=test_out_path, file_nm='test_{}-mc{}.png'.format(ep, nickname)
        #                , ylog=False, ls_method='ls-mc_5_20', plot_all_features=True)

    @done_decorator_with_logger
    def test_plot_maml(self, performer, model, ep, is_monthly, is_insample=False):
        # self=ds; ep=0; is_monthly = False
        model.eval()

        if is_monthly:
            # mode = 'test_monthly'
            performer_func = performer.predict_plot_monthly
            raise NotImplementedError
        else:
            if is_insample:
                mode = 'test_insample2'
            else:
                mode = 'test'
            performer_func = performer.predict_plot_maml

        self.logger.info("[test_plot_maml][Ep %d][%s]", ep, mode)
        self.logger_train.info("[test_plot_maml][Ep %d][%s]", ep, mode)
        if (ep == 0) or (self.dataloader.get(mode) is None):
            self.dataloader[mode] = self._dataloader_maml(mode)

        if self.dataloader[mode] is False:
            return False

        dataloader_set = self.dataloader[mode]
        test_out_path = os.path.join(self.data_out_path, '{}/{}'.format(self.base_idx, mode))
        os.makedirs(test_out_path, exist_ok=True)

        performer_func(model, dataloader_set, self.labels_torch, save_dir=test_out_path, file_nm='test_{}.png'.format(ep)
                       , ylog=False, ls_method='ls_5_20', plot_all_features=True)

    def labels_torch_multi(self, f_list, labels, add_infos):
        c = self.configs
        labels_mtl = dict()
        for cls in c.features_structure.keys():
            for arr_base in c.features_structure[cls].keys():
                for key in c.features_structure[cls][arr_base].keys():
                    n_arr = c.features_structure[cls][arr_base][key]
                    if cls == 'classification':    # classification
                        for n in n_arr:
                            f_nm = '{}_{}'.format(key, n)
                            labels_mtl[f_nm] = (labels[:, :, f_list.index(f_nm)] > 0).long()
                    else:
                        labels_mtl[key] = torch.stack([labels[:, :, f_list.index("{}_{}".format(key, n))] for n in n_arr], axis=-1)

        labels_mtl['size_rnk'] = add_infos['size_rnk'].reshape(-1, 1, 1)
        if not c.use_maml:
            labels_mtl['importance_wgt'] = add_infos['importance_wgt'].reshape(-1, 1, 1)

        return labels_mtl

    def labels_torch(self, f_list, labels, add_infos):
        # 카테고리별 대표값만
        c = self.configs
        labels_mtl = dict()
        for cls in c.features_structure.keys():
            for arr_base in c.features_structure[cls].keys():
                for key in c.features_structure[cls][arr_base].keys():
                    if key in ['logp', 'fft', 'mdd'] or arr_base != 'logp_base':
                        n = c.features_structure[cls][arr_base][key][0]
                    else:
                        n = c.k_days
                    if cls == 'classification':    # classification
                        f_nm = '{}_{}'.format(key, n)
                        labels_mtl[f_nm] = (labels[:, :, f_list.index(f_nm)] > 0).long()
                    else:
                        labels_mtl[key] = labels[:, :, f_list.index("{}_{}".format(key, n))].unsqueeze(-1)

        labels_mtl['size_rnk'] = add_infos['size_rnk'].reshape(-1, 1, 1)
        if not c.use_maml:
            labels_mtl['importance_wgt'] = add_infos['importance_wgt'].reshape(-1, 1, 1)

        return labels_mtl

    def test(self, performer, model, dataset=None, dataset_m=None, use_label=True, out_dir=None, file_nm='out.png', ylog=False, save_type=None, table_nm=None):
        if out_dir is None:
            test_out_path = os.path.join(self.data_out_path, '{}/test'.format(self.base_idx))
        else:
            test_out_path = out_dir

        os.makedirs(test_out_path, exist_ok=True)
        if use_label:
            if dataset is None:
                _dataset_list = self._dataset('test')
            else:
                _dataset_list = dataset

            if dataset_m is None:
                _dataset_list_m = self._dataset_monthly('test')
            else:
                _dataset_list_m = dataset_m

            if _dataset_list is False:
                print('[test] no test data')
                return False
            performer.predict_plot_mtl(model, _dataset_list, save_dir=test_out_path, file_nm=file_nm
                                       , ylog=ylog, ls_method='ls_5_20', plot_all_features=True)
            performer.predict_plot_mtl(model, _dataset_list, save_dir=test_out_path + "2", file_nm=file_nm,
                                       ylog=ylog, ls_method='l_60', plot_all_features=True)
            if _dataset_list_m is not None:
                performer.predict_plot_monthly(model, _dataset_list_m, save_dir=test_out_path + "_ml", file_nm=file_nm,
                                                              ylog=ylog, ls_method='l_60', plot_all_features=True, rate_=self.configs.app_rate)
                performer.predict_plot_monthly(model, _dataset_list_m, save_dir=test_out_path + "_mls", file_nm=file_nm,
                                                              ylog=ylog, ls_method='ls_5_20', plot_all_features=True, rate_=self.configs.app_rate)

        if save_type is not None:
            _dataset_list = self._dataset('predict')
            if _dataset_list is False:
                print('[predict] no test data')
                return False

            if save_type == 'db':
                self.save_score_to_db(model, _dataset_list, table_nm=table_nm)
            elif save_type == 'csv':
                self.save_score_to_csv(model, _dataset_list, out_dir=test_out_path)

    def save_score_to_csv(self, model, dataset_list, out_dir=None):
        input_enc_list, output_dec_list, _, _, additional_infos, start_date, _ = dataset_list
        df_infos = pd.DataFrame(columns={'start_d', 'base_d', 'infocode', 'score'})
        for i, (input_enc_t, output_dec_t) in enumerate(zip(input_enc_list, output_dec_list)):
            assert np.sum(input_enc_t[:, -1, :] - output_dec_t[:, 0, :]) == 0
            assert np.sum(output_dec_t[:, 1:, :]) == 0

            new_output_t = np.zeros_like(output_dec_t)
            new_output_t[:, 0, :] = output_dec_t[:, 0, :]

            features = {'input': input_enc_t, 'output': new_output_t}
            predictions = model.predict_mtl(features)
            df_infos = pd.concat([df_infos, pd.DataFrame({
                'start_d': start_date,
                'base_d': additional_infos[i]['date'],
                'infocode': additional_infos[i]['assets_list'],
                'score': predictions[self.features_cls.pred_feature][:, 0, 0]})], ignore_index=True, sort=True)
        df_infos.to_csv(os.path.join(out_dir, 'out_{}.csv'.format(str(start_date))))

    def save_score_to_db(self, model, dataset_list, table_nm='kr_weekly_score_temp'):
        if table_nm is None:
            table_nm = 'kr_weekly_score_temp'

        input_enc_list, output_dec_list, _, _, additional_infos, start_date, _ = dataset_list
        df_infos = pd.DataFrame(columns={'start_d', 'base_d', 'infocode', 'score'})
        for i, (input_enc_t, output_dec_t) in enumerate(zip(input_enc_list, output_dec_list)):
            assert np.sum(input_enc_t[:, -1, :] - output_dec_t[:, 0, :]) == 0
            assert np.sum(output_dec_t[:, 1:, :]) == 0
            new_output_t = np.zeros_like(output_dec_t)
            new_output_t[:, 0, :] = output_dec_t[:, 0, :]

            features = {'input': input_enc_t, 'output': new_output_t}
            predictions = model.predict_mtl(features)
            df_infos = pd.concat([df_infos, pd.DataFrame({
                'start_d': start_date,
                'base_d': additional_infos[i]['date'],
                'infocode': additional_infos[i]['assets_list'],
                'score': predictions[self.features_cls.pred_feature][:, 0, 0]})], ignore_index=True, sort=True)

            # db insert
            # sqlm = SqlManager()
            # sqlm.set_db_name('passive')
            # sqlm.db_insert(df_infos[['start_d', 'base_d', 'infocode', 'score']], table_nm, fast_executemany=True)

    def save(self, ep, model, optimizer):
        save_path = os.path.join(self.data_out_path, "saved_model.pt")
        torch.save({
            'ep': ep,
            'model_state_dict': model.optim_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)

        self.logger.info("[load] Model Saved. %s", save_path)

    def load(self, model, optimizer):
        load_path = os.path.join(self.data_out_path, "saved_model.pt")
        if not os.path.exists(load_path):
            return False

        checkpoint = torch.load(load_path)
        model.optim_state_dict = checkpoint['model_state_dict']
        model.load_from_optim()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(tu.device)
        model.eval()

        self.logger.info("[load] Model Loaded. %s", load_path)

    def next(self):
        self.logger.debug('[next][before] base_idx: %d / train_begin_idx: %d / eval_begin_idx: %d / test_begin_idx: %d / test_end_idx: %d'
                         , self.base_idx, self.train_begin_idx, self.eval_begin_idx, self.test_begin_idx, self.test_end_idx)

        self.base_idx += self.retrain_days
        self.train_begin_idx += self.retrain_days
        self.eval_begin_idx += self.retrain_days
        self.test_begin_idx += self.retrain_days
        self.test_end_idx = min(self.test_end_idx + self.retrain_days, self.data_generator.max_length - self.configs.k_days - 1)

        self.logger.debug('[next][after] base_idx: %d / train_begin_idx: %d / eval_begin_idx: %d / test_begin_idx: %d / test_end_idx: %d'
                         , self.base_idx, self.train_begin_idx, self.eval_begin_idx, self.test_begin_idx, self.test_end_idx)

        self.dataloader = {'train': False, 'eval': False}

    def get_date(self):
        return self.date_[self.base_d]

    def __del__(self):
        for h in self.logger.handlers:
            self.logger.removeHandler(h)

        for h in self.logger_train.handlers:
            self.logger_train.removeHandler(h)

    @property
    def date_(self):
        return self.data_generator.date_

    @property
    def done(self):
        # if self.test_end_idx > self.data_generator.max_length:
        if self.test_end_idx <= self.test_begin_idx + self.retrain_days:
            return True
        else:
            return False


class PrepareDataFromDB:
    def __init__(self, data_type='kr_stock'):
        from dbmanager import SqlManager
        self.sqlm = SqlManager()
        self.data_type = data_type

    def get_all_csv(self):
        if self.data_type == 'kr_stock':
            # date
            self.get_datetable()
            # return data
            self.get_close_y(90, country='kr')
            # factor wgt & univ
            self.get_factorwgt_and_univ('CAP_300_100')  # 'CAP_100_150'
            # mktcap
            self.get_mktcap_daily(country='kr')
            self.get_ivol()
            # macro data
            self.get_macro_daily(country='kr')
        elif self.data_type == 'us_stock':
            # date
            self.get_datetable()
            # return data
            self.get_close_y(90, country='us')
            # mktcap
            self.get_mktcap_daily(country='us')

    def run_procedure(self):
        # universe
        print('[proc] EquityUniverse start')
        self.sqlm.set_db_name('qinv')
        self.sqlm.db_execute('EXEC qinv..SP_EquityUniverse')
        print('[proc] EquityUniverse done')

        # date
        print('[proc] EquityTradeDateDaily start')
        self.sqlm.set_db_name('qinv')
        self.sqlm.db_execute('EXEC qinv..SP_EquityTradeDateDaily')
        print('[proc] EquityTradeDateDaily done')

        # EquityReturnDaily
        print('[proc] EquityReturnDaily start')
        self.sqlm.set_db_name('qinv')
        self.sqlm.db_execute('EXEC qinv..SP_batch_EquityReturnDaily')
        print('[proc] EquityReturnDaily done')

        # EquityMarketValueMonthly
        print('[proc] EquityMarketValueMonthly start')
        self.sqlm.set_db_name('qinv')
        self.sqlm.db_execute('EXEC qinv..SP_EquityMarketValueMonthly')
        print('[proc] EquityMarketValueMonthly done')

    def get_equityuniverse(self):
        sql_ = """
        select infocode, region, dslocalcode, DsQtName as name_ 
	        from qinv..EquityUniverse
	        order by infocode
        """
        self.sqlm.set_db_name('qinv')
        df = self.sqlm.db_read(sql_)
        df.to_csv('./data/equityuniverse.csv', index=False)

    @done_decorator
    def get_close_y(self, top_npercent=90, country='kr'):
        sql_ = """
        select date_, infocode, y
            from (
                select distinct infocode
                    from (
                        select infocode
                            from qinv..EquityUniverse 
                            where region = '{}' 
                            and typecode = 'eq'
                    ) U
                    cross apply (
                        select eval_d, size_port
                            from qinv..EquityMarketValueMonthly 
                            where infocode = u.infocode
                            and month(eval_d) = 12
                            and size_port <= {}
                    ) M
            ) U
            cross apply (
                select marketdate as date_, y
                    from qinv..equityreturndaily 
                    where infocode = u.infocode
            ) A
            order by date_, infocode
        """.format(country, top_npercent)
        self.sqlm.set_db_name('qinv')
        df = self.sqlm.db_read(sql_)
        df.to_csv('./data/{}_close_y_{}.csv'.format(country, top_npercent), index=False)

    @done_decorator
    def get_factorwgt_and_univ(self, univ_nm='CAP_300_100'):
        sql_ = """
        select m.work_d as work_m, univ_nm, gicode, infocode, wgt
            from (
                select work_d, univ_nm, gicode, infocode, stock_weight as wgt
                    from passive..active_factor_univ_weight 
                    where work_d >= '2001-08-01' and univ_nm = '{}'
            ) A
            join (
                select eval_d, work_d 
                    from qdb..T_CALENDAR_EVAL_D
                    where is_m_end = 'y'
            ) M
            on datediff(month, a.work_d, m.eval_d) = 0
            order by m.work_d, a.wgt desc""".format(univ_nm)
        self.sqlm.set_db_name('passive')
        df = self.sqlm.db_read(sql_)
        df.to_csv('./data/kr_factor_wgt.csv', index=False)

        df.ix[:, ['work_m', 'infocode']].to_csv('./data/kr_univ_monthly.csv', index=False)

    @done_decorator
    def get_datetable(self):
        sql_ = """
        select  *
            from (
                select eval_d as eval_m, work_d as work_m
                    from qdb..T_CALENDAR_EVAL_D
                    where is_m_end = 'y'
            ) M
            join (
                select eval_d as eval_y, work_d as work_y
                    from qdb..T_CALENDAR_EVAL_D
                    where is_y_end = 'y'
            ) Y
            on datediff(month, eval_y, eval_m) <= 12 and eval_m > eval_y
            order by eval_m"""
        self.sqlm.set_db_name('qdb')
        df = self.sqlm.db_read(sql_)
        df.to_csv('./data/date.csv', index=False)

    @done_decorator
    def get_ivol(self):
        sql_ = """
        select cast(a.base_d as date) as eval_d, u.infocode, a.ivol
            from (
                SELECT infocode, case when len(dslocalcode) = 7 then 'A' else 'A0' end + substring(dslocalcode,2, 6)  as gicode
                    FROM qinv..EquityUniverse 
                    where region = 'KR'
            ) u
            JOIN (
                select * 
                    from WMS..IVOLData 
            ) A
            on u.gicode = a.gicode
            order by eval_d, infocode"""
        self.sqlm.set_db_name('qdb')
        df = self.sqlm.db_read(sql_)
        df.to_csv('./data/kr_ivol.csv', index=False)

    @done_decorator
    def get_mktcap_daily(self, country='kr'):
        sql_ = """
        select d.eval_d
            , U.infocode
            , isnull(P.Volume / N.NUMSHRS / 1000, -1.) as turnover
            , N.NUMSHRS * P.CLOSE_ / 1000 AS mktcap
            , NTILE(100) OVER (PARTITION BY d.eval_d, u.REGION ORDER BY N.NUMSHRS * P.CLOSE_ DESC) AS size
            from  (
                select eval_d from qdb..T_CALENDAR_EVAL_D where eval_d = work_d and eval_d <= getdate()
            ) D
            cross apply (
                select * 
                    from qinv..EquityUniverse U
                    where region = '{}' AND typecode = 'EQ'
                    and u.StartDate <= d.eval_d and u.EndDate >= d.eval_d
            ) U
            cross apply (
            select *
                from qinv..EquityTradeDate T
                where t.infocode = u.infocode
                and t.eval_d = d.eval_d
            ) T
            cross apply (
                select *
                    from (
                        select p.INFOCODE, P.MarketDate, P.CLOSE_, P.VOLUME
                                , case when REGION = 'US' AND p.close_ >= 5 then 0 
                                    when REGION = 'KR' AND p.close_ >= 2000 then 0 
                                    when REGION = 'JP' AND p.close_ >= 200 then 0 else 1 end as penny_flag
                                , case when p.Marketdate >= dateadd(day, -31, t.eval_d) then 1 else 0 end as is_active
                            from qai..ds2primqtprc p
                            where p.infocode = u.Infocode
                                and p.MarketDate = T.buy_d
                    ) P
                    WHERE penny_flag = 0 and is_active = 1
            ) P
            cross apply (
                select	top 1 N.*
                    from qai..DS2NumShares N
                    where n.infocode = u.infocode
                    and EventDate <= d.eval_d
                    order by EventDate desc
            ) N            
            order by eval_d, infocode
        """.format(country)
        self.sqlm.set_db_name('qinv')
        df = self.sqlm.db_read(sql_)
        df.to_csv('./data/{}_mktcap_daily.csv'.format(country), index=False)

    @done_decorator
    def get_macro_daily(self, country='kr'):
        if country != 'kr':
            raise NotImplementedError

        sql_ = """
        select b.*, f.mkt_rf as exmkt, f.smb, f.hml, f.wml, f.rmw, f.callrate, x.usdkrw
            from wms..MktSentiData A
            pivot (
                min(value_) 
                for macro_id in ([ConfIndex], [ConfIndex52], [MomStr], [MomStr52], [PrStr], [PrStr52], [VolStr], [VolStr52], [CS3YAAm], [CS3YBBBm], [PCratioW], [VKospi])
            ) B
            left join (	
                select return_date as date_, mkt_rf, smb, hml, rmw, wml, call_rate as callrate
                    from passive..factor_timeseries_wise 
            ) F
            on b.Date_ = f.date_
            left join (	
                select cast(base_d as date) as date_, CLSPRC as 'usdkrw'
                    from wms..indexData A
                    where idxcd = 'usdkrw'
            ) X
            on b.date_ = x.date_
            order by b.date_"""
        self.sqlm.set_db_name('qdb')
        df = self.sqlm.db_read(sql_)
        df.to_csv('./data/{}_macro_daily.csv'.format(country), index=False)


class DataGeneratorDynamic:
    def __init__(self, features_cls):
        self.features_cls = features_cls
        self.mkt_features = ['mktcap', 'turnover']  # sample_data에서 계산할때 사용
        self.initialize = False  # 속도 및 메모리를 위해 계산 필요하지 않은 경우 데이터로딩 생략

    def _initialize(self, configs):
        self.initialize = True

        data_type = configs.data_type
        univ_type = configs.univ_type
        self.use_macro = configs.use_macro
        min_size_port = configs.min_size_port

        add_path = dict()
        univ_path = None
        date_mapping_path = './data/date.csv'           # [eval_m / work_m / eval_y / work_y]
        equityuniverse_path = './data/equityuniverse.csv'
        if data_type == 'kr_stock':
            # 가격데이터
            return_path = './data/kr_close_y_90.csv'    # [date_ / infocode / y]
            market_path = './data/kr_mktcap_daily.csv'    # [eval_d / infocode / turnover / mktcap / size]
            macro_path = './data/kr_macro_daily.csv'
            ivol_path = './data/kr_ivol.csv'

            # if os.path.exists(ivol_path):
            #     add_path['ivol'] = ivol_path

            if univ_type == 'selected':
                univ_path = './data/kr_factor_wgt.csv'  # [work_m / univ_nm / gicode / infocode / wgt] # monthly

        elif data_type == 'us_stock':
            # 가격데이터
            return_path = './data/us_close_y_90.csv'
            market_path = './data/us_mktcap_daily.csv'

        self.set_return_and_date(return_path)
        self.set_marketdata(market_path, **add_path)
        self.set_univ(univ_path, date_mapping_path, min_size_port)
        if self.use_macro:
            self.set_macrodata(macro_path)

    def set_return_and_date(self, return_path, min_stocks_per_day=10):
        df = pd.read_csv(return_path)
        df = df[df.infocode > 0]  # 잘못된 infocode 제거

        # 날짜별 종목수
        date_ = df[['date_', 'infocode']].groupby('date_').count()
        date_ = date_[date_.infocode >= min_stocks_per_day]
        date_.columns = ['cnt']

        # 수익률 계산
        self.date_ = list(date_.index)
        data_df = pd.merge(date_, df, on='date_')
        data_df['y'] = data_df['y'] + 1
        data_df['cum_y'] = data_df[['date_', 'infocode', 'y']].groupby('infocode').cumprod(axis=0)

        self.df_pivoted_all = data_df[['date_', 'infocode', 'cum_y']].pivot(index='date_', columns='infocode')
        self.df_pivoted_all.columns = self.df_pivoted_all.columns.droplevel(0).to_numpy(dtype=np.int32)

    def set_marketdata(self, market_path, **add_path):
        self.size_df = pd.read_csv(market_path)
        self.size_df.columns = ['eval_d', 'infocode', 'turnover', 'mktcap', 'size_port']

        add_data = self._additional_data(**add_path)
        for key in add_path.keys():
            self.mkt_features.append(key)
            self.size_df = pd.merge(self.size_df, add_data[key], on=['eval_d', 'infocode'])

        self.size_df = self.size_df.loc[self.size_df.eval_d >= min(self.date_), :]  # TODO: 임시 (date_보다 이른 데이터 제거)

    def _additional_data(self, **add_path):
        add_data = dict()
        for key in add_path.keys():
            add_data[key] = pd.read_csv(add_path[key])

        return add_data

    def set_univ(self, univ_path, date_mapping_path, min_size_port=90):
        assert hasattr(self, 'size_df'), '[set_univ] run set_marketdata first.'

        # month end / year end mapping table
        date_mapping = pd.read_csv(date_mapping_path)

        if univ_path is None:
            univ_df = self.size_df.loc[self.size_df.size_port <= min_size_port, ['eval_d', 'infocode']]
            left_on = 'eval_d'
        else:
            univ_df = pd.read_csv(univ_path)
            univ_df = univ_df[univ_df.infocode > 0]
            left_on = 'work_m'

        univ_mapping = pd.merge(univ_df, date_mapping, left_on=left_on, right_on='work_m')
        # univ_mapping = univ_mapping.loc[:, ['work_m', 'eval_m', 'infocode']]
        # daily basis
        univ_w_size = pd.merge(univ_mapping, self.size_df,
                               left_on=['infocode', 'work_m'],
                               right_on=['infocode', 'eval_d'])

        univ_w_size = univ_w_size[univ_w_size.infocode > 0]
        univ_w_size['mktcap'] = univ_w_size['mktcap'] / 1000.
        if 'wgt' not in univ_w_size.columns:
            univ_w_size['wgt'] = univ_w_size.loc[:, ['work_m', 'mktcap']].groupby('work_m').apply(lambda x: x / x.sum())

        if 'gicode' not in univ_w_size.columns:
            univ_w_size['gicode'] = univ_w_size['infocode']

        self.univ = univ_w_size.loc[:, ['eval_m', 'infocode', 'gicode', 'mktcap', 'wgt']]
        self.univ.columns = ['eval_m', 'infocode', 'gicode', 'mktcap', 'wgt']

    def set_macrodata(self, macro_path):
        assert hasattr(self, 'date_'), '[set_macrodata] run set_return_and_date first'
        df = pd.read_csv(macro_path).set_index('date_')
        df.columns = [col.lower() for col in df.columns]

        # returns인 값들 logp로 전처리  # => sample_data로 이관
        # value_y = ['exmkt', 'smb', 'hml', 'wml', 'rmw', 'callrate']
        # df.loc[:, value_y] = np.log(df.loc[:, value_y] + 1).cumsum(axis=0)
        data_df = pd.merge(pd.DataFrame({'eval_d': self.date_}), df, how='left', left_on='eval_d', right_on='date_')
        data_df = data_df.set_index('eval_d').ffill()
        self.macro_df = data_df

    def sample_data(self, date_i, configs, debug=True):
        if self.initialize is False:
            self._initialize(configs)

        # get attributes to local variables
        date_ = self.date_
        univ = self.univ
        df_pivoted = self.df_pivoted_all
        features_cls = self.features_cls
        c = configs

        base_d = date_[date_i]
        univ_d = univ.eval_m[univ.eval_m <= base_d].max()
        univ_code = list(univ[univ.eval_m == univ_d].infocode)

        size_d = self.size_df.eval_d[self.size_df.eval_d <= base_d].max()
        size_code = self.size_df[self.size_df.eval_d == size_d][['infocode', 'mktcap']].set_index('infocode')

        # set local parameters
        m_days = c.m_days
        calc_length = c.calc_length
        calc_length_label = c.calc_length_label
        delay_days = c.delay_days

        len_data = calc_length + m_days
        len_label = calc_length_label + delay_days
        # k_days_adj = k_days + delay_days
        # len_label = k_days_adj

        start_d = date_[max(0, date_i - len_data)]
        end_d = date_[min(date_i + len_label, len(date_) - 1)]

        # data cleansing
        select_where = ((df_pivoted.index >= start_d) & (df_pivoted.index <= end_d))
        df_logp = cleansing_missing_value(df_pivoted.loc[select_where, :], n_allow_missing_value=20, to_log=True)

        if df_logp.empty or len(df_logp) <= calc_length + m_days:
            return False

        univ_list = sorted(list(set.intersection(set(univ_code), set(df_logp.columns), set(size_code.index))))
        if len(univ_list) < 10:
            return False

        # macro
        if self.use_macro:
            # 연산 속도 위해 feature_calc 실행 전에 일단 데이터 유무 체크
            select_where_macro = ((self.macro_df.index >= start_d) & (self.macro_df.index <= end_d))
            df_macro = self.macro_df.loc[select_where_macro, :]

            # returns값들 logp로 변환
            y_values = c.macro_dict['returns']
            p_values = c.macro_dict['values']
            df_macro.loc[:, y_values] = (1+df_macro.loc[:, y_values]).cumprod(axis=0)

            # assert np.sum(select_where) == np.sum(select_where_macro), 'selected data not matched (macro vs stocks)'
            if len(df_logp) != len(df_macro):
                df_macro = pd.merge(df_logp.reindex(columns=[]), df_macro, how='left', left_index=True, right_index=True)

            df_macro.loc[:, p_values] = cleansing_missing_value(df_macro.loc[:, p_values], reset_first_value=False, n_allow_missing_value=20, to_log=False)
            df_macro.loc[:, y_values] = cleansing_missing_value(df_macro.loc[:, y_values], n_allow_missing_value=20, to_log=True)

            if df_macro.empty or ((df_macro.isna().sum(axis=1) > 0).sum() > 0):
                return False

        print('[{}] univ size: {}'.format(base_d, len(univ_list)))

        logp_arr = df_logp.reindex(columns=univ_list).to_numpy(dtype=np.float32)

        mkt_arr_dict = dict()
        select_where_mkt = ((self.size_df.eval_d >= start_d) & (self.size_df.eval_d <= end_d))
        for mkt_f in self.mkt_features:
            df_mkt = self.size_df.loc[select_where_mkt, ['eval_d', 'infocode', mkt_f]].pivot(index='eval_d', columns='infocode')
            df_mkt.columns = df_mkt.columns.droplevel(0)
            if len(df_logp) != len(df_mkt):
                df_mkt = pd.merge(df_logp.reindex(columns=[]), df_mkt, how='left', left_index=True, right_index=True)

            df_mkt = df_mkt.fillna(-1.)
            df_mkt = cleansing_missing_value(df_mkt, n_allow_missing_value=20, to_log=False, reset_first_value=False)
            mkt_arr = df_mkt.reindex(columns=univ_list).to_numpy(dtype=np.float32)
            assert logp_arr.shape == mkt_arr.shape
            mkt_arr_dict[mkt_f] = mkt_arr

        # mc_adj_y
        wgt_arr = np.abs(mkt_arr_dict['mktcap']) / np.sum(np.abs(mkt_arr_dict['mktcap']), axis=1, keepdims=True)
        mkt_arr_dict['wlogy'] = features_cls.get_weighted_arr(logp_arr, wgt_arr)
        order = np.argsort(mkt_arr_dict['wlogy'], axis=1)
        mkt_arr_dict['wlogyrnk'] = np.argsort(order, axis=1)

        # calculate features
        features_dict, labels_dict = features_cls.calc_features(logp_arr, debug=debug)

        for key in mkt_arr_dict:
            if key == 'mktcap':
                calc_list = ['nmsize_0']
            elif key == 'turnover':
                calc_list = ['nmturnover_0', 'tsturnover_0']
            elif key == 'ivol':
                calc_list = ['nmivol_0']
            elif key == 'wlogy':
                calc_list = ['wlogy_0', 'nmwlogy_0']
            elif key == 'wlogyrnk':
                calc_list = ['nmwlogyrnk_0']
            else:
                continue

            f_, l_ = features_cls.calc_features(mkt_arr_dict[key], debug=debug, calc_list=calc_list)
            features_dict.update(f_)
            labels_dict.update(l_)

        spot_dict = dict()
        spot_dict['base_d'] = base_d
        spot_dict['asset_list'] = univ_list
        spot_dict['mktcap'] = size_code.loc[univ_list]
        spot_dict['size_rnk'] = spot_dict['mktcap'].rank() / len(spot_dict['mktcap'])
        spot_dict['mc_wgt'] = size_code.loc[univ_list] / size_code.loc[univ_list].sum()

        # macro
        if self.use_macro:
            features_dict['macro_dict'] = dict()
            labels_dict['macro_dict'] = dict()
            spot_dict['macro_list'] = dict()
            for key in c.macro_dict:
                macro_arr = df_macro.loc[:, c.macro_dict[key]].to_numpy(dtype=np.float32)
                assert len(logp_arr) == len(macro_arr)

                macro_features_dict, macro_labels_dict = features_cls.calc_features(macro_arr
                                                                                    , debug=debug
                                                                                    , calc_list=c.macro_features[key])

                features_dict['macro_dict'][key] = macro_features_dict
                labels_dict['macro_dict'][key] = macro_labels_dict

                spot_dict['macro_list'][key] = c.macro_dict[key]

        return features_dict, labels_dict, spot_dict

    @property
    def max_length(self):
        return len(self.date_)


class Noise:
    @staticmethod
    def random_noise(arr, p):
        # arr shape: (batch_size , seq_size, n_features)
        assert arr.dim() == 3
        # add random noise
        if np.random.random() <= p:
            # normal with mu=0 and sig=sigma
            sample_sigma = torch.std(arr, axis=[0, 1], keepdims=True)
            eps = sample_sigma * torch.randn_like(arr)
        else:
            eps = 0

        return arr + eps

    @staticmethod
    # @profile
    def adversarial_noise(features_dict, labels_mtl, model):
        # arr shape: (batch_size , seq_size, n_features)
        assert features_dict['input'].dim() == 3
        features_dict['input'].requires_grad = True
        features_dict['output'].requires_grad = True

        out, loss_noise = model.forward_with_loss(features_dict, labels_mtl)
        loss_total = 0
        for key in loss_noise.keys():
            loss_total += loss_noise[key].mean()

        loss_total.backward(retain_graph=True)
        inputs_grad = torch.sign(features_dict['input'].grad)
        outputs_grad = torch.sign(features_dict['output'].grad)

        sample_sigma = torch.std(features_dict['input'], axis=[0, 1], keepdims=True)
        eps = 0.01
        scaled_eps = eps * sample_sigma  # [1, 1, n_features]

        inputs_perturbed = features_dict['input'] + scaled_eps * inputs_grad
        outputs_perturbed = features_dict['output'] + scaled_eps * outputs_grad

        features_dict['input'].grad.zero_()
        features_dict['output'].grad.zero_()

        return {'input': inputs_perturbed, 'output': outputs_perturbed}

    @staticmethod
    def _get_mask(arr_shape, mask_p):
        mask = np.random.choice([False, True], size=arr_shape, p=[1 - mask_p, mask_p])
        return mask

    @classmethod
    def random_mask(cls, arr, p, mask_p=0.2):
        """p의 확률로 mask_p만큼의 값을 0처리"""
        # deep copy
        new_arr = torch.zeros_like(arr)
        new_arr[:] = arr[:]

        # randomly masked input data
        if np.random.random() <= p:
            mask = cls._get_mask(new_arr.shape, mask_p)
            new_arr[[mask]] = 0

        return new_arr

    @classmethod
    def random_flip(cls, arr, p, flip_p=0.2):
        """p의 확률로 flip_p만큼의 값을 flip"""

        # deep copy
        new_arr = torch.zeros_like(arr)
        new_arr[:] = arr[:]

        if np.random.random() <= p:
            mask = cls._get_mask(arr.shape, flip_p)
            new_arr[[mask]] = arr[[mask]] * -1

        return new_arr


class AssetDataset(Dataset):
    def __init__(self, enc_in, dec_in, dec_out, add_infos_dict):
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.dec_out = dec_out
        self.add_infos = add_infos_dict

    def __len__(self):
        return len(self.enc_in)

    def __getitem__(self, idx):
        features = {'input': self.enc_in[idx], 'output': self.dec_in[idx]}
        out_addinfos = dict()
        for key in self.add_infos.keys():
            out_addinfos[key] = self.add_infos[key][idx]

        return features, self.dec_out[idx], out_addinfos


class MetaDataset(Dataset):
    def __init__(self, spt_dataset, tgt_dataset):
        self.spt_dataset = spt_dataset
        self.tgt_dataset = tgt_dataset

    def __len__(self):
        return len(self.tgt_dataset)

    def __getitem__(self, idx):
        spt_ds = self.spt_dataset[idx]
        tgt_ds = self.tgt_dataset[idx]
        spt_features = {'input': spt_ds[0], 'output': spt_ds[1]}
        spt_labels = spt_ds[2]

        tgt_features = {'input': tgt_ds[0], 'output': tgt_ds[1]}
        tgt_labels = tgt_ds[2]

        spt_addinfos = dict()
        tgt_addinfos = dict()
        for key in spt_ds[3].keys():
            spt_addinfos[key] = spt_ds[3][key]

        for key in tgt_ds[3].keys():
            tgt_addinfos[key] = tgt_ds[3][key]

        spt_data = (spt_features, spt_labels, spt_addinfos)
        tgt_data = (tgt_features, tgt_labels, tgt_addinfos)
        return spt_data, tgt_data


def data_loader(enc_in, dec_in, dec_out, add_infos_dict, batch_size=1, shuffle=True, num_workers=0):
    asset_dataset = AssetDataset(enc_in, dec_in, dec_out, add_infos_dict)
    return DataLoader(asset_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

# spt_list, tgt_list, features_list, importance_wgt, start_date, end_date = ds._dataset_maml('train')
def data_loader_maml(spt_dataset, tgt_dataset, sampler, shuffle=False, num_workers=0):
    asset_dataset = MetaDataset(spt_dataset, tgt_dataset)
    if sampler is None:
        dataloader = DataLoader(asset_dataset, batch_size=1, shuffle=shuffle, pin_memory=False, num_workers=num_workers)
    else:
        dataloader = DataLoader(asset_dataset, batch_size=1, sampler=sampler, pin_memory=False, num_workers=num_workers)  # sampler가 있으면 shuffle은 반드시 False

    return dataloader


def to_device(device, list_to_device):
    assert isinstance(list_to_device, list)

    for i, value_ in enumerate(list_to_device):
        if isinstance(value_, dict):
            for key in value_.keys():
                value_[key] = value_[key].to(device)
        elif isinstance(value_, torch.Tensor):
            list_to_device[i] = value_.to(device)
        else:
            raise NotImplementedError

    return list_to_device