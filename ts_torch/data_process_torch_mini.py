
import pandas as pd
import pickle
import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


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
        f(*args, **kwargs)
        print("{} ...done".format(f.__name__))
    return decorated


class DataScheduler:
    def __init__(self, configs, features_cls):
        self.data_generator = DataGeneratorDynamic(features_cls, configs.data_type, configs.univ_type)
        self.data_market = DataGeneratorMarket(features_cls, configs.data_type_mm)
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

        self.dataloader = {'train': False, 'eval': False}

    def _make_dir(self, configs):
        # data path for fetching data
        self.data_path = os.path.join(os.getcwd(), 'data', '{}_{}_{}'.format(configs.univ_type, configs.sampling_days, configs.m_days))
        os.makedirs(self.data_path, exist_ok=True)
        # make a directory for outputs
        self.data_out_path = os.path.join(os.getcwd(), configs.data_out_path, self.configs.f_name)
        os.makedirs(self.data_out_path, exist_ok=True)

    def set_idx(self, base_idx):
        c = self.configs

        self.base_idx = base_idx
        self.train_begin_idx = np.max([0, base_idx - c.train_set_length])
        # self.train_begin_idx = 4500
        self.eval_begin_idx = int(c.train_set_length * c.trainset_rate) + np.max([0, base_idx - c.train_set_length])
        self.test_begin_idx = base_idx - c.m_days
        self.test_end_idx = base_idx + c.retrain_days

    def get_data_params(self, mode='train'):
        c = self.configs
        dg = self.data_generator
        data_params = dict()

        if mode == 'train':
            start_idx = self.train_begin_idx + c.m_days
            end_idx = self.eval_begin_idx - c.k_days
            data_params['balance_class'] = True
            data_params['label_type'] = 'trainable_label'   # trainable: calc_length 반영
            decaying_factor = 0.995   # 기간별 샘플 중요도
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
            # start_idx = self.test_begin_idx
            end_idx = self.test_begin_idx - c.k_days
            data_params['balance_class'] = False
            data_params['label_type'] = 'test_label'        # test: 예측하고자 하는 것만 반영 (k_days)
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

    def _fetch_data(self, date_i):
        dg = self.data_generator
        data_path = self.data_path
        key_list = self.configs.key_list
        configs = self.configs

        file_nm = os.path.join(self.data_path, '{}.pkl'.format(date_i))
        if os.path.exists(file_nm):
            result = pickle.load(open(file_nm, 'rb'))
        else:
            result = dg.sample_data(date_i)
            if result is False:
                return None

            pickle.dump(result, open(os.path.join(data_path, '{}.pkl'.format(date_i)), 'wb'))

        features_dict, labels_dict, spot_dict = result

        n_assets = len(spot_dict['asset_list'])
        n_features = len(key_list)
        M = configs.m_days // configs.sampling_days + 1

        question = np.stack([features_dict[key] for key in key_list], axis=-1).astype(np.float32)
        question = np.transpose(question, axes=(1, 0, 2))
        assert question.shape == (n_assets, M, n_features)

        answer = np.zeros([n_assets, 2, n_features], dtype=np.float32)

        answer[:, 0, :] = question[:, -1, :]  # temporary
        answer[:, 1, :] = np.stack(
            [labels_dict[key] if labels_dict[key] is not None else np.zeros(n_assets) for key in key_list],
            axis=-1)

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

    def _dataset_t(self, base_d):
        univ = self.data_generator.univ

        features_list = self.configs.key_list
        recent_d, recent_idx = self.nearest_d_from_m_end([base_d])
        recent_d, recent_idx = recent_d[0], recent_idx[0]

        fetch_data = self._fetch_data(recent_idx)
        if fetch_data is None:
            return False

        enc_in, dec_in, dec_out, add_info = fetch_data
        add_info['factor_d'] = base_d
        add_info['model_d'] = recent_d
        add_info['univ'] = univ[univ.eval_m == base_d]
        add_info['importance_wgt'] = np.array([1 for _ in range(len(enc_in))], dtype=np.float32)

        return enc_in, dec_in, dec_out, features_list, add_info

    def _dataset_monthly(self, mode='test'):
        assert mode in ['test', 'test_insample', 'predict']
        c = self.configs
        dg = self.data_generator
        prc_df = dg.df_pivoted_all
        univ = dg.univ

        # parameter setting
        enc_in, dec_in, dec_out = [], [], []
        additional_infos = []  # test/predict 인경우 list, train/eval인 경우 dict
        start_idx, end_idx, data_params, decaying_factor = self.get_data_params(mode)
        features_list = c.key_list

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
                add_info[nm_] = torch.from_numpy(np.array(add_info[nm_], dtype=np.float32).squeeze())

            enc_in.append(torch.from_numpy(tmp_ein))
            dec_in.append(torch.from_numpy(tmp_din))
            dec_out.append(torch.from_numpy(tmp_dout))
            additional_infos.append(add_info)

        if len(enc_in) == 0:
            return False

        start_date = self.date_[start_idx]
        end_date = self.date_[end_idx]

        return enc_in, dec_in, dec_out, features_list, additional_infos, start_date, end_date

    def _dataset(self, mode='train'):
        c = self.configs

        enc_in, dec_in, dec_out = [], [], []
        add_infos_list = []  # test/predict 인경우 list, train/eval인 경우 dict
        start_idx, end_idx, data_params, decaying_factor = self.get_data_params(mode)
        features_list = c.key_list

        idx_balance = c.key_list.index(c.balancing_key)

        balancing_list = ['mktcap', 'size_rnk', 'importance_wgt']
        n_loop = np.ceil((end_idx - start_idx) / c.sampling_days)
        for i, d in enumerate(range(start_idx, end_idx, c.sampling_days)):
            fetch_data = self._fetch_data(d)
            if fetch_data is None:
                continue

            tmp_ein, tmp_din, tmp_dout, add_info = fetch_data
            add_info['importance_wgt'] = np.array([decaying_factor ** (n_loop - i - 1)
                                                          for _ in range(len(tmp_ein))], dtype=np.float32)
            if data_params['balance_class'] is True and c.balancing_method == 'each':
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
            # test의 경우 전부 torch로 변환
            enc_in = [torch.from_numpy(ei_t) for ei_t in enc_in]
            dec_in = [torch.from_numpy(di_t) for di_t in dec_in]
            dec_out = [torch.from_numpy(do_t) for do_t in dec_out]
            
            add_infos = []
            for add_info in add_infos_list:
                add_info_temp = add_info.copy()
                for nm_ in balancing_list:
                    add_info_temp[nm_] = torch.from_numpy(np.array(add_info[nm_], dtype=np.float32).squeeze())
                add_infos.append(add_info_temp)

        start_date = self.date_[start_idx]
        end_date = self.date_[end_idx]

        return enc_in, dec_in, dec_out, features_list, add_infos, start_date, end_date

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

    def _dataloader(self, mode):
        c = self.configs
        if mode == 'train':
            batch_size = c.train_batch_size
        elif mode == 'eval':
            batch_size = c.eval_batch_size
        else:
            batch_size = 1

        _dataset = self._dataset(mode)

        if _dataset is False:
            print('[train] no {} data'.format(mode))
            return False

        enc_in, dec_in, dec_out, features_list, add_infos, _, _ = _dataset
        # TODO: 임시 처리 (nmsize nan값 0 처리)
        enc_in[np.isnan(enc_in)] = 0
        dec_in[np.isnan(dec_in)] = 0
        dec_out[np.isnan(dec_out)] = 0
        assert np.nanmax(np.abs(enc_in[:, -1, :] - dec_in[:, 0, :])) == 0
        dataloader = data_loader(enc_in, dec_in, dec_out, add_infos, batch_size=batch_size)

        print('dataloader: mode-{} batchsize-{}'.format(mode, batch_size))
        return dataloader, features_list

    def train(self, model, optimizer, performer, num_epochs, early_stopping_count=2):
        min_eval_loss = 99999
        stop_count = 0
        print('train start...')
        for ep in range(num_epochs):
            if ep % 2 == 0:
                print('[Ep {}] plot'.format(ep))
                self.test_plot(performer, model, ep)

            print('[Ep {}] model evaluation ...'.format(ep))
            eval_loss = self.step_epoch(ep, model, optimizer, is_train=False)
            if eval_loss is False:
                return False

            if eval_loss > min_eval_loss:
                stop_count += 1
            else:
                min_eval_loss = eval_loss
                stop_count = 0

            print('[Ep {}] count: {}/{}'.format(ep, stop_count, early_stopping_count))
            if stop_count >= early_stopping_count:
                print('[Ep {}] Early Stopped'.format(ep))
                self.test_plot(performer, model, ep)
                break

            print('[Ep {}] model train ...'.format(ep))
            train_loss = self.step_epoch(ep, model, optimizer, is_train=True)
            if train_loss is False:
                return False

    def step_epoch(self, ep, model, optimizer, is_train=True):
        if is_train:
            mode = 'train'
            model.train()
        else:
            mode = 'eval'
            model.eval()

        if ep == 0:
            self.dataloader[mode] = self._dataloader(mode)
            if self.dataloader[mode] is False:
                return False

        dataloader, features_list = self.dataloader[mode]
        if ep == 0:
            print('f_list: {}'.format(features_list))

        total_loss = 0
        i = 0
        for features, labels, add_infos in dataloader:
            #  features, labels, add_infos = next(iter(dataloader))
            with torch.set_grad_enabled(is_train):
                # 미래데이터 안 땡겨쓰게
                new_out = torch.zeros_like(features['output'])
                new_out[:] = features['output']

                # add random noise for features
                features_with_noise = {'input': None, 'output': new_out}

                features_with_noise['input'] = Noise.random_noise(features['input'], p=0.5)
                features_with_noise['input'] = Noise.random_mask(features_with_noise['input'], p=0.9, mask_p=0.2)

                # add random noise for labels
                labels_with_noise = Noise.random_noise(labels, p=0.2)
                labels_with_noise = Noise.random_flip(labels_with_noise, p=0.9, flip_p=0.2)

                labels_mtl = self.labels_torch(features_list, labels_with_noise, add_infos)
                to_device(model.device, [features_with_noise, labels_mtl])
                # pred, _, _, _ = model.forward(features_with_noise, labels_mtl)
                pred, loss_each = model.forward_with_loss(features_with_noise, labels_mtl)

                losses = 0
                for key in loss_each.keys():
                    losses += loss_each[key].mean()

                if is_train:
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                total_loss += losses
                i += 1

        total_loss = total_loss.detach().cpu().numpy() / i
        if is_train:
            print_str = "[Ep {}][{}] ".format(ep, mode)
            size_str = "[Ep {}][{}][size] ".format(ep, mode)
            for key in loss_each.keys():
                print_str += "{}- {:.4f} / ".format(key, loss_each[key].mean().detach().cpu().numpy())
                size_str += "{} - {} / ".format(key, loss_each[key].shape)
            print(print_str)
            print(size_str)
            return total_loss
        else:
            print('[Ep {}][{}] total - {:.4f}'.format(ep, mode, total_loss))
            return total_loss

    def test_plot(self, performer, model, ep):
        # self=ds; ep=0; ylog = False
        model.eval()
        model.to('cpu')

        _dataset_list = self._dataset('test')
        _dataset_list_m = self._dataset_monthly('test')

        test_out_path = os.path.join(self.data_out_path, '{}/test'.format(self.base_idx))
        os.makedirs(test_out_path, exist_ok=True)

        if _dataset_list is False:
            print('[test] no test data')
            return False

        performer.predict_plot_mtl_cross_section_test(model, _dataset_list
                                                      , save_dir=test_out_path
                                                      , file_nm='test_{}.png'.format(ep)
                                                      , ylog=False
                                                      , ls_method='ls_5_20'
                                                      , plot_all_features=True)
        if _dataset_list_m is not None:
            performer.predict_plot_monthly(model, _dataset_list_m
                                           , save_dir=test_out_path + "_mls"
                                           , file_nm='test_{}.png'.format(ep)
                                           , ylog=True
                                           , ls_method='ls_5_20'
                                           , plot_all_features=True
                                           , rate_=self.configs.app_rate)

        model.to(model.device)

    def labels_torch(self, f_list, labels, add_infos):
        c = self.configs
        labels_mtl = dict()
        for cls in c.features_structure.keys():
            for key in c.features_structure[cls].keys():
                n_arr = c.features_structure[cls][key]
                if cls == 'classification':    # classification
                    for n in n_arr:
                        f_nm = '{}_{}'.format(key, n)
                        labels_mtl[f_nm] = (labels[:, :, f_list.index(f_nm)] > 0).long()
                else:
                    labels_mtl[key] = torch.stack([labels[:, :, f_list.index("{}_{}".format(key, n))] for n in n_arr], axis=-1)

        labels_mtl['size_rnk'] = add_infos['size_rnk'].reshape(-1, 1, 1)
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
            performer.predict_plot_mtl_cross_section_test(model, _dataset_list,  save_dir=test_out_path, file_nm=file_nm
                                                          , ylog=ylog, ls_method='ls_5_20', plot_all_features=True)
            performer.predict_plot_mtl_cross_section_test(model, _dataset_list, save_dir=test_out_path + "2", file_nm=file_nm,
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

    def next(self):
        self.base_idx += self.retrain_days
        self.train_begin_idx += self.retrain_days
        self.eval_begin_idx += self.retrain_days
        self.test_begin_idx += self.retrain_days
        self.test_end_idx = min(self.test_end_idx + self.retrain_days, self.data_generator.max_length - self.configs.k_days - 1)

    def get_date(self):
        return self.date_[self.base_d]

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
            self.get_kr_close_y(90)
            # factor wgt & univ
            self.get_factorwgt_and_univ('CAP_300_100')  # 'CAP_100_150'
            # mktcap
            self.get_mktcap_daily()

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

    @done_decorator
    def get_kr_close_y(self, top_npercent=90):
        sql_ = """
        select date_, infocode, y
            from (
                select distinct infocode
                    from (
                        select infocode
                            from qinv..EquityUniverse 
                            where region = 'KR' 
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
        """.format(top_npercent)
        self.sqlm.set_db_name('qinv')
        df = self.sqlm.db_read(sql_)
        df.to_csv('./data/kr_close_y_{}.csv'.format(top_npercent), index=False)

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
        df.to_csv('./data/factor_wgt.csv', index=False)

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
            on datediff(month, eval_y, eval_m) <= 12 and eval_m > eval_y"""
        self.sqlm.set_db_name('qdb')
        df = self.sqlm.db_read(sql_)
        df.to_csv('./data/date.csv', index=False)

    @done_decorator
    def get_mktcap_daily(self):
        sql_ = """
        select d.eval_d
            , U.infocode
            , N.NUMSHRS * P.CLOSE_ / 1000 AS mktcap
            , NTILE(100) OVER (PARTITION BY d.eval_d, u.REGION ORDER BY N.NUMSHRS * P.CLOSE_ DESC) AS size
            from  (
                select eval_d from qdb..T_CALENDAR_EVAL_D where eval_d = work_d and eval_d <= getdate()
            ) D
            cross apply (
                select * 
                    from qinv..EquityUniverse U
                    where region = 'KR' AND typecode = 'EQ'
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
        """
        self.sqlm.set_db_name('qinv')
        df = self.sqlm.db_read(sql_)
        df.to_csv('./data/kr_mktcap_daily.csv', index=False)


class DataSamplerMarket:
    def __init__(self, configs, features_cls):
        self.configs = configs
        self.data_type = 'kr_market'
        self.data_generator_mm = DataGeneratorMarket(features_cls, data_type='kr_market')

        self._initialize(configs)

    def _initialize(self, configs):
        self.base_idx = 1250
        self.train_begin_idx = 250
        self.eval_begin_idx = 250 + int(1000 * configs.trainset_rate)
        self.test_begin_idx = self.base_idx - configs.m_days
        self.test_end_idx = self.base_idx + configs.retrain_days

        self._make_path(configs)

    def _make_path(self, configs):
        # make a directory for outputs
        self.data_out_path = os.path.join(os.getcwd(), configs.data_out_path)
        os.makedirs(self.data_out_path, exist_ok=True)

    def get_data_params(self, mode='train'):
        c = self.configs
        data_params = dict()

        if mode == 'train':
            start_idx = self.train_begin_idx + c.m_days
            end_idx = self.eval_begin_idx - c.k_days
            data_params['balance_class'] = True
            data_params['label_type'] = 'trainable_label'   # trainable: calc_length 반영
            decaying_factor = 0.99   # 기간별 샘플 중요도
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
            # start_idx = self.test_begin_idx
            end_idx = self.test_begin_idx - c.k_days
            data_params['balance_class'] = False
            data_params['label_type'] = 'test_label'        # test: 예측하고자 하는 것만 반영 (k_days)
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

        print("start idx:{} ({}) / end idx: {} ({})".format(start_idx, self.date_[start_idx], end_idx, self.date_[end_idx]))

        return start_idx, end_idx, data_params, decaying_factor

    def _fetch_data(self, base_d):
        # q, a가 주식과는 transpose관계. 여기- [T, ts, features] // 주식- [assets, T, features]
        c = self.configs
        dgmm = self.data_generator_mm
        key_list = c.key_list

        date_i = self.date_.index(base_d)

        result = dgmm.sample_data(base_d)
        if result is False:
            return None

        features_dict, labels_dict = result

        n_features = len(key_list)
        M = c.m_days // c.sampling_days + 1
        ts_list = list(dgmm.data_df.columns)
        question = np.stack([features_dict[key] for key in key_list], axis=-1).astype(np.float32)
        assert question.shape == (M, len(ts_list), n_features)

        answer = np.zeros([1, len(ts_list), n_features], dtype=np.float32)

        answer[0, :, :] = question[-1, :, :]
        if labels_dict[c.label_feature] is not None:
            label_ = labels_dict[c.label_feature][ts_list.index('kospi')]
        else:
            label_ = 0

        return question[:], answer[:], label_

    def _dataset(self, mode='train'):
        c = self.configs

        input_enc, output_dec, target_dec, additional_info = [], [], [], []
        start_idx, end_idx, data_params, decaying_factor = self.get_data_params(mode)
        features_list = c.key_list

        idx_balance = c.key_list.index(c.balancing_key)

        n_loop = np.ceil((end_idx - start_idx) / c.sampling_days)
        for i, d in enumerate(range(start_idx, end_idx, c.sampling_days)):
            base_d = self.date_[d]
            fetch_data = self._fetch_data(base_d)
            if fetch_data is None:
                continue

            tmp_ie, tmp_od, tmp_td = fetch_data
            addi_info = decaying_factor ** (n_loop - i - 1)

            if data_params['balance_class'] is True and c.balancing_method == 'each':
                idx_bal = self.balanced_index(tmp_td[:, 0, idx_balance])
                tmp_ie, tmp_od, tmp_td = tmp_ie[idx_bal], tmp_od[idx_bal], tmp_td[idx_bal]

            input_enc.append(tmp_ie)
            output_dec.append(tmp_od)
            target_dec.append(tmp_td)
            additional_info.append(addi_info)

        if len(input_enc) == 0:
            return False

        if mode in ['train', 'eval']:
            additional_infos = dict()
            input_enc = np.concatenate(input_enc, axis=0)
            output_dec = np.concatenate(output_dec, axis=0)
            target_dec = np.concatenate(target_dec, axis=0)

            importance_wgt = np.concatenate([additional_info['importance_wgt'] for additional_info in additional_infos_list], axis=0)

            if data_params['balance_class'] is True and c.balancing_method == 'once':
                idx_bal = self.balanced_index(target_dec[:, 0, idx_balance])
                input_enc, output_dec, target_dec = input_enc[idx_bal], output_dec[idx_bal], target_dec[idx_bal]
                additional_infos['importance_wgt'] = importance_wgt[idx_bal]
            else:
                additional_infos['importance_wgt'] = importance_wgt[:]
        else:
            additional_infos = additional_infos_list

        start_date = self.date_[start_idx]
        end_date = self.date_[end_idx]

        return input_enc, output_dec, target_dec, features_list, additional_infos, start_date, end_date

    def get_date(self):
        return self.date_[self.base_d]

    @property
    def date_(self):
        return self.data_generator_mm.date_


class DataGeneratorMarket:
    def __init__(self, features_cls, data_type='kr_market'):
        self.features_cls = features_cls
        if data_type == 'kr_market':
            data_path = './data/data_for_metarl.csv'
            data_df_temp = pd.read_csv(data_path).set_index('eval_d')

            self.date_ = list(data_df_temp.index)
            features_mm = ['mkt_rf', 'smb', 'hml', 'rmw', 'wml', 'call_rate', 'kospi', 'usdkrw']
            self.data_df = pd.DataFrame(data_df_temp.loc[:, features_mm], dtype=np.float32)

    def sample_data(self, base_d, debug=True):
        # get attributes to local variables
        date_ = self.date_
        data_df = self.data_df
        features_cls = self.features_cls

        date_i = date_.index(base_d)

        # set local parameters
        m_days = features_cls.m_days
        k_days = features_cls.k_days
        calc_length = features_cls.calc_length
        calc_length_label = features_cls.calc_length_label
        delay_days = features_cls.delay_days

        len_data = calc_length + m_days
        len_label = calc_length_label + delay_days
        # k_days_adj = k_days + delay_days
        # len_label = k_days_adj

        start_d = date_[max(0, date_i - len_data)]
        end_d = date_[min(date_i + len_label, len(date_) - 1)]

        # data cleansing
        select_where = ((data_df.index >= start_d) & (data_df.index <= end_d))
        selected_df = (1 + data_df.ix[select_where, :]).cumprod()
        df_logp = np.log(selected_df / selected_df.iloc[0])

        if df_logp.empty or len(df_logp) <= calc_length + m_days:
            return False

        # calculate features
        features_dict, labels_dict = features_cls.calc_features(df_logp.to_numpy(dtype=np.float32), transpose=False, debug=debug)

        return features_dict, labels_dict

    @property
    def max_length(self):
        return len(self.date_)


class DataGeneratorDynamic:
    def __init__(self, features_cls, data_type='kr_stock', univ_type='selected'):
        if data_type == 'kr_stock':
            # 가격데이터
            data_path = './data/kr_close_y_90.csv'
            data_df_temp = pd.read_csv(data_path)
            data_df_temp = data_df_temp[data_df_temp.infocode > 0]
            # 가격데이터 날짜 오류 수정
            date_temp = data_df_temp[['date_', 'infocode']].groupby('date_').count()
            date_temp = date_temp[date_temp.infocode >= 10]
            date_temp.columns = ['cnt']
            # 수익률 계산
            self.date_ = list(date_temp.index)
            data_df = pd.merge(date_temp, data_df_temp, on='date_')  # 최소 10종목 이상 존재 하는 날짜만
            data_df['y'] = data_df['y'] + 1
            data_df['cum_y'] = data_df[['date_', 'infocode', 'y']].groupby('infocode').cumprod(axis=0)
            #
            self.df_pivoted_all = data_df[['date_', 'infocode', 'cum_y']].pivot(index='date_', columns='infocode')
            self.df_pivoted_all.columns = self.df_pivoted_all.columns.droplevel(0).to_numpy(dtype=np.int32)

            size_df = pd.read_csv('./data/kr_mktcap_daily.csv')
            size_df.columns = ['date_', 'infocode', 'mktcap', 'size_port']
            data_df = data_df.set_index(['date_', 'infocode'])
            size_df = size_df.set_index(['date_', 'infocode'])

            if univ_type == 'all':
                self.data_code = pd.read_csv('./data/kr_sizeinfo_90.csv')
                self.data_code = self.data_code[self.data_code.infocode > 0]
            elif univ_type == 'selected':
                # selected universe (monthly)
                # univ = pd.read_csv('./data/kr_univ_monthly.csv')
                univ = pd.read_csv('./data/factor_wgt.csv')
                univ = univ[univ.infocode > 0]

                # month end / year end mapping table
                date_mapping = pd.read_csv('./data/date.csv')
                univ_mapping = pd.merge(univ, date_mapping, left_on='work_m', right_on='work_m')

                # size_data = pd.read_csv('./data/kr_sizeinfo_90.csv')
                if os.path.exists('./data/kr_mktcap_daily.csv'):
                    # daily basis
                    size_data = pd.read_csv('./data/kr_mktcap_daily.csv')
                    size_data.columns = ['eval_d', 'infocode', 'mktcap', 'size_port']
                    univ_w_size = pd.merge(univ_mapping, size_data,
                                           left_on=['infocode', 'work_m'],
                                           right_on=['infocode', 'eval_d'])
                else:
                    # year-end basis
                    size_data = pd.read_csv('./data/kr_sizeinfo_90.csv')
                    univ_w_size = pd.merge(univ_mapping, size_data,
                                           left_on=['infocode', 'eval_y'],
                                           right_on=['infocode', 'eval_d'])

                univ_w_size = univ_w_size[univ_w_size.infocode > 0]
                univ_w_size['mktcap'] = univ_w_size['mktcap'] / 1000.
                self.univ = univ_w_size.loc[:, ['eval_m', 'infocode', 'gicode', 'size_port', 'mktcap', 'wgt']]
                self.univ.columns = ['eval_m', 'infocode', 'gicode', 'size_port', 'mktcap', 'wgt']
                self.size_data = size_data

            self.features_cls = features_cls

    def sample_data(self, date_i, debug=True):
        # get attributes to local variables
        date_ = self.date_
        univ = self.univ
        df_pivoted = self.df_pivoted_all
        features_cls = self.features_cls

        base_d = date_[date_i]
        univ_d = univ.eval_m[univ.eval_m <= base_d].max()
        univ_code = list(univ[univ.eval_m == univ_d].infocode)

        size_d = self.size_data.eval_d[self.size_data.eval_d <= base_d].max()
        size_df = self.size_data[self.size_data.eval_d == size_d][['infocode', 'mktcap']].set_index('infocode')

        # set local parameters
        m_days = features_cls.m_days
        k_days = features_cls.k_days
        calc_length = features_cls.calc_length
        calc_length_label = features_cls.calc_length_label
        delay_days = features_cls.delay_days

        len_data = calc_length + m_days
        len_label = calc_length_label + delay_days
        # k_days_adj = k_days + delay_days
        # len_label = k_days_adj

        start_d = date_[max(0, date_i - len_data)]
        end_d = date_[min(date_i + len_label, len(date_) - 1)]

        # data cleansing
        select_where = ((df_pivoted.index >= start_d) & (df_pivoted.index <= end_d))
        df_logp = cleansing_missing_value(df_pivoted.loc[select_where, :], n_allow_missing_value=5, to_log=True)

        if df_logp.empty or len(df_logp) <= calc_length + m_days:
            return False

        univ_list = sorted(list(set.intersection(set(univ_code), set(df_logp.columns), set(size_df.index))))
        if len(univ_list) < 10:
            return False
        print('[{}] univ size: {}'.format(base_d, len(univ_list)))

        logp_arr = df_logp.reindex(columns=univ_list).to_numpy(dtype=np.float32)

        # size
        selected_where_sz = ((self.size_data.eval_d >= start_d) & (self.size_data.eval_d <= end_d))
        df_sz = self.size_data.loc[selected_where_sz, ['eval_d', 'infocode', 'mktcap']].pivot(index='eval_d', columns='infocode')
        df_sz.columns = df_sz.columns.droplevel(0)
        if len(df_logp) != len(df_sz):
            df_sz = pd.merge(df_logp.reindex(columns=[]), df_sz, how='left', left_index=True, right_index=True)

        df_sz = cleansing_missing_value(df_sz, n_allow_missing_value=20, to_log=False, reset_first_value=False)
        sz_arr = df_sz.reindex(columns=univ_list).to_numpy(dtype=np.float32)
        assert logp_arr.shape == sz_arr.shape

        # calculate features
        features_dict, labels_dict = features_cls.calc_features(logp_arr, transpose=False, debug=debug)
        f_size, l_size = features_cls.calc_func_size(sz_arr)
        features_dict['nmsize'] = f_size
        labels_dict['nmsize'] = l_size

        spot_dict = dict()
        spot_dict['base_d'] = base_d
        spot_dict['asset_list'] = univ_list
        spot_dict['mktcap'] = size_df.loc[univ_list]
        spot_dict['size_rnk'] = spot_dict['mktcap'].rank() / len(spot_dict['mktcap'])

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


def data_loader(enc_in, dec_in, dec_out, add_infos_dict, batch_size=1, shuffle=True):
    asset_dataset = AssetDataset(enc_in, dec_in, dec_out, add_infos_dict)
    return DataLoader(asset_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def to_device(device, list_to_device):
    assert isinstance(list_to_device, list)

    for value_ in list_to_device:
        if isinstance(value_, dict):
            for key in value_.keys():
                value_[key] = value_[key].to(device)
        elif isinstance(value_, torch.Tensor):
            value_ = value_.to(device)
        else:
            raise NotImplementedError

