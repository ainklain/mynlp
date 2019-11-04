
import pandas as pd
import pickle
import tensorflow as tf
import time
import numpy as np
import os
# from dbmanager import SqlManager


def cleansing_missing_value(df_selected, n_allow_missing_value=5, to_log=True):
    mask = np.sum(df_selected.isna(), axis=0) <= n_allow_missing_value
    df = df_selected.ix[:, mask].ffill().bfill()
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

        self._make_path(configs)

    def _make_path(self, configs):
        # data path for fetching data
        self.data_path = os.path.join(os.getcwd(), 'data', '{}_{}'.format(configs.univ_type, configs.sampling_days))
        os.makedirs(self.data_path, exist_ok=True)
        # make a directory for outputs
        self.data_out_path = os.path.join(os.getcwd(), configs.data_out_path)
        os.makedirs(self.data_out_path, exist_ok=True)

    def set_idx(self, base_idx):
        c = self.configs

        self.base_idx = base_idx
        self.train_begin_idx = np.max([0, base_idx - c.train_set_length])
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

    def _dataset(self, mode='train'):
        c = self.configs

        input_enc, output_dec, target_dec, additional_info = [], [], [], []
        additional_infos_list = []  # test/predict 인경우 list, train/eval인 경우 dict
        start_idx, end_idx, data_params, decaying_factor = self.get_data_params(mode)
        features_list = c.key_list

        idx_balance = c.key_list.index(c.balancing_key)

        n_loop = np.ceil((end_idx - start_idx) / c.sampling_days)
        for i, d in enumerate(range(start_idx, end_idx, c.sampling_days)):
            fetch_data = self._fetch_data(d)
            if fetch_data is None:
                continue

            tmp_ie, tmp_od, tmp_td, additional_info = fetch_data
            additional_info['importance_wgt'] = np.array([decaying_factor ** (n_loop - i - 1)
                                                          for _ in range(len(tmp_ie))], dtype=np.float32)

            if data_params['balance_class'] is True and c.balancing_method == 'each':
                idx_bal = self.balanced_index(tmp_td[:, 0, idx_balance])
                tmp_ie, tmp_od, tmp_td = tmp_ie[idx_bal], tmp_od[idx_bal], tmp_td[idx_bal]
                additional_info['size_factor'] = additional_info['size_factor'].iloc[idx_bal]
                additional_info['size_factor_mktcap'] = additional_info['size_factor_mktcap'].iloc[idx_bal]
                additional_info['importance_wgt'] = additional_info['importance_wgt'][idx_bal]

            input_enc.append(tmp_ie)
            output_dec.append(tmp_od)
            target_dec.append(tmp_td)
            additional_infos_list.append(additional_info)

        if len(input_enc) == 0:
            return False

        if mode in ['train', 'eval']:
            additional_infos = dict()
            input_enc = np.concatenate(input_enc, axis=0)
            output_dec = np.concatenate(output_dec, axis=0)
            target_dec = np.concatenate(target_dec, axis=0)

            size_factor = np.concatenate([additional_info['size_factor'] for additional_info in additional_infos_list], axis=0)
            size_factor_mktcap = np.concatenate([additional_info['size_factor_mktcap'] for additional_info in additional_infos_list], axis=0)
            importance_wgt = np.concatenate([additional_info['importance_wgt'] for additional_info in additional_infos_list], axis=0)

            if data_params['balance_class'] is True and c.balancing_method == 'once':
                idx_bal = self.balanced_index(target_dec[:, 0, idx_balance])
                input_enc, output_dec, target_dec = input_enc[idx_bal], output_dec[idx_bal], target_dec[idx_bal]
                additional_infos['size_factor'] = size_factor[idx_bal]
                additional_infos['size_factor_mktcap'] = size_factor_mktcap[idx_bal]
                additional_infos['importance_wgt'] = importance_wgt[idx_bal]
            else:
                additional_infos['size_factor'] = size_factor[:]
                additional_infos['size_factor_mktcap'] = size_factor_mktcap[:]
                additional_infos['importance_wgt'] = importance_wgt[:]
        else:
            additional_infos = additional_infos_list

        start_date = self.date_[start_idx]
        end_date = self.date_[end_idx]

        return input_enc, output_dec, target_dec, features_list, additional_infos, start_date, end_date

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

    def train(self,
              model,
              trainset=None,
              evalset=None,
              model_name='ts_model_v1.0',
              epoch=True):

        c = self.configs

        # make directories for graph results (both train and test one)
        train_out_path = os.path.join(self.data_out_path, model_name, '{}'.format(self.base_idx))
        os.makedirs(train_out_path, exist_ok=True)

        if trainset is None:
            _train_dataset = self._dataset('train')
        else:
            _train_dataset = trainset

        if evalset is None:
            _eval_dataset = self._dataset('eval')
        else:
            _eval_dataset = evalset

        if _train_dataset is False or _eval_dataset is False:
            print('[train] no train/eval data')
            return False

        train_input_enc, train_output_dec, train_target_dec, features_list, train_add_infos, _, _ = _train_dataset
        eval_input_enc, eval_output_dec, eval_target_dec, _, eval_add_infos, _, _ = _eval_dataset

        assert np.max(np.abs(train_input_enc[:, -1, :] - train_output_dec[:, 0, :])) == 0
        assert np.max(np.abs(eval_input_enc[:, -1, :] - eval_output_dec[:, 0, :])) == 0

        train_size_factor = train_add_infos['size_factor'].reshape([-1, 1, 1]).astype(np.float32)
        train_importance_wgt = train_add_infos['importance_wgt']
        eval_size_factor = eval_add_infos['size_factor'].reshape([-1, 1, 1]).astype(np.float32)
        eval_importance_wgt = eval_add_infos['importance_wgt']

        # K > 1인 경우 미래데이터 안 땡겨쓰게.
        train_new_output = np.zeros_like(train_output_dec)
        eval_new_output = np.zeros_like(eval_output_dec)
        # if weight_scheme == 'ew':
        #     train_new_output[:, 0, :] = train_output_dec[:, 0, :]
        #     eval_new_output[:, 0, :] = eval_output_dec[:, 0, :]
        # elif weight_scheme == 'mw':
        train_new_output[:, 0, :] = train_output_dec[:, 0, :] + train_size_factor[:, 0, :]
        eval_new_output[:, 0, :] = eval_output_dec[:, 0, :] + eval_size_factor[:, 0, :]

        train_dataset = dataset_process(train_input_enc, train_new_output, train_target_dec, train_size_factor, batch_size=c.train_batch_size, importance_wgt=train_importance_wgt)
        eval_dataset = dataset_process(eval_input_enc, eval_new_output, eval_target_dec, eval_size_factor, batch_size=c.eval_batch_size, importance_wgt=eval_importance_wgt, iter_num=1)
        print("train step: {}  eval step: {}".format(len(train_input_enc) // c.train_batch_size,
                                                     len(eval_input_enc) // c.eval_batch_size))
        if epoch:
            train_steps = len(train_input_enc) // c.train_batch_size
            eval_steps = train_steps
            save_steps = train_steps
        else:
            train_steps = c.train_steps
            eval_steps = c.eval_steps
            save_steps = c.save_steps

        for i, (features, labels, size_factors, importance_wgt) in enumerate(train_dataset.take(train_steps)):
            print_loss = False
            if i % save_steps == 0:
                model.save_model(model_name)

            if i % eval_steps == 0:
                print_loss = True
                model.evaluate_mtl(eval_dataset, features_list, steps=len(eval_input_enc) // c.eval_batch_size)

                print("[t: {} / i: {}] min_eval_loss:{} / count:{}".format(self.base_idx, i, model.eval_loss, model.eval_count))
                if model.eval_count >= c.early_stopping_count:
                    print("[t: {} / i: {}] train finished.".format(self.base_idx, i))
                    model.weight_to_optim()
                    model.save_model(model_name)
                    break

            features_with_noise = {'input': None, 'output': features['output']}

            # add random noise
            if np.random.random() <= 0.4:
                # normal with mu=0 and sig=sigma
                sample_sigma = tf.math.reduce_std(features['input'], axis=[0, 1], keepdims=True)
                eps = sample_sigma * tf.random.normal(features['input'].shape, mean=0, stddev=1)
            else:
                eps = 0

            features_with_noise['input'] = features['input'] + eps

            # randomly masked input data
            if np.random.random() <= 0.1:
                t_size = features['input'].shape[1]
                mask = np.ones_like(features['input'])
                masked_idx = np.random.choice(t_size, size=int(t_size * 0.2), replace=False)
                for mask_i in masked_idx:
                    mask[:, mask_i, :] = 0

                    features_with_noise['input'] = features_with_noise['input'] * mask

            # randomly flip label: not implemented

            labels_mtl = self.features_cls.labels_for_mtl(features_list, labels, size_factors, importance_wgt)
            model.train_mtl(features_with_noise, labels_mtl, print_loss=print_loss)

    def test(self, performer, model, dataset=None, use_label=True, out_dir=None, file_nm='out.png', ylog=False, save_type=None, table_nm=None, t_stepsize=1):
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

            if _dataset_list is False:
                print('[test] no test data')
                return False
            performer.predict_plot_mtl_cross_section_test(model, _dataset_list,  save_dir=test_out_path, file_nm=file_nm
                                                          , ylog=ylog, t_stepsize=t_stepsize, ls_method='ls_5_20', plot_all_features=True)
            performer.predict_plot_mtl_cross_section_test(model, _dataset_list, save_dir=test_out_path + "2", file_nm=file_nm,
                                                          ylog=ylog, t_stepsize=t_stepsize, ls_method='l_60', plot_all_features=True)
            # performer.predict_plot_mtl_cross_section_test_long(model, _dataset_list, save_dir=test_out_path + "2", file_nm=file_nm, ylog=ylog, time_step=time_step, invest_rate=0.8)
            # performer.predict_plot_mtl_cross_section_test_long(model, _dataset_list, save_dir=test_out_path + "3", file_nm=file_nm, ylog=ylog, t_stepsize=time_step, invest_rate=0.6)

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
        size_factor_list = [add_info['size_factor'] for add_info in additional_infos]
        df_infos = pd.DataFrame(columns={'start_d', 'base_d', 'infocode', 'score'})
        for i, (input_enc_t, output_dec_t, size_factor) in enumerate(zip(input_enc_list, output_dec_list, size_factor_list)):
            assert np.sum(input_enc_t[:, -1, :] - output_dec_t[:, 0, :]) == 0
            assert np.sum(output_dec_t[:, 1:, :]) == 0
            new_output_t = np.zeros_like(output_dec_t)
            new_output_t[:, 0, :] = output_dec_t[:, 0, :] + size_factor[:, 0, :]

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
        size_factor_list = [add_info['size_factor'] for add_info in additional_infos]
        df_infos = pd.DataFrame(columns={'start_d', 'base_d', 'infocode', 'score'})
        for i, (input_enc_t, output_dec_t, size_factor) in enumerate(zip(input_enc_list, output_dec_list, size_factor_list)):
            assert np.sum(input_enc_t[:, -1, :] - output_dec_t[:, 0, :]) == 0
            assert np.sum(output_dec_t[:, 1:, :]) == 0
            new_output_t = np.zeros_like(output_dec_t)
            new_output_t[:, 0, :] = output_dec_t[:, 0, :] + size_factor[:, 0, :]
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
        df.to_csv('./data/kr_close_y_{}.csv'.format(top_npercent))

    @done_decorator
    def get_factorwgt_and_univ(self, univ_nm='CAP_300_100'):
        sql_ = """
        select work_d as eval_d, univ_nm, gicode, infocode, stock_weight as wgt
        from passive..active_factor_univ_weight 
        where univ_nm = '{}'
        order by work_d, stock_weight desc""".format(univ_nm)
        self.sqlm.set_db_name('passive')
        df = self.sqlm.db_read(sql_)
        df.to_csv('./data/factor_wgt.csv')

        df.ix[:, ['eval_d', 'infocode']].to_csv('./data/kr_univ_monthly.csv')

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
        df.to_csv('./data/date.csv')

    @done_decorator
    def get_mktcap_daily(self):
        sql_ = """
        select d.eval_d
            , U.infocode
            , N.NUMSHRS * P.CLOSE_ / 1000 AS mktcap
            , NTILE(100) OVER (PARTITION BY d.eval_d, u.REGION ORDER BY N.NUMSHRS * P.CLOSE_ DESC) AS size
            from  (
                select eval_d
                from qinv..DateTable where is_m_end = 1 and eval_d <= getdate()
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
        df.to_csv('./data/kr_mktcap_daily.csv')


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
            self.data_df = pd.DataFrame(data_df_temp.ix[:, features_mm], dtype=np.float32)

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
            data_path = './data/kr_close_y_90.csv'
            data_df_temp = pd.read_csv(data_path)
            data_df_temp = data_df_temp[data_df_temp.infocode > 0]

            date_temp = data_df_temp[['date_', 'infocode']].groupby('date_').count()
            date_temp = date_temp[date_temp.infocode >= 10]
            date_temp.columns = ['cnt']

            self.date_ = list(date_temp.index)
            data_df = pd.merge(date_temp, data_df_temp, on='date_')  # 최소 10종목 이상 존재 하는 날짜만
            data_df['y'] = data_df['y'] + 1
            data_df['cum_y'] = data_df[['date_', 'infocode', 'y']].groupby('infocode').cumprod(axis=0)

            self.df_pivoted_all = data_df[['date_', 'infocode', 'cum_y']].pivot(index='date_', columns='infocode')
            self.df_pivoted_all.columns = self.df_pivoted_all.columns.droplevel(0).to_numpy(dtype=np.int32)

            if univ_type == 'all':
                self.data_code = pd.read_csv('./data/kr_sizeinfo_90.csv')
                self.data_code = self.data_code[self.data_code.infocode > 0]
            elif univ_type == 'selected':
                # selected universe (monthly)
                univ = pd.read_csv('./data/kr_univ_monthly.csv')
                univ = univ[univ.infocode > 0]

                # month end / year end mapping table
                date_mapping = pd.read_csv('./data/date.csv')
                univ_mapping = pd.merge(univ, date_mapping, left_on='eval_d', right_on='work_m')

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
                self.univ = univ_w_size.ix[:, ['eval_m', 'infocode', 'size_port', 'mktcap']]
                self.univ.columns = ['eval_d', 'infocode', 'size_port', 'mktcap']
                self.size_data = size_data

            self.features_cls = features_cls

    def sample_data(self, date_i, debug=True):
        # get attributes to local variables
        date_ = self.date_
        df_pivoted = self.df_pivoted_all
        features_cls = self.features_cls

        base_d = date_[date_i]
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
        df_logp = cleansing_missing_value(df_pivoted.ix[select_where, :], n_allow_missing_value=5, to_log=True)

        if df_logp.empty or len(df_logp) <= calc_length + m_days:
            return False

        univ_list = sorted(list(set.intersection(set(df_logp.columns), set(size_df.index))))

        # calculate features
        features_dict, labels_dict = features_cls.calc_features(df_logp.ix[:, univ_list].to_numpy(dtype=np.float32), transpose=False, debug=debug)

        spot_dict = dict()
        spot_dict['asset_list'] = univ_list
        spot_dict['size_factor'] = size_df.loc[univ_list].mktcap.rank() / len(univ_list)
        spot_dict['size_factor_mktcap'] = size_df.loc[univ_list]

        return features_dict, labels_dict, spot_dict

    @property
    def max_length(self):
        return len(self.date_)


def rearrange(input, output, target, size_factor, importance_wgt):
    features = {"input": input, "output": output}
    return features, target, size_factor, importance_wgt


# 학습에 들어가 배치 데이터를 만드는 함수이다.
def dataset_process(input_enc, output_dec, target_dec, size_factor, batch_size, importance_wgt=None, shuffle=True, iter_num=None):
    # Dataset을 생성하는 부분으로써 from_tensor_slices부분은
    # 각각 한 문장으로 자른다고 보면 된다.
    # train_input_enc, train_output_dec, train_target_dec
    # 3개를 각각 한문장으로 나눈다.
    dataset = tf.data.Dataset.from_tensor_slices((input_enc, output_dec, target_dec, size_factor, importance_wgt))

    # 전체 데이터를 섞는다.
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=len(input_enc))
    # 배치 인자 값이 없다면  에러를 발생 시킨다.
    assert batch_size is not None, "train batchSize must not be None"
    # from_tensor_slices를 통해 나눈것을
    # 배치크기 만큼 묶어 준다.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # 데이터 각 요소에 대해서 rearrange 함수를
    # 통해서 요소를 변환하여 맵으로 구성한다.
    dataset = dataset.map(rearrange)
    # repeat()함수에 원하는 에포크 수를 넣을수 있으면
    # 아무 인자도 없다면 무한으로 이터레이터 된다.
    if iter_num is None:
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat(iter_num)
    # make_one_shot_iterator를 통해 이터레이터를
    # 만들어 준다.
    # 이터레이터를 통해 다음 항목의 텐서
    # 개체를 넘겨준다.
    return dataset
