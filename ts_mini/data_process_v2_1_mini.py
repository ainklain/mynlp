
import pandas as pd
import tensorflow as tf
import numpy as np
import os


class DataScheduler:
    def __init__(self, configs, features_cls):
        # make a directory for outputs
        self.data_out_path = os.path.join(os.getcwd(), configs.data_out_path)
        os.makedirs(self.data_out_path, exist_ok=True)

        # self.data_generator = DataGenerator(data_type)    # infocode
        self.data_generator = DataGeneratorDynamic(features_cls, configs.data_type, configs.univ_type)
        self.train_set_length = configs.train_set_length
        self.retrain_days = configs.retrain_days
        self.m_days = configs.m_days
        self.k_days = configs.k_days
        self.sampling_days = configs.sampling_days
        self.balancing_method = configs.balancing_method

        self.train_batch_size = configs.batch_size
        self.eval_batch_size = configs.eval_batch_size
        self.trainset_rate = configs.trainset_rate

        self.features_cls = features_cls
        self.buffer = dict()

        self._initialize()

    def _initialize(self):
        self.base_idx = self.train_set_length

        self.train_begin_idx = 0
        self.eval_begin_idx = int(self.train_set_length * self.trainset_rate)
        self.test_begin_idx = self.base_idx - self.m_days
        self.test_end_idx = self.base_idx + self.retrain_days

    def set_idx(self, base_idx):
        self.base_idx = base_idx

        self.train_begin_idx = np.max([0, base_idx - self.train_set_length])
        self.eval_begin_idx = int(self.train_set_length * self.trainset_rate) + np.max([0, base_idx - self.train_set_length])
        self.test_begin_idx = self.base_idx - self.m_days
        self.test_end_idx = self.base_idx + self.retrain_days

    def get_data_params(self, mode='train'):
        dg = self.data_generator
        data_params = dict()
        data_params['sampling_days'] = self.sampling_days
        data_params['m_days'] = self.m_days
        data_params['k_days'] = self.k_days
        data_params['calc_length'] = 250
        data_params['univ_idx'] = self.test_begin_idx
        # data_params['univ_idx'] = None
        if mode == 'train':
            start_idx = self.train_begin_idx + self.m_days
            end_idx = self.eval_begin_idx - self.k_days
            data_params['balance_class'] = True
            data_params['label_type'] = 'trainable_label'   # trainable: calc_length 반영
            decaying_factor = 0.99   # 기간별 샘플 중요도
        elif mode == 'eval':
            start_idx = self.eval_begin_idx + self.m_days
            end_idx = self.test_begin_idx - self.k_days
            data_params['balance_class'] = True
            data_params['label_type'] = 'trainable_label'   # trainable: calc_length 반영
            decaying_factor = 1.   # 기간별 샘플 중요도
        elif mode == 'test':
            start_idx = self.test_begin_idx + self.m_days
            # start_idx = self.test_begin_idx
            end_idx = self.test_end_idx
            data_params['balance_class'] = False
            data_params['label_type'] = 'test_label'        # test: 예측하고자 하는 것만 반영 (k_days)
            decaying_factor = 1.   # 기간별 샘플 중요도
        elif mode == 'test_insample':
            start_idx = self.train_begin_idx + self.m_days
            # start_idx = self.test_begin_idx
            end_idx = self.test_begin_idx - self.k_days
            data_params['balance_class'] = False
            data_params['label_type'] = 'test_label'        # test: 예측하고자 하는 것만 반영 (k_days)
            decaying_factor = 1.   # 기간별 샘플 중요도
        elif mode == 'predict':
            start_idx = self.test_begin_idx + self.m_days
            # start_idx = self.test_begin_idx
            end_idx = self.test_end_idx
            data_params['balance_class'] = False
            data_params['label_type'] = None            # label 없이 과거데이터만으로 스코어 산출
            decaying_factor = 1.   # 기간별 샘플 중요도
        else:
            raise NotImplementedError

        print("start idx:{} ({}) / end idx: {} ({})".format(start_idx, dg.date_[start_idx], end_idx, dg.date_[end_idx]))

        return start_idx, end_idx, data_params, decaying_factor

    def _dataset(self, mode='train'):
        input_enc, output_dec, target_dec = [], [], []  # test/predict 인경우 list, train/eval인 경우 array
        self.date_idx = []
        features_list = []
        additional_infos_list = []  # test/predict 인경우 list, train/eval인 경우 dict
        sampling_wgt = []  # time decaying factor
        start_idx, end_idx, data_params, decaying_factor = self.get_data_params(mode)

        n_loop = np.ceil((end_idx - start_idx) / self.sampling_days)
        for i, d in enumerate(range(start_idx, end_idx, self.sampling_days)):
            if d in self.buffer.keys():
                tmp_ie, tmp_od, tmp_td, features_list, additional_info = self.buffer[d]
            else:
                if self.balancing_method in ['once', 'nothing']:
                    _sampled_data = self.data_generator.sample_inputdata_split_new3(d, **data_params)
                elif self.balancing_method == 'each':
                    _sampled_data = self.data_generator.sample_inputdata_split_new2(d, **data_params)
                else:
                    raise NotImplementedError

                if _sampled_data is False:
                    continue
                else:
                    tmp_ie, tmp_od, tmp_td, features_list, additional_info = _sampled_data
                    additional_info['importance_wgt'] = np.array([decaying_factor ** (n_loop - i - 1) for _ in range(len(tmp_ie))], dtype=np.float32)
                    self.buffer[d] = [tmp_ie, tmp_od, tmp_td, features_list, additional_info]

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

            size_value = np.concatenate([additional_info['size_value'] for additional_info in additional_infos_list], axis=0)
            mktcap = np.concatenate([additional_info['mktcap'] for additional_info in additional_infos_list], axis=0)
            importance_wgt = np.concatenate([additional_info['importance_wgt'] for additional_info in additional_infos_list], axis=0)

            if self.balancing_method == 'once':
                idx_label = features_list.index(self.features_cls.label_feature)
                where_p = (np.squeeze(target_dec)[:, idx_label] > 0)
                where_n = (np.squeeze(target_dec)[:, idx_label] <= 0)
                n_max = np.max([np.sum(where_p), np.sum(where_n)])
                idx_pos = np.concatenate([np.random.choice(np.where(where_p)[0], np.sum(where_p), replace=False),
                                          np.random.choice(np.where(where_p)[0], n_max - np.sum(where_p),
                                                           replace=True)])
                idx_neg = np.concatenate([np.random.choice(np.where(where_n)[0], np.sum(where_n), replace=False),
                                          np.random.choice(np.where(where_n)[0], n_max - np.sum(where_n),
                                                           replace=True)])

                idx_bal = np.concatenate([idx_pos, idx_neg])
                input_enc, output_dec, target_dec = input_enc[idx_bal], output_dec[idx_bal], target_dec[idx_bal]
                additional_infos['size_value'] = size_value[idx_bal]
                additional_infos['mktcap'] = mktcap[idx_bal]
                importance_wgt['importance_wgt'] = importance_wgt[idx_bal]
            else:
                additional_infos['size_value'] = size_value[:]
                additional_infos['mktcap'] = mktcap[:]
                additional_infos['importance_wgt'] = importance_wgt[:]
        else:
            additional_infos = additional_infos_list


        start_date = self.data_generator.date_[start_idx]
        end_date = self.data_generator.date_[end_idx]
        return input_enc, output_dec, target_dec, features_list, additional_infos, start_date, end_date

    def balancing_data(self):
        pass

    def train(self,
              model,
              trainset=None,
              evalset=None,
              train_steps=1,
              eval_steps=10,
              save_steps=50,
              early_stopping_count=10,
              model_name='ts_model_v1.0'):

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

        assert np.sum(train_input_enc[:, -1, :] - train_output_dec[:, 0, :]) == 0
        assert np.sum(eval_input_enc[:, -1, :] - eval_output_dec[:, 0, :]) == 0

        train_size_value = train_add_infos['size_value']
        train_importance_wgt = train_add_infos['importance_wgt']
        eval_size_value = eval_add_infos['size_value']
        eval_importance_wgt = eval_add_infos['importance_wgt']

        # train_size_value = np.concatenate([add_info['size_value'] for add_info in train_add_infos], axis=0)
        # eval_size_value = np.concatenate([add_info['size_value'] for add_info in eval_add_infos], axis=0)

        # K > 1인 경우 미래데이터 안 땡겨쓰게.
        train_new_output = np.zeros_like(train_output_dec)
        eval_new_output = np.zeros_like(eval_output_dec)
        # if weight_scheme == 'ew':
        #     train_new_output[:, 0, :] = train_output_dec[:, 0, :]
        #     eval_new_output[:, 0, :] = eval_output_dec[:, 0, :]
        # elif weight_scheme == 'mw':
        train_new_output[:, 0, :] = train_output_dec[:, 0, :] + train_size_value[:, 0, :]
        eval_new_output[:, 0, :] = eval_output_dec[:, 0, :] + eval_size_value[:, 0, :]

        train_dataset = dataset_process(train_input_enc, train_new_output, train_target_dec, train_size_value, batch_size=self.train_batch_size, importance_wgt=train_importance_wgt)
        eval_dataset = dataset_process(eval_input_enc, eval_new_output, eval_target_dec, eval_size_value, batch_size=self.eval_batch_size, importance_wgt=eval_importance_wgt, iter_num=1)
        print("train step: {}  eval step: {}".format(len(train_input_enc) // self.train_batch_size,
                                                     len(eval_input_enc) // self.eval_batch_size))
        for i, (features, labels, size_values, importance_wgt) in enumerate(train_dataset.take(train_steps)):
            print_loss = False
            if i % save_steps == 0:
                model.save_model(model_name)

            if i % eval_steps == 0:
                print_loss = True
                model.evaluate_mtl(eval_dataset, features_list, steps=len(eval_input_enc) // self.eval_batch_size)

                print("[t: {} / i: {}] min_eval_loss:{} / count:{}".format(self.base_idx, i, model.eval_loss, model.eval_count))
                if model.eval_count >= early_stopping_count:
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

            labels_mtl = self.features_cls.labels_for_mtl(features_list, labels, size_values, importance_wgt)
            model.train_mtl(features_with_noise, labels_mtl, print_loss=print_loss)

    def test(self, model, dataset=None, use_label=True, out_dir=None, file_nm='out.png', ylog=False, save_type=None, table_nm=None, time_step=1):
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
            self.features_cls.predict_plot_mtl_cross_section_test(model, _dataset_list,  save_dir=test_out_path, file_nm=file_nm, ylog=ylog, time_step=time_step)
            self.features_cls.predict_plot_mtl_cross_section_test_long(model, _dataset_list, save_dir=test_out_path + "2", file_nm=file_nm, ylog=ylog, time_step=time_step, invest_rate=0.8)
            self.features_cls.predict_plot_mtl_cross_section_test_long(model, _dataset_list, save_dir=test_out_path + "3", file_nm=file_nm, ylog=ylog, time_step=time_step, invest_rate=0.6)

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
        size_value_list = [add_info['size_value'] for add_info in additional_infos]
        df_infos = pd.DataFrame(columns={'start_d', 'base_d', 'infocode', 'score'})
        for i, (input_enc_t, output_dec_t, size_value) in enumerate(zip(input_enc_list, output_dec_list, size_value_list)):
            assert np.sum(input_enc_t[:, -1, :] - output_dec_t[:, 0, :]) == 0
            assert np.sum(output_dec_t[:, 1:, :]) == 0
            new_output_t = np.zeros_like(output_dec_t)
            new_output_t[:, 0, :] = output_dec_t[:, 0, :] + size_value[:, 0, :]

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
        size_value_list = [add_info['size_value'] for add_info in additional_infos]
        df_infos = pd.DataFrame(columns={'start_d', 'base_d', 'infocode', 'score'})
        for i, (input_enc_t, output_dec_t, size_value) in enumerate(zip(input_enc_list, output_dec_list, size_value_list)):
            assert np.sum(input_enc_t[:, -1, :] - output_dec_t[:, 0, :]) == 0
            assert np.sum(output_dec_t[:, 1:, :]) == 0
            new_output_t = np.zeros_like(output_dec_t)
            new_output_t[:, 0, :] = output_dec_t[:, 0, :] + size_value[:, 0, :]
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
        self.test_end_idx = min(self.test_end_idx + self.retrain_days, self.data_generator.max_length - self.k_days - 1)

    def get_date(self):
        return self.date_[self.base_d]

    @property
    def date_(self):
        return self.data_generator.date_

    @property
    def done(self):
        # if self.test_end_idx > self.data_generator.max_length:
        if self.test_end_idx <= self.test_begin_idx:
            return True
        else:
            return False


def cleansing_missing_value(df_selected, n_allow_missing_value=5, to_log=True):
    mask = np.sum(df_selected.isna(), axis=0) <= n_allow_missing_value
    df = df_selected.ix[:, mask].ffill().bfill()
    df = df / df.iloc[0]
    if to_log:
        df = np.log(df)

    return df


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

    def sample_data(self, date_i):
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
        len_label = calc_length_label
        # k_days_adj = k_days + delay_days
        # len_label = k_days_adj

        start_d = date_[(date_i - len_data)]
        end_d = date_[min(date_i + len_label, len(date_) - 1)]

        # data cleansing
        select_where = ((df_pivoted.index >= start_d) & (df_pivoted.index <= end_d))
        df_logp = cleansing_missing_value(df_pivoted.ix[select_where, :], n_allow_missing_value=5, to_log=True)

        if df_logp.empty:
            return False

        univ_list = sorted(list(set.intersection(set(df_logp.columns), set(size_df.index))))

        # calculate features
        features_dict, labels_dict = features_cls.calc_features(df_logp.ix[:, univ_list].to_numpy(dtype=np.float32), transpose=False)

        spot_dict = dict()
        spot_dict['asset_list'] = univ_list
        spot_dict['size_factor'] = size_df.loc[univ_list].mktcap.rank() / len(univ_list)
        spot_dict['size_factor_mktcap'] = size_df.loc[univ_list]

        return features_dict, labels_dict, spot_dict

    @property
    def max_length(self):
        return len(self.date_)


def rearrange(input, output, target, size_value, importance_wgt):
    features = {"input": input, "output": output}
    return features, target, size_value, importance_wgt


# 학습에 들어가 배치 데이터를 만드는 함수이다.
def dataset_process(input_enc, output_dec, target_dec, size_value, batch_size, importance_wgt=None, shuffle=True, iter_num=None):
    # Dataset을 생성하는 부분으로써 from_tensor_slices부분은
    # 각각 한 문장으로 자른다고 보면 된다.
    # train_input_enc, train_output_dec, train_target_dec
    # 3개를 각각 한문장으로 나눈다.
    dataset = tf.data.Dataset.from_tensor_slices((input_enc, output_dec, target_dec, size_value, importance_wgt))

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
