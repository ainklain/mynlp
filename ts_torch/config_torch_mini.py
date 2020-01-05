
class Config:
    def __init__(self):
        # time series parameter
        self.train_set_length = 2500    # previous 10 years data
        self.retrain_days = 250         # re-train every year
        self.m_days = 120                # input length of encoding layer (key, value)
        self.k_days = 20                # input and output length of decoding layer (query)
        self.calc_length = 250
        self.calc_length_label = 120
        self.delay_days = 1

        self.sampling_days = 20          # get data every 'sampling_days' days
        self.trainset_rate = 0.8
        self.cost_rate = 0.003

        self.train_batch_size = 256
        self.eval_batch_size = 256
        self.train_steps = 200000
        self.eval_steps = 50
        self.save_steps = 50
        self.early_stopping_count = 5
        self.dropout = 0.5
        self.learning_rate = 1e-4

        self.train_decaying_factor = 0.99

        self.use_beta = False
        self.univ_type = 'selected'     # all / selected
        self.balancing_method = 'each'  # each / once / nothing
        self.data_type = 'kr_stock'
        self.weight_scheme = 'mw'       # mw / ew
        self.size_encoding = False          # decoder input에 size_value add할지 여부
        self.app_rate = 1.     # 적용 비율
        # # features info
        # self.set_features_info(self.k_days)

        self.shuffle_seek = 1000
        self.d_model = 64
        self.n_heads = 4
        self.n_layers = 2
        # self.model_hidden_size = 128
        # self.model_hidden_size = self.embedding_size    # self.set_features_info 에서 재설정
        self.d_ff = self.d_model
        self.d_k = self.d_v = self.d_q = self.d_model // self.n_heads
        self.data_path = './timeseries/asset_data.csv'
        self.data_out_path = './out/'
        # self.vocabulary_path = './data/vocabularyData.txt'
        self.checkpoint_path = './ckpt/'
        self.f_name = 'ts_model_v1.0'
        self.tokenize_as_morph = False

        self.max_input_seq_len = 24
        self.max_output_seq_len = 12

        self.weighted_model = False     #  torch: use weighted model
        self.set_kdays(self.k_days)

        # MARKET MODEL
        self.data_type_mm = 'kr_market'
        self.d_model_mm = 32
        self.model_hidden_size_mm = self.d_model_mm
        self.ffn_hidden_size_mm = 32
        self.attention_head_size_mm = 4
        self.layer_size_mm = 2
        self.learning_rate_mm = 1e-4
        self.dropout_mm = 0.5
        self.embedding_size_mm = 10

        # meta
        self.use_maml = False
        self.n_tasks = 5
        self.inner_lr = 1e-2  # 5e-3
        self.meta_lr = 1e-3  # 1e-4

        # self.train_set_length = 1000    # previous 10 years data
        # self.sampling_days = 20          # get data every 'sampling_days' days
        # self.trainset_rate = 0.5


    @property
    def balancing_key(self):
        return 'pos_{}'.format(self.k_days)

    @property
    def key_list(self):
        # key_list = []
        # if self.k_days == 20:
        #     key_list += ['logy_{}'.format(n) for n in [20, 60, 120]]
        #     key_list += ['std_{}'.format(n) for n in [20, 60, 120]]
        #     key_list += ['mdd_{}'.format(n) for n in [20, 60, 120]]
        #     key_list += ['stdnew_{}'.format(n) for n in [20, 60, 120]]
        #     key_list += ['pos_{}'.format(n) for n in [20, 60]]

            # label_keys = ['logy_{}'.format(n) for n in [20, 60, 120]]
            # label_keys += ['stdnew_{}'.format(n) for n in [20, 60, 120]]
            # label_keys += ['pos_{}'.format(n) for n in [20, 60]]
        key_list = self._parse_features_structure() + ['nmsize']
        return key_list

    def _parse_features_structure(self):
        key_list = []
        for key_cls in self.features_structure.keys():
            for subkey in self.features_structure[key_cls].keys():
                for nd in self.features_structure[key_cls][subkey]:
                    key_list.append("{}_{}".format(subkey, nd))

        return key_list

    def set_features_info(self, k_days=5):
        if k_days == 5:
            # self.model_predictor_list = ['logy', 'pos_20', 'pos_60', 'pos_120', 'std', 'mdd', 'fft']
            # self.model_predictor_list = ['logy', 'cslogy_5', 'pos_5', 'pos_10', 'pos_20', 'std', 'mdd', 'fft']
            # self.model_predictor_list = ['logy', 'cslogy', 'csstd', 'std', 'stdnew', 'mdd', 'fft', 'pos_5', 'pos_20']
            # self.model_predictor_list = ['cslogy', 'csstd', 'pos_5']

            self.model_predictor_list = ['nmlogy', 'nmstd', 'pos_5']

            # self.features_structure = \
            #     {'regression':
            #          {'logy': [5],
            #           'std': [5],
            #           'stdnew': [5],
            #           'mdd': [5],
            #           'fft': [100],
            #           'nmlogy': [5],
            #           'nmstd': [5],
            #           },
            #      'classification':
            #          {'pos': [5]}}

            self.features_structure = \
                    {'regression':
                         # {'logy': [20, 60, 120, 250],
                         {'logy': [5, 10, 20, 60, 120],
                          'logp': [0],
                          'std': [5, 20, 60, 120],
                          'stdnew': [5, 20],
                          'mdd': [20, 60, 120],
                          'fft': [100, 3],
                          'nmlogy': [5, 20],
                          'nmstd': [5, 20],
                          },
                     'classification':
                         # {'pos': [20, 60, 120, 250]}}
                         {'pos': [5, 10, 20, 60]}, }
                     # 'crosssection':
                     #     {'cslogy': [5, 20]}}

        elif k_days == 10:
            # self.model_predictor_list = ['logy', 'pos_10', 'pos_20', 'pos_60', 'std', 'mdd', 'fft']
            self.model_predictor_list = ['logy', 'cslogy', 'csstd', 'std', 'stdnew', 'mdd', 'fft']

            self.features_structure = \
                {'regression':
                     {'logy': [10, 20, 60, 120],
                      'std': [10, 20, 60],
                      'stdnew': [10, 20],
                      'mdd': [20, 60, 120],
                      'fft': [3, 100],
                      'cslogy': [10, 20],
                      'csstd': [10, 20],
                      },
                 'classification':
                     {'pos': [10, 20, 60, 120]}}

        elif k_days == 20:
            # self.model_predictor_list = ['logy', 'pos_20', 'pos_60', 'pos_120', 'std', 'mdd', 'fft']
            # self.model_predictor_list = ['logy', 'cslogy', 'csstd', 'std', 'stdnew', 'mdd', 'fft', 'pos_20', 'pos_60']

            # self.model_predictor_list = ['logy', 'cslogy', 'std', 'stdnew', 'mdd', 'fft', 'pos_20', 'pos_60']

            # self.model_predictor_list = ['std']
            self.model_predictor_list = ['nmlogy', 'nmstd', 'pos_20']
            # self.model_predictor_list = ['nmlogy']

            # self.features_structure = \
            #     {'regression':
            #          {'logy': [20],
            #           'std': [20],
            #           'stdnew': [20],
            #           'mdd': [20],
            #           'fft': [100],
            #           'nmlogy': [20],
            #           'nmstd': [20],
            #           },
            #      'classification':
            #          {'pos': [20]}}

            self.features_structure = \
                {'regression':
                     {'logp': [0],
                      'logy': [20, 60, 120, 250],
                      'std': [20, 60, 120],
                      'stdnew': [20, 60],
                      'mdd': [20, 60, 120],
                      'fft': [100, 3],
                      'nmlogy': [20, 60],
                      'nmstd': [20, 60],
                      },
                 'classification':
                     {'pos': [20, 60, 120, 250]}}

        self.embedding_size = len(self._parse_features_structure()) + 1 # 1: nm_size

    def set_kdays(self, k_days, pred='pos'):
        self.k_days = k_days
        self.label_feature = 'logy_{}'.format(self.k_days)
        if pred == 'pos':
            self.pred_feature = 'pos_{}'.format(self.k_days)
        # elif pred == 'cslogy':
        #     self.pred_feature = 'cslogy_{}'.format(self.k_days)
        else:
            self.pred_feature = pred

        self.set_features_info(k_days)

    def export(self):
        return_str = ""
        for key in self.__dict__.keys():
            return_str += "{}: {}\n".format(key, self.__dict__[key])

        return return_str

    def generate_name(self):
        return "M{}_K{}_COST{}_BS{}_LR{}_BM{}_UT{}".format(self.m_days, self.k_days, self.cost_rate,
                                                      self.train_batch_size, self.learning_rate, self.balancing_method, self.univ_type)


