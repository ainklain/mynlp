
from collections import OrderedDict
import os

class Config:
    def __init__(self, use_maml=False, use_macro=False, use_swa=False):
        self.num_workers = 0
        # time series parameter
        self.train_set_length = 2500    # previous 10 years data
        self.retrain_days = 250         # re-train every year
        self.m_days = 120               # input length of encoding layer (key, value)
        self.k_days = 20                # input and output length of decoding layer (query)
        self.calc_length = 250
        self.calc_length_label = 120
        self.delay_days = 1

        self.sampling_days = 20          # get data every 'sampling_days' days
        self.trainset_rate = 0.8
        self.cost_rate = 0.003

        # self.train_batch_size = 4096
        # self.eval_batch_size = 4096
        self.train_steps = 100
        self.eval_steps = 100
        self.save_steps = 100
        self.early_stopping_count = 5
        self.dropout = 0.4
        self.learning_rate = 1e-4

        self.train_decaying_factor = 0.99

        self.set_datatype('us_stock')

        self.use_beta = False
        self.univ_type = 'selected'     # all / selected
        self.balancing_method = 'nothing'  # each / once / nothing
        self.weight_scheme = 'ew'       # mw / ew
        self.size_encoding = False          # decoder input에 size_value add할지 여부
        self.app_rate = 1.     # 적용 비율
        # # features info
        # self.set_features_info(self.k_days)

        self.shuffle_seek = 1000
        self.d_model = 64
        self.n_heads = 8
        self.n_layers = 6
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

        # meta
        self.use_maml = use_maml
        if self.use_maml is True:
            self.n_tasks = 5
            self.inner_lr = 1e-2  # 5e-3
            self.meta_lr = 1e-3  # 1e-4

            self.train_set_length = 1000    # previous 10 years data
            self.sampling_days = 20          # get data every 'sampling_days' days
            self.trainset_rate = 0.5

        # uncertainty
        self.use_uncertainty = False

        self.adversarial_training = True    # input data augmentation

        # additional features

        self.possible_func = {'logp_base': ['logp', 'tsp', 'nmy', 'logy', 'std', 'stdnew', 'pos', 'mdd', 'fft', 'cslogy', 'csstd', 'nmlogy', 'nmstd', 'tsnormal', 'csnormal', 'value', 'ir', 'nmir', 'nmirnew'],
                              'size_base': ['nmsize'],
                              'turnover_base': ['nmturnover', 'tsturnover'],
                              'ivol_base': ['nmivol'],
                              'wlogy_base': ['wlogy', 'nmwlogy'],
                              'wlogyrnk_base': ['nmwlogyrnk'],
                              }
        # macro
        # TODO: 전체 데이터 (파일 저장용 - 변수명 및 위치 변경)
        self.macro_dict = {
            'returns': ['exmkt', 'smb', 'hml', 'wml', 'rmw', 'callrate'],  # returns값을 logp로 변환
            'values': ['confindex', 'confindex52', 'momstr', 'momstr52', 'prstr', 'prstr52',
                       'volstr', 'volstr52', 'cs3yaam', 'cs3ybbbm', 'pcratiow', 'vkospi', 'usdkrw'],  # values 그 자체로 사용
        }

        self.macro_features = {
            'returns': [self.get_main_feature(key) for key in ['logp', 'logy', 'std', 'stdnew', 'pos', 'mdd', 'fft']],
            'values': [self.get_main_feature(key) for key in ['value', 'tsnormal']]}

        self.use_macro = use_macro
        if self.use_macro:

            # TODO: 실제 사용 데이터 (전체데이터로부터 로드 - 변수명 및 위치 변경)
            # format: (macro_list, calc_feature_list)
            self.add_features = OrderedDict({
                'returns': (['exmkt'], ['logp_0']),
                'values': (['confindex', 'momstr', 'prstr', 'volstr', 'cs3yaam', 'pcratiow', 'usdkrw'], ['tsnormal_0'])
                # 'returns': (['exmkt', 'smb', 'hml', 'wml', 'rmw', 'callrate'], ['logp_0']),
                # 'values': (['confindex', 'confindex52', 'momstr', 'momstr52', 'prstr', 'prstr52', 'volstr', 'volstr52', 'cs3yaam', 'cs3ybbbm', 'pcratiow', 'vkospi', 'usdkrw'], ['tsnormal_0'])
            })
            # self.add_features_list = [id + '-' + key for id in self.macro_id_list for key in self.macro_key_list]


        # logger
        self.log_level = 'info'
        self.log_maxbytes = 10 * 1024 * 1024
        self.log_backupcount = 10
        self.log_format = "%(asctime)s[%(levelname)s|%(name)s,%(lineno)s] %(message)s"

        # swa
        self.use_swa = use_swa
        if self.use_swa is True:
            self.lr_init = 0.01
            self.momentum = 0.9  # SGD momentum
            self.wd = 1e-4  # weight decay
            self.swa_lr = 0.005
            self.swa_start = 20
            self.swa_c_epochs = 1
            self.eval_freq = 5


    def set_datatype(self, data_type):
        self.data_type = data_type
        # US  일단 덮어쓰기
        if self.data_type == 'us_stock':
            self.train_batch_size = 512
            self.eval_batch_size = 512
            self.min_size_port = 30
        elif self.data_type == 'kr_stock':
            self.train_batch_size = 2048
            self.eval_batch_size = 2048
            self.min_size_port = 100

    def log_filename(self, name_='log'):
        return os.path.join(os.getcwd(), self.data_out_path, self.f_name, f'{name_}.log')

    @property
    def add_features_list(self):
        l = []
        for base_key in self.add_features:
            tmp_m_list, tmp_f_list = self.add_features[base_key]
            for f_ in tmp_f_list:
                l += [m_ + '/' + f_ for m_ in tmp_m_list]

        return l

    @property
    def balancing_key(self):
        return self.get_main_feature('pos')

    @property
    def key_list(self):
        # TODO: key_list_with_macro와 통합
        key_list = self._parse_features_structure()
        return key_list

    @property
    def key_list_with_macro(self):
        if self.use_macro:
            key_list = self._parse_features_structure() + self.add_features_list
        else:
            key_list = self._parse_features_structure()

        return key_list

    def _parse_features_structure(self):
        key_list = []
        for key_cls in self.features_structure.keys():
            for base in self.features_structure[key_cls].keys():
                for subkey in self.features_structure[key_cls][base].keys():
                    for nd in self.features_structure[key_cls][base][subkey]:
                        key_list.append("{}_{}".format(subkey, nd))

        return key_list

    def set_features_info(self, k_days=5, model_predictor_list=None, features_structure=None):
        if model_predictor_list is not None and features_structure is not None:
            self.model_predictor_list = model_predictor_list
            self.features_structure = features_structure

        else:
            if k_days == 5:
                self.model_predictor_list = ['logp', 'nmlogy', 'nmstd', 'pos_5']

                self.features_structure = \
                    {'regression':
                         {'logp_base':
                             {'logp': [0],
                              'logy': [5, 10, 20, 60],
                              'std': [20, 60],
                              'stdnew': [20, 60],
                              'mdd': [20, 60],
                              'fft': [100, 3],
                              'nmlogy': [5, 10, 20, 60],
                              'nmstd': [5, 10, 20, 60],
                              },
                          'size_base': {'nmsize': [0]},
                          'turnover_base': {'nmturnover': [0],
                                            'tsturnover': [0]},
                          'ivol_base': {'nmivol': [0]},
                          },
                     'classification':
                         {'logp_base':
                              {'pos': [5, 10, 20, 60, 120]}
                          }
                     }

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
                self.model_predictor_list = ['logp', 'nmlogy', 'nmstd', 'pos']
                # self.model_predictor_list = ['nmlogy', 'nmstd', 'pos_20']
                # self.model_predictor_list = ['nmlogy']

                self.features_structure = \
                    {'regression':
                         {'logp_base':
                             {'logp': [0],
                              'logy': [20],
                              'nmlogy': [20],
                              'nmstd': [20],
                              },
                          },
                     'classification':
                         {'logp_base':
                            {'pos': [20]}
                          }
                     }

                # self.features_structure = \
                #     {'regression':
                #          {'logp_base':
                #              {'logp': [0],
                #               'logy': [20, 60, 120, 250],
                #               'std': [20, 60, 120],
                #               'stdnew': [20, 60],
                #               'mdd': [20, 60, 120],
                #               'fft': [100, 3],
                #               'nmlogy': [20, 60],
                #               'nmstd': [20, 60],
                #               },
                #           'size_base': {'nmsize': [0]},
                #           'turnover_base': {'nmturnover': [0],
                #                             'tsturnover': [0]},
                #           # 'ivol_base': {'nmivol': [0]},
                #           },
                #      'classification':
                #          {'logp_base':
                #               {'pos': [20, 60, 120, 250]}
                #           }
                #      }

    def set_kdays(self, k_days, pred='pos', **kwargs):
        self.k_days = k_days
        self.label_feature = 'logy_{}'.format(self.k_days)
        self.pred_feature = pred

        self.set_features_info(k_days, **kwargs)
        self.embedding_size = len(self.key_list_with_macro)

    def get_main_feature(self, feature):
        if feature in ['logp', 'tsp', 'nmsize', 'nmturnover', 'tsturnover', 'nmivol', 'value', 'tsnormal', 'csnormal', 'nmwlogy', 'wlogy', 'nmwlogyrnk']:
            return '{}_0'.format(feature)
        elif feature in ['fft']:
            return '{}_100'.format(feature)
        elif feature[:3] == 'pos':
            if len(feature) == 3:
                return '{}_{}'.format(feature, self.k_days)
            else:
                return feature
        else:
            return '{}_{}'.format(feature, self.k_days)

    def export(self):
        return_str = ""
        for key in self.__dict__.keys():
            return_str += "{}: {}\n".format(key, self.__dict__[key])

        return return_str

    def load(self):
        file_path = os.path.join(os.getcwd(), self.data_out_path, self.f_name, 'config.txt')
        if os.path.exists(file_path):
            f = open(file_path, 'r')



    def generate_name(self):
        return "M{}_K{}_COST{}_BS{}_LR{}_BM{}_UT{}".format(self.m_days, self.k_days, self.cost_rate,
                                                      self.train_batch_size, self.learning_rate, self.balancing_method, self.univ_type)


