
class Config:
    def __init__(self):
        # time series parameter
        self.train_set_length = 2500    # previous 10 years data
        self.retrain_days = 250         # re-train every year
        self.m_days = 60                # input length of encoding layer (key, value)
        self.k_days = 5                # input and output length of decoding layer (query)
        self.set_kdays(self.k_days)

        self.sampling_days = 5          # get data every 'sampling_days' days
        self.trainset_rate = 0.8

        self.batch_size = 64
        self.train_steps = 200000
        self.eval_steps = 200
        self.early_stopping_count = 10
        self.dropout = 0.5
        self.learning_rate = 1e-4

        self.delayed_days = 1
        self.use_beta = False

        # features info
        self.set_features_info()

        self.shuffle_seek = 1000
        # self.model_hidden_size = 128
        self.model_hidden_size = self.embedding_size
        self.ffn_hidden_size = 128
        self.attention_head_size = 2
        self.layer_size = 2
        self.data_path = './timeseries/asset_data.csv'
        self.data_out_path = './out/'
        # self.vocabulary_path = './data/vocabularyData.txt'
        self.checkpoint_path = './ckpt/'
        self.f_name = 'ts_model_v1.0'
        self.tokenize_as_morph = False

    def set_features_info(self):
        self.model_predictor_list = ['logy', 'pos_5', 'pos_20', 'std', 'mdd', 'fft']

        self.features_structure = \
            {'regression':
                 {'logy': [5, 20, 60, 120, 250],
                  'std': [20, 60, 120],
                  'mdd': [20, 60, 120],
                  'fft': [3, 100]},
             'classification':
                 {'pos': [5, 20, 60]}}

        self.embedding_size = 0
        for cls in self.features_structure.keys():
            for key in self.features_structure[cls].keys():
                self.embedding_size += len(self.features_structure[cls][key])

    def set_kdays(self, k_days):
        self.k_days = k_days
        self.label_feature = 'logy_{}'.format(self.k_days)
        self.pred_feature = 'pos_{}'.format(self.k_days)

    def export(self):
        return_str = ""
        for key in self.__dict__.keys():
            return_str += "{}: {}\n".format(key, self.__dict__[key])

        return return_str
