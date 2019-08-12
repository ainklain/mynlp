
class Config:
    def __init__(self):
        # time series parameter
        self.train_set_length = 2500    # previous 10 years data
        self.retrain_days = 250         # re-train every year
        self.m_days = 60                # input length of encoding layer (key, value)
        self.k_days = 5                # input and output length of decoding layer (query)

        self.sampling_days = 5          # get data every 'sampling_days' days
        self.trainset_rate = 0.8

        self.batch_size = 64
        self.train_steps = 200000
        self.eval_steps = 200
        self.early_stopping_count = 10
        self.dropout = 0.5
        self.embedding_size = 14
        self.learning_rate = 1e-4

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

    def export(self):
        return_str = ""
        for key in self.__dict__.keys():
            return_str += "{}: {}\n".format(key, self.__dict__[key])

        return return_str
