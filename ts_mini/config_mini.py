
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
        return_str = """
        train_set_length={}
        retrain_days={}
        m_days={}
        k_days={}
        sampling_days={}
        trainset_rate={}
        batch_size={}
        train_steps={}
        dropout={}
        embedding_size={}
        learning_rate={}
        """.format(self.train_set_length, self.retrain_days, self.m_days,
                   self.k_days, self.sampling_days, self.trainset_rate,
                   self.batch_size, self.train_steps, self.dropout, self.embedding_size, self.learning_rate)
        return return_str