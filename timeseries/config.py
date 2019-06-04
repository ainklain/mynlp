
class Config:
    def __init__(self):
        self.batch_size = 32
        self.train_steps = 200000
        self.dropout_width = 0.5
        self.embedding_per_feature = 5
        self.n_features = 2
        self.embedding_size = self.n_features * self.embedding_per_feature
        self.learning_rate = 1e-3
        self.shuffle_seek = 1000
        self.max_sequence_length_in = 12
        self.max_sequence_length_out = 3
        # self.model_hidden_size = 128
        self.model_hidden_size = 10
        self.ffn_hidden_size = 512
        self.attention_head_size = 2
        self.layer_size = 2
        self.data_path = './timeseries/data.csv'
        # self.vocabulary_path = './data/vocabularyData.txt'
        self.checkpoint_path = './ckpt/'
        self.f_name = 'ts_model_v1.0'
        self.tokenize_as_morph = False
        self.xavier_initializer = True
