
class Config:
    def __init__(self):
        self.batch_size = 64
        self.train_steps = 200000
        self.dropout_width = 0.5
        self.embedding_size = 128
        self.learning_rate = 1e-3
        self.shuffle_seek = 1000
        self.max_sequence_length = 25
        self.model_hidden_size = 128
        self.ffn_hidden_size = 512
        self.attention_head_size = 4
        self.layer_size = 2
        self.data_path = './data/ChatBotData.csv'
        self.vocabulary_path = './data/vocabularyData.txt'
        self.checkpoint_path = './ckpt/'
        self.f_name = 'nlp_model_v1.0'
        self.tokenize_as_morph = False
        self.xavier_initializer = True
