

import os

# ## TORCH TEST
import torch
from torch import nn, optim

# scheduler test
from ts_torch.model_torch_mini import TSModel
from ts_mini.features_mini import FeatureNew

# vtorch test
from ts_torch.data_process_torch_mini import DataScheduler, Noise
from ts_torch.config_torch_mini import Config
from ts_torch.performance_torch_mini import Performance
from ts_torch import torch_util_mini as tu


configs = Config()

k_days = 20; w_scheme = 'mw'; univ_type='selected'; pred='nmlogy'; balancing_method='nothing'; head=8
configs.set_kdays(k_days)
configs.pred_feature = pred
configs.weight_scheme = w_scheme
configs.balancing_method = balancing_method
# configs.learning_rate = 1e-4
configs.f_name = 'kr_{}_{}_{}_{}_h{}_torch_maml_005'.format(k_days, univ_type, balancing_method, pred, head)
configs.train_steps = 100
configs.eval_steps = 100
configs.save_steps = 100
configs.size_encoding = True
configs.attention_head_size = head
configs.early_stopping_count = 50
configs.learning_rate = 5e-4
configs.update_comment = 'save and load'
config_str = configs.export()


features_cls = FeatureNew(configs)
ds = DataScheduler(configs, features_cls)
ds.set_idx(6500)

os.makedirs(os.path.join(ds.data_out_path), exist_ok=True)
with open(os.path.join(ds.data_out_path, 'config.txt'), 'w') as f:
    f.write(config_str)


model = TSModel(configs, features_cls, weight_scheme=configs.weight_scheme)

performer = Performance(configs)
if configs.use_maml:
    optimizer = optim.Adam(model.parameters(), lr=configs.meta_lr)
else:
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)

ds.load(model, optimizer)

# ds.train_maml(model, optimizer, performer, num_epochs=50, early_stopping_count=configs.early_stopping_count)

while True:
    ds.save(0, model, optimizer)
    if configs.use_maml:
        ds.train_maml(model, optimizer, performer, num_epochs=50, early_stopping_count=configs.early_stopping_count)
    else:
        ds.train(model, optimizer, performer, num_epochs=50, early_stopping_count=configs.early_stopping_count)
    ds.next()
    if ds.done:
        break

    if ds.base_idx >= 10000:
        print('something wrong')
        break



# recent value extraction
import numpy as np
recent_month_end = '2019-10-31'
dataloader_t = ds.dataloader_t(recent_month_end)
# dataset_t = ds._dataset_t(recent_month_end)
#
# enc_in, dec_in, dec_out, features_list, add_infos = dataset_t
# new_dec_in = np.zeros_like(enc_in)
# new_dec_in[:, 0, :] = enc_in[:, 0, :]
# new_dec_in[:] += np.array(add_infos['size_rnk']).reshape(-1, 1, 1)
#
# features = {'input': torch.from_numpy(enc_in), 'output': torch.from_numpy(new_dec_in)}
# dataloader = [features, add_infos]
# all_assets_list = add_infos['asset_list']
# dataloader_t = (dataloader, features_list, all_assets_list, _, _)
x = performer.extract_portfolio(model, dataloader_t, rate_=configs.app_rate)
x.to_csv('./out/{}/result_10.csv'.format(configs.f_name))




if os.path.exists(os.path.join(ds.data_out_path, configs.f_name, configs.f_name + '.pkl')):
    model.load_model(os.path.join(ds.data_out_path, configs.f_name, configs.f_name))

testset_insample = ds._dataset('test_insample')
testset_insample_m = ds._dataset_monthly('test_insample')
testset = ds._dataset('test')
testset_m = ds._dataset_monthly('test')


while not ds.done:
    if ii > 100 or (ii > 1 and model.eval_loss > 10000):
        jj += 1
        ii = 0
        ds.next()

        print("jj: {}".format(jj))
        trainset = ds._dataset('train')
        evalset = ds._dataset('eval')
        testset_insample = ds._dataset('test_insample')
        testset_insample_m = ds._dataset_monthly('test_insample')
        testset = ds._dataset('test')
        testset_m = ds._dataset_monthly('test')

    # if trainset is None:
    #     trainset = ds._dataset('train')
    #     evalset = ds._dataset('eval')

    if ii > 0:
        is_trained = ds.train(model, trainset, evalset
                              , model_name=os.path.join(ds.data_out_path, configs.f_name, configs.f_name)
                              , epoch=True)

        if is_trained is not False:
            model.save_model(os.path.join(ds.data_out_path, configs.f_name, configs.f_name, str(ds.base_idx), configs.f_name))

    ds.test(performer, model, testset_insample, testset_insample_m,
            use_label=True,
            out_dir=os.path.join(ds.data_out_path, configs.f_name, str(jj), 'test_insample'),
            file_nm='test_{}.png'.format(ii),
            ylog=False,
            # save_type='csv',
            table_nm='kr_weekly_score_temp')

    ds.test(performer, model, testset, testset_m,
            use_label=True,
            out_dir=os.path.join(ds.data_out_path, configs.f_name, str(jj), 'test'),
            file_nm='test_{}.png'.format(ii),
            ylog=False,
            # save_type='csv',
            table_nm='kr_weekly_score_temp')

    ii += 1

















# MODEL TEST
from ts_torch.model_torch_mini import ConvEmbeddingLayer, Encoder, Decoder

features = {'input': torch.zeros(2, 25, configs.embedding_size), 'output': torch.zeros(2, 1, configs.embedding_size)}
print(features['input'].shape, features['output'].shape)
conv_embedding = ConvEmbeddingLayer(n_features=configs.embedding_size, d_model=configs.d_model)
encoder = Encoder(configs.n_layers, configs.d_k, configs.d_v, configs.d_model, configs.d_ff,
                               configs.n_heads, configs.max_input_seq_len, configs.dropout, configs.weighted_model)
decoder = Decoder(configs.n_layers, configs.d_k, configs.d_v, configs.d_model, configs.d_ff,
                               configs.n_heads, configs.max_output_seq_len, configs.dropout, configs.weighted_model)
weighted_model = configs.weighted_model

# input_seq_size = configs.m_days // configs.sampling_days + 1
# self.output_seq_size = configs.k_days // configs.sampling_days
output_seq_size = 1
conv_in = conv_embedding(features['input'])
conv_out = conv_embedding(features['output'])

input_seq_size = torch.Tensor([conv_in.shape[1] for _ in range(conv_in.shape[0])])
output_seq_size = torch.Tensor([conv_out.shape[1] for _ in range(conv_out.shape[0])])
enc_outputs, enc_self_attns = encoder(conv_in, input_seq_size, return_attn=False)

dec_outputs, dec_self_attns, dec_enc_attns = decoder(conv_out, output_seq_size, conv_in, enc_outputs, return_attn=False)


# PosEncoding Test
from ts_torch.model_torch_mini import PosEncoding
pos_emb = PosEncoding(configs.max_input_seq_len * 10, configs.d_model)
input_len = torch.Tensor([10, 14, 12, 20])
pe = pos_emb(input_len)


# MultiHeadAttention Test
from ts_torch.model_torch_mini import MultiHeadAttention, get_attn_pad_mask
enc_self_attn = MultiHeadAttention(configs.d_k, configs.d_v, configs.d_model, configs.n_heads, configs.dropout)

input_seq_len = torch.Tensor([conv_in.shape[1] for _ in range(conv_in.shape[0])])
pe = pos_emb(input_seq_len)
conv_in_pe = conv_in + pe
attn_mask = get_attn_pad_mask(conv_in_pe, conv_in_pe)

enc, attn = enc_self_attn(conv_in_pe, conv_in_pe, conv_in_pe, attn_mask)

# EncoderLayer TEST
from ts_torch.model_torch_mini import EncoderLayer
el = EncoderLayer(configs.d_k, configs.d_v, configs.d_model, configs.d_ff, configs.n_heads, configs.dropout)

el(conv_in_pe, attn_mask)


# ENCODER TEST
from ts_torch.model_torch_mini import WeightedEncoderLayer

dropout_emb = nn.Dropout(configs.dropout)
layer_type = EncoderLayer if not weighted_model else WeightedEncoderLayer
layers = nn.ModuleList(
            [layer_type(configs.d_k, configs.d_v, configs.d_model, configs.d_ff, configs.n_heads, configs.dropout)
             for _ in range(configs.n_layers)])

enc_self_attn_mask = get_attn_pad_mask(conv_in, conv_in)
enc_self_attns = []
for layer in layers:
    enc_outputs, enc_self_attn = layer(conv_in_pe, enc_self_attn_mask)


# DECODER TEST
from ts_torch.model_torch_mini import get_attn_subsequent_mask, get_attn_pad_mask
dec_inputs, dec_inputs_len, enc_inputs, enc_outputs = conv_out, output_seq_size, conv_in, enc_outputs
return_attn = False

dropout_emb = nn.Dropout(configs.dropout)
dec_outputs = dec_inputs + pos_emb(dec_inputs_len)  # Adding positional encoding # TODO: note
dec_outputs = dropout_emb(dec_outputs)

dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)




