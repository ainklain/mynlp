

import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os
import pandas as pd
from pyts.image import GramianAngularField
import time
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from torch.optim import lr_scheduler
from torchvision import models, datasets, transforms
from torchsummary import summary


# define device
class Configs:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.image_size = 64

        self.m_days = 1000           # 이걸 input_length로 정정예정
        self.lookback_period = 500  # class balance를 위해 보는 기간, 이걸 m_days로 정정예정
        self.k_days = 5

        self.downsampling_size = 5

        self.k_shot = 20
        self.n_tasks = 50
        self.lr_inner = 0.02
        self.lr_meta = 0.0005
        self.num_epochs = 200
        self.test_period = 260  # test 적용기간

        self.early_stopping = 20
        self.print_step = 5

def prepare_dataset():
    data_path = './data/data_for_metarl.csv'
    data_df = pd.read_csv(data_path)
    data_df.set_index('eval_d', inplace=True)
    date_ = list(data_df.index)
    return data_df


def minmax_scaler(arr, axis=0):
    return (arr - np.min(arr, axis=axis, keepdims=True)) / (np.max(arr, axis=axis, keepdims=True) - np.min(arr, axis=axis, keepdims=True))


def arr_to_dataset_ntasks(log_arr, m_days, k_days=5):
    delay_days = 1

    k_days_adj = k_days + delay_days
    sampling_days = k_days

    arr_list, label_list = [], []
    for i in range(m_days, len(log_arr) - (k_days_adj + 1), sampling_days):
        arr_list.append(log_arr[(i - m_days):(i + 1)] - log_arr[i-m_days])
        label_list.append(log_arr[(i + delay_days):][k_days] - log_arr[(i + delay_days):][0])
    return arr_list, label_list


def arr_to_dataset_ntasks2(log_arr, m_days, k_days=5):
    delay_days = 1
    k_days_adj = k_days + delay_days
    sampling_days = k_days

    log_arr_spt, log_arr_qry = log_arr[:(-k_days_adj)], log_arr[-(k_days_adj+m_days+1):]  # QRY LABEL 분리

    assert len(log_arr_spt) > m_days + k_days_adj + 1
    assert len(log_arr_qry) == m_days + k_days_adj + 1

    arr_list_spt, label_list_spt, arr_list_qry, label_list_qry = [], [], [], []
    for i in range(m_days, len(log_arr_spt) - k_days_adj, sampling_days):
        arr_list_spt.append(log_arr[(i - m_days):(i + 1)] - log_arr[i-m_days])
        label_list_spt.append(log_arr[(i + delay_days):][k_days] - log_arr[(i + delay_days):][0])

    arr_list_qry.append(log_arr_qry[:(m_days + 1)] - log_arr_qry[0])
    label_list_qry.append(log_arr_qry[(m_days + delay_days):][k_days] - log_arr_qry[(m_days + delay_days)])
    return arr_list_spt, label_list_spt, arr_list_qry, label_list_qry


def convert_to_image(data_arr, image_size, downsampling_size=5):
    # Gramian Angular Field
    gasf = GramianAngularField(image_size=image_size, method='summation')

    # original
    logp_scaled_spt = minmax_scaler(data_arr, axis=1)
    fig1 = gasf.fit_transform(logp_scaled_spt)

    # smoothing
    logp_smoothed = data_arr[:, 60:] - data_arr[:, :-60]
    logp_smoothed_scaled = minmax_scaler(logp_smoothed, axis=1)
    fig2 = gasf.fit_transform(logp_smoothed_scaled)

    # downsampling
    logp_downsampled = data_arr[:, ::downsampling_size]
    logp_downsampled_scaled = minmax_scaler(logp_downsampled, axis=1)
    fig3 = gasf.fit_transform(logp_downsampled_scaled)
    figs = np.stack([fig1, fig2, fig3], axis=1)

    assert figs.shape[1:] == (3, image_size, image_size)

    return figs


def preprocessing_ntasks(arr, image_size=64):
    m_days = 500
    downsampling_size = 5
    assert m_days > image_size * downsampling_size
    data_, label_ = arr_to_dataset_ntasks(arr, m_days=m_days, k_days=5)
    data_arr = np.stack(data_, axis=0)
    label_arr = np.stack(label_, axis=0)

    n_classes = 2
    label_class = ((label_arr < 0) * 1).astype(np.float32)  # 0보다 크면 0, 작으면 1

    figs = convert_to_image(data_arr, image_size, downsampling_size=downsampling_size).astype(np.float32)

    transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    figs = torch.stack([transformer(torch.from_numpy(fig)) for fig in figs]).numpy()

    n_spt = 48  # n_spt * sampling_freq = n_days for support

    k_shot = 5
    d_spt_list, l_spt_list, d_qry_list, l_qry_list = [], [], [], []
    y_spt_list, y_qry_list = [], []
    for i in range(n_spt, len(figs)):
        selected = label_class[(i - n_spt):i]
        idx = np.arange(len(selected))

        random_sampled = []
        for class_i in range(n_classes):
            idx_c = np.random.choice(idx[selected == class_i], k_shot, replace=True)
            random_sampled.append(idx_c)
        idx_balanced = np.random.permutation(np.concatenate(random_sampled))
        d_spt_list.append(figs[(i - n_spt):i][idx_balanced])
        l_spt_list.append(label_class[(i - n_spt):i][idx_balanced])
        d_qry_list.append(figs[i:(i+1)])
        l_qry_list.append(label_class[i:(i+1)])
        y_spt_list.append(label_arr[(i - n_spt):i][idx_balanced])
        y_qry_list.append(label_arr[i:(i+1)])

    spt_x = torch.from_numpy(np.stack(d_spt_list))  # transform to torch tensors
    spt_y = torch.from_numpy(np.stack(l_spt_list))
    qry_x = torch.from_numpy(np.stack(d_qry_list))  # transform to torch tensors
    qry_y = torch.from_numpy(np.stack(l_qry_list))
    spt_logy = torch.from_numpy(np.stack(y_spt_list))
    qry_logy = torch.from_numpy(np.stack(y_qry_list))

    return spt_x, spt_y, qry_x, qry_y, spt_logy, qry_logy


def make_onetask(arr, configs):
    # 항상 가장 마지막 값을 qry, 이전의 데이터를 spt
    image_size = configs.image_size
    k_shot = configs.k_shot

    m_days = configs.m_days
    downsampling_size = 5
    assert m_days > image_size * downsampling_size
    data_list_spt, label_list_spt, data_list_qry, label_list_qry = arr_to_dataset_ntasks2(arr, m_days=m_days, k_days=5)
    data_arr_spt = np.stack(data_list_spt, axis=0)
    label_arr_spt = np.stack(label_list_spt, axis=0)
    data_arr_qry = np.stack(data_list_qry, axis=0)
    label_arr_qry = np.stack(label_list_qry, axis=0)

    n_classes = 2
    labels_spt = ((label_arr_spt < 0) * 1).astype(np.int32)  # 0보다 크면 0, 작으면 1
    labels_qry = ((label_arr_qry < 0) * 1).astype(np.int32)  # 0보다 크면 0, 작으면 1

    figs_spt = convert_to_image(data_arr_spt, image_size, downsampling_size=downsampling_size).astype(np.float32)
    figs_qry = convert_to_image(data_arr_qry, image_size, downsampling_size=downsampling_size).astype(np.float32)

    transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    figs_spt = torch.stack([transformer(torch.from_numpy(fig)) for fig in figs_spt]).numpy()
    figs_qry = torch.stack([transformer(torch.from_numpy(fig)) for fig in figs_qry]).numpy()

    y_spt, y_qry = label_arr_spt, label_arr_qry

    idx = np.arange(len(labels_spt))
    idx_p = np.array([np.power(0.99, len(idx) - i - 1) for i in idx])
    random_sampled = []
    for class_i in range(n_classes):
        selected = (labels_spt == class_i)
        idx_c = np.random.choice(idx[selected], k_shot, replace=True, p=idx_p[selected] / np.sum(idx_p[selected]))
        random_sampled.append(idx_c)
    idx_balanced = np.random.permutation(np.concatenate(random_sampled))

    spt_x = torch.from_numpy(figs_spt[idx_balanced])  # transform to torch tensors
    spt_y = torch.from_numpy(labels_spt[idx_balanced])
    qry_x = torch.from_numpy(figs_qry)  # transform to torch tensors
    qry_y = torch.from_numpy(labels_qry)
    spt_logy = torch.from_numpy(y_spt[idx_balanced])
    qry_logy = torch.from_numpy(y_qry)

    return spt_x, spt_y, qry_x, qry_y, spt_logy, qry_logy


def make_dataloader_ntasks(arr_y, split_r=[0.6, 0.8]):
    # arr_y = np.array(data_df['kospi'], dtype=np.float32);split_r=[0.6, 0.8]; batch_size=32
    arr_logp = np.log(np.cumprod(1 + arr_y) / (1 + arr_y[0]))

    train_arr = arr_logp[:int(len(arr_logp) * split_r[0])]
    eval_arr = arr_logp[int(len(arr_logp) * split_r[0]):int(len(arr_logp) * split_r[1])]
    test_arr = arr_logp[int(len(arr_logp) * split_r[1]):]

    image_size = 64
    dataset_train = preprocessing_ntasks(train_arr, image_size=image_size)
    dataset_eval = preprocessing_ntasks(eval_arr, image_size=image_size)
    dataset_test = preprocessing_ntasks(test_arr, image_size=image_size)

    return dataset_train, dataset_eval, dataset_test


def make_dataloader_ntasks2(arr_y, split_r=[0.5]):
    # test는 차트용, 전체데이터를 train, eval로 나눔
    # arr_y = np.array(data_df['kospi'], dtype=np.float32);split_r=[0.6, 0.8]; batch_size=32
    arr_logp = np.log(np.cumprod(1 + arr_y) / (1 + arr_y[0]))

    train_arr = arr_logp[:int(len(arr_logp) * split_r[0])]
    eval_arr = arr_logp[int(len(arr_logp) * split_r[0]):]
    test_arr = arr_logp[:]

    image_size = 64
    dataset_train = preprocessing_ntasks(train_arr, image_size=image_size)
    dataset_eval = preprocessing_ntasks(eval_arr, image_size=image_size)
    dataset_test = preprocessing_ntasks(test_arr, image_size=image_size)

    return dataset_train, dataset_eval, dataset_test


def np_to_tensor(list_of_numpy_objs):
    return (torch.from_numpy(np.array(obj, dtype=np.float32)) for obj in list_of_numpy_objs)


def compute_loss(model, x, y, loss_fn=nn.MSELoss()):
    logits = model.forward(x)
    mse = loss_fn(y, logits)
    return mse, logits


def plot_model(model, dataloader_test, criterion, ep, lr_inner=0.01):

    x_spt, y_spt, x_qry, y_qry, spt_logy, logy_qry = dataloader_test
    result1, result2 = [], []
    pred1, pred2 = [], []
    acc1, acc2 = [], []

    device = 'cuda:0'
    model.to(device)
    s_t = time.time()
    for i in range(len(x_spt)):
        x_spt_i, y_spt_i, x_qry_i, y_qry_i, logy_spt_i, logy_qry_i = x_spt[i], y_spt[i], x_qry[i], y_qry[i], spt_logy[i], logy_qry[i]

        x_spt_i = x_spt_i.to(device)
        y_spt_i = y_spt_i.long().to(device)
        x_qry_i = x_qry_i.to(device)
        y_qry_i = y_qry_i.long().to(device)

        logy_spt_i = logy_spt_i.to(device)
        logy_qry_i = logy_qry_i.cpu()

        # # 매개변수 경사도를 0으로 설정
        # optimizer.zero_grad()

        with torch.no_grad():
            # before train
            # 0보다 크면 0, 작으면 1
            out_qry_i_before = model.forward(x_qry_i)
            _, pred_before = torch.max(out_qry_i_before.cpu(), 1)
            pred1.append(pred_before.numpy()[0])
            acc1.append((pred_before == y_qry_i.cpu()).numpy()[0])
            result1.append(int(not (pred1[-1])) * logy_qry_i.numpy()[0])
            # result1.append(int(not(pred1[-1])) * logy_qry_i.numpy()[0] + int(pred1[-1]) * logy_qry_i.numpy()[0])

        outputs_spt_i = model.forward(x_spt_i)

        _, preds = torch.max(outputs_spt_i, 1)
        train_loss = criterion(outputs_spt_i, y_spt_i) * torch.exp(logy_spt_i)
        train_loss = torch.mean(train_loss)
        # Step 6
        grad = torch.autograd.grad(train_loss, model.parameters(), retain_graph=True, create_graph=True)

        fast_weights = list(map(lambda p: p[1] - lr_inner * p[0], zip(grad, model.parameters())))

        with torch.no_grad():
            out_qry_i_after = model.forward(x_qry_i, fast_weights)
            _, pred_after = torch.max(out_qry_i_after.cpu(), 1)
            pred2.append(pred_after.numpy()[0])
            acc2.append((pred_after == y_qry_i.cpu()).numpy()[0])
            result2.append(int(not(pred2[-1])) * logy_qry_i.numpy()[0])
            # result2.append(int(not(pred2[-1])) * logy_qry_i.numpy()[0] + int(pred2[-1]) * logy_qry_i.numpy()[0])

    e_t = time.time()
    print("{} {} sec".format(i, e_t - s_t))

    print("acc1: {}, acc2: {}".format(np.sum(acc1) / len(acc1), np.sum(acc2) / len(acc2)))
    result_arr1 = np.array(result1, dtype=np.float32)
    result_arr2 = np.array(result2, dtype=np.float32)
    result_plot1 = np.cumsum(result_arr1) - result_arr1[0]
    result_plot2 = np.cumsum(result_arr2) - result_arr2[0]
    result_bm = np.cumsum(logy_qry.squeeze()) - logy_qry.squeeze()[0]

    fig = plt.figure()
    plt.plot(np.arange(len(result_bm)), result_bm, 'b')
    plt.plot(np.arange(len(result_bm)), result_plot1, 'k')
    plt.plot(np.arange(len(result_bm)), result_plot2, 'r')
    fig.legend(['bm', 'before', 'after'])
    fig.savefig('./out/maml/{}.png'.format(ep))
    plt.close(fig)


def plot_model2(configs, model, dataloaders, criterion, ep, use_maml=True):
    print('plot: use_maml {}'.format(use_maml))
    lr_inner = configs.lr_inner

    before = {'result': [], 'pred': [], 'acc': [], 'prob': []}
    after = {'result': [], 'pred': [], 'acc': [], 'prob': []}
    bm = []

    dataloader_test = dataloaders['train'] + dataloaders['eval'] + dataloaders['test']
    data_len = [len(dataloaders['train']), len(dataloaders['eval']), len(dataloaders['test'])]
    device = 'cuda:0'
    model.to(device)
    s_t = time.time()
    for i, dataset in enumerate(dataloader_test):
        x_spt_i, y_spt_i, x_qry_i, y_qry_i, logy_spt_i, logy_qry_i = dataset

        x_spt_i = x_spt_i.to(device)
        y_spt_i = y_spt_i.long().to(device)
        x_qry_i = x_qry_i.to(device)
        y_qry_i = y_qry_i.long().to(device)

        logy_spt_i = logy_spt_i.to(device)
        logy_qry_i = logy_qry_i.cpu()
        # bm.append(logy_qry_i.numpy()) # TODO: real return
        bm.append(np.sign(logy_qry_i.numpy()) * 0.01)

        # # 매개변수 경사도를 0으로 설정
        # optimizer.zero_grad()

        with torch.no_grad():
            # before train
            # 0보다 크면 0, 작으면 1
            out_qry_i_before = model.forward(x_qry_i)
            _, pred_before = torch.max(out_qry_i_before.cpu(), 1)
            before['prob'].append(F.sigmoid(out_qry_i_before).cpu().numpy()[0][0])
            before['pred'].append(pred_before.numpy()[0])
            before['acc'].append((pred_before == y_qry_i.cpu()).numpy()[0])
            # before['result'].append(int(not(before['pred'][-1])) * logy_qry_i.numpy()[0]) # TODO: real return
            before['result'].append(int(not(before['pred'][-1])) * np.sign(logy_qry_i.numpy()[0]) * 0.01)
            # result1.append(int(not(pred1[-1])) * logy_qry_i.numpy()[0] + int(pred1[-1]) * logy_qry_i.numpy()[0])

        if use_maml:
            outputs_spt_i = model.forward(x_spt_i)

            _, preds = torch.max(outputs_spt_i, 1)
            train_loss = criterion(outputs_spt_i, y_spt_i) # * torch.exp(logy_spt_i) / torch.sum(torch.exp(logy_spt_i)) # TODO: real return
            train_loss = torch.sum(train_loss) / len(train_loss)
            # Step 6
            grad = torch.autograd.grad(train_loss, model.parameters(), retain_graph=True, create_graph=True)

            fast_weights = list(map(lambda p: p[1] - lr_inner * p[0], zip(grad, model.parameters())))

        with torch.no_grad():
            if use_maml:
                out_qry_i_after = model.forward(x_qry_i, fast_weights)
            else:
                out_qry_i_after = model.forward(x_qry_i)

            _, pred_after = torch.max(out_qry_i_after.cpu(), 1)
            after['prob'].append(F.sigmoid(out_qry_i_after).cpu().numpy()[0][0])
            after['pred'].append(pred_after.numpy()[0])
            after['acc'].append((pred_after == y_qry_i.cpu()).numpy()[0])
            # after['result'].append(int(not(after['pred'][-1])) * logy_qry_i.numpy()[0]) # TODO: real return
            after['result'].append(int(not(after['pred'][-1])) * np.sign(logy_qry_i.numpy()[0]) * 0.01)
            # result2.append(int(not(pred2[-1])) * logy_qry_i.numpy()[0] + int(pred2[-1]) * logy_qry_i.numpy()[0])

    e_t = time.time()
    print("{} {} sec".format(i, e_t - s_t))

    print("acc1: {}, acc2: {}".format(np.sum(before['acc']) / len(before['acc']), np.sum(after['acc']) / len(after['acc'])))
    before['result'] = np.array(before['result'], dtype=np.float32)
    after['result'] = np.array(after['result'], dtype=np.float32)
    before['prob'] = np.array(before['prob'], dtype=np.float32)
    after['prob'] = np.array(after['prob'], dtype=np.float32)
    bm_arr = np.array(bm, dtype=np.float32).squeeze()

    result_bm = np.cumsum(bm_arr) - bm_arr[0]
    len1 = data_len[0]
    len2 = data_len[0] + data_len[1]
    # before
    result_plot1 = np.cumsum(before['result']) - before['result'][0]
    result_plot1[len1:] = result_plot1[len1:] - result_plot1[len1] + result_bm[len1]
    result_plot1[len2:] = result_plot1[len2:] - result_plot1[len2] + result_bm[len2]
    # after
    result_plot2 = np.cumsum(after['result']) - after['result'][0]
    result_plot2[len1:] = result_plot2[len1:] - result_plot2[len1] + result_bm[len1]
    result_plot2[len2:] = result_plot2[len2:] - result_plot2[len2] + result_bm[len2]
    # prob after
    before['result_prob'] = 0.5 * bm_arr + 0.3 * (before['prob'] > 0.6) * bm_arr - 0.3 * (before['prob'] < 0.4) * bm_arr
    after['result_prob'] = 0.5 * bm_arr + 0.3 * (after['prob'] > 0.6) * bm_arr - 0.3 * (after['prob'] < 0.4) * bm_arr
    result_plot3 = np.cumsum(after['result_prob']) - after['result_prob'][0]
    result_plot3[len1:] = result_plot3[len1:] - result_plot3[len1] + result_bm[len1]
    result_plot3[len2:] = result_plot3[len2:] - result_plot3[len2] + result_bm[len2]
    # ls
    before['result_ls'] = 0.5 * (before['prob'] > 0.5) * bm_arr - 0.5 * (before['prob'] < 0.5) * bm_arr
    after['result_ls'] = 0.5 * (after['prob'] > 0.5) * bm_arr - 0.5 * (after['prob'] < 0.5) * bm_arr
    result_plot3 = np.cumsum(after['result_prob']) - after['result_prob'][0]
    result_plot3[len1:] = result_plot3[len1:] - result_plot3[len1] + result_bm[len1]
    result_plot3[len2:] = result_plot3[len2:] - result_plot3[len2] + result_bm[len2]
    #

    fig = plt.figure()
    plt.plot(np.arange(len(result_bm)), result_bm, 'b')
    plt.plot(np.arange(len(result_bm)), result_plot1, 'k')
    plt.plot(np.arange(len(result_bm)), result_plot2, 'r')
    plt.plot(np.arange(len(result_bm)), result_plot3, 'g')
    plt.axvline(x=len1)
    plt.axvline(x=len2)
    fig.legend(['bm', 'before', 'after', 'prob_after'])
    fig.savefig('./out/maml/{}.png'.format(ep))
    plt.close(fig)

    # prob 적용
    result_plot1 = np.cumsum(before['prob']) - before['result'][0]
    result_plot1[len1:] = result_plot1[len1:] - result_plot1[len1] + result_bm[len1]
    result_plot1[len2:] = result_plot1[len2:] - result_plot1[len2] + result_bm[len2]
    result_plot2 = np.cumsum(after['result']) - after['result'][0]
    result_plot2[len1:] = result_plot2[len1:] - result_plot2[len1] + result_bm[len1]
    result_plot2[len2:] = result_plot2[len2:] - result_plot2[len2] + result_bm[len2]

    return before, after, bm


# many tasks
def train_maml2(model, criterion, optimizer, scheduler, dataloaders, lr_inner=0.01, num_epochs=25):
    # num_epochs=25;phase='train';lr_inner=0.01
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_eval_loss = 9999
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if epoch % 20 == 0:
            model.eval()
            dataloader_test = dataloaders['test']
            plot_model(model, dataloader_test, criterion, ep=epoch, lr_inner=lr_inner)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'eval']:
            if phase == 'train':
                scheduler.step()
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            x_spt, y_spt, x_qry, y_qry, logy_spt, logy_qry = dataloaders[phase]

            if phase == 'train':
                n_tasks = 20
                tasks_idxs = np.random.choice(len(x_spt), n_tasks, replace=True)
            else:
                n_tasks = 100
                tasks_idxs = np.arange(n_tasks)

            task_losses = None
            for i, task_i in enumerate(tasks_idxs):
                # print(i)
                x_spt_i, y_spt_i, x_qry_i, y_qry_i, logy_spt_i, logy_qry_i = x_spt[task_i], y_spt[task_i], x_qry[task_i], y_qry[task_i], logy_spt[task_i], logy_qry[task_i]
                # print('after balancing: {}'.format(torch.sum(labels_balanced, axis=0)))
                x_spt_i = x_spt_i.to(device)
                y_spt_i = y_spt_i.long().to(device)
                logy_spt_i = logy_spt_i.to(device)
                x_qry_i = x_qry_i.to(device)
                y_qry_i = y_qry_i.long().to(device)
                logy_qry_i = logy_qry_i.to(device)

                # # 매개변수 경사도를 0으로 설정
                # optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                outputs_spt = model.forward(x_spt_i)
                # outputs = F.softmax(outputs)
                _, preds = torch.max(outputs_spt, 1)
                train_loss = criterion(outputs_spt, y_spt_i) * torch.exp(logy_spt_i)
                train_loss = torch.mean(train_loss)
                # Step 6
                grad = torch.autograd.grad(train_loss, model.parameters(), retain_graph=True, create_graph=True)

                fast_weights = list(map(lambda p: p[1] - lr_inner * p[0], zip(grad, model.parameters())))

                with torch.set_grad_enabled(phase == 'train'):
                    outputs_qry = model.forward(x_qry_i, fast_weights)  # run forward pass to initialize weights
                    test_loss = criterion(outputs_qry, y_qry_i) * torch.exp(logy_qry_i)
                    if task_losses is None:
                        task_losses = test_loss
                    else:
                        task_losses += test_loss

                with torch.no_grad():
                    # 통계
                    running_loss += test_loss.item() * x_qry_i.size(0)
                    # running_corrects += torch.sum(preds == labels.data)
                    running_corrects += torch.sum(torch.max(outputs_qry, 1)[1] == y_qry_i.data)

            # 학습 단계인 경우 역전파 + 최적화
            if phase == 'train':
                total_losses = task_losses / n_tasks
                optimizer.zero_grad()
                total_losses.backward()
                optimizer.step()

            epoch_loss = running_loss / n_tasks
            epoch_acc = running_corrects.double() / n_tasks

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'eval':
                if epoch_loss < min_eval_loss:
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    model.load_state_dict(best_model_wts)

            # 모델을 깊은 복사(deep copy)함
            if phase == 'eval' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)

    return model


def train_maml3(configs, model, criterion, optimizer, scheduler, dataloaders, base_i, use_maml=True):
    # num_epochs=25;phase='train';lr_inner=0.01;base_i=2000; use_maml=True;ep=0
    since = time.time()

    c = configs
    device = c.device
    num_epochs = c.num_epochs
    lr_inner = c.lr_inner
    early_stopping = c.early_stopping

    test_result = {'before': [], 'after': [], 'bm': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_eval_loss = 9999
    stop_count = 0
    for ep in range(num_epochs):
        print('Epoch {}/{}'.format(ep, num_epochs - 1))
        print('-' * 10)

        if ep % c.print_step == 0:
            model.eval()
            _ = plot_model2(configs, model, dataloaders, criterion, ep=base_i * num_epochs + ep, use_maml=use_maml)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'eval']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            weights_per_task = []
            task_losses = None
            for i_task in dataloaders[phase]:
                # print(i)
                x_spt_i, y_spt_i, x_qry_i, y_qry_i, logy_spt_i, logy_qry_i = i_task
                # print('after balancing: {}'.format(torch.sum(labels_balanced, axis=0)))
                x_spt_i = x_spt_i.to(device)
                y_spt_i = y_spt_i.long().to(device)
                logy_spt_i = logy_spt_i.to(device)
                x_qry_i = x_qry_i.to(device)
                y_qry_i = y_qry_i.long().to(device)
                logy_qry_i = logy_qry_i.to(device)

                # # 매개변수 경사도를 0으로 설정
                # optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                outputs_spt = model.forward(x_spt_i)
                # outputs = F.softmax(outputs)
                _, preds = torch.max(outputs_spt, 1)
                train_loss = criterion(outputs_spt, y_spt_i) # * torch.exp(logy_spt_i) / torch.sum(torch.exp(logy_spt_i)) # TODO: real return
                train_loss = torch.sum(train_loss) / len(train_loss)

                if use_maml:
                    # Step 6
                    grad = torch.autograd.grad(train_loss, model.parameters(), retain_graph=True, create_graph=True)

                    fast_weights = list(map(lambda p: p[1] - lr_inner * p[0], zip(grad, model.parameters())))

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs_qry = model.forward(x_qry_i, fast_weights)  # run forward pass to initialize weights
                        test_loss = criterion(outputs_qry, y_qry_i) # * torch.exp(logy_qry_i) # TODO: real return
                        weights_per_task.append(torch.exp(logy_qry_i).cpu().numpy())
                        if task_losses is None:
                            task_losses = test_loss
                        else:
                            task_losses += test_loss

                else:
                    outputs_qry = model.forward(x_qry_i)  # run forward pass to initialize weights
                    weights_per_task.append(torch.exp(logy_qry_i).cpu().numpy())
                    test_loss = criterion(outputs_qry, y_qry_i)
                    if task_losses is None:
                        task_losses = test_loss
                    else:
                        task_losses += test_loss

                with torch.no_grad():
                    # 통계
                    running_loss += test_loss.item() * x_qry_i.size(0)
                    # running_corrects += torch.sum(preds == labels.data)
                    running_corrects += torch.sum(torch.max(outputs_qry, 1)[1] == y_qry_i.data)

            # 학습 단계인 경우 역전파 + 최적화
            if phase == 'train':
                total_losses = task_losses / len(weights_per_task) # / np.sum(weights_per_task) # TODO: real return
                optimizer.zero_grad()
                total_losses.backward()
                optimizer.step()

            epoch_loss = running_loss / len(weights_per_task)  # / np.sum(weights_per_task) # TODO: real return
            epoch_acc = running_corrects.double() / len(dataloaders[phase])


            if phase == 'eval':
                if epoch_loss < min_eval_loss:
                    min_eval_loss = epoch_loss
                    stop_count = 0
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    stop_count += 1
                    # model.load_state_dict(best_model_wts)

            print('{} Loss: {:.4f} Acc: {:.4f} count: {}/{}'.format(
                phase, epoch_loss, epoch_acc, stop_count, early_stopping))

            # 모델을 깊은 복사(deep copy)함
            if phase == 'eval' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if stop_count >= early_stopping:
            print('early stopped')
            model.load_state_dict(best_model_wts)
            before, after, bm = plot_model2(configs, model, dataloaders, criterion, ep=base_i * num_epochs + ep + 1, use_maml=use_maml)
            test_result['before'].append(before['result'][-len(dataloaders['test']):])
            test_result['after'].append(after['result'][-len(dataloaders['test']):])
            test_result['bm'].append(bm[-len(dataloaders['test']):])

            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)

    return model, test_result


class ImageModel(nn.Module):
    def __init__(self, output_size, base_model='resnet50', configs=None):
        super().__init__()
        if base_model == 'resnet50':
            base_model = models.resnet50(pretrained=True)

            # 마지막 fc 삭제
            num_ftrs = base_model.fc.in_features
            self.base_model = nn.Sequential(*list(base_model.children())[:-1])
            for param in self.base_model.parameters():
                param.requires_grad = False

        # 새로 생성된 모듈의 매개변수는 기본값이 requires_grad=True 임
        self.vars = nn.ParameterList()
        if configs is None:
            self.structure = [('linear', [64, num_ftrs]),
                              ('relu', []),
                              ('dropout', [0.5]),
                              ('layer_norm', [64]),
                              ('linear', [64, 64]),
                              ('relu', []),
                              ('dropout', [0.5]),
                              ('layer_norm', [64]),
                              ('linear', [32, 64]),
                              ('relu', []),
                              ('dropout', [0.5]),
                              ('layer_norm', [32]),
                              ('linear', [output_size, 32])]
        else:
            self.structure = configs
        for name, param in self.structure:
            self.set_layer_wgt(name, param)

    def set_layer_wgt(self, type_='linear', param=[]):
        if len(param) == 0:
            return None

        if type_.lower() == 'linear':
            # [ch_out, ch_in]
            w = nn.Parameter(torch.ones(*param))
            b = nn.Parameter(torch.zeros(param[0]))
            # gain=1 according to cbfinn's implementation
            # torch.nn.init.kaiming_uniform_(w)
            torch.nn.init.xavier_uniform_(w)
            self.vars.append(w)
            self.vars.append(b)

    def forward(self, x, vars=None):

        x = self.base_model(x)
        with torch.no_grad():
            x = x.view(x.size(0), -1)  # flatten

        if vars is None:
            vars = self.vars

        idx = 0
        for name, param in self.structure:
            if name.lower() == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
            elif name.lower() == 'relu':
                x = F.relu(x)
            elif name.lower() == 'dropout':
                x = F.dropout(x, param[0], training=self.training)
            elif name.lower() == 'layer_norm':
                x = F.layer_norm(x, param)
            else:
                raise NotImplementedError

        assert idx == len(self.vars)

        return x

    def zero_grad(self, vars=None):
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


def main_maml2():

    # get dataset
    data_df = prepare_dataset()

    arr_y = np.array(data_df['mkt_rf'], dtype=np.float32)
    # arr_y = np.array(data_df['mom'], dtype=np.float32)

    classification = True
    if classification:
        output_size = 2
    else:
        output_size = 1

    # make dataloader
    dl_train, dl_eval, dl_test = make_dataloader_ntasks2(arr_y)
    dataloaders = {'train': dl_train, 'eval': dl_eval, 'test': dl_test}


    model = ImageModel(output_size, base_model='resnet50')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(reduce=False)

    # 이전과는 다르게 마지막 계층의 매개변수들만 최적화되는지 관찰
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 7 에폭마다 0.1씩 학습율 감소
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model_conv = train_maml2(model, criterion, optimizer, scheduler,
                             dataloaders, lr_inner=0.01, num_epochs=1000)


def main_maml3():
    configs = Configs()
    c = configs
    # get dataset
    data_df = prepare_dataset()

    arr_y = np.array(data_df['mkt_rf'], dtype=np.float32)
    arr_logp = np.log(np.cumprod(1 + arr_y) / (1 + arr_y[0]))
    # arr_y = np.array(data_df['mom'], dtype=np.float32)

    classification = True
    if classification:
        output_size = 2
    else:
        output_size = 1

    result_plot = dict()
    results_all = {'before': [], 'after': [], 'bm': []}

    model = ImageModel(output_size, base_model='resnet50')

    model = model.to(c.device)

    criterion = nn.CrossEntropyLoss(reduce=False)

    # 이전과는 다르게 마지막 계층의 매개변수들만 최적화되는지 관찰
    optimizer = optim.Adam(model.parameters(), lr=c.lr_meta)

    begin_i = 2000
    for i, t in enumerate(range(begin_i, len(arr_logp) - c.test_period, c.test_period)):
        # i = 0; t = begin_i
        train_tasks = np.random.choice(np.arange(t - c.lookback_period, t - c.k_days * (c.n_tasks + 2), c.k_days), c.n_tasks)
        eval_tasks = np.arange(t - c.k_days * (c.n_tasks + 2), t - c.k_days * 2, c.k_days)
        test_tasks = np.arange(t, t + c.test_period, c.k_days)
        dl_train, dl_eval, dl_test = [], [], []
        len_arr_per_task = c.m_days + c.lookback_period  # 500: for image, 250: for lookback period
        print('preparing train set...')
        for j in train_tasks:
            arr_train = arr_logp[(j-len_arr_per_task):(j+1)]
            dl_train.append(make_onetask(arr_train, c))
        print('preparing train set...done')
        print('preparing eval set...')
        for j in eval_tasks:
            arr_eval = arr_logp[(j-len_arr_per_task):(j+1)]
            dl_eval.append(make_onetask(arr_eval, c))
        print('preparing eval set...done')
        print('preparing test set...')
        for j in test_tasks:
            arr_test = arr_logp[(j-len_arr_per_task):(j + 1)]
            dl_test.append(make_onetask(arr_test, c))
        print('preparing test set...done')

        dataloaders = {'train': dl_train, 'eval': dl_eval, 'test': dl_test}
        # 7 에폭마다 0.1씩 학습율 감소
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        scheduler = None
        model_conv, test_result = train_maml3(c, model, criterion, optimizer, scheduler, dataloaders, i, use_maml=False)

        for key in ['before', 'after', 'bm']:
            results_all[key] += test_result[key]

    for key in ['before', 'after', 'bm']:
        result_plot[key] = np.cumsum(np.concatenate(results_all[key])) - np.concatenate(results_all[key])[0]

    fig = plt.figure()
    ax1 = fig.subplots()
    ax1.plot(np.arange(len(result_plot['bm'])), result_plot['bm'], 'b')
    ax1.plot(np.arange(len(result_plot['bm'])), result_plot['before'], 'k')
    ax1.plot(np.arange(len(result_plot['bm'])), result_plot['after'], 'r')
    ax1.legend(['bm', 'before', 'after'])

    ax1_1 = ax1.twinx()
    ax1_1.plot(np.arange(len(result_plot['bm'])), result_plot['after'] - result_plot['bm'])

# dataset example
# def ex_dataloader():
#     my_x = [np.array([[1.0, 2], [3, 4]]), np.array([[5., 6], [7, 8]])]  # a list of numpy arrays
#     my_y = [np.array([4.]), np.array([2.])]  # another list of numpy arrays (targets)
#
#     tensor_x = torch.stack([torch.Tensor(i) for i in my_x])  # transform to torch tensors
#     tensor_y = torch.stack([torch.Tensor(i) for i in my_y])
#
#     my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
#     my_dataloader = DataLoader(my_dataset)  # create your dataloader
#
#
# def arr_to_dataset(log_arr, sampling_freq=20):
#     m_days = 500
#     k_days = 20
#     delay_days = 1
#
#     arr_list = []
#     label_list = []
#     for i in range(m_days, len(log_arr) - (k_days + delay_days + 1), sampling_freq):
#         arr_list.append(log_arr[(i - m_days):(i + 1)])
#         label_list.append(log_arr[(i + delay_days):][k_days] - log_arr[(i + delay_days):][0])
#
#     return arr_list, label_list
#
#
# def preprocessing(arr, image_size=64, to_dataloader=True, balancing=True, batch_size=1, classification=True):
#     # Gramian Angular Field
#     gasf = GramianAngularField(image_size=image_size, method='summation')
#
#     data_, label_ = arr_to_dataset(arr, 5)
#     data_arr = np.stack(data_, axis=0)
#     label_arr = np.stack(label_, axis=0)
#
#     # original
#     logp_scaled = minmax_scaler(data_arr, axis=1)
#     fig1 = gasf.fit_transform(logp_scaled)
#
#     # smoothing
#     logp_smoothed = data_arr[:, 60:] - data_arr[:, :-60]
#     logp_smoothed_scaled = minmax_scaler(logp_smoothed, axis=1)
#     fig2 = gasf.fit_transform(logp_smoothed_scaled)
#
#     # downsampling
#     logp_downsampled = data_arr[:, ::5]
#     logp_downsampled_scaled = minmax_scaler(logp_downsampled, axis=1)
#     fig3 = gasf.fit_transform(logp_downsampled_scaled)
#     figs = np.stack([fig1, fig2, fig3], axis=1)
#
#     assert figs.shape[1:] == (3, image_size, image_size)
#
#
#     if classification:
#         # 고정값
#         cp = 0.02
#         cp_l, cp_h = -cp, cp
#         # 변동값 (class수량 맞추기)
#         # cp_l, cp_h = np.percentile(label_arr, q=[33, 66])
#         # pos / zero / neg
#         label_class = np.stack([label_arr > cp_h, (label_arr >= cp_l) & (label_arr <= cp_h), label_arr < cp_l], axis=-1) * 1
#         print(np.sum(label_class, axis=0))
#     else:
#         balancing = False
#         label_class = label_arr.reshape([-1, 1])
#
#     d, l = figs.astype(np.float32), label_class.astype(np.float32)
#
#     if balancing:
#         idx = np.arange(len(l))
#         n_per_class = np.sum(l, axis=0)
#         max_class = np.max(n_per_class)
#
#         random_sampled = []
#         for i in range(len(n_per_class)):
#             if len(idx[l[:, i] == 1]) == max_class:
#                 replace_ = False
#             else:
#                 replace_ = True
#
#             idx_c = np.random.choice(idx[l[:, i] == 1], int(max_class), replace=replace_)
#             random_sampled.append(idx_c)
#         idx_balanced = np.random.permutation(np.concatenate(random_sampled))
#         d = d[idx_balanced]
#         l = l[idx_balanced]
#         print('after balancing: {}'.format(np.sum(l, axis=0)))
#
#     if to_dataloader:
#         transformer = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor()])
#
#         tensor_x = torch.from_numpy(d)  # transform to torch tensors
#         tensor_y = torch.from_numpy(l)
#
#         # resize image_size to 224 for resnet
#         tensor_x = torch.stack([transformer(tx) for tx in tensor_x])
#
#         my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
#         my_dataloader = DataLoader(my_dataset, shuffle=True, batch_size=batch_size, num_workers=0)  # create your dataloader
#
#         return my_dataloader, len(tensor_x)
#     else:
#         return (d, l)
#
#
# def make_dataloader(arr_y, split_r=[0.4, 0.8], batch_size=32, classification=True):
#     # arr_y = np.array(data_df['kospi'], dtype=np.float32);split_r=[0.6, 0.8]; batch_size=32
#     arr_logp = np.log(np.cumprod(1 + arr_y) / (1 + arr_y[0]))
#
#     train_arr = arr_logp[:int(len(arr_logp) * split_r[0])]
#     valid_arr = arr_logp[int(len(arr_logp) * split_r[0]):int(len(arr_logp) * split_r[1])]
#     test_arr = arr_logp[int(len(arr_logp) * split_r[1]):]
#
#     image_size = 64
#     dataloader_train, len_train = preprocessing(train_arr, image_size=image_size, to_dataloader=True, batch_size=batch_size, classification=classification)
#     dataloader_valid, len_valid = preprocessing(valid_arr, image_size=image_size, to_dataloader=True, classification=classification)
#     dataloader_test, len_test = preprocessing(test_arr, image_size=image_size, balancing=False, to_dataloader=True, classification=classification)
#
#     return dataloader_train, dataloader_valid, dataloader_test, [len_train, len_valid, len_test]
#
#
# def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
#     # model = model_ft; optimizer = optimizer_ft;num_epochs=25;phase='train'
#     since = time.time()
#
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)
#
#         # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
#         for phase in ['train', 'valid']:
#             if phase == 'train':
#                 scheduler.step()
#                 model.train()  # 모델을 학습 모드로 설정
#             else:
#                 model.eval()   # 모델을 평가 모드로 설정
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             # 데이터를 반복
#             for inputs, labels in dataloaders[phase]:
#                 # inputs, labels = next(iter(dataloaders[phase]))
#
#                 # print('after balancing: {}'.format(torch.sum(labels_balanced, axis=0)))
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#                 # 매개변수 경사도를 0으로 설정
#                 optimizer.zero_grad()
#
#                 # 순전파
#                 # 학습 시에만 연산 기록을 추적
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     outputs = F.softmax(outputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)
#
#                     # 학습 단계인 경우 역전파 + 최적화
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#
#                 # 통계
#                 running_loss += loss.item() * inputs.size(0)
#                 # running_corrects += torch.sum(preds == labels.data)
#                 running_corrects += torch.sum(preds == torch.max(labels, 1)[1])
#
#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]
#
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))
#
#             # 모델을 깊은 복사(deep copy)함
#             if phase == 'valid' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#
#         print()
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))
#
#
#     # 가장 나은 모델 가중치를 불러옴
#     model.load_state_dict(best_model_wts)
#
#     phase = 'test'
#     if phase == 'test':
#         model.eval()  # 모델을 평가 모드로 설정
#
#         running_loss = 0.0
#         running_corrects = 0
#
#         # 데이터를 반복
#         for inputs, labels in dataloaders[phase]:
#             # inputs, labels = next(iter(dataloaders[phase]))
#
#             # print('after balancing: {}'.format(torch.sum(labels_balanced, axis=0)))
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             # 매개변수 경사도를 0으로 설정
#             optimizer.zero_grad()
#
#             # 순전파
#             # 학습 시에만 연산 기록을 추적
#             with torch.no_grad():
#                 outputs = model(inputs)
#                 outputs = F.softmax(outputs)
#                 _, preds = torch.max(outputs, 1)
#                 loss = criterion(outputs, labels)
#
#             # 통계
#             running_loss += loss.item() * inputs.size(0)
#             # running_corrects += torch.sum(preds == labels.data)
#             running_corrects += torch.sum(preds == torch.max(labels, 1)[1])
#
#         epoch_loss = running_loss / dataset_sizes[phase]
#         epoch_acc = running_corrects.double() / dataset_sizes[phase]
#         print('test= {} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
#
#     return model
#
#
# # one task
# def train_maml(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, lr_inner=0.01, num_epochs=25):
#     # num_epochs=25;phase='train';lr_inner=0.01
#     since = time.time()
#
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)
#
#         # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
#         for phase in ['train', 'valid']:
#             if phase == 'train':
#                 scheduler.step()
#                 model.train()  # 모델을 학습 모드로 설정
#             else:
#                 model.eval()   # 모델을 평가 모드로 설정
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             # 데이터를 반복
#             for inputs, labels in dataloaders[phase]:
#                 # inputs, labels = next(iter(dataloaders[phase]))
#
#                 # print('after balancing: {}'.format(torch.sum(labels_balanced, axis=0)))
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#                 # # 매개변수 경사도를 0으로 설정
#                 # optimizer.zero_grad()
#
#                 # 순전파
#                 # 학습 시에만 연산 기록을 추적
#                 outputs = model.forward(inputs)
#                 # outputs = F.softmax(outputs)
#                 _, preds = torch.max(outputs, 1)
#                 train_loss = criterion(outputs, labels)
#
#                 # Step 6
#                 grad = torch.autograd.grad(train_loss, model.parameters(), create_graph=True)
#
#                 fast_weights = list(map(lambda p: p[1] - lr_inner * p[0], zip(grad, model.parameters())))
#
#                 with torch.set_grad_enabled(phase == 'train'):
#                     logits = model.forward(inputs, fast_weights)  # run forward pass to initialize weights
#                     test_loss = criterion(logits, labels)
#
#                     # 학습 단계인 경우 역전파 + 최적화
#                     if phase == 'train':
#                         optimizer.zero_grad()
#                         test_loss.backward()
#                         optimizer.step()
#
#                 with torch.no_grad():
#                     # 통계
#                     running_loss += test_loss.item() * inputs.size(0)
#                     # running_corrects += torch.sum(preds == labels.data)
#                     running_corrects += torch.sum(preds == torch.max(labels, 1)[1])
#
#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]
#
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))
#
#             # 모델을 깊은 복사(deep copy)함
#             if phase == 'valid' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#
#         print()
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))
#
#     # 가장 나은 모델 가중치를 불러옴
#     model.load_state_dict(best_model_wts)
#
#     phase = 'test'
#     if phase == 'test':
#         model.eval()  # 모델을 평가 모드로 설정
#
#         running_loss = 0.0
#         running_corrects = 0
#
#         # 데이터를 반복
#         for inputs, labels in dataloaders[phase]:
#             # inputs, labels = next(iter(dataloaders[phase]))
#
#             # print('after balancing: {}'.format(torch.sum(labels_balanced, axis=0)))
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             # 매개변수 경사도를 0으로 설정
#             optimizer.zero_grad()
#
#             # 순전파
#             # 학습 시에만 연산 기록을 추적
#             with torch.no_grad():
#                 outputs = model(inputs)
#                 # outputs = F.softmax(outputs)
#                 _, preds = torch.max(outputs, 1)
#                 loss = criterion(outputs, labels)
#
#             # 통계
#             running_loss += loss.item() * inputs.size(0)
#             # running_corrects += torch.sum(preds == labels.data)
#             running_corrects += torch.sum(preds == torch.max(labels, 1)[1])
#
#         epoch_loss = running_loss / dataset_sizes[phase]
#         epoch_acc = running_corrects.double() / dataset_sizes[phase]
#         print('test= {} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
#
#     return model
#
# def main():
#     # get dataset
#     data_df = prepare_dataset()
#
#     # arr_y = np.array(data_df['mkt_rf'], dtype=np.float32)
#     arr_y = np.array(data_df['mom'], dtype=np.float32)
#
#     classification = False
#     if classification:
#         output_size = 3
#     else:
#         output_size = 1
#
#     # make dataloader
#     dl_train, dl_valid, dl_test, ds_sizes = make_dataloader(arr_y, batch_size=32, classification=classification)
#     dataloaders = {'train': dl_train, 'valid': dl_valid, 'test': dl_test}
#     dataset_sizes = {'train': ds_sizes[0], 'valid': ds_sizes[1], 'test': ds_sizes[2]}
#
#     model_ft = models.resnet50(pretrained=True)
#     num_ftrs = model_ft.fc.in_features
#     model_ft.fc = nn.Linear(num_ftrs, output_size)
#     model_ft = model_ft.to(device)
#
#     criterion = nn.MSELoss()
#
#     optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
#     # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#     scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
#
#     model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler,
#                            dataloaders, dataset_sizes,
#                            num_epochs=25)
#
#     model_conv = models.resnet50(pretrained=True)
#     for param in model_conv.parameters():
#         param.requires_grad = False
#
#     # 새로 생성된 모듈의 매개변수는 기본값이 requires_grad=True 임
#     num_ftrs = model_conv.fc.in_features
#     model_conv.fc = nn.Linear(num_ftrs, output_size)
#
#     model_conv = model_conv.to(device)
#
#     criterion = nn.MSELoss()
#
#     # 이전과는 다르게 마지막 계층의 매개변수들만 최적화되는지 관찰
#     optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
#
#     # 7 에폭마다 0.1씩 학습율 감소
#     exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
#     model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,
#                              dataloaders, dataset_sizes, num_epochs=25)
#
#
# def main_maml():
#
#     # get dataset
#     data_df = prepare_dataset()
#
#     # arr_y = np.array(data_df['mkt_rf'], dtype=np.float32)
#     arr_y = np.array(data_df['mom'], dtype=np.float32)
#
#     classification = False
#     if classification:
#         output_size = 3
#     else:
#         output_size = 1
#
#     # make dataloader
#     dl_train, dl_valid, dl_test, ds_sizes = make_dataloader(arr_y, batch_size=32, classification=classification)
#     dataloaders = {'train': dl_train, 'valid': dl_valid, 'test': dl_test}
#     dataset_sizes = {'train': ds_sizes[0], 'valid': ds_sizes[1], 'test': ds_sizes[2]}
#
#
#     model = ImageModel(output_size, base_model='resnet50')
#     model = model.to(device)
#
#     criterion = nn.MSELoss()
#
#     # 이전과는 다르게 마지막 계층의 매개변수들만 최적화되는지 관찰
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     # 7 에폭마다 0.1씩 학습율 감소
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#     model_conv = train_maml(model, criterion, optimizer, scheduler,
#                              dataloaders, dataset_sizes, lr_inner=0.01, num_epochs=25)
