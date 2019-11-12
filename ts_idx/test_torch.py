
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os
import pandas as pd
from pyts.image import GramianAngularField
import time
import torch
from torch import nn, optim, utils

from torch.optim import lr_scheduler
from torchvision import models, datasets, transforms

# define device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# dataset example
def ex_dataloader():
    my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
    my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets)

    tensor_x = torch.stack([torch.Tensor(i) for i in my_x]) # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor(i) for i in my_y])

    my_dataset = utils.data.TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = utils.data.DataLoader(my_dataset) # create your dataloader


def prepare_dataset():
    data_path = './data/data_for_metarl.csv'
    data_df = pd.read_csv(data_path)
    data_df.set_index('eval_d', inplace=True)
    date_ = list(data_df.index)
    return data_df


def minmax_scaler(arr, axis=0):
    return (arr - np.min(arr, axis=axis, keepdims=True)) / (np.max(arr, axis=axis, keepdims=True) - np.min(arr, axis=axis, keepdims=True))


def arr_to_dataset(log_arr, sampling_freq=20):
    m_days = 500
    k_days = 20
    delay_days = 1

    arr_list = []
    label_list = []
    for i in range(m_days, len(log_arr) - (k_days + delay_days + 1), sampling_freq):
        arr_list.append(log_arr[(i - m_days):(i + 1)])
        label_list.append(log_arr[(i + delay_days):][k_days] - log_arr[(i + delay_days):][0])

    return arr_list, label_list


def preprocessing(arr, image_size=64, to_dataloader=True):
    # Gramian Angular Field
    gasf = GramianAngularField(image_size=image_size, method='summation')

    data_, label_ = arr_to_dataset(arr)
    data_arr = np.stack(data_, axis=0)
    label_arr = np.stack(label_, axis=0)

    # original
    logp_scaled = minmax_scaler(data_arr, axis=1)
    fig1 = gasf.fit_transform(logp_scaled)

    # smoothing
    logp_smoothed = data_arr[:, 60:] - data_arr[:, :-60]
    logp_smoothed_scaled = minmax_scaler(logp_smoothed, axis=1)
    fig2 = gasf.fit_transform(logp_smoothed_scaled)

    # downsampling
    logp_downsampled = data_arr[:, ::5]
    logp_downsampled_scaled = minmax_scaler(logp_downsampled, axis=1)
    fig3 = gasf.fit_transform(logp_downsampled_scaled)
    figs = np.stack([fig1, fig2, fig3], axis=-1)

    assert figs.shape[1:] == (image_size, image_size, 3)

    # 고정값
    # cp = 0.02
    # cp_l, cp_h = -cp, cp
    # 변동값 (class수량 맞추기)
    cp_l, cp_h = np.percentile(label_arr, q=[33, 66])
    # pos / zero / neg
    label_class = np.stack([label_arr > cp_h, (label_arr >= cp_l) & (label_arr <= cp_h), label_arr < cp_l], axis=-1) * 1
    d, l = figs.astype(np.float32), label_class.astype(np.float32)

    if to_dataloader:
        tensor_x = torch.from_numpy(d)  # transform to torch tensors
        tensor_y = torch.from_numpy(l)

        my_dataset = utils.data.TensorDataset(tensor_x, tensor_y)  # create your datset
        my_dataloader = utils.data.DataLoader(my_dataset, shuffle=True)  # create your dataloader

        return my_dataloader
    else:
        return (d, l)


def make_dataloader(arr_y, split_r=[0.6, 0.8]):
    # arr_y = np.array(data_df['kospi'], dtype=np.float32)
    arr_list = []
    arr_logp = np.log(np.cumprod(1 + arr_y) / (1 + arr_y[0]))

    train_arr = arr_logp[:int(len(arr_logp) * split_r[0])]
    valid_arr = arr_logp[int(len(arr_logp) * split_r[0]):int(len(arr_logp) * split_r[1])]
    test_arr = arr_logp[int(len(arr_logp) * split_r[1]):]

    image_size = 64
    dataloader_train = preprocessing(train_arr, image_size=image_size, to_dataloader=True)
    dataloader_valid = preprocessing(valid_arr, image_size=image_size, to_dataloader=True)
    dataloader_test = preprocessing(test_arr, image_size=image_size, to_dataloader=True)

    return dataloader_train, dataloader_valid, dataloader_test


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
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


def main():

    # get dataset
    data_df = prepare_dataset()
    arr_y = np.array(data_df['mkt_rf'], dtype=np.float32)

    # make dataloader
    dl_train, dl_valid, dl_test = make_dataloader(arr_y)
    dataloader = {'train': dl_train, 'valid': dl_valid}

    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 3)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)

