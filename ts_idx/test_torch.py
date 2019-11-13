
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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# dataset example
def ex_dataloader():
    my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
    my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets)

    tensor_x = torch.stack([torch.Tensor(i) for i in my_x]) # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor(i) for i in my_y])

    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = DataLoader(my_dataset) # create your dataloader


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


def preprocessing(arr, image_size=64, to_dataloader=True, balancing=True, batch_size=1):
    # Gramian Angular Field
    gasf = GramianAngularField(image_size=image_size, method='summation')

    data_, label_ = arr_to_dataset(arr, 5)
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
    figs = np.stack([fig1, fig2, fig3], axis=1)

    assert figs.shape[1:] == (3, image_size, image_size)

    # 고정값
    cp = 0.02
    cp_l, cp_h = -cp, cp
    # 변동값 (class수량 맞추기)
    # cp_l, cp_h = np.percentile(label_arr, q=[33, 66])
    # pos / zero / neg
    label_class = np.stack([label_arr > cp_h, (label_arr >= cp_l) & (label_arr <= cp_h), label_arr < cp_l], axis=-1) * 1
    print(np.sum(label_class, axis=0))
    d, l = figs.astype(np.float32), label_class.astype(np.float32)

    if balancing:
        idx = np.arange(len(l))
        n_per_class = np.sum(l, axis=0)
        max_class = np.max(n_per_class)

        random_sampled = []
        for i in range(len(n_per_class)):
            if len(idx[l[:, i] == 1]) == max_class:
                replace_ = False
            else:
                replace_ = True

            idx_c = np.random.choice(idx[l[:, i] == 1], int(max_class), replace=replace_)
            random_sampled.append(idx_c)
        idx_balanced = np.random.permutation(np.concatenate(random_sampled))
        d = d[idx_balanced]
        l = l[idx_balanced]
        print('after balancing: {}'.format(np.sum(l, axis=0)))

    if to_dataloader:
        transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()])

        tensor_x = torch.from_numpy(d)  # transform to torch tensors
        tensor_y = torch.from_numpy(l)

        # resize image_size to 224 for resnet
        tensor_x = torch.stack([transformer(tx) for tx in tensor_x])

        my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
        my_dataloader = DataLoader(my_dataset, shuffle=True, batch_size=batch_size, num_workers=0)  # create your dataloader

        return my_dataloader, len(tensor_x)
    else:
        return (d, l)


def make_dataloader(arr_y, split_r=[0.6, 0.8], batch_size=32):
    # arr_y = np.array(data_df['kospi'], dtype=np.float32);split_r=[0.6, 0.8]; batch_size=32
    arr_list = []
    arr_logp = np.log(np.cumprod(1 + arr_y) / (1 + arr_y[0]))

    train_arr = arr_logp[:int(len(arr_logp) * split_r[0])]
    valid_arr = arr_logp[int(len(arr_logp) * split_r[0]):int(len(arr_logp) * split_r[1])]
    test_arr = arr_logp[int(len(arr_logp) * split_r[1]):]

    image_size = 64
    dataloader_train, len_train = preprocessing(train_arr, image_size=image_size, to_dataloader=True, batch_size=batch_size)
    dataloader_valid, len_valid = preprocessing(valid_arr, image_size=image_size, to_dataloader=True)
    dataloader_test, len_test = preprocessing(test_arr, image_size=image_size, balancing=False, to_dataloader=True)

    return dataloader_train, dataloader_valid, dataloader_test, [len_train, len_valid, len_test]


def np_to_tensor(list_of_numpy_objs):
    return (torch.from_numpy(np.array(obj, dtype=np.float32)) for obj in list_of_numpy_objs)


def compute_loss(model, x, y, loss_fn=nn.MSELoss()):
    logits = model.forward(x)
    mse = loss_fn(y, logits)
    return mse, logits


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    # model = model_ft; optimizer = optimizer_ft;num_epochs=25;phase='train'
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            for inputs, labels in dataloaders[phase]:
                # inputs, labels = next(iter(dataloaders[phase]))

                # print('after balancing: {}'.format(torch.sum(labels_balanced, axis=0)))
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = F.softmax(outputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)
                running_corrects += torch.sum(preds == torch.max(labels, 1)[1])

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 모델을 깊은 복사(deep copy)함
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)

    phase = 'test'
    if phase == 'test':
        model.eval()  # 모델을 평가 모드로 설정

        running_loss = 0.0
        running_corrects = 0

        # 데이터를 반복
        for inputs, labels in dataloaders[phase]:
            # inputs, labels = next(iter(dataloaders[phase]))

            # print('after balancing: {}'.format(torch.sum(labels_balanced, axis=0)))
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 매개변수 경사도를 0으로 설정
            optimizer.zero_grad()

            # 순전파
            # 학습 시에만 연산 기록을 추적
            with torch.no_grad():
                outputs = model(inputs)
                outputs = F.softmax(outputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # 통계
            running_loss += loss.item() * inputs.size(0)
            # running_corrects += torch.sum(preds == labels.data)
            running_corrects += torch.sum(preds == torch.max(labels, 1)[1])

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        print('test= {} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    return model


def train_maml(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, lr_inner=0.01, num_epochs=25):
    # num_epochs=25;phase='train';lr_inner=0.01
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            for inputs, labels in dataloaders[phase]:
                # inputs, labels = next(iter(dataloaders[phase]))

                # print('after balancing: {}'.format(torch.sum(labels_balanced, axis=0)))
                inputs = inputs.to(device)
                labels = labels.to(device)

                # # 매개변수 경사도를 0으로 설정
                # optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                outputs = model.forward(inputs)
                outputs = F.softmax(outputs)
                _, preds = torch.max(outputs, 1)
                train_loss = criterion(outputs, labels)

                # Step 6
                grad = torch.autograd.grad(train_loss, model.parameters(), create_graph=True)

                fast_weights = list(map(lambda p: p[1] - lr_inner * p[0], zip(grad, model.parameters())))

                with torch.set_grad_enabled(phase == 'train'):
                    logits = model.forward(inputs, fast_weights)  # run forward pass to initialize weights
                    test_loss = criterion(logits, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        optimizer.zero_grad()
                        test_loss.backward()
                        optimizer.step()

                with torch.no_grad():
                    # 통계
                    running_loss += test_loss.item() * inputs.size(0)
                    # running_corrects += torch.sum(preds == labels.data)
                    running_corrects += torch.sum(preds == torch.max(labels, 1)[1])

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 모델을 깊은 복사(deep copy)함
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)

    phase = 'test'
    if phase == 'test':
        model.eval()  # 모델을 평가 모드로 설정

        running_loss = 0.0
        running_corrects = 0

        # 데이터를 반복
        for inputs, labels in dataloaders[phase]:
            # inputs, labels = next(iter(dataloaders[phase]))

            # print('after balancing: {}'.format(torch.sum(labels_balanced, axis=0)))
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 매개변수 경사도를 0으로 설정
            optimizer.zero_grad()

            # 순전파
            # 학습 시에만 연산 기록을 추적
            with torch.no_grad():
                outputs = model(inputs)
                outputs = F.softmax(outputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # 통계
            running_loss += loss.item() * inputs.size(0)
            # running_corrects += torch.sum(preds == labels.data)
            running_corrects += torch.sum(preds == torch.max(labels, 1)[1])

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        print('test= {} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    return model


class ImageModel(nn.Module):
    def __init__(self, base_model='resnet50', configs=None):
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
            self.structure = [('linear', [32, num_ftrs]),
                              ('relu', []),
                              ('linear', [16, 32]),
                              ('relu', []),
                              ('linear', [3, 16])]
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


def main():
    # get dataset
    data_df = prepare_dataset()

    # arr_y = np.array(data_df['mkt_rf'], dtype=np.float32)
    arr_y = np.array(data_df['mom'], dtype=np.float32)

    # make dataloader
    dl_train, dl_valid, dl_test, ds_sizes = make_dataloader(arr_y, batch_size=32)
    dataloaders = {'train': dl_train, 'valid': dl_valid, 'test': dl_test}
    dataset_sizes = {'train': ds_sizes[0], 'valid': ds_sizes[1], 'test': ds_sizes[2]}

    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 3)
    model_ft = model_ft.to(device)

    criterion = nn.MSELoss()

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler,
                           dataloaders, dataset_sizes,
                           num_epochs=25)

    model_conv = models.resnet50(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # 새로 생성된 모듈의 매개변수는 기본값이 requires_grad=True 임
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 3)

    model_conv = model_conv.to(device)

    criterion = nn.MSELoss()

    # 이전과는 다르게 마지막 계층의 매개변수들만 최적화되는지 관찰
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # 7 에폭마다 0.1씩 학습율 감소
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,
                             dataloaders, dataset_sizes, num_epochs=25)



def main_maml():

    # get dataset
    data_df = prepare_dataset()

    # arr_y = np.array(data_df['mkt_rf'], dtype=np.float32)
    arr_y = np.array(data_df['mom'], dtype=np.float32)

    # make dataloader
    dl_train, dl_valid, dl_test, ds_sizes = make_dataloader(arr_y, batch_size=32)
    dataloaders = {'train': dl_train, 'valid': dl_valid, 'test': dl_test}
    dataset_sizes = {'train': ds_sizes[0], 'valid': ds_sizes[1], 'test': ds_sizes[2]}



    model = ImageModel(base_model='resnet50')
    model = model.to(device)

    criterion = nn.MSELoss()

    # 이전과는 다르게 마지막 계층의 매개변수들만 최적화되는지 관찰
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 7 에폭마다 0.1씩 학습율 감소
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model_conv = train_maml(model, criterion, optimizer, scheduler,
                             dataloaders, dataset_sizes, lr_inner=0.01, num_epochs=25)


