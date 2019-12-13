
from collections import OrderedDict
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F



class Net(nn.Module):
    def __init__(self, dim_in=3, dim_h=4, dim_out=2):
        super(Net, self).__init__()
        self.in_layer = nn.Linear(dim_in, dim_h)
        self.out_layer = nn.Linear(dim_h, dim_out)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.in_layer(x))
        return self.out_layer(x)

    def compute(self, inputs, params):
        x = F.linear(inputs, weight=params[0], bias=params[1])
        x = F.relu(x)
        return F.linear(x, weight=params[2], bias=params[3])


n_batch = 5

dim_in = 3
dim_h = 4
dim_out = 2
net = Net(dim_in, dim_h, dim_out)
opt = optim.Adam(net.parameters())
loss_func = nn.MSELoss()

x = (torch.arange(dim_in * n_batch).float() / 10.).reshape([n_batch, dim_in])
y = (torch.arange(dim_out * n_batch).float() / 10.).reshape([n_batch, dim_out])

a = nn.ParameterList()
for p in net.parameters():
    a.append(p)


opt.zero_grad()
print('----- test 1 ----------------------'.format(a[0].grad))
print('before backward 1: {}'.format(a[0].grad))
pred = net(x)
loss = loss_func(pred, y)
print('before backward 2: {}'.format(a[0].grad))
loss.backward(retain_graph=True)
loss.backward()
print('after backward: {}'.format(a[0].grad))
opt.zero_grad()
print('after zero_grad: {}'.format(a[0].grad))




print('----- test 2 ----------------------'.format(a[0].grad))
opt.zero_grad()
print('before backward: {}'.format(a[0].grad))
pred = net.compute(x, a)
loss = loss_func(pred, y)
loss.backward(retain_graph=True)
print('after backward: {}'.format(a[0].grad))
opt.zero_grad()
print('zero grad: {}'.format(a[0].grad))
g = torch.autograd.grad(loss, a, create_graph=True)
fast_weights = list(map(lambda p: p[1] - 1 * p[0], zip(g, a)))

pred2 = net.compute(x, fast_weights)
loss2 = loss_func(pred2, y)
print('before 2backward: {}'.format(a[0].grad))
loss2.backward(retain_graph=True)
print('after 2backward : {}'.format(a[0].grad))
opt.step()
print('after step : {}'.format(a[0].grad))



print('----- test 3 ----------------------'.format(a[0].grad))

opt.zero_grad()
print('before backward: {}'.format(a[0].grad))
pred = net.compute(x, a)
loss = loss_func(pred, y)
loss.backward(retain_graph=True)
print('after backward: {}'.format(a[0].grad))
opt.zero_grad()
print('zero grad: {}'.format(a[0].grad))
g = torch.autograd.grad(loss, a, create_graph=True)
fast_weights = list(map(lambda p: p[1] - 1 * p[0], zip(g, a)))

pred2 = net.compute(x, fast_weights)
loss2 = loss_func(pred2, y)
torch.autograd.grad(loss2, fast_weights, retain_graph=True)
print('before 2backward: {}'.format(a[0].grad))
loss2.backward(retain_graph=True)
print('after 2backward : {}'.format(a[0].grad))
opt.step()
print('after step : {}'.format(a[0].grad))


hooks = []

for param, grad_ in zip(net.parameters(), g) :
    hooks.append(param.register_hook(replace_grad(grad_)))

model.train()
optimiser.zero_grad()
# Dummy pass in order to create `loss` variable
# Replace dummy gradients with mean task gradients using hooks
logits = model(torch.zeros((k_way,) + data_shape).to(device, dtype=torch.double))
loss = loss_fn(logits, create_nshot_task_label(k_way, 1).to(device))
loss.backward()
optimiser.step()

for h in hooks:
    h.remove()

a2 = OrderedDict(net.named_parameters())