
from collections import OrderedDict

# Other dependencies
import random
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.nn import functional as F, init
from torch.autograd import Variable


np.random.seed(333)


print('Python version: ', sys.version)
print('Pytorch version: ', torch.__version__)

# device_name = tf.test.gpu_device_name()
if not torch.cuda.is_available():
  raise SystemError('GPU device not found')


class SinusoidGenerator():
    '''
        Sinusoid Generator.

        p(T) is continuous, where the amplitude varies within [0.1, 5.0]
        and the phase varies within [0, π].

        This abstraction is the basically the same defined at:
        https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0
    '''

    def __init__(self, K=10, amplitude=None, phase=None):
        '''
        Args:
            K: batch size. Number of values sampled at every batch.
            amplitude: Sine wave amplitude. If None is uniformly sampled from
                the [0.1, 5.0] interval.
            pahse: Sine wave phase. If None is uniformly sampled from the [0, π]
                interval.
        '''
        self.K = K
        self.amplitude = amplitude if amplitude else np.random.uniform(0.1, 5.0)
        self.phase = phase if amplitude else np.random.uniform(0, np.pi)
        self.sampled_points = None
        self.x = self._sample_x()

    def _sample_x(self):
        return np.random.uniform(-5, 5, self.K)

    def f(self, x):
        '''Sinewave function.'''
        return self.amplitude * np.sin(x - self.phase)

    def batch(self, x=None, force_new=False):
        '''Returns a batch of size K.

        It also changes the sape of `x` to add a batch dimension to it.

        Args:
            x: Batch data, if given `y` is generated based on this data.
                Usually it is None. If None `self.x` is used.
            force_new: Instead of using `x` argument the batch data is
                uniformly sampled.

        '''
        if x is None:
            if force_new:
                x = self._sample_x()
            else:
                x = self.x
        y = self.f(x)
        return x[:, None], y[:, None]

    def equally_spaced_samples(self, K=None):
        '''Returns `K` equally spaced samples.'''
        if K is None:
            K = self.K
        return self.batch(x=np.linspace(-5, 5, K))


def plot(data, *args, **kwargs):
    '''Plot helper.'''
    x, y = data
    return plt.plot(x, y, *args, **kwargs)


def generate_dataset(K, train_size=20000, test_size=10):
    '''Generate train and test dataset.

    A dataset is composed of SinusoidGenerators that are able to provide
    a batch (`K`) elements at a time.
    '''

    def _generate_dataset(size):
        return [SinusoidGenerator(K=K) for _ in range(size)]

    return _generate_dataset(train_size), _generate_dataset(test_size)


train_ds, test_ds = generate_dataset(K=10)



class SineModel(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        self.vars = nn.ParameterList()
        if configs is None:
            self.structure = [('linear', [40, 1]),
                              ('relu', []),
                              ('linear', [40, 40]),
                              ('relu', []),
                              ('linear', [1, 40])]
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


def np_to_tensor(list_of_numpy_objs):
    return (torch.from_numpy(np.array(obj, dtype=np.float32)) for obj in list_of_numpy_objs)


def compute_loss(model, x, y, loss_fn=nn.MSELoss()):
    logits = model.forward(x)
    mse = loss_fn(y, logits)
    return mse, logits

# def compute_loss(model, x, y, loss_fn=loss_function):
#     logits = model.forward(x)
#     mse = loss_fn(y, logits)
#     return mse, logits

# model = maml; dataset=train_ds; lr_inner=0.01; batch_size=1; log_steps=1000
def train_maml(model, epochs, dataset, lr_inner=0.01, batch_size=1, log_steps=1000):

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Step 2: instead of checking for convergence, we train for a number
    # of epochs
    for _ in range(epochs):
        total_loss = 0
        losses = []
        start = time.time()
        # Step 3 and 4
        for i, t in enumerate(random.sample(dataset, len(dataset))):
            # Step 8
            x, y = np_to_tensor(t.batch())
            logits = model.forward(x)  # run forward pass to initialize weights
            # Step 5
            train_loss = loss_fn(logits, y)

            # Step 6
            grad = torch.autograd.grad(train_loss, model.parameters(), create_graph=True)
            # grad = torch.autograd.grad(train_loss, model.parameters())

            fast_weights = list(map(lambda p: p[1] - lr_inner * p[0], zip(grad, model.parameters())))

            logits = model.forward(x, fast_weights)  # run forward pass to initialize weights
            test_loss = loss_fn(logits, y)

            # grad_test = torch.autograd.grad(test_loss, model.parameters())
            optimizer.zero_grad()
            test_loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Logs
                total_loss += test_loss
                loss = total_loss / (i + 1.0)
                losses.append(loss)

            if i % log_steps == 0 and i > 0:
                print('Step {}: loss = {}, Time to run {} steps = {}'.format(i, loss, log_steps, time.time() - start))
                start = time.time()
        plt.plot(losses)
        plt.show()

maml = SineModel()
train_maml(maml, 1, train_ds)

