import numpy as np
from collections import OrderedDict
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from absl import flags
FLAGS = flags.FLAGS

class MyNet(Model):
    def __init__(self, ):
        super().__init__()

        self.h = Dense(3, name='h', kernel_initializer='ones')
        self.o = Dense(1, name='o', kernel_initializer='ones')

    def call(self, inputs):
        x = inputs

        return self.o(self.h(x))

net = MyNet()
net(tf.zeros([1, 3]))
import re
layer_list = []
attr_list = []
for var in net.trainable_weights:
    nm_ = list(filter(lambda x: x != '', re.split('/|:\d', var.name)))
    if len(nm_) == 3:
        model_nm, layer_nm, attr_nm = nm_
    elif len(nm_) == 2:
        layer_nm, attr_nm = nm_

    layer_list.append(layer_nm)
    attr_list.append(attr_nm)

add_value = []
with tf.GradientTape(persistent=True) as tape2:
    with tf.GradientTape() as tape:
        var_lists = net.trainable_variables
        y = net(tf.convert_to_tensor(np.array([[1, 0, 0]]), dtype=tf.float32))
        loss = tf.square(y - 2)
    grad = tape.gradient(loss, var_lists)
    print('var:{}'.format(var_lists[0]))
    print('grad:{}'.format(grad[0]))
    print('add:{}'.format(var_lists[0] + grad[0]))
    new_var_lists = []
    for i in range(len(attr_list)):
        obj = net.get_layer(layer_list[i])
        print('before:{}'.format(getattr(obj, attr_list[i])))
        setattr(obj, attr_list[i], getattr(obj, attr_list[i]) + grad[i])
        print('after:{}'.format(getattr(obj, attr_list[i])))
        new_var_lists.append(getattr(obj, attr_list[i]))
        if i == 0:
            print('var:{}\nreal:{}'.format(var_lists[i], net.h.kernel))

    y2 = net(tf.convert_to_tensor(np.array([[1, 0, 0]]), dtype=tf.float32))
    grad2 = tape2.gradient(y2, var_lists)
    grad3 = tape2.gradient(y2, new_var_lists)
    print('var:{}'.format(var_lists[0]))
    print('grad2:{}'.format(grad2[0]))
    print('grad3:{}'.format(grad3[0]))
    print('add:{}'.format(var_lists[0] + grad2[0]))

    for i in range(len(attr_list)):
        obj = net.get_layer(layer_list[i])
        print('before:{}'.format(getattr(obj, attr_list[i])))
        setattr(obj, attr_list[i], getattr(obj, attr_list[i]) + grad2[i])
        print('after:{}'.format(getattr(obj, attr_list[i])))

        if i == 0:
            print('var:{}\nreal:{}'.format(var_lists[i], net.h.kernel))
#