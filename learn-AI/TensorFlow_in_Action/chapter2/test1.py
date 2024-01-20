# Section 2.1
# Code listing 2.1

import numpy as np
import tensorflow as tf

@tf.function
def forward(x, W, b, act):
    """ Encapsulates the computations of a single layer in a multilayer perceptron """
    return act(tf.matmul(x,W)+b)

# Input (numpy array)
# np.random.normal() 函数用于生成服从正态分布的随机数
x = np.random.normal(size=[1,4]).astype('float32')
print(x)
# Variable initializer
# 在 TensorFlow 的 Keras 模块中，tf.keras.initializers.RandomNormal 函数用于创建一个初始化器，该初始化器用从正态分布中随机采样的值来初始化模型权重。
#
# 主要参数：
#
# mean（float，默认值：0）： 正态分布的均值。
# stddev（float，默认值：0.05）： 正态分布的标准差。
# seed（int，可选）： 用于随机数生成的种子，用于控制随机性的可重复性
init = tf.keras.initializers.RandomNormal()

# Defining layer 1 variables
w1 = tf.Variable(init(shape=[4,3]))
b1 = tf.Variable(init(shape=[1,3]))

# Defining layer 2 variables
w2 = tf.Variable(init(shape=[3,2]))
b2 = tf.Variable(init(shape=[1,2]))

# Computing h
h = forward(x, w1, b1, tf.nn.sigmoid)

# Computing y
y = forward(h, w2, b2, tf.nn.softmax)

print(y)