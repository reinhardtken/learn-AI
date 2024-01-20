import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 在 Python 中，axis 参数用于指定操作的轴。在二维数组中，axis=0 表示沿着列的方向进行操作，axis=1 表示沿着行的方向进行操作。

x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y = tf.reduce_mean(x, axis=0)
print(x)
print(y)


x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y = tf.reduce_mean(x, axis=1)
print(y)

#不指定就是全部平均
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y = tf.reduce_mean(x)
print(y)