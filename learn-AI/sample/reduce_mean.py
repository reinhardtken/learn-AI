import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# axis=0，的意思是x轴每个元素的其他维度求和平均
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y = tf.reduce_mean(x, axis=0)
print(x)
print(y)

# axis=y，的意思是y轴每个元素的其他维度求和平均
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y = tf.reduce_mean(x, axis=1)
print(y)

#不指定就是全部平均
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y = tf.reduce_mean(x)
print(y)