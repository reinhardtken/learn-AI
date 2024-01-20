import tensorflow as tf
import numpy as np


v = tf.Variable(np.ones(shape=[4,3]), dtype='float32')
b = v * 3.0
print(type(b).__name__)

# 一维，长度是4，初始值是2和3
a = tf.constant(2, shape=[4], dtype='float32')
b = tf.constant(3, shape=[4], dtype='float32')
# 标准的加法
c = tf.add(a,b)
print(c)

# 这里操作的结果都是向量
# Defining two TensorFlow constants
a = tf.constant(4, shape=[4], dtype='float32')
b = tf.constant(2, shape=[4], dtype='float32')
print(a)
print(b)

# Arithmatic operations

c = a + b  # Addition
print(c)
d = a - b  # Subtraction
e = a * b  # Multiplication
f = a / b  # Division

# Logical operations

a = tf.constant([[1,2,3],[4,5,6]])
b = tf.constant([[5,4,3],[3,2,1]])

equal_check = (a == b) # Element-wise equality
print(equal_check)
leq_check = (a <= b) # Element-wise less than or equal