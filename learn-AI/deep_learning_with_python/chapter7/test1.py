from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

from tensorflow import keras
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 首先声明一个Input（注意，你也可以对输入对象命名，就像对其他对象一样）
# 这个inputs对象保存了关于模型将处理的数据的形状和数据类型的信息。
# 我们将这样的对象叫作符号张量（symbolic tensor）。它不包含任何实际数据，
# 但编码了调用模型时实际数据张量的详细信息。它代表的是未来的数据张量。
inputs = keras.Input(shape=(3,), name="my_input")
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs)


vocabulary_size = 10000
num_tags = 100
num_departments = 4

title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")

features = layers.Concatenate()([title, text_body, tags])
features = layers.Dense(64, activation="relu")(features)

priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(
    num_departments, activation="softmax", name="department")(features)

model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])
tf.keras.utils.plot_model(model, show_shapes=True, to_file="model1.png")

# 假设你想对前一个模型增加一个输出——估算某个问题工单的解决时长，这是一种难度评分。
# 实现方法是利用包含3个类别的分类层，这3个类别分别是“快速”“中等”和“困难”。
# 你无须从头开始重新创建和训练模型。你可以从前一个模型的中间特征开始（这些中间特征是可以访问的），如代码清单7-14所示。
features = model.layers[4].output
difficulty = layers.Dense(3, activation="softmax", name="difficulty")(features)

new_model = keras.Model(
    inputs=[title, text_body, tags],
    outputs=[priority, department, difficulty])

tf.keras.utils.plot_model(new_model, show_shapes=True, to_file="model1.png")