from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

from tensorflow import keras
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
(train_data, train_labels), _ = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
train_data = vectorize_sequences(train_data)

# 16个参数
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_original = model.fit(train_data, train_labels,
                             epochs=20, batch_size=512, validation_split=0.4)

# 4个参数
model = keras.Sequential([
    layers.Dense(4, activation="relu"),
    layers.Dense(4, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_smaller_model = model.fit(
    train_data, train_labels,
    epochs=20, batch_size=512, validation_split=0.4)


# 较小模型开始过拟合的时间要晚于初始模型（前者6轮后开始过拟合，而后者4轮后就开始过拟合），而且开始过拟合之后，它的性能下降速度也更慢。

smaller_val_loss = history_smaller_model.history["val_loss"]
origin_val_loss = history_original.history["val_loss"]

epochs = range(1, 21)
plt.plot(epochs, smaller_val_loss, "b--",
         label="smaller Validation loss")
plt.plot(epochs, origin_val_loss, "b",
         label="origin Validation loss")
plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 512个参数
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(512, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_larger_model = model.fit(
    train_data, train_labels,
    epochs=20, batch_size=512, validation_split=0.4)

# 我们现在添加一个容量更大的模型——其容量远大于问题所需。虽然过度参数化的模型很常见，
# 但肯定会有这样一种情况：模型的记忆容量过大。如果模型立刻开始过拟合，而且它的验证损失曲线看起来很不稳定、方差很大，你就知道模型容量过大了
larger_val_loss = history_larger_model.history["val_loss"]

plt.plot(epochs, larger_val_loss, "b--",
         label="larger Validation loss")
plt.plot(epochs, origin_val_loss, "b",
         label="origin Validation loss")
plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()