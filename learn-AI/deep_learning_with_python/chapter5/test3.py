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
origin_val_loss = history_original.history["val_loss"]



from tensorflow.keras import regularizers
# regularizers.l2(0.002)的含义是该层权重矩阵的每个系数都会使模型总损失值增加0.002 * weight_coefficient_value ** 22。
# 注意，因为只在训练时添加这个惩罚项，所以该模型的训练损失会比测试损失大很多。图5-19展示了L2正则化惩罚项的影响。
# 如你所见，虽然两个模型的参数个数相同，但具有L2正则化的模型比初始模型更不容易过拟合。
model = keras.Sequential([
    layers.Dense(16,
                 kernel_regularizer=regularizers.l2(0.002),
                 activation="relu"),
    layers.Dense(16,
                 kernel_regularizer=regularizers.l2(0.002),
                 activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_l2_reg = model.fit(
    train_data, train_labels,
    epochs=20, batch_size=512, validation_split=0.4)


l2_val_loss = history_l2_reg.history["val_loss"]


epochs = range(1, 21)
plt.plot(epochs, l2_val_loss, "b--",
         label="l2 Validation loss")
plt.plot(epochs, origin_val_loss, "b",
         label="origin Validation loss")
plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()