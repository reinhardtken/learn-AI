# 影评分类：二分类问题示例

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.datasets import imdb

# 参数num_words=10000的意思是仅保留训练数据中前10 000个最常出现的单词。低频词将被舍弃。
# 这样一来，我们得到的向量数据不会太大，便于处理。如果没有这个限制，那么我们需要处理训练数据中的88 585个单词。
# 这个数字太大，且没有必要。许多单词只出现在一个样本中，它们对于分类是没有意义的。
# train_data和test_data这两个变量都是由评论组成的列表，每条评论又是由单词索引组成的列表（表示单词序列）。
# train_labels和test_labels都是由0和1组成的列表，其中0代表负面（negative），1代表正面（positive）。
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)

# train_data是每个评论中，词汇的index的组合
print(train_data[0])
# train_labels是每个评论属于正面还是负面
print(train_labels[0])

word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]])
print(decoded_review)

# 因为总共的评论词是10000个，所以每个评论的词汇总量也不会超过10000，就用10000的array，1表示某个词汇存在，0表示不存在处理
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 你还应该将标签向量化，其实下面两行只是改变了数据格式，从int64变float32了，
# 神经网络只能处理浮点？
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# history_dict = history.history
# history_dict.keys()

history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 模型
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
# 参数
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
# 训练
model.fit(x_train, y_train, epochs=4, batch_size=512)

# 评估模型在测试集上的表现，整体表现情况就是88%正确
# results:
# [0.2929924130630493, 0.88327999999999995]  ←----第一个数字是测试损失，第二个数字是测试精度
results = model.evaluate(x_test, y_test)
print(results)

# 输出评估结果，就是每条的分数是多少
test_result = model.predict(x_test)
print(test_result)