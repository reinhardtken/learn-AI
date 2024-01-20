## 新闻分类：多分类问题示例

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.datasets import reuters


import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = r'..\\'
abs_path = os.path.join(current_dir, relative_path)
sys.path.append(abs_path)
from common.function import vectorize_sequences, to_one_hot

# 与IMDB数据集一样，参数num_words=10000将数据限定为前10 000个最常出现的单词。我们有8982个训练样本和2246个测试样本。
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000)
print(len(train_data))
print(len(test_data))

# 样本对应的标签是一个介于0和45之间的整数，即话题索引编号。
print(len(train_labels))
print(train_labels)

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = " ".join([reverse_word_index.get(i - 3, "?") for i in
    train_data[0]])

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)

# keras的内置方法，等同于to_one_hot，那这个方法还需要先把序列里面的最大值找出来？？？
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

# 餐馆评论问题的用的是sigmoid，这里用的是softmax，因为餐馆问题的要的是0~1之间的概率，表示好不好
# 这里要的是46个概率，加总为1，也就是46个标签中分别为xxx的概率，但也不对啊，一个新闻当然既可以是a类又可以是b类？？？
# 感觉多分类的意思是在46个分类中，新闻会属于某一个，但是不会同时属于好几个，应该是如此。所以求和概率为一是讲的通的

# 第二，最后一层使用了softmax激活函数。你在MNIST例子中见过这种用法。
# 模型将输出一个在46个输出类别上的概率分布——对于每个输入样本，模型都会生成一个46维输出向量，
# 其中output[i]是样本属于第i个类别的概率。46个概率值的总和为1。
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
])

# 对于这个例子，最好的损失函数是categorical_crossentropy（分类交叉熵），
# 如代码清单4-16所示。它衡量的是两个概率分布之间的距离，这里两个概率分布分别是模型输出的概率分布和标签的真实分布。
# 我们训练模型将这两个分布的距离最小化，从而让输出结果尽可能接近真实标签。
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 绘制训练损失和验证损失
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 绘制训练精度和验证精度
plt.clf()
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

model = keras.Sequential([
  layers.Dense(64, activation="relu"),
  layers.Dense(64, activation="relu"),
  layers.Dense(46, activation="softmax")
])
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train,
          y_train,
          epochs=9,
          batch_size=512)
results = model.evaluate(x_test, y_test)

# 处理标签和损失的另一种方法
# 前面提到过另一种编码标签的方法，也就是将其转换为整数张量，如下所示。
# 对于这种编码方法，唯一需要改变的就是损失函数的选择。对于代码清单4-21使用的损失函数categorical_crossentropy，
# 标签应遵循分类编码。对于整数标签，你应该使用sparse_categorical_crossentropy（稀疏分类交叉熵）损失函数。

y_train = np.array(train_labels)
y_test = np.array(test_labels)

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])