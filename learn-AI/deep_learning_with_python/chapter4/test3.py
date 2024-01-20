# 预测房价：标量归问题示例

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers


from tensorflow.keras.datasets import boston_housing

# 可以看到，我们有404个训练样本和102个测试样本，每个样本都有13个数值特征，比如人均犯罪率、住宅的平均房间数、高速公路可达性等。
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data)

#将取值范围差异很大的数据输入到神经网络中，这是有问题的。
# 模型可能会自动适应这种取值范围不同的数据，但这肯定会让学习变得更加困难。对于这类数据，
# 普遍采用的最佳处理方法是对每个特征进行标准化，即对于输入数据的每个特征（输入数据矩阵的每一列），
# 减去特征平均值，再除以标准差，这样得到的特征平均值为0，标准差为1。用NumPy可以很容易实现数据标准化，如代码清单4-24所示

# 这里有两件事情，一个是均值为0，一个是标准差为1
# 均值为0就是：
# mean = ∑(x)/n # 此处的x是某个xi，比如x[0]
# x_norm = (x - mean) # 此处的x是向量
# 0 = ∑(x_norm)/n

# 另一个事情是标准差为1
# 标准差是衡量一组数据离散程度的一个指标。它是数据的平均离散程度的平方根。
#
# 标准差的公式如下：
#
# σ = √(∑(x - μ)^2 / n)
# 其中：
#
# σ 是标准差
# x 是数据中的每个值
# μ 是数据的均值
# n 是数据的数量
# 标准差越大，数据的离散程度越大。标准差越小，数据的离散程度越小。



mean = train_data.mean(axis=0)
train_data -= mean
# 走完上面两步，此时数据的均值已经是0了
print(train_data)
std = train_data.std(axis=0)
train_data /= std
# 走完上面三步，此时数据的均值是0，标准差是1，1 怎么来的，数学替换出来就是如此，那笔演算一下就知道了
print(train_data)
test_data -= mean
test_data /= std

# 模型的最后一层只有一个单元且没有激活，它是一个线性层。这是标量回归（标量回归是预测单一连续值的回归）的典型设置。
# 添加激活函数将限制输出范围。如果向最后一层添加sigmoid激活函数，那么模型只能学会预测0到1的值。
# 这里最后一层是纯线性的，所以模型可以学会预测任意范围的值。
# 注意，我们编译模型用的是mse损失函数，即均方误差（mean squared error，MSE），预测值与目标值之差的平方。
# 这是回归问题常用的损失函数。在训练过程中还要监控一个新指标：平均绝对误差（mean absolute error，MAE）。
# 它是预测值与目标值之差的绝对值。如果这个问题的MAE等于0.5，就表示预测房价与实际价格平均相差500美元。
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=16, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

num_epochs = 500
all_mae_histories = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=16, verbose=0)
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)