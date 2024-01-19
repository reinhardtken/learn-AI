#3.5.4　一个端到端的例子：用TensorFlow编写线性分类器
# https://bard.google.com/chat/1903ebc4f4e9b41e
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Generating two classes of random points in a 2D plane
# multivariate_normal大概就是以均值，协防差的方式产生size个随机数
# 直观上来看，协方差矩阵描述了点云的形状，均值则描述了点云在平面上的位置
# 这里的参数给了2个，所以产生的是2维数据
# negative_samples，它的x均值是0，y的均值是3，所以它是x=0，y=3为中心的一团点。
# positive_samples，它的x均值是3，y的均值是0，所以它是x=3，y=0为中心的一团点
# cov应该是决定斜向上形状的，不深究了
num_samples_per_class = 1000
half_num_samples_per_class = 500
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5],[0.5, 1]],
    size=num_samples_per_class)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5],[0.5, 1]],
    size=num_samples_per_class)

# positive_samples = np.random.multivariate_normal(
#     mean=[0, 2],
#     cov=[[0.5, 0.5], [1, 0.5]],
#     size=num_samples_per_class)

print(negative_samples.shape)

# vstack就是垂直堆叠
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
# print(inputs)

#Generating the corresponding targets (0 and 1)
# targets 2行，1000列，第一行都是0，第二行1
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                     np.ones((num_samples_per_class, 1), dtype="float32")))
print(targets)

#Plotting the two point classes
# 这里c=targets[:, 0]是设置颜色，但是没看懂怎么就设置颜色了。。。，可能1表示是一个颜色，0表示是另一个颜色
# import matplotlib.pyplot as plt
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
# plt.scatter(inputs[:, 0], inputs[:, 1])
plt.show()

input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))
print(W)
def model(inputs):
    # inputs 是 1000*2，W是2*1，所以结果就是1000*1
    # 因为这个线性分类器处理的是二维输入，所以W实际上只包含两个标量系数W1和W2：
    # W = [[w1], [w2]]。b则是一个标量系数。因此，对于给定的输入点[x,y]，
    # 其预测值为：prediction = [[w1], [w2]]•[x, y] + b = w1 * x + w2 * y + b。
    return tf.matmul(inputs, W) + b

def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)

learning_rate = 0.1

def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss

for step in range(40):
    loss = training_step(inputs, targets)
    print(f"Loss at step {step}: {loss:.4f}")

predictions = model(inputs)
# plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
# plt.show()

x = np.linspace(-1, 4, 100)
y = - W[0] /  W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()