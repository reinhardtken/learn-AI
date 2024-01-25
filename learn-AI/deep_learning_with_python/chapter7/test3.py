


from tensorflow.keras.datasets import mnist

# ←----创建模型（我们将其包装为一个单独的函数，以便后续复用）
def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model

# ←----加载数据，保留一部分数据用于验证
(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

model = get_mnist_model()
# ←---- (本行及以下2行)编译模型，指定模型的优化器、需要最小化的损失函数和需要监控的指标
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
# ←---- (本行及以下2行)使用fit()训练模型，可以选择提供验证数据来监控模型在前所未见的数据上的性能
model.fit(train_images, train_labels,
          epochs=3,
          validation_data=(val_images, val_labels))
# ←----使用evaluate()计算模型在新数据上的损失和指标
test_metrics = model.evaluate(test_images, test_labels)
# ←----使用predict()计算模型在新数据上的分类概率
predictions = model.predict(test_images)