# TensorFlow and tf.keras

import tensorflow as tf

# Helper libraries

import numpy as np

import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape

len(train_labels)

train_labels

plt.figure()

plt.imshow(train_images[0])

plt.colorbar()

plt.grid(False)

plt.show()

plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

plt.xticks([])

plt.yticks([])

plt.grid(False)

plt.imshow(train_images[i], cmap=plt.cm.binary)

plt.xlabel(class_names[train_labels[i]])

plt.show()

model = tf.keras.Sequential([

tf.keras.layers.Flatten(input_shape=(28, 28)),

tf.keras.layers.Dense(1280, activation='relu'),

tf.keras.layers.Dense(1280, activation='relu'),

tf.keras.layers.Dense(1280, activation='relu'),

tf.keras.layers.Dense(128, activation='relu'),

tf.keras.layers.Dense(10)

])

model.compile(optimizer='adam',

loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

metrics=['accuracy'])

# import timeit

import time

aa=time.time()

model.fit(train_images, train_labels, epochs=10)

bb=time.time()

cc=bb-aa

cc

aa=time.time()

with tf.device('/CPU:0'):
    model.fit(train_images, train_labels, epochs=10)

bb=time.time()

cc=bb-aa

cc

aa=time.time()

with tf.device('/GPU:0'):
    model.fit(train_images, train_labels, epochs=10)

bb=time.time()

cc=bb-aa

cc