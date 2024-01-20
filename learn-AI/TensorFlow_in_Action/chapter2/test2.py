from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = r'..\\'
abs_path = os.path.join(current_dir, relative_path)
sys.path.append(abs_path)

# Loading the image and converting to a TensorFlow tensor
pic_path = os.path.join(current_dir, "2024-01-20-171936.jpg")
x_rgb = np.array(Image.open(pic_path)).astype('float32')
x_rgb = tf.constant(x_rgb)

# Defining the matrix used to convert the image to grayscale
grays = tf.constant([[0.3], [0.59] ,[0.11]])

# Converting the image to grayscale
x = tf.matmul(x_rgb, grays)
x = tf.squeeze(x)
print("The size of the final image: {}".format(x.shape))

# Plotting the image
f, axes = plt.subplots(1,1, figsize=(5,5))

axes.imshow(x.numpy(),cmap='gray')
axes.axis('off')
plt.show()