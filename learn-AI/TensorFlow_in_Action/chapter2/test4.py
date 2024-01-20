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
# x = tf.squeeze(x)


# Defining the input image
y = tf.constant(x)
print(y.shape)

# Defining the convolution kernel as a TensorFlow variable
f = tf.Variable(np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]).astype('float32'))

# Reshaping the input and the kernel to meet tf.nn.convolution requirements
# y_reshaped = tf.reshape(y, [1,512,512,1]) # [batch size, height, width, channels]
y_reshaped = tf.reshape(y, [1,530,531,1]) # [batch size, height, width, channels]
f_reshaped = tf.reshape(f, [3,3,1,1]) # [height, width, in channels, out channels]

# Convolving the images
y_conv = tf.nn.convolution(y_reshaped, f_reshaped)

def improve_contrast(x,n=3):
    """ This is function to improve the contrast in the image for visual purposes. """
    return np.clip(x*n, np.min(x), np.max(x))

y_conv_clipped = improve_contrast(y_conv, 4)
print("The size of the final image: {}".format(y_conv.shape))

f, axes = plt.subplots(1,2, figsize=(10,10))

axes[0].imshow(np.squeeze(x.numpy()),cmap='gray')
axes[1].imshow(np.squeeze(y_conv_clipped),cmap='gray')

axes[0].axis('off')
axes[0].set_title('Original Image')
axes[1].axis('off')
axes[1].set_title('Result after Edge Detection')
plt.show()