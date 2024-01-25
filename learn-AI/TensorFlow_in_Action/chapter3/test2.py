from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model


import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = r'..\\'
abs_path = os.path.join(current_dir, relative_path)
sys.path.append(abs_path)

# Loading the image and converting to a TensorFlow tensor
data_path = os.path.join(current_dir, "iris.data")

K.clear_session() # Making sure we are clearing out the TensorFlow graph

# The two input layers
# inp1: This defines the first input layer with a shape of (4,). This means it expects a one-dimensional array with 4 features.
# inp2: This defines the second input layer with a shape of (2,). This means it expects a one-dimensional array with 2 features.
inp1 = Input(shape=(4,), name="inp1")
inp2 = Input(shape=(2,), name="inp2")

# Two parallel dense layers
# out1: This creates a Dense layer with 16 neurons and a ReLU activation function for the inp1 input.
# out2: This creates a Dense layer with 16 neurons and a ReLU activation function for the inp2 input.
# units: Positive integer, dimensionality of the output space.
#       activation: Activation function to use.

# (inp1) 调用的是基类keras.engine.base_layer.Layer.__call__
out1 = Dense(16, activation='relu')(inp1)
out2 = Dense(16, activation='relu')(inp2)

# Concatenate the two outputs from the parallel layers
# out = Concatenate(axis=1)([out1, out2]): This combines the outputs from both out1 and out2 along the first dimension (axis=1).
# This results in a single one-dimensional array with 32 features (16 from each input).
# out1和out1都是[1,16]，它们堆叠要么变成一个[2,16]要么是[1,32]，按照书上的说法是变成32了
out = Concatenate(axis=1)([out1,out2])

# The single dense layer
# 等于创建了一个新层，output是16，input是32
out = Dense(16, activation='relu', name="my_first_layer")(out)

# The final output layer
out = Dense(3, activation='softmax', name="my_second_layer")(out)

# Create and compile the model
model = Model(inputs=[inp1, inp2], outputs=out)

# Compiling the model with a loss function, an optimizer and a performance metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.summary()


# try:
#   # pydot-ng is a fork of pydot that is better maintained.
#   import pydot_ng as pydot
# except ImportError:
#   print("import pydot_ng failed")
#   # pydotplus is an improved version of pydot
#   try:
#     import pydotplus as pydot
#   except ImportError:
#     # Fall back on pydot if necessary.
#     print("import pydotplus failed")
#     try:
#       import pydot
#     except ImportError:
#       print("import pydot failed")
#       pydot = None
#
# def check_pydot():
#   """Returns True if PyDot and Graphviz are available."""
#   if pydot is None:
#     print("pydot is None")
#     return False
#   try:
#     # Attempt to create an image of a blank graph
#     # to check the pydot/graphviz installation.
#     pydot.Dot.create(pydot.Dot())
#     return True
#   except (OSError, pydot.InvocationException):
#     return False
#
# check_pydot()

# 不会show，是会产生一张同目录的图
# need：pip install pydot
# pip install graphviz
tf.keras.utils.plot_model(model, show_shapes=True)

# Read the data in
iris_df = pd.read_csv(data_path, header=None)

# Set the column names
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_width', 'petal_length', 'label']
print(iris_df.label.unique())

# Convert labels to integers
iris_df["label"] = iris_df["label"].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

# Shuffle the data
# Shuffling the data is an important step: the data is in a very specific order, with each
# class appearing one after another. But you achieve the best results when data has been
# shuffled so that each batch presented to the network has a good mix of all classes
# found in the full data set.
iris_df = iris_df.sample(frac=1.0, random_state=4321)

# Normalize the features by subtracting the mean
x = iris_df[["sepal_length", "sepal_width", "petal_width", "petal_length"]]
x = x - x.mean(axis=0)

# Converting integer labels to one-hot vectors
y = tf.one_hot(iris_df["label"], depth=3)

from sklearn.decomposition import PCA
# Defining a PCA transformer
pca_model = PCA(n_components=2, random_state=4321)
# Generating the PCA features from data
x_pca = pca_model.fit_transform(x)

model.fit([x, x_pca], y, batch_size=64, epochs=25)