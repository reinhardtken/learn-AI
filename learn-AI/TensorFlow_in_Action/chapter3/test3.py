
from tensorflow.keras import layers
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




class MulBiasDense(layers.Layer):
    """ The layer with the new multiplicative bias we want to test """
    # First, we have the __init__() function. There are two parameters for the layer: the
    # number of hidden units and the type of activation. The activation defaults to None,
    # meaning that if unspecified, there will be no nonlinear activation (i.e., only a linear
    # transformation):
    def __init__(self, units=32, activation=None):
        """ Defines various hyperparameters of the layer"""

        super(MulBiasDense, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        """ Defines the parameters (weights and biases)"""

        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b_mul = self.add_weight(shape=(self.units,),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, inputs):
        """ Defines the computations that happen in the layer"""

        out = (tf.matmul(inputs, self.w) + self.b) * self.b_mul
        return layers.Activation(self.activation)(out)


from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf

K.clear_session() # Making sure we are clearing out the TensorFlow graph

# The input layer
inp = Input(shape=(4,))

# Using the newly defined MulBiasDense layer to create two layers
out = MulBiasDense(units=32, activation='relu')(inp)
out = MulBiasDense(units=16, activation='relu')(out)

# The softmax layer
out = Dense(3, activation='softmax')(out)

model = Model(inputs=inp, outputs=out)
# Compiling the model with a loss function, an optimizer and a performance metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.summary()

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

model.fit(x, y, batch_size=64, epochs=25)