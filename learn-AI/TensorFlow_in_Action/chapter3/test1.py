
import requests
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


def download():
  if not os.path.exists(data_path):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    r = requests.get(url)

    with open(data_path, 'wb') as f:
      f.write(r.content)

# Retrieve the data
download()

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

print(iris_df.head())
# print(y)

## 训练
K.clear_session() # Making sure we are clearing out the TensorFlow graph

# Defining Model A with the Sequential API
model = Sequential([
    Dense(32, activation='relu', input_shape=(4,)),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

# Print out a summary of the model
# model.summary()

# Compiling the model with a loss function, an optimizer and a performance metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

tf.keras.utils.plot_model(model)

# Fitting the model with data
model.fit(x, y, batch_size=64, epochs=25)