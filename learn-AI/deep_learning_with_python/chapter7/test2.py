from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

from tensorflow import keras
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = r'..\\'
abs_path = os.path.join(current_dir, relative_path)
sys.path.append(abs_path)
from common.data import data1


class CustomerTicketModel(keras.Model):

    def __init__(self, num_departments):
        super().__init__()
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid")
        self.department_classifier = layers.Dense(
            num_departments, activation="softmax")

    def call(self, inputs):
        title = inputs["title_data"]
        text_body = inputs["text_body_data"]
        tags = inputs["tags_data"]

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department

model = CustomerTicketModel(num_departments=4)

# priority, department = model(
#     {"title": title_data, "text_body": text_body_data, "tags": tags_data})
data = data1()
priority, department = model(data)

model.compile(optimizer="rmsprop",
              loss=["mean_squared_error", "categorical_crossentropy"],
              metrics=[["mean_absolute_error"], ["accuracy"]])
model.fit(data,
          [data["priority_data"], data["department_data"]],
          epochs=1)
model.evaluate(data,
               [data["priority_data"], data["department_data"]])
priority_preds, department_preds = model.predict(data)
tf.keras.utils.plot_model(model, show_shapes=True, to_file="CustomerTicketModel.png")