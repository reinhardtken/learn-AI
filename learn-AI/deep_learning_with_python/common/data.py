import numpy as np


def data1():
    vocabulary_size = 10000
    num_tags = 100
    num_departments = 4

    num_samples = 1280

    title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
    text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
    tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

    priority_data = np.random.random(size=(num_samples, 1))
    department_data = np.random.randint(0, 2, size=(num_samples, num_departments))
    return {"title_data": title_data, "text_body_data": text_body_data, "tags_data": tags_data, "priority_data": priority_data, "department_data": department_data}

    # model.compile(optimizer="rmsprop",
    #               loss=["mean_squared_error", "categorical_crossentropy"],
    #               metrics=[["mean_absolute_error"], ["accuracy"]])
    # model.fit([title_data, text_body_data, tags_data],
    #           [priority_data, department_data],
    #           epochs=1)
    # model.evaluate([title_data, text_body_data, tags_data],
    #                [priority_data, department_data])
    # priority_preds, department_preds = model.predict([title_data, text_body_data, tags_data])