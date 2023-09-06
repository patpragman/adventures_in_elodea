import os

import numpy as np
import matplotlib.pyplot as plt

from cm_handler import display_and_save_cm  # handles confusion matrices

# preparation steps (we'll need these a lot)
from sklearn.metrics import classification_report
import pathlib
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from skimage.io import imread

from sklearn.model_selection import train_test_split

sizes = [1024, 512, 256, 128, 64]
sizes.reverse()
for size in sizes:
    print(f'Building SVM Model for Image size: {size} x {size}')

    img_width = size
    img_height = size

    data_dir = pathlib.Path(f"../resized_data/data_{size}")
    image_count = len(list(data_dir.glob('*/*.JPG')))

    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [0.0001, 0.001, 0.1, 1],
                  'kernel': ['rbf', 'poly']}

    flat_data_array = []  # input array
    target_data_array = []  # output array

    # set up the categories now
    categories = [f.path.split("/")[-1] for f in os.scandir(data_dir) if f.is_dir()]

    for category in categories:
        path = os.path.join(data_dir, category)

        for image_file_name in os.listdir(path):
            image_array = imread(os.path.join(path, image_file_name))
            flat_data_array.append(image_array.flatten())
            target_data_array.append(categories.index(category))

    # convert to numpy arrays
    X = np.array(flat_data_array)
    y = np.array(target_data_array)

    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=42,
                                                        stratify=y)

    svc = svm.SVC(probability=True)
    model = GridSearchCV(svc, param_grid)

    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluation
    report = classification_report(y_test, y_pred)
    print(report)

    display_and_save_cm(y_test, y_pred, labels=["Vegetation", "No Vegetation"])