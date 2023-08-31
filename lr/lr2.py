# imports
from sklearn.linear_model import LogisticRegression
from skimage import color, io
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sizes = [1024, 512, 256, 128, 64]
folder_paths = [f"resized_data/data_{size}" for size in sizes]

for size, dataset_path in zip(sizes, folder_paths):
    print(f"Training Logistic Regression Model for {size} x {size} images")
    # variables to hold our data
    data = []
    Y = []

    classifier = LogisticRegression(class_weight='balanced')
    classes = os.listdir(dataset_path)

    if ".DS_Store" in classes:
        # for macs
        classes.remove(".DS_Store")

    mapping = {n: i for i, n in enumerate(classes)}
    demapping = {i: n for i, n in enumerate(classes)}

    encoder = lambda s: mapping[s]
    decoder = lambda i: demapping[i]

    # now walk through and load the data in the containers we constructed above
    for root, dirs, files in os.walk(dataset_path):

        for file in files:
            if ".JPEG" in file.upper() or ".JPG" in file.upper() or ".PNG" in file.upper():
                key = root.split("/")[-1]
                img = io.imread(f"{root}/{file}", as_gray=True)
                arr = np.asarray(img).reshape(size*size, )  # reshape into an array
                data.append(arr)

                Y.append(encoder(key))  # simple one hot encoding

    y = np.array(Y)
    X = np.array(data)

    # now we've loaded all the X values into a single array
    # and all the Y values into another one, let's do a train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42)  # for consistency

    # now fit the classifier
    # fit the model with data
    classifier.fit(X_train, y_train)



    y_pred = classifier.predict(X_test)

    print(classification_report(
        y_test, y_pred,
    ))



