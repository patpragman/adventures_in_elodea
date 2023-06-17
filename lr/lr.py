# imports
from sklearn.linear_model import LogisticRegression
from skimage import color, io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 5568 × 4176
k = 8
IMG_WIDTH = 5568 // k
IMG_HEIGHT = 4176 // k
print(IMG_WIDTH, IMG_HEIGHT)

# let's do logistic regression... I'll be kind of non-plussed if this ends up being better
datasets = [
    "../ml/data"]


def make_encoder(classes):
    mapping = {}
    for i, class_name in enumerate(classes):
        mapping[class_name] = i

    return lambda s: mapping[s]


for i, data_path in enumerate(datasets):
    print(f"Training Model {i}")
    # variables to hold our data
    data = []
    Y = []
    classifier = LogisticRegression()
    classes = os.listdir(data_path)

    if ".DS_Store" in classes:
        # for macs
        classes.remove(".DS_Store")

    print('found the following classes in the data path')
    for c in classes:
        print(c)

    # encoder = make_encoder(os.listdir(data_path))

    mapping = {n: i for i, n in enumerate(classes)}
    demapping = {i: n for i, n in enumerate(classes)}

    encoder = lambda s: mapping[s]
    decoder = lambda i: demapping[i]

    # now walk through and load the data in the containers we constructed above
    for root, dirs, files in os.walk(data_path):

        for file in tqdm(files):

            if ".JPEG" in file.upper() or ".JPG" in file.upper() or ".PNG" in file.upper():
                key = root.split("/")[-1]
                img = io.imread(f"{root}/{file}", as_gray=True)
                img = resize(img, (IMG_WIDTH, IMG_HEIGHT))
                arr = np.asarray(img).reshape(IMG_WIDTH * IMG_HEIGHT, )  # reshape into an array
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

    # Evaluate the model on the test set
    print('Test accuracy:', accuracy_score(y_test, y_pred))

    # Evaluation and make a confusion matrix
    print(classification_report(y_test, y_pred))
    confusion_mtx = confusion_matrix(y_test, y_pred)
    print(confusion_mtx)

    plt.imshow(confusion_mtx, cmap='binary', interpolation='nearest')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, [c for c in classes], rotation=45)
    plt.yticks(tick_marks, [c for c in classes])

    thresh = confusion_mtx.max() // 2.
    for j in range(confusion_mtx.shape[0]):
        for k in range(confusion_mtx.shape[1]):
            plt.text(k, j, format(confusion_mtx[j, k]), ha="center", va="center",
                     color="white" if confusion_mtx[i, j] > thresh else "black")

    plt.title('Logistic Regression Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # save it
    with open(f"logistic_regression_{i}.pkl", "wb") as file:
        pickle.dump(classifier, file)
