import os

import numpy as np
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# preparation steps (we'll need these a lot)
from sklearn.metrics import classification_report, confusion_matrix

import pathlib

# constants
batch_size = 32
img_width = 227
img_height = 227
number_of_classes = 2

data_dir = pathlib.Path("data")
image_count = len(list(data_dir.glob('*/*.JPG')))
print(f'working with {image_count} images')

my_callbacks = [
    EarlyStopping(monitor="val_categorical_accuracy",
                  patience=5,
                  restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_categorical_accuracy",
                      factor=0.50, patience=3,
                      verbose=1,
                      min_delta=0.0001),
    #ModelCheckpoint(filepath=f'/content/drive/MyDrive/checkpoints/{model_name}.{epoch:02d}-{val_categorical_accuracy:.2f}.h5', save_best_only=True),
]

datagen = ImageDataGenerator(rescale=1. / 255,
                             validation_split=0.2,
                             zoom_range=0.1,  # Randomly zoom image
                             width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                             height_shift_range=0.1,
                             rotation_range=30
                             )

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_width),  # resize for alexnet
    batch_size=batch_size,
    subset='training',
    )

test_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_width),  # resize for alexnet
    batch_size=batch_size,
    subset='validation',
    )

train_class_counts = train_generator.classes
test_class_counts = test_generator.classes

train_class_count = dict(zip(train_generator.class_indices.keys(), np.zeros(len(train_generator.class_indices), dtype=int)))
test_class_count = dict(zip(test_generator.class_indices.keys(), np.zeros(len(test_generator.class_indices), dtype=int)))

for label in train_class_counts:
    train_class_count[list(train_generator.class_indices.keys())[int(label)]] += 1

for label in test_class_counts:
    test_class_count[list(test_generator.class_indices.keys())[int(label)]] += 1

print('Number of training samples in each class in the training set:', train_class_count)
print('Number of test samples in each class in the testing set:', test_class_count)

from collections import Counter
counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
print(class_weights)

data_augmentation = keras.Sequential(
    [
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),

        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
    ]
)

model = Sequential([
    layers.Conv2D(64, 4, activation='relu'),
    layers.Dropout(0.65),
    layers.MaxPooling2D(4),

    layers.Conv2D(256, 3, activation='relu'),
    layers.Dropout(0.25),
    layers.MaxPooling2D(2),

    layers.Conv2D(256, 3, activation='relu'),
    layers.Dropout(0.25),
    layers.MaxPooling2D(2),

    layers.Conv2D(256, 2, activation='relu'),
    layers.Conv2D(256, 2, activation='relu'),
    layers.MaxPooling2D(2),

    layers.Flatten(),

    layers.Dense(32, activation='relu'),
    layers.Dropout(0.25),

    layers.Dense(number_of_classes, activation='softmax')
])

model.compile(optimizer='sgd',
              loss="categorical_crossentropy",
              metrics=['categorical_accuracy'],
              )
model.build(input_shape=(None, img_height, img_width, 3))

print(model.summary())

epochs = 20
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=my_callbacks
)

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

model.save_weights('model3.h5')


# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

# Make predictions on the test set
y_pred = model.predict(test_generator)
y_actual = test_generator.classes

# should round to 0 or 1...
y_pred = np.round(y_pred)
y_pred = np.argmax(y_pred, axis=1)


confusion_mtx = confusion_matrix(y_actual, y_pred)
print(confusion_mtx)

# Evaluation
print(classification_report(test_generator.classes, y_pred))

plt.imshow(confusion_mtx, cmap='binary', interpolation='nearest')
plt.colorbar()

# manually set - could be better
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Elodea', 'No Elodea'], rotation=45)
plt.yticks(tick_marks, ['Elodea', 'No Elodea'])

thresh = confusion_mtx.max() / 2.
for i in range(confusion_mtx.shape[0]):
    for j in range(confusion_mtx.shape[1]):
        plt.text(j, i, format(confusion_mtx[i, j]), ha="center", va="center", color="white" if confusion_mtx[i, j] > thresh else "black")

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix')
plt.savefig('confusionmtx.png')
plt.show()

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title(f'accuracy chart')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('accuracy_chart.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(f'loss chart')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss.png')
plt.show()