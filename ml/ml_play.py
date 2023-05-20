import os

import numpy as np
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pathlib

# print out gpu status
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))




# constants
batch_size = 32
img_width = 227
img_height = 227
number_of_classes = 2

data_dir = pathlib.Path("data")
image_count = len(list(data_dir.glob('*/*.JPG')))
print(f'working with {image_count} images')

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
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

model = Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(number_of_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=['categorical_accuracy']
              )
print(model.summary())

epochs = 20
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs,
    class_weight=class_weights,
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save_weights('model2.h5')
