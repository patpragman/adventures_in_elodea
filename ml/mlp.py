import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from cm_handler import display_and_save_cm  # handles confusion matrices

# preparation steps (we'll need these a lot)
from sklearn.metrics import classification_report
import pathlib

# print out gpu status
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))

# constants
batch_size = 32
k = 32
number_of_classes = 2
dropout_value = 0.1
epochs = 200




datagen = ImageDataGenerator(validation_split=0.2)

"""
rescale=1. / 255,
                             validation_split=0.2,
                             zoom_range=0.1,  # Randomly zoom image
                             width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                             height_shift_range=0.1,
                             rotation_range=30,
                             horizontal_flip=True,
                             vertical_flip=True,
"""
sizes = [512, 256, 128, 64]
for size in sizes:
    print(f'Building Model for Image size: {size} x {size}')

    img_width = size
    img_height = size

    data_dir = pathlib.Path(f"../resized_data/data_{size}")
    image_count = len(list(data_dir.glob('*/*.JPG')))
    train_generator = datagen.flow_from_directory(
        data_dir,
        batch_size=batch_size,
        subset='training',
        color_mode='grayscale'
    )

    test_generator = datagen.flow_from_directory(
        data_dir,
        batch_size=batch_size,
        subset='validation',
        color_mode='grayscale'
    )

    train_class_counts = train_generator.classes
    test_class_counts = test_generator.classes

    data_augmentation = keras.Sequential(
        [
            layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),

            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
        ]
    )

    model = models.Sequential()
    model.add(tf.keras.Input(shape=(img_width, img_height, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam',
                  loss="categorical_crossentropy",
                  metrics=['categorical_accuracy'],
                  )

    model.build(input_shape=(None, img_height, img_width, 3))
    print(model.summary())

    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=epochs,
    )

    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_generator)
    print('Test accuracy:', test_acc)

    # Make predictions on the test set
    y_pred = model.predict(test_generator)
    y_actual = test_generator.classes

    # should round to 0 or 1...
    y_pred = np.round(y_pred)
    y_pred = np.argmax(y_pred, axis=1)


    # Evaluation
    report = classification_report(test_generator.classes, y_pred)
    print(report)

    display_and_save_cm(y_actual, y_pred, labels=["Vegetation", "No Vegetation"])

    # display the accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title(f'accuracy chart')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'accuracy_chart_{size}x{size}.png')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'loss chart')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'loss_{size}x{size}.png')
    plt.show()
