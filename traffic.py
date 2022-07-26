import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

DENSE_LAYERS = 1
DENSE_UNITS = 512

DROPOUT = 0.5

CONV_LAYERS = 3
CONV_FILTERS = 32

MAX_POOLING = False

LEARNING_RATE = 1e-3

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    # Loop categories
    for cat in range(NUM_CATEGORIES):
        cat_path = os.path.join(data_dir, str(cat))
        # Loop images in category
        for img_name in os.listdir(cat_path):
            img_path = os.path.join(cat_path, img_name)
            # Load image
            img = cv2.imread(img_path)
            # Resize image
            resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # Normalise colour values
            resized = resized / 255

            # Add to images
            images.append(resized)
            # Add to labels
            labels.append(cat)
    # Return data
    return images, labels

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Create model, with input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    ])

    model = add_conv(model)

    model.add(tf.keras.layers.Flatten())

    model = add_dense(model)

    # Add output layer, which represents probability distribution of each image category
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))

    print(f"Compiling model with learning rate: {LEARNING_RATE}...")

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    # Return compiled model
    return model


def add_conv(model):
    print(f"Adding {CONV_LAYERS} convolution layers with {CONV_FILTERS} filters and {'' if MAX_POOLING else 'no '}max-pooling...")
    if CONV_LAYERS == 0 and MAX_POOLING:
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    else:
        for i in range(CONV_LAYERS):
            model.add(tf.keras.layers.Conv2D(CONV_FILTERS, (3, 3), activation="relu"))
            if MAX_POOLING:
                model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    return model


def add_dense(model):
    print(f"Adding {DENSE_LAYERS} hidden layers with {DENSE_UNITS} units in each{' and a dropout rate of ' + str(DROPOUT) if DROPOUT > 0 else ''}...")
    for i in range(DENSE_LAYERS):
        model.add(tf.keras.layers.Dense(DENSE_UNITS, activation="relu"))
        if DROPOUT > 0:
            model.add(tf.keras.layers.Dropout(DROPOUT))
    return model


if __name__ == "__main__":
    main()
