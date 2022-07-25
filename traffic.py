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

X_PARAMS = [
    (0, 0),
    (128, 1),
    (128, 2),
    (128, 3),
    (256, 1),
    (256, 2),
    (256, 3),
    (512, 1),
    (512, 2),
    (512, 3)
]

Y_PARAMS = [
    (0, False, 0),
    (0, True, 0),
    (8, False, 1),
    (8, True, 1),
    (16, False, 1),
    (16, True, 1),
    (32, False, 1),
    (32, True, 1),
    (8, False, 2),
    (8, True, 2),
    (16, False, 2),
    (16, True, 2),
    (32, False, 2),
    (32, True, 2),
    (8, False, 3),
    (8, True, 3),
    (16, False, 3),
    (16, True, 3),
    (32, False, 3),
    (32, True, 3)
]

DENSE_LAYERS = 0
DENSE_UNITS = 0

CONV_LAYERS = 0
CONV_FILTERS = 0

MAX_POOLING = False

NUM_SAMPLES = 3

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)


    results = dict()
    global CONV_FILTERS
    global MAX_POOLING
    global CONV_LAYERS
    global DENSE_UNITS
    global DENSE_LAYERS

    for y in Y_PARAMS:
        CONV_FILTERS = y[0]
        MAX_POOLING = y[1]
        CONV_LAYERS = y[2]
        for x in X_PARAMS:
            DENSE_UNITS = x[0]
            DENSE_LAYERS = x[1]

            for i in range(NUM_SAMPLES):

                x_train, x_test, y_train, y_test = train_test_split(
                    np.array(images), np.array(labels), test_size=TEST_SIZE
                )

                # Get a compiled neural network
                model = get_model()

                # Fit model on training data
                training = model.fit(x_train, y_train, epochs=EPOCHS)

                # Evaluate neural network performance
                testing = model.evaluate(x_test,  y_test, verbose=2)

                if (x, y) in results:
                    temp = results.get((x, y))
                    results[(x, y)] = (temp[0] + training.history['accuracy'][-1], temp[1] + testing[-1])
                else:
                    results[(x, y)] = (training.history['accuracy'][-1], testing[-1])
            temp = results.get((x, y))
            results[(x, y)] = (temp[0] / NUM_SAMPLES, temp[1] / NUM_SAMPLES)

    for res in results:
        print(f"{res}: {results[res]}")

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

    # Compile model
    model.compile(
        optimizer="adam",
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
    print(
        f"Adding {DENSE_LAYERS} hidden layers with {DENSE_UNITS} units in each...")
    for i in range(DENSE_LAYERS):
        model.add(tf.keras.layers.Dense(DENSE_UNITS, activation="relu"))
        # model.Add(tf.keras.layers.Dropout(0.5))
    return model


if __name__ == "__main__":
    main()
