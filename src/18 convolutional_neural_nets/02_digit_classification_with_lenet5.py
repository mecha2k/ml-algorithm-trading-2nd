# Basic Image Classification with Feedforward NN and LetNet5
# All libraries we introduced in the last chapter provide support for convolutional layers. We are going to illustrate
# the LeNet5 architecture using the most basic MNIST handwritten digit dataset, and then use AlexNet on CIFAR10,
# a simplified version of the original ImageNet to demonstrate the use of data augmentation.
# LeNet5 and MNIST using Keras.

from pathlib import Path
from random import randint
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Dropout, Flatten
import matplotlib.pyplot as plt
import seaborn as sns


gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if gpu_devices:
    print("Using GPU")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print("Using CPU")

idx = pd.IndexSlice
np.random.seed(seed=42)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
pd.options.display.float_format = "{:,.2f}".format


results_path = Path("../data/ch18")
mnist_path = results_path / "mnist"
if not mnist_path.exists():
    mnist_path.mkdir(parents=True)

if __name__ == "__main__":
    ## Load MNIST Database
    # The original MNIST dataset contains 60,000 images in 28x28 pixel resolution with a single grayscale containing
    # handwritten digits from 0 to 9. A good alternative is the more challenging but structurally similar Fashion MNIST
    # dataset that we encountered in Chapter 12 on Unsupervised Learning.
    # We can load it in keras out of the box:

    # use Keras to import pre-shuffled MNIST database
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print("The MNIST database has a training set of %d examples." % len(X_train))
    print("The MNIST database has a test set of %d examples." % len(X_test))
    print(X_train.shape, X_test.shape)

    ## Visualize Data
    ### Visualize First 10 Training Images
    # The below figure shows the first ten images in the dataset and highlights significant variation among instances
    # of the same digit. On the right, it shows how the pixel values for an indivual image range from 0 to 255.
    fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(20, 8))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(X_train[i], cmap="gray")
        ax.axis("off")
        ax.set_title("Digit: {}".format(y_train[i]), fontsize=16)
    fig.suptitle("First 10 Digits", fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig("images/02-01.png")

    ### Show random image in detail
    fig, ax = plt.subplots(figsize=(14, 14))

    i = randint(0, len(X_train))
    img = X_train[i]

    ax.imshow(img, cmap="gray")
    ax.set_title("Digit: {}".format(y_train[i]), fontsize=16)

    width, height = img.shape
    thresh = img.max() / 2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(
                "{:2}".format(img[x][y]),
                xy=(y, x),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if img[x][y] < thresh else "black",
            )
    plt.savefig("images/02-02.png")

    ## Prepare Data
    ### Rescale pixel values
    # We rescale the pixel values to the range [0, 1] to normalize the training data and faciliate the backpropagation
    # process and convert the data to 32-bit floats that reduce memory requirements and computational cost while
    # providing sufficient precision for our use case:

    # rescale [0,255] --> [0,1]
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    ### One-Hot Label Encoding using Keras
    # Print first ten labels
    print("Integer-valued labels:")
    print(y_train[:10])

    ## Feed-Forward NN
    ### Model Architecture
    ffnn = Sequential(
        [
            Flatten(input_shape=X_train.shape[1:]),
            Dense(512, activation="relu"),
            Dropout(0.2),
            Dense(512, activation="relu"),
            Dropout(0.2),
            Dense(10, activation="softmax"),
        ]
    )
    print(ffnn.summary())

    ### Compile the Model
    ffnn.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    ### Calculate Baseline Classification Accuracy

    # evaluate test accuracy
    baseline_accuracy = ffnn.evaluate(X_test, y_test, verbose=0)[1]

    # print test accuracy
    print(f"Test accuracy: {baseline_accuracy:.2%}")

    ### Callback for model persistence
    ffn_path = mnist_path / "ffn.best.hdf5"

    checkpointer = ModelCheckpoint(filepath=ffn_path.as_posix(), verbose=1, save_best_only=True)

    ### Early Stopping Callback
    early_stopping = EarlyStopping(monitor="val_loss", patience=20)

    ### Train the Model
    epochs = 100
    batch_size = 32

    ffnn_history = ffnn.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[checkpointer, early_stopping],
        verbose=1,
        shuffle=True,
    )

    ### Plot CV Results
    pd.DataFrame(ffnn_history.history)[["accuracy", "val_accuracy"]].plot(figsize=(14, 4))
    sns.despine()
    plt.savefig("images/02-03.png")

    ### Load the Best Model
    # load the weights that yielded the best validation accuracy
    ffnn.load_weights(ffn_path.as_posix())

    ### Test Classification Accuracy
    # evaluate test accuracy
    ffnn_accuracy = ffnn.evaluate(X_test, y_test, verbose=0)[1]
    print(f"Test accuracy: {ffnn_accuracy:.2%}")

    ## LeNet5
    K.clear_session()

    ### Model Architecture
    # We can define a simplified version of LeNet5 that omits the original final layer containing radial basis
    # functions as follows, using the default ‘valid’ padding and single step strides unless defined otherwise:
    lenet5 = Sequential(
        [
            Conv2D(
                filters=6, kernel_size=5, activation="relu", input_shape=(28, 28, 1), name="CONV1"
            ),
            AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid", name="POOL1"),
            Conv2D(filters=16, kernel_size=(5, 5), activation="tanh", name="CONV2"),
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="POOL2"),
            Conv2D(filters=120, kernel_size=(5, 5), activation="tanh", name="CONV3"),
            Flatten(name="FLAT"),
            Dense(units=84, activation="tanh", name="FC6"),
            Dense(units=10, activation="softmax", name="FC7"),
        ]
    )

    # The summary indicates that the model thus defined has over 300,000 parameters:
    print(lenet5.summary())

    # We compile using crossentropy loss and the original stochastic gradient optimizer:
    lenet5.compile(loss="sparse_categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

    ### Define checkpoint callback
    lenet_path = mnist_path / "lenet.best.hdf5"
    checkpointer = ModelCheckpoint(filepath=lenet_path.as_posix(), verbose=1, save_best_only=True)

    # Now we are ready to train the model. The model expects 4D input so we reshape accordingly. We use the standard
    # batch size of 32, 80-20 train-validation split, use checkpointing to store the model weights if the validation
    # error improves, and make sure the dataset is randomly shuffled:

    ### Train Model
    batch_size = 32
    epochs = 100

    lenet_history = lenet5.fit(
        X_train.reshape(-1, 28, 28, 1),
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,  # use 0 to train on all data
        callbacks=[checkpointer, early_stopping],
        verbose=1,
        shuffle=True,
    )

    ### Plot CV Results
    # On a single GPU, 50 epochs take around 2.5 minutes, resulting in a test accuracy of 99.09%, slightly below the
    # same result as for the original LeNet5:
    pd.DataFrame(lenet_history.history)[["accuracy", "val_accuracy"]].plot(figsize=(14, 4))
    plt.savefig("images/02-04.png")

    ### Test Classification Accuracy
    # evaluate test accuracy
    lenet_accuracy = lenet5.evaluate(X_test.reshape(-1, 28, 28, 1), y_test, verbose=0)[1]
    print("Test accuracy: {:.2%}".format(lenet_accuracy))

    ## Summary
    # For comparison, a simple two-layer feedforward network achieves only 37.36% test accuracy.
    # The LeNet5 improvement on MNIST is, in fact, modest. Non-neural methods have also achieved classification
    # accuracies greater than or equal to 99%, including K-Nearest Neighbours or Support Vector Machines. CNNs really
    # shine with more challenging datasets as we will see next.
