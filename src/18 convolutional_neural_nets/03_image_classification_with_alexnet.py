# CIFAR10 Image Classification
# Fast-forward to 2012, and we move on to the deeper and more modern AlexNet architecture. We will use the CIFAR10
# dataset that uses 60,000 ImageNet samples, compressed to 32x32 pixel resolution (from the original 224x224), but still
# with three color channels. There are only 10 of the original 1,000 classes.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from pathlib import Path

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


results_path = Path("../data/ch18", "cifar10")
mnist_path = results_path / "mnist"
if not mnist_path.exists():
    mnist_path.mkdir(parents=True)

if __name__ == "__main__":
    ## Load CIFAR-10 Data
    # CIFAR10 can also be downloaded from keras, and we similarly rescale the pixel values and one-hot encode
    # the ten class labels.
    # load the pre-shuffled train and test data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    ### Visualize the First 30 Training Images
    cifar10_labels = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }
    num_classes = len(cifar10_labels)

    height, width, channels = X_train.shape[1:]
    input_shape = height, width, channels
    print(input_shape)

    fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(20, 5))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(np.squeeze(X_train[i]))
        ax.axis("off")
        ax.set_title(cifar10_labels[y_train[i, 0]])
    plt.savefig("images/03-01.png")

    ### Rescale the Images
    # rescale [0,255] --> [0,1]
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    ### Train-Test split
    X_train, X_valid = X_train[5000:], X_train[:5000]
    y_train, y_valid = y_train[5000:], y_train[:5000]

    # shape of training set
    print(X_train.shape)
    print(X_train.shape[0], "train samples")
    print(X_test.shape[0], "test samples")
    print(X_valid.shape[0], "validation samples")

    ## Feedforward Neural Network
    # We first train a two-layer feedforward network on 50,000 training samples for training for 20 epochs to achieve
    # a test accuracy of 44.22%. We also experiment with a three-layer convolutional net with 500K parameters
    # for 67.07% test accuracy.

    ### Model Architecture
    mlp = Sequential(
        [
            Flatten(input_shape=input_shape, name="input"),
            Dense(1000, activation="relu", name="hidden_layer_1"),
            Dropout(0.2, name="droput_1"),
            Dense(512, activation="relu", name="hidden_layer_2"),
            Dropout(0.2, name="dropout_2"),
            Dense(num_classes, activation="softmax", name="output"),
        ]
    )
    print(mlp.summary())

    ### Compile the Model
    mlp.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    ### Define Callbacks
    mlp_path = (results_path / "mlp.weights.best.hdf5").as_posix()

    checkpointer = ModelCheckpoint(
        filepath=mlp_path, verbose=1, monitor="val_accuracy", save_best_only=True
    )

    tensorboard = TensorBoard(
        log_dir=results_path / "logs" / "mlp",
        histogram_freq=1,
        write_graph=True,
        write_grads=False,
        update_freq="epoch",
    )

    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10)

    ### Train the Model
    batch_size = 32
    epochs = 100

    mlp_history = mlp.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_valid, y_valid),
        callbacks=[checkpointer, tensorboard, early_stopping],
        verbose=1,
        shuffle=True,
    )

    ### Plot CV Results
    pd.DataFrame(mlp_history.history)[["accuracy", "val_accuracy"]].plot(figsize=(14, 4))
    sns.despine()
    plt.savefig("images/03-02.png")

    ### Load best model
    # load the weights that yielded the best validation accuracy
    mlp.load_weights(mlp_path)

    ### Test Classification Accuracy
    # evaluate and print test accuracy
    mlp_accuracy = mlp.evaluate(X_test, y_test, verbose=0)[1]
    print("Test accuracy: {:.2%}".format(mlp_accuracy))

    ## Convolutional Neural Network
    # https://stackoverflow.com/questions/35114376/error-when-computing-summaries-in-tensorflow/35117760#35117760
    K.clear_session()

    ### Model Architecture
    cnn = Sequential(
        [
            Conv2D(
                filters=16,
                kernel_size=2,
                padding="same",
                activation="relu",
                input_shape=input_shape,
                name="CONV1",
            ),
            MaxPooling2D(pool_size=2, name="POOL1"),
            Conv2D(filters=32, kernel_size=2, padding="same", activation="relu", name="CONV2"),
            MaxPooling2D(pool_size=2, name="POOL2"),
            Conv2D(filters=64, kernel_size=2, padding="same", activation="relu", name="CONV3"),
            MaxPooling2D(pool_size=2, name="POOL3"),
            Dropout(0.3, name="DROP1"),
            Flatten(name="FLAT1"),
            Dense(500, activation="relu", name="FC1"),
            Dropout(0.4, name="DROP2"),
            Dense(10, activation="softmax", name="FC2"),
        ]
    )
    print(cnn.summary())

    ### Compile the Model
    cnn.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    ### Define Callbacks
    cnn_path = (results_path / "cnn.weights.best.hdf5").as_posix()

    checkpointer = ModelCheckpoint(
        filepath=cnn_path, verbose=1, monitor="val_accuracy", save_best_only=True
    )

    tensorboard = TensorBoard(
        log_dir=results_path / "logs" / "cnn",
        histogram_freq=1,
        write_graph=True,
        write_grads=False,
        update_freq="epoch",
    )

    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10)

    ### Train the Model
    batch_size = 32
    epochs = 100

    cnn_history = cnn.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_valid, y_valid),
        callbacks=[checkpointer, tensorboard, early_stopping],
        verbose=2,
        shuffle=True,
    )

    ### Plot CV Results
    pd.DataFrame(cnn_history.history)[["accuracy", "val_accuracy"]].plot(figsize=(14, 4))
    sns.despine()
    plt.savefig("images/03-03.png")

    ### Load best model
    cnn.load_weights(cnn_path)

    ### Test set accuracy
    cnn_accuracy = cnn.evaluate(X_test, y_test, verbose=0)[1]
    print("Accuracy: {:.2%}".format(cnn_accuracy))

    ### Evaluate Predictions
    y_hat = cnn.predict(X_test)

    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(20, 8))
    axes = axes.flatten()
    images = np.random.choice(X_test.shape[0], size=32, replace=False)
    for i, (ax, idx) in enumerate(zip(axes, images)):
        ax.imshow(np.squeeze(X_test[idx]))
        ax.axis("off")
        pred_idx, true_idx = np.argmax(y_hat[idx]), np.argmax(y_test[idx])
        if pred_idx == true_idx:
            ax.set_title("{} (✓)".format(cifar10_labels[pred_idx]), color="green")
        else:
            ax.set_title(
                "{} ({})".format(cifar10_labels[pred_idx], cifar10_labels[true_idx]), color="red"
            )
    plt.savefig("images/03-04.png")

    ## CNN with Image Augmentation
    # A common trick to enhance performance is to artificially increase the size of the training set by creating synthetic
    # data. This involves randomly shifting or horizontally flipping the image, or introducing noise into the image.

    ### Create and configure augmented image generator
    # Keras includes an ImageDataGenerator for this purpose that we can configure and fit to the training data as follows:
    datagen = ImageDataGenerator(
        width_shift_range=0.1,  # randomly horizontal shift
        height_shift_range=0.1,  # randomly vertial shift
        horizontal_flip=True,
    )  # randomly horizontalflip

    # fit augmented image generator on data
    datagen.fit(X_train)

    ### Visualize subset of training data
    # The result shows how the augmented images have been altered in various ways as expected:
    n_images = 6
    x_train_subset = X_train[:n_images]

    # original images
    fig, axes = plt.subplots(nrows=1, ncols=n_images, figsize=(20, 4))
    for i, (ax, img) in enumerate(zip(axes, x_train_subset)):
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle("Subset of Original Training Images", fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig("images/03_original_images.png")

    # augmented images
    fig, axes = plt.subplots(nrows=1, ncols=n_images, figsize=(20, 4))
    for x_batch in datagen.flow(x_train_subset, batch_size=n_images, shuffle=False):
        for i, ax in enumerate(axes):
            ax.imshow(x_batch[i])
            ax.axis("off")
        #     fig.suptitle('Augmented Images', fontsize=20)
        break
    fig.suptitle("Augmented Images", fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig("images/03_augmented_images.png")

    ### Define Callbacks
    K.clear_session()

    cnn_aug_path = (results_path / "augmented.cnn.weights.best.hdf5").as_posix()

    checkpointer = ModelCheckpoint(
        filepath=cnn_aug_path, verbose=1, monitor="val_accuracy", save_best_only=True
    )

    tensorboard = TensorBoard(
        log_dir=results_path / "logs" / "cnn_aug",
        histogram_freq=1,
        write_graph=True,
        write_grads=False,
        update_freq="epoch",
    )

    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10)

    ### Train Augmented Images
    batch_size = 32
    epochs = 100

    cnn_aug_history = cnn.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=X_train.shape[0] // batch_size,
        epochs=epochs,
        validation_data=(X_valid, y_valid),
        callbacks=[checkpointer, tensorboard, early_stopping],
        verbose=2,
    )

    ### Plot CV Result
    pd.DataFrame(cnn_aug_history.history)[["accuracy", "val_accuracy"]].plot(figsize=(14, 4))
    sns.despine()
    plt.savefig("images/03-05.png")

    ### Load best model
    cnn.load_weights(cnn_aug_path)

    ### Test set accuracy
    # The test accuracy for the three-layer CNN improves markedly to 74.79% after training on the larger, augmented data.
    cnn_aug_accuracy = cnn.evaluate(X_test, y_test, verbose=0)[1]
    print("Test Accuracy: {:.2%}".format(cnn_aug_accuracy))

    ## AlexNet
    # We also need to simplify the AlexNet architecture in response to the lower dimensionality of CIFAR10 images
    # relative to the ImageNet samples used in the competition. We use the original number of filters but make them
    # smaller (see notebook for implementation). The summary shows the five convolutional layers followed by two
    # fully-connected layers with frequent use of batch normalization, for a total of 21.5 million parameters:

    ### Define Architecture
    K.clear_session()

    alexnet = Sequential(
        [
            # 1st Convolutional Layer
            Conv2D(
                96,
                (3, 3),
                strides=(2, 2),
                activation="relu",
                padding="same",
                input_shape=input_shape,
                name="CONV_1",
            ),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="POOL_1"),
            BatchNormalization(name="NORM_1"),
            # 2nd Convolutional Layer
            Conv2D(
                filters=256, kernel_size=(5, 5), padding="same", activation="relu", name="CONV2"
            ),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="POOL2"),
            BatchNormalization(name="NORM_2"),
            # 3rd Convolutional Layer
            Conv2D(
                filters=384, kernel_size=(3, 3), padding="same", activation="relu", name="CONV3"
            ),
            # 4th Convolutional Layer
            Conv2D(
                filters=384, kernel_size=(3, 3), padding="same", activation="relu", name="CONV4"
            ),
            # 5th Convolutional Layer
            Conv2D(
                filters=256, kernel_size=(3, 3), padding="same", activation="relu", name="CONV5"
            ),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="POOL5"),
            BatchNormalization(name="NORM_5"),
            # Fully Connected Layers
            Flatten(name="FLAT"),
            Dense(4096, input_shape=(32 * 32 * 3,), activation="relu", name="FC1"),
            Dropout(0.4, name="DROP1"),
            Dense(4096, activation="relu", name="FC2"),
            Dropout(0.4, name="DROP2"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    print(alexnet.summary())

    ### Compile Model
    alexnet.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    ### Define Callbacks
    alexnet_path = (results_path / "alexnet.weights.best.hdf5").as_posix()

    checkpointer = ModelCheckpoint(
        filepath=alexnet_path, verbose=1, monitor="val_accuracy", save_best_only=True
    )

    tensorboard = TensorBoard(
        log_dir=results_path / "logs" / "alexnet",
        histogram_freq=1,
        write_graph=True,
        write_grads=False,
        update_freq="epoch",
    )

    early_stopping = EarlyStopping(monitor="val_accuracy", mode="max", patience=10)

    ### Train Model
    batch_size = 32
    epochs = 100

    alex_history = alexnet.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_valid, y_valid),
        callbacks=[checkpointer, tensorboard, early_stopping],
        verbose=1,
    )

    pd.DataFrame(alex_history.history)[["accuracy", "val_accuracy"]].plot(figsize=(14, 5))
    sns.despine()
    plt.savefig("images/03-06.png")

    alexnet.load_weights(alexnet_path)

    # After training for 20 episodes, each of which takes a little under 30 seconds on a single GPU, we obtain 76.84% test accuracy.
    alex_accuracy = alexnet.evaluate(X_test, y_test, verbose=0)[1]
    print("Test Accuracy: {:.2%}".format(alex_accuracy))

    ## Compare Results
    cv_results = pd.DataFrame(
        {
            "Feed-Forward NN": pd.Series(mlp_history.history["val_accuracy"]),
            "CNN": pd.Series(cnn_history.history["val_accuracy"]),
            "CNN Aug.": pd.Series(cnn_aug_history.history["val_accuracy"]),
            "Alex Net": pd.Series(alex_history.history["val_accuracy"]),
        }
    )

    test_accuracy = pd.Series(
        {
            "Feed-Forward NN": mlp_accuracy,
            "CNN": cnn_accuracy,
            "CNN Aug.": cnn_aug_accuracy,
            "Alex Net": alex_accuracy,
        }
    )

    fig, axes = plt.subplots(ncols=2, figsize=(14, 4))
    cv_results.plot(ax=axes[0], title="CV Validation Performance")
    test_accuracy.plot.barh(ax=axes[1], xlim=(0.3, 0.8), title="Test Accuracy")
    fig.tight_layout()
    sns.despine()
    fig.savefig("images/03_comparison.jpg", dpi=300)

    ## TensorBoard visualization
    # get_ipython().run_line_magic("load_ext", "tensorboard")
    # get_ipython().run_line_magic("tensorboard", "--logdir results/cifar10/logs")
