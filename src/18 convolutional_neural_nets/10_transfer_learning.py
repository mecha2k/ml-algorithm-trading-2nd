# How to further train a pre-trained model
# We will demonstrate how to freeze some or all of the layers of a pre-trained model and continue training using a new
# fully-connected set of layers and data with a different format.
# Adapted from the Tensorflow 2.0 [transfer learning tutorial]
# (https://www.tensorflow.org/tutorials/images/transfer_learning).

from sklearn.datasets import load_files
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

import tensorflow_datasets as tfds


gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if gpu_devices:
    print("Using GPU")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print("Using CPU")

results_path = Path("../data/ch18", "transfer_learning")
if not results_path.exists():
    results_path.mkdir(parents=True)

idx = pd.IndexSlice
np.random.seed(seed=42)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
pd.options.display.float_format = "{:,.2f}".format

if __name__ == "__main__":
    ## Load TensorFlow Cats vs Dog Dataset
    # TensorFlow includes a large number of built-in dataset:
    tfds.list_builders()

    # We will use a set of cats and dog images for binary classification.
    (raw_train, raw_validation, raw_test), metadata = tfds.load(
        "cats_vs_dogs",
        split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
        with_info=True,
        as_supervised=True,
        data_dir="../data/tensorflow",
    )
    print("Raw train:\t", raw_train)
    print("Raw validation:\t", raw_validation)
    print("Raw test:\t", raw_test)

    ### Show sample images
    get_label_name = metadata.features["label"].int2str
    for image, label in raw_train.take(2):
        plt.figure()
        plt.imshow(image)
        plt.title(get_label_name(label))
        plt.grid(False)
        plt.axis("off")
    plt.savefig("images/10-01.png")

    ## Preprocessing
    # All images will be resized to 160x160:
    IMG_SIZE = 160
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

    def format_example(image, label):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        return image, label

    train = raw_train.map(format_example)
    validation = raw_validation.map(format_example)
    test = raw_test.map(format_example)

    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000

    train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    validation_batches = validation.batch(BATCH_SIZE)
    test_batches = test.batch(BATCH_SIZE)

    image_batch = None
    for image_batch, label_batch in train_batches.take(1):
        pass
    print(image_batch.shape)

    ## Load the VGG-16 Bottleneck Features
    # We use the VGG16 weights, pre-trained on ImageNet with the much smaller 32 x 32 CIFAR10 data. Note that we
    # indicate the new input size upon import and set all layers to not trainable:
    vgg16 = VGG16(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    vgg16.summary()

    feature_batch = vgg16(image_batch)
    print(feature_batch.shape)

    ## Freeze model layers
    vgg16.trainable = False
    vgg16.summary()

    ## Add new layers to model
    ### Using the Sequential model API
    global_average_layer = GlobalAveragePooling2D()
    dense_layer = Dense(64, activation="relu")
    dropout = Dropout(0.5)
    prediction_layer = Dense(1, activation="sigmoid")

    seq_model = tf.keras.Sequential(
        [vgg16, global_average_layer, dense_layer, dropout, prediction_layer]
    )

    seq_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer="Adam",
        metrics=["accuracy"],
    )
    seq_model.summary()

    ### Using the Functional model API
    # We use Keras’ functional API to define the vgg16 output as input into a new set of fully-connected layers like so:

    # Adding custom Layers
    x = vgg16.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation="sigmoid")(x)

    # We define a new model in terms of inputs and output, and proceed from there on as before:
    transfer_model = Model(inputs=vgg16.input, outputs=predictions)
    transfer_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer="Adam",
        metrics=["accuracy"],
    )
    transfer_model.summary()

    ### Compute baseline metrics
    initial_epochs = 10
    validation_steps = 20

    initial_loss, initial_accuracy = transfer_model.evaluate(
        validation_batches, steps=validation_steps
    )
    print(f"Initial loss: {initial_loss:.2f} | initial_accuracy accuracy: {initial_accuracy:.2%}")

    ## Train VGG16 transfer model
    history = transfer_model.fit(
        train_batches, epochs=initial_epochs, validation_data=validation_batches
    )

    ### Plot Learning Curves
    def plot_learning_curves(df):
        fig, axes = plt.subplots(ncols=2, figsize=(15, 4))
        df[["loss", "val_loss"]].plot(ax=axes[0], title="Cross-Entropy")
        df[["accuracy", "val_accuracy"]].plot(ax=axes[1], title="Accuracy")
        for ax in axes:
            ax.legend(["Training", "Validation"])
        sns.despine()
        fig.tight_layout()

    metrics = pd.DataFrame(history.history)
    plot_learning_curves(metrics)
    plt.savefig("images/10-02.png")

    ## Fine-tune VGG16 weights
    ### Unfreeze selected layers
    vgg16.trainable = True

    # How many layers are in the base model:
    print(f"Number of layers in the base model: {len(vgg16.layers)}")

    # Fine-tune from this layer onwards
    start_fine_tuning_at = 12

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in vgg16.layers[:start_fine_tuning_at]:
        layer.trainable = False

    base_learning_rate = 0.0001
    transfer_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate / 10),
        metrics=["accuracy"],
    )

    ### Define callbacks
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10)
    transfer_model.summary()

    ### Continue Training
    # And now we proceed to train the model:
    fine_tune_epochs = 50
    total_epochs = initial_epochs + fine_tune_epochs

    history_fine_tune = transfer_model.fit(
        train_batches,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=validation_batches,
        callbacks=[early_stopping],
    )

    metrics_tuned = metrics.append(pd.DataFrame(history_fine_tune.history), ignore_index=True)

    fig, axes = plt.subplots(ncols=2, figsize=(15, 4))
    metrics_tuned[["loss", "val_loss"]].plot(ax=axes[1], title="Cross-Entropy Loss")
    metrics_tuned[["accuracy", "val_accuracy"]].plot(
        ax=axes[0], title=f"Accuracy (Best: {metrics_tuned.val_accuracy.max():.2%})"
    )
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    axes[0].set_ylabel("Accuracy")
    axes[1].set_ylabel("Loss")
    for ax in axes:
        ax.axvline(10, ls="--", lw=1, c="k")
        ax.legend(["Training", "Validation", "Start Fine Tuning"])
        ax.set_xlabel("Epoch")
    sns.despine()
    fig.tight_layout()
    fig.savefig("images/10_transfer_learning.png")
