# Designing and training autoencoders using Python
# In this notebook, we illustrate how to implement several of the autoencoder models introduced in the preceding
# section using Keras. We first load and prepare an image dataset that we use throughout this section because it makes
# it easier to visualize the results of the encoding process.
# We then proceed to build autoencoders using deep feedforward nets, sparsity constraints, and convolutions and then
# apply the latter to denoise images.
# Source: https://blog.keras.io/building-autoencoders-in-keras.html

from pathlib import Path

import numpy as np
from numpy.random import choice
from numpy.linalg import norm
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import fashion_mnist

from sklearn.preprocessing import minmax_scale
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, cdist

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings

idx = pd.IndexSlice
np.random.seed(seed=42)
tf.random.set_seed(seed=42)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
pd.options.display.float_format = "{:,.2f}".format
warnings.filterwarnings("ignore")

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if gpu_devices:
    print("Using GPU")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print("Using CPU")

results_path = Path("../data/ch20", "fashion_mnist")
if not results_path.exists():
    results_path.mkdir(parents=True)


if __name__ == "__main__":
    n_classes = 10  # all examples have 10 classes
    cmap = sns.color_palette("Paired", n_classes)

    ## Fashion MNIST Data
    # For illustration, we'll use the Fashion MNIST dataset, a modern drop-in replacement for the classic MNIST
    # handwritten digit dataset popularized by Yann LeCun with LeNet in the 1990s. We also relied on this dataset
    # in Chapter 12, Unsupervised Learning.
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Keras makes it easy to access the 60,000 train and 10,000 test grayscale samples with a resolution of 28 x 28 pixels:
    print(X_train.shape, X_test.shape)

    image_size = 28  # size of image (pixels per side)
    input_size = image_size ** 2  # Compression factor: 784 / 32 = 24.5

    class_dict = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }
    classes = list(class_dict.keys())

    ### Plot sample images
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(14, 5))
    axes = axes.flatten()
    for row, label in enumerate(classes):
        label_idx = np.argwhere(y_train == label).squeeze()
        axes[row].imshow(X_train[choice(label_idx)], cmap="gray")
        axes[row].axis("off")
        axes[row].set_title(class_dict[row])

    fig.suptitle("Fashion MNIST Samples", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.savefig("images/01-01.png")

    n_samples = 15
    fig, axes = plt.subplots(nrows=n_classes, figsize=(15, 15))
    axes = axes.flatten()
    for row, label in enumerate(classes):
        class_imgs = np.empty(shape=(image_size, n_samples * image_size))
        label_idx = np.argwhere(y_train == label).squeeze()
        class_samples = choice(label_idx, size=n_samples, replace=False)
        for col, sample in enumerate(class_samples):
            i = col * image_size
            class_imgs[:, i : i + image_size] = X_train[sample]
        axes[row].imshow(class_imgs, cmap="gray")
        axes[row].axis("off")
        axes[row].set_title(class_dict[row])

    fig.suptitle("Fashion MNIST Samples", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0)
    plt.savefig("images/01-02.png")

    ## Reshape & normalize Fashion MNIST data
    # We reshape the data so that each image is represented by a flat one-dimensional pixel vector with 28 x 28 = 784
    # elements normalized to the range of [0, 1]:
    encoding_size = 32  # Size of encoding

    def data_prep(x, size=input_size):
        return x.reshape(-1, size).astype("float32") / 255

    X_train_scaled = data_prep(X_train)
    X_test_scaled = data_prep(X_test)
    print(X_train_scaled.shape, X_test_scaled.shape)

    ## Vanilla single-layer autoencoder
    # We start with a vanilla feedforward autoencoder with a single hidden layer to illustrate the general design
    # approach using the functional Keras API and establish a performance baseline.
    # Encoding 28 x 28 images to a 32 value representation for a compression factor of 24.5

    ### Single-layer Model
    #### Input Layer
    input_ = Input(shape=(input_size,), name="Input")

    #### Dense Encoding Layer
    # The encoder part of the model consists of a fully-connected layer that learns the new, compressed representation
    # of the input. We use 32 units for a compression ratio of 24.5:
    encoding = Dense(units=encoding_size, activation="relu", name="Encoder")(input_)

    #### Dense Reconstruction Layer
    # The decoding part reconstructs the compressed data to its original size in a single step:
    decoding = Dense(units=input_size, activation="sigmoid", name="Decoder")(encoding)

    #### Autoencoder Model
    autoencoder = Model(inputs=input_, outputs=decoding, name="Autoencoder")

    # The thus defined encoder-decoder computation uses almost 51,000 parameters:
    autoencoder.summary()

    ### Encoder Model
    # The functional API allows us to use parts of the model's chain as separate encoder and decoder models that use
    # the autoencoder's parameters learned during training.
    # The encoder just uses the input and hidden layer with about half of the total parameters:
    encoder = Model(inputs=input_, outputs=encoding, name="Encoder")
    encoder.summary()

    # Once we train the autoencoder, we can use the encoder to compress the data.

    ### Decoder Model
    # The decoder consists of the last autoencoder layer, fed by a placeholder for the encoded data:

    #### Placeholder for encoded input
    encoded_input = Input(shape=(encoding_size,), name="Decoder_Input")

    #### Extract last autoencoder layer
    decoder_layer = autoencoder.layers[-1](encoded_input)

    #### Define Decoder Model
    decoder = Model(inputs=encoded_input, outputs=decoder_layer)
    decoder.summary()

    ### Compile the Autoencoder Model
    autoencoder.compile(optimizer="adam", loss="mse")

    ### Train the autoencoder
    # We compile the model to use the Adam optimizer (see Chapter 17, Deep Learning) to minimize the MSE between
    # the input data and the reproduction achieved by the autoencoder. To ensure that the autoencoder learns to
    # reproduce the input, we train the model using the same input and output data:

    #### Create `early_stopping` callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-5,
        patience=5,
        verbose=0,
        restore_best_weights=True,
        mode="auto",
    )

    #### Create TensorBard callback to visualize network performance
    tb_callback = TensorBoard(
        log_dir=results_path / "logs", histogram_freq=5, write_graph=True, write_images=True
    )

    #### Create checkpoint callback
    filepath = (results_path / "autencoder.32.weights.hdf5").as_posix()
    checkpointer = ModelCheckpoint(
        filepath=filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
    )

    #### Fit the Model
    # To avoid running time, you can load the pre-computed results in the 'model' folder (see below)
    training = autoencoder.fit(
        x=X_train_scaled,
        y=X_train_scaled,
        epochs=100,
        batch_size=32,
        shuffle=True,
        validation_split=0.1,
        callbacks=[tb_callback, early_stopping, checkpointer],
    )

    ### Reload weights from best-performing model
    autoencoder.load_weights(filepath)

    ### Evaluate trained model
    # Training stops after some 20 epochs with a test RMSE of 0.1122:
    mse = autoencoder.evaluate(x=X_test_scaled, y=X_test_scaled)
    print(f"MSE: {mse:.4f} | RMSE {mse**.5:.4f}")

    ### Encode and decode test images
    # To encode data, we use the encoder we just defined, like so:
    encoded_test_img = encoder.predict(X_test_scaled)
    print(encoded_test_img.shape)

    # The decoder takes the compressed data and reproduces the output according to the autoencoder training results:
    decoded_test_img = decoder.predict(encoded_test_img)
    print(decoded_test_img.shape)

    #### Compare Original with Reconstructed Samples
    # The following figure shows ten original images and their reconstruction by the autoencoder and illustrates
    # the loss after compression:
    fig, axes = plt.subplots(ncols=n_classes, nrows=2, figsize=(20, 4))
    for i in range(n_classes):
        axes[0, i].imshow(X_test_scaled[i].reshape(image_size, image_size), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(decoded_test_img[i].reshape(28, 28), cmap="gray")
        axes[1, i].axis("off")
    fig.suptitle("Original and Reconstructed Images", fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig("images/01_reconstructed.png", dpi=300)

    ## Combine training steps into function
    # The helper function `train_autoencoder` just summarizes some repetitive steps.
    def train_autoencoder(path, model, x_train=X_train_scaled, x_test=X_test_scaled):
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(filepath=path, save_best_only=True, save_weights_only=True),
        ]
        model.fit(x=x_train, y=x_train, epochs=100, validation_split=0.1, callbacks=callbacks)
        model.load_weights(path)
        mse = model.evaluate(x=x_test, y=x_test)
        return model, mse

    ## Autoencoders with Sparsity Constraints
    ### Encoding Layer with L1 activity regularizer
    # The addition of regularization is fairly straightforward. We can apply it to the dense encoder layer using Keras'
    # `activity_regularizer`, as follows:
    encoding_l1 = Dense(
        units=encoding_size,
        activation="relu",
        activity_regularizer=regularizers.l1(10e-5),
        name="Encoder_L1",
    )(input_)

    ### Decoding Layer
    decoding_l1 = Dense(units=input_size, activation="sigmoid", name="Decoder_L1")(encoding_l1)
    autoencoder_l1 = Model(input_, decoding_l1)

    ### Autoencoder Model
    autoencoder_l1.summary()

    autoencoder_l1.compile(optimizer="adam", loss="mse")

    ### Encoder & Decoder Models
    encoder_l1 = Model(inputs=input_, outputs=encoding_l1, name="Encoder")
    encoded_input = Input(shape=(encoding_size,), name="Decoder_Input")
    decoder_l1_layer = autoencoder_l1.layers[-1](encoded_input)
    decoder_l1 = Model(inputs=encoded_input, outputs=decoder_l1_layer)

    ### Train Model
    path = (results_path / "autencoder_l1.32.weights.hdf5").as_posix()
    autoencoder_l1, mse = train_autoencoder(path, autoencoder_l1)

    ### Evaluate Model
    # The input and decoding layers remain unchanged. In this example, with a compression of factor 24.5,
    # regularization negatively affects performance with a test RMSE of 0.0.1229.
    print(f"MSE: {mse:.4f} | RMSE {mse**.5:.4f}")

    encoded_test_img = encoder_l1.predict(X_test_scaled)

    fig, axes = plt.subplots(ncols=n_classes, nrows=2, figsize=(20, 4))
    for i in range(n_classes):
        axes[0, i].imshow(X_test_scaled[i].reshape(image_size, image_size), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(decoded_test_img[i].reshape(28, 28), cmap="gray")
        axes[1, i].axis("off")
    plt.savefig("images/01-03.png")

    ##  Deep Autoencoder
    # To illustrate the benefit of adding depth to the autoencoder, we build a three-layer feedforward model that
    # successively compresses the input from 784 to 128, 64, and 34 units, respectively:

    ### Define three-layer architecture
    input_ = Input(shape=(input_size,))
    x = Dense(128, activation="relu", name="Encoding1")(input_)
    x = Dense(64, activation="relu", name="Encoding2")(x)
    encoding_deep = Dense(32, activation="relu", name="Encoding3")(x)

    x = Dense(64, activation="relu", name="Decoding1")(encoding_deep)
    x = Dense(128, activation="relu", name="Decoding2")(x)
    decoding_deep = Dense(input_size, activation="sigmoid", name="Decoding3")(x)

    autoencoder_deep = Model(input_, decoding_deep)
    autoencoder_deep.compile(optimizer="adam", loss="mse")

    # The resulting model has over 222,000 parameters, more than four times the capacity of the preceding single-layer
    # model:
    autoencoder_deep.summary()

    ### Encoder & Decoder Models
    encoder_deep = Model(inputs=input_, outputs=encoding_deep, name="Encoder")
    encoded_input = Input(shape=(encoding_size,), name="Decoder_Input")

    x = autoencoder_deep.layers[-3](encoded_input)
    x = autoencoder_deep.layers[-2](x)
    decoded = autoencoder_deep.layers[-1](x)
    decoder_deep = Model(inputs=encoded_input, outputs=decoded)
    decoder_deep.summary()

    ### Train Model
    path = (results_path / "autencoder_deep.32.weights.hdf5").as_posix()
    autoencoder_deep, mse = train_autoencoder(path, autoencoder_deep)
    autoencoder_deep.load_weights(path)

    ### Evaluate Model
    # Training stops after 54 epochs and results in a ~10% reduction of the test RMSE to 0.1026. Due to the low
    # resolution, it is difficult to visually note the better reconstruction.
    print(f"MSE: {mse:.4f} | RMSE {mse**.5:.4f}")

    reconstructed_images = autoencoder_deep.predict(X_test_scaled)
    print(reconstructed_images.shape)

    fig, axes = plt.subplots(ncols=n_classes, nrows=2, figsize=(20, 4))
    for i in range(n_classes):
        axes[0, i].imshow(X_test_scaled[i].reshape(image_size, image_size), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed_images[i].reshape(image_size, image_size), cmap="gray")
        axes[1, i].axis("off")
    plt.savefig("images/01-04.png")

    ### Compute t-SNE Embedding
    # We can use the t-distributed Stochastic Neighbor Embedding (t-SNE) manifold learning technique, see Chapter 12,
    # Unsupervised Learning, to visualize and assess the quality of the encoding learned by the autoencoder's hidden
    # layer. If the encoding is successful in capturing the salient features of the data, the compressed representation
    # of the data should still reveal a structure aligned with the 10 classes that differentiate the observations.
    # We use the output of the deep encoder we just trained to obtain the 32-dimensional representation of the test set:
    # Since t-SNE can take a long time to run (~15-20 min), we are providing pre-computed results

    # alternatively, compute the result yourself
    tsne = TSNE(perplexity=25, n_iter=5000)
    train_embed = tsne.fit_transform(encoder_deep.predict(X_train_scaled))

    #### Persist result
    # store results given computational intensity (different location to avoid overwriting the pre-computed results)
    pd.DataFrame(train_embed).to_hdf(results_path / "tsne.h5", "autoencoder_deep")

    #### Load pre-computed embeddings
    # Load the pre-computed results here:
    train_embed = pd.read_hdf(results_path / "tsne.h5", "autoencoder_deep")

    #### Visualize Embedding
    def plot_embedding(X, y=y_train, title=None, min_dist=0.1, n_classes=10, cmap=cmap):
        X = minmax_scale(X)
        inner = outer = 0
        for c in range(n_classes):
            inner += np.mean(pdist(X[y == c]))
            outer += np.mean(cdist(X[y == c], X[y != c]))
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis("off")
        ax.set_title(title + " | Distance: {:.2%}".format(inner / outer))
        sc = ax.scatter(*X.T, c=y, cmap=ListedColormap(cmap), s=5)
        shown_images = np.ones((1, 2))
        images = X_train.reshape(-1, 28, 28)
        for i in range(0, X.shape[0]):
            dist = norm(X[i] - shown_images, axis=1)
            if (dist > min_dist).all():
                shown_images = np.r_[shown_images, [X[i]]]
                imagebox = AnnotationBbox(OffsetImage(images[i], cmap=plt.cm.gray_r), X[i])
                ax.add_artist(imagebox)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        plt.colorbar(sc, cax=cax)
        fig.tight_layout()
        fig.savefig("images/01_tsne_autoencoder_deep.png", dpi=300)

    # The following figure shows that t-SNE manages to separate the 10 classes well, suggesting that the encoding is
    # useful as a lower-dimensional representation that preserves key characteristics of the data:
    plot_embedding(X=train_embed, title="t-SNE & Deep Autoencoder")
