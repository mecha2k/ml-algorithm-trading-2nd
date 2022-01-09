# LSTM & Word Embeddings for Sentiment Classification
# RNNs are commonly applied to various natural language processing tasks. We've already encountered sentiment analysis
# using text data in part three of this book.
# We are now going to illustrate how to apply an RNN model to text data to detect positive or negative sentiment
# (which can easily be extended to a finer-grained sentiment scale). We are going to use word embeddings to represent
# the tokens in the documents. We covered word embeddings in Chapter 15, Word Embeddings. They are an excellent
# technique to convert text into a continuous vector representation such that the relative location of words in the
# latent space encodes useful semantic aspects based on the words' usage in context.
# We saw in the previous RNN example that Keras has a built-in embedding layer that allows us to train vector
# representations specific to the task at hand. Alternatively, we can use pretrained vectors.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

idx = pd.IndexSlice
np.random.seed(seed=42)
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


results_path = Path("../data/ch19", "sentiment_imdb")
if not results_path.exists():
    results_path.mkdir(parents=True)


if __name__ == "__main__":
    # To keep the data manageable, we will illustrate this use case with the IMDB reviews dataset, which contains 50,000
    # positive and negative movie reviews evenly split into a train and a test set, and with balanced labels in each
    # dataset. The vocabulary consists of 88,586 tokens.
    # The dataset is bundled into Keras and can be loaded so that each review is represented as an integer-encoded
    # sequence. We can limit the vocabulary to num_words while filtering out frequent and likely less informative words
    # using skip_top, as well as sentences longer than maxlen. We can also choose oov_char, which represents tokens we
    # chose to exclude from the vocabulary on frequency grounds, as follows:
    vocab_size = 20000

    (X_train, y_train), (X_test, y_test) = imdb.load_data(
        seed=42, skip_top=0, maxlen=None, oov_char=2, index_from=3, num_words=vocab_size
    )

    ax = sns.hisplot(data=[len(review) for review in X_train])
    ax.set(xscale="log")
    plt.savefig("images/05-01.png")

    ## Prepare Data
    # In the second step, convert the lists of integers into fixed-size arrays that we can stack and provide as input
    # to our RNN. The pad_sequence function produces arrays of equal length, truncated, and padded to conform to maxlen.
    maxlen = 100

    X_train_padded = pad_sequences(X_train, truncating="pre", padding="pre", maxlen=maxlen)
    X_test_padded = pad_sequences(X_test, truncating="pre", padding="pre", maxlen=maxlen)
    print(X_train_padded.shape, X_test_padded.shape)

    ## Define Model Architecture
    # Now we can define our RNN architecture. The first layer learns the word embeddings. We define the embedding
    # dimension as previously using the input_dim keyword to set the number of tokens that we need to embed,
    # the output_dim keyword, which defines the size of each embedding, and how long each input sequence is going to be.

    K.clear_session()

    ### Custom Loss Metric
    embedding_size = 100

    # Note that we are using GRUs this time, which train faster and perform better on smaller data. We are also using
    # dropout for regularization, as follows:
    rnn = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=maxlen),
            GRU(
                units=32,
                dropout=0.2,  # comment out to use optimized GPU implementation
                recurrent_dropout=0.2,
            ),
            Dense(1, activation="sigmoid"),
        ]
    )
    rnn.summary()

    # The resulting model has over 2 million parameters.
    # We compile the model to use our custom AUC metric, which we introduced previously, and train with early stopping:
    rnn.compile(
        loss="binary_crossentropy",
        optimizer="RMSProp",
        metrics=["accuracy", tf.keras.metrics.AUC(name="AUC")],
    )

    rnn_path = (results_path / "lstm.h5").as_posix()
    checkpointer = ModelCheckpoint(
        filepath=rnn_path, verbose=1, monitor="val_AUC", mode="max", save_best_only=True
    )

    early_stopping = EarlyStopping(
        monitor="val_AUC", mode="max", patience=5, restore_best_weights=True
    )

    # Training stops after eight epochs, and we recover the weights for the best models to find a high test AUC of 0.9346:
    training = rnn.fit(
        X_train_padded,
        y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_test_padded, y_test),
        callbacks=[early_stopping, checkpointer],
        verbose=1,
    )

    ## Evaluate Results
    history = pd.DataFrame(training.history)
    history.index += 1

    fig, axes = plt.subplots(ncols=2, figsize=(14, 4))
    df1 = history[["accuracy", "val_accuracy"]].rename(
        columns={"accuracy": "Training", "val_accuracy": "Validation"}
    )
    df1.plot(ax=axes[0], title="Accuracy", xlim=(1, len(history)))

    axes[0].axvline(df1.Validation.idxmax(), ls="--", lw=1, c="k")

    df2 = history[["AUC", "val_AUC"]].rename(columns={"AUC": "Training", "val_AUC": "Validation"})
    df2.plot(ax=axes[1], title="Area under the ROC Curve", xlim=(1, len(history)))

    axes[1].axvline(df2.Validation.idxmax(), ls="--", lw=1, c="k")

    for i in [0, 1]:
        axes[i].set_xlabel("Epoch")

    sns.despine()
    fig.tight_layout()
    fig.savefig("images/05_rnn_imdb_cv.png", dpi=300)

    y_score = rnn.predict(X_test_padded)
    print(y_score.shape)

    print(roc_auc_score(y_score=y_score.squeeze(), y_true=y_test))
