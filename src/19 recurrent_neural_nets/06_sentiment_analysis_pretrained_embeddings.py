# Sentiment analysis with pretrained word vectors
# In Chapter 15, Word Embeddings, we discussed how to learn domain-specific word embeddings. Word2vec, and related
# learning algorithms, produce high-quality word vectors, but require large datasets. Hence, it is common that research
# groups share word vectors trained on large datasets, similar to the weights for pretrained deep learning models that
# we encountered in the section on transfer learning in the previous chapter.
# We are now going to illustrate how to use pretrained Global Vectors for Word Representation (GloVe) provided by the
# Stanford NLP group with the IMDB review dataset.

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
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
    # We are going to load the IMDB dataset from the source for manual preprocessing.
    # Data source: [Stanford IMDB Reviews Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
    # Dowload extract, and place the content in a newly created `data` folder so that your directory structure looks as:
    # 19_recurrent_neural_nets
    #  |-data
    #      |-aclimdb
    #           |-train
    #               |-neg
    #               |-pos
    #               ...
    #           |-test
    #           |-imdb.vocab

    path = Path("..", "data", "aclImdb")

    files = path.glob("*/**/*.txt")
    print("files :", len(list(files)))

    # data = []
    # for f in files:
    #     if f.stem.startswith(("urls_", "imdbEr")):
    #         continue
    #     aa = f.parent.as_posix().split("/")
    #     _, _, _, data_set, outcome = f.parent.as_posix().split("/")
    #     if outcome == "unsup":
    #         continue
    #     data.append([data_set, int(outcome == "pos"), f.read_text(encoding="latin1")])
    # data = pd.DataFrame(data, columns=["dataset", "label", "review"])
    # data.to_pickle(results_path / "aclimdb_data.pkl")

    data = pd.read_pickle(results_path / "aclimdb_data.pkl")
    data.info()

    train_data = data.loc[data.dataset == "train", ["label", "review"]]
    test_data = data.loc[data.dataset == "test", ["label", "review"]]

    print(train_data.label.value_counts())
    print(test_data.label.value_counts())

    ## Prepare Data
    ### Tokenizer
    # Keras provides a tokenizer that we use to convert the text documents to integer-encoded sequences, as shown here:
    num_words = 10000
    t = Tokenizer(num_words=num_words, lower=True, oov_token=2)
    t.fit_on_texts(train_data.review)

    vocab_size = len(t.word_index) + 1
    print(vocab_size)

    train_data_encoded = t.texts_to_sequences(train_data.review)
    test_data_encoded = t.texts_to_sequences(test_data.review)

    max_length = 100

    ### Pad Sequences
    # We also use the pad_sequences function to convert the list of lists (of unequal length) to stacked sets of padded
    # and truncated arrays for both the train and test datasets:
    X_train_padded = pad_sequences(
        train_data_encoded, maxlen=max_length, padding="post", truncating="post"
    )
    y_train = train_data["label"]
    print(X_train_padded.shape)

    X_test_padded = pad_sequences(
        test_data_encoded, maxlen=max_length, padding="post", truncating="post"
    )
    y_test = test_data["label"]
    print(X_test_padded.shape)

    ## Load Embeddings
    # Assuming we have downloaded and unzipped the GloVe data to the location indicated in the code, we now create a
    # dictionary that maps GloVe tokens to 100-dimensional real-valued vectors, as follows:

    # load the whole embedding into memory
    glove_path = Path("..", "data", "glove", "glove.6B.100d.txt")
    embeddings_index = dict()

    for line in glove_path.open(encoding="latin1"):
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype="float32")
        except:
            continue
        embeddings_index[word] = coefs
    print("Loaded {:,d} word vectors.".format(len(embeddings_index)))

    # There are around 340,000 word vectors that we use to create an embedding matrix that matches the vocabulary so
    # that the RNN model can access embeddings by the token index:
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print(embedding_matrix.shape)

    ## Define Model Architecture
    # The difference between this and the RNN setup in the previous example is that we are going to pass the embedding
    # matrix to the embedding layer and set it to non-trainable, so that the weights remain fixed during training:
    embedding_size = 100

    rnn = Sequential(
        [
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_size,
                input_length=max_length,
                weights=[embedding_matrix],
                trainable=False,
            ),
            GRU(units=32, dropout=0.2, recurrent_dropout=0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    rnn.summary()

    rnn.compile(
        loss="binary_crossentropy",
        optimizer="RMSProp",
        metrics=["accuracy", tf.keras.metrics.AUC(name="AUC")],
    )

    rnn_path = (results_path / "lstm.pretrained.h5").as_posix()
    checkpointer = ModelCheckpoint(
        filepath=rnn_path, verbose=1, monitor="val_AUC", mode="max", save_best_only=True
    )

    early_stopping = EarlyStopping(
        monitor="val_AUC", patience=5, mode="max", restore_best_weights=True
    )

    training = rnn.fit(
        X_train_padded,
        y_train,
        batch_size=256,
        epochs=100,
        validation_data=(X_test_padded, y_test),
        callbacks=[early_stopping, checkpointer],
        verbose=1,
    )

    y_score = rnn.predict(X_test_padded)
    print(roc_auc_score(y_score=y_score.squeeze(), y_true=y_test))

    df = pd.DataFrame(training.history)
    best_auc = df.val_AUC.max()
    best_acc = df.val_accuracy.max()

    fig, axes = plt.subplots(ncols=2, figsize=(14, 4))
    df.index = df.index.to_series().add(1)
    df[["AUC", "val_AUC"]].plot(
        ax=axes[0],
        title=f"AUC | Best: {best_auc:.4f}",
        legend=False,
        xlim=(1, 33),
        ylim=(0.7, 0.95),
    )

    axes[0].axvline(df.val_AUC.idxmax(), ls="--", lw=1, c="k")
    df[["accuracy", "val_accuracy"]].plot(
        ax=axes[1],
        title=f"Accuracy | Best: {best_acc:.2%}",
        legend=False,
        xlim=(1, 33),
        ylim=(0.7, 0.9),
    )
    axes[1].axvline(df.val_accuracy.idxmax(), ls="--", lw=1, c="k")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    fig.suptitle("Sentiment Analysis - Pretrained Vectors", fontsize=14)
    fig.legend(["Train", "Validation"], loc="center right")

    sns.despine()
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig("images/06_imdb_pretrained.png", dpi=300)
