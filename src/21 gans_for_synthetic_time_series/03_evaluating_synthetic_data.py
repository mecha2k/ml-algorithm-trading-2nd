import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from pathlib import Path


idx = pd.IndexSlice
np.random.seed(seed=42)
tf.random.set_seed(seed=42)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
pd.options.display.float_format = "{:,.2f}".format

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if gpu_devices:
    print("Using GPU")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print("Using CPU")


results_path = Path("../data/ch21", "time_gan")
if not results_path.exists():
    results_path.mkdir()
hdf_store = results_path / "TimeSeriesGAN.h5"

if __name__ == "__main__":
    seq_len = 24
    n_seq = 6
    experiment = 0

    def get_real_data():
        df = pd.read_hdf(hdf_store, "data/real").sort_index()

        # Preprocess the dataset:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        data = []
        for i in range(len(df) - seq_len):
            data.append(scaled_data[i : i + seq_len])
        return data

    real_data = get_real_data()
    n = len(real_data)
    print(np.asarray(real_data).shape)

    synthetic_data = np.load(results_path / f"experiment_{experiment:02d}" / "generated_data.npy")
    print(synthetic_data.shape)
    real_data = real_data[: synthetic_data.shape[0]]

    # Prepare Sample
    sample_size = 250
    idx = np.random.permutation(len(real_data))[:sample_size]

    # Data preprocessing
    real_sample = np.asarray(real_data)[idx]
    synthetic_sample = np.asarray(synthetic_data)[idx]

    real_sample_2d = real_sample.reshape(-1, seq_len)
    synthetic_sample_2d = synthetic_sample.reshape(-1, seq_len)
    print(real_sample_2d.shape, synthetic_sample_2d.shape)

    # Visualization in 2D: A Qualitative Assessment of Diversity
    ## Run PCA
    pca = PCA(n_components=2)
    pca.fit(real_sample_2d)
    pca_real = pd.DataFrame(pca.transform(real_sample_2d)).assign(Data="Real")
    pca_synthetic = pd.DataFrame(pca.transform(synthetic_sample_2d)).assign(Data="Synthetic")
    pca_result = pca_real.append(pca_synthetic).rename(
        columns={0: "1st Component", 1: "2nd Component"}
    )

    ## Run t-SNE
    tsne_data = np.concatenate((real_sample_2d, synthetic_sample_2d), axis=0)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40)
    tsne_result = tsne.fit_transform(tsne_data)

    tsne_result = pd.DataFrame(tsne_result, columns=["X", "Y"]).assign(Data="Real")
    tsne_result.loc[sample_size * 6 :, "Data"] = "Synthetic"

    ## Plot Result
    fig, axes = plt.subplots(ncols=2, figsize=(14, 5))
    sns.scatterplot(
        x="1st Component", y="2nd Component", data=pca_result, hue="Data", style="Data", ax=axes[0]
    )
    sns.despine()
    axes[0].set_title("PCA Result")

    sns.scatterplot(x="X", y="Y", data=tsne_result, hue="Data", style="Data", ax=axes[1])
    sns.despine()
    for i in [0, 1]:
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    axes[1].set_title("t-SNE Result")
    fig.suptitle(
        "Assessing Diversity: Qualitative Comparison of Real and Synthetic Data Distributions",
        fontsize=14,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig("images/03-01.png")

    # Time Series Classification: A quantitative Assessment of Fidelity
    ## Prepare Data
    real_data = get_real_data()
    real_data = np.array(real_data)[: len(synthetic_data)]
    print(real_data.shape)
    print(synthetic_data.shape)

    n_series = real_data.shape[0]
    idx = np.arange(n_series)
    n_train = int(0.8 * n_series)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    train_data = np.vstack((real_data[train_idx], synthetic_data[train_idx]))
    test_data = np.vstack((real_data[test_idx], synthetic_data[test_idx]))
    n_train, n_test = len(train_idx), len(test_idx)
    train_labels = np.concatenate((np.ones(n_train), np.zeros(n_train)))
    test_labels = np.concatenate((np.ones(n_test), np.zeros(n_test)))

    ## Create Classifier
    ts_classifier = Sequential(
        [GRU(6, input_shape=(24, 6), name="GRU"), Dense(1, activation="sigmoid", name="OUT")],
        name="Time_Series_Classifier",
    )
    ts_classifier.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=[AUC(name="AUC"), "accuracy"]
    )
    ts_classifier.summary()

    result = ts_classifier.fit(
        x=train_data,
        y=train_labels,
        validation_data=(test_data, test_labels),
        epochs=250,
        batch_size=128,
        verbose=0,
    )
    ts_classifier.evaluate(x=test_data, y=test_labels)

    history = pd.DataFrame(result.history)
    history.info()

    sns.set_style("white")
    fig, axes = plt.subplots(ncols=2, figsize=(14, 4))
    history[["AUC", "val_AUC"]].rename(columns={"AUC": "Train", "val_AUC": "Test"}).plot(
        ax=axes[1], title="ROC Area under the Curve", style=["-", "--"], xlim=(0, 250)
    )
    history[["accuracy", "val_accuracy"]].rename(
        columns={"accuracy": "Train", "val_accuracy": "Test"}
    ).plot(ax=axes[0], title="Accuracy", style=["-", "--"], xlim=(0, 250))
    for i in [0, 1]:
        axes[i].set_xlabel("Epoch")

    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].set_ylabel("AUC")
    sns.despine()
    fig.suptitle("Assessing Fidelity: Time Series Classification Performance", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.savefig("images/03-02.png")

    # Train on Synthetic, test on real: Assessing usefulness
    real_data = get_real_data()
    real_data = np.array(real_data)[: len(synthetic_data)]
    print(real_data.shape, synthetic_data.shape)

    real_train_data = real_data[train_idx, :23, :]
    real_train_label = real_data[train_idx, -1, :]

    real_test_data = real_data[test_idx, :23, :]
    real_test_label = real_data[test_idx, -1, :]
    print(
        real_train_data.shape, real_train_label.shape, real_test_data.shape, real_test_label.shape
    )

    synthetic_train = synthetic_data[:, :23, :]
    synthetic_label = synthetic_data[:, -1, :]
    print(synthetic_train.shape, synthetic_label.shape)

    def get_model():
        model = Sequential([GRU(12, input_shape=(seq_len - 1, n_seq)), Dense(6)])
        model.compile(optimizer=Adam(), loss=MeanAbsoluteError(name="MAE"))
        return model

    ts_regression = get_model()
    synthetic_result = ts_regression.fit(
        x=synthetic_train,
        y=synthetic_label,
        validation_data=(real_test_data, real_test_label),
        epochs=100,
        batch_size=128,
        verbose=0,
    )

    ts_regression = get_model()
    real_result = ts_regression.fit(
        x=real_train_data,
        y=real_train_label,
        validation_data=(real_test_data, real_test_label),
        epochs=100,
        batch_size=128,
        verbose=0,
    )

    synthetic_result = pd.DataFrame(synthetic_result.history).rename(
        columns={"loss": "Train", "val_loss": "Test"}
    )
    real_result = pd.DataFrame(real_result.history).rename(
        columns={"loss": "Train", "val_loss": "Test"}
    )

    fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharey=True)
    synthetic_result.plot(
        ax=axes[0], title="Train on Synthetic, Test on Real", logy=True, xlim=(0, 100)
    )
    real_result.plot(ax=axes[1], title="Train on Real, Test on Real", logy=True, xlim=(0, 100))
    for i in [0, 1]:
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Mean Absolute Error (log scale)")

    sns.despine()
    fig.suptitle("Assessing Usefulness: Time Series Prediction Performance", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.savefig("images/03-03.png")
