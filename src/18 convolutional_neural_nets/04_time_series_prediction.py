# How to use CNN with time series data
# The regular measurements of time series result in a similar grid-like data structure as for the image data we have
# focused on so far. As a result, we can use CNN architectures for univariate and multivariate time series.
# In the latter case, we consider different time series as channels, similar to the different color signals.

import sys
from time import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

import tensorflow as tf

tf.autograph.set_verbosity(0, True)
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv1D,
    MaxPooling1D,
    Dropout,
    BatchNormalization,
)

import matplotlib.pyplot as plt
import seaborn as sns


gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if gpu_devices:
    print("Using GPU")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print("Using CPU")


def format_time(t):
    """Return a formatted time string 'HH:MM:SS
    based on a numeric time() value"""
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f"{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}"


class MultipleTimeSeriesCV:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes"""

    def __init__(
        self,
        n_splits=3,
        train_period_length=126,
        test_period_length=21,
        lookahead=None,
        date_idx="date",
        shuffle=False,
    ):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle
        self.date_idx = date_idx

    def split(self, X, y=None, groups=None):
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days = sorted(unique_dates, reverse=True)
        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            split_idx.append([train_start_idx, train_end_idx, test_start_idx, test_end_idx])

        dates = X.reset_index()[[self.date_idx]]
        for train_start, train_end, test_start, test_end in split_idx:

            train_idx = dates[
                (dates[self.date_idx] > days[train_start])
                & (dates[self.date_idx] <= days[train_end])
            ].index
            test_idx = dates[
                (dates[self.date_idx] > days[test_start]) & (dates[self.date_idx] <= days[test_end])
            ].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx.to_numpy(), test_idx.to_numpy()

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


idx = pd.IndexSlice
np.random.seed(seed=42)
tf.random.set_seed(seed=42)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
# pd.options.display.float_format = "{:,.2f}".format


results_path = Path("../data/ch18", "time_series")
mnist_path = results_path / "mnist"
if not mnist_path.exists():
    mnist_path.mkdir(parents=True)


if __name__ == "__main__":
    ## Prepare Data
    prices = pd.read_hdf("../data/assets.h5", "quandl/wiki/prices").adj_close.unstack().loc["2000":]
    prices.info()

    ### Compute monthly returns
    returns = (
        prices.resample("M")
        .last()
        .pct_change()
        .dropna(how="all")
        .loc["2000":"2017"]
        .dropna(axis=1)
        .sort_index(ascending=False)
    )

    # remove outliers likely representing data errors
    returns = returns.where(returns < 1).dropna(axis=1)
    returns.info()

    ### Create model data
    n = len(returns)
    nlags = 12
    lags = list(range(1, nlags + 1))

    cnn_data = []
    for i in range(n - nlags - 1):
        df = returns.iloc[i : i + nlags + 1]  # select outcome and lags
        date = df.index.max()  # use outcome date
        cnn_data.append(
            df.reset_index(drop=True)  # append transposed series
            .transpose()
            .assign(date=date)
            .set_index("date", append=True)
            .sort_index(axis=1, ascending=True)
        )
    cnn_data = pd.concat(cnn_data).rename(columns={0: "label"}).sort_index()
    cnn_data.info(show_counts=True)

    ## Evaluate features
    ### Mutual Information
    mi = mutual_info_regression(X=cnn_data.drop("label", axis=1), y=cnn_data.label)
    mi = pd.Series(mi, index=cnn_data.drop("label", axis=1).columns)

    ### Information Coefficient
    ic = {}
    for lag in lags:
        ic[lag] = spearmanr(cnn_data.label, cnn_data[lag])
    ic = pd.DataFrame(ic, index=["IC", "p-value"]).T

    ax = ic.plot.bar(rot=0, figsize=(14, 4), ylim=(-0.05, 0.05), title="Feature Evaluation")
    ax.set_xlabel("Lag")
    plt.tight_layout()
    plt.savefig("images/04_cnn_ts1d_feature_ic.png", dpi=300)

    ### Plot Metrics
    metrics = pd.concat(
        [mi.to_frame("Mutual Information"), ic.IC.to_frame("Information Coefficient")], axis=1
    )

    ax = metrics.plot.bar(figsize=(12, 4), rot=0)
    ax.set_xlabel("Lag")
    plt.tight_layout()
    plt.savefig("images/04_ts1d_metrics.png", dpi=300)

    ## CNN: Model Architecture
    # We design a simple one-layer CNN that uses one-dimensional convolutions combined with max pooling to learn
    # time series patterns:
    def get_model(filters=32, kernel_size=5, pool_size=2):
        model = Sequential(
            [
                Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation="relu",
                    padding="causal",
                    input_shape=input_shape,
                    use_bias=True,
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
                ),
                MaxPooling1D(pool_size=pool_size),
                Flatten(),
                BatchNormalization(),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(loss="mse", optimizer="Adam")
        return model

    ### Set up CV
    cv = MultipleTimeSeriesCV(
        n_splits=12 * 3, train_period_length=12 * 5, test_period_length=1, lookahead=1
    )

    input_shape = nlags, 1

    ### Train Model
    def get_train_valid_data(X, y, train_idx, test_idx):
        x_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        x_val, y_val = X.iloc[test_idx, :], y.iloc[test_idx]
        m = X.shape[1]
        return x_train.values.reshape(-1, m, 1), y_train, x_val.values.reshape(-1, m, 1), y_val

    batch_size = 128
    epochs = 100

    filters = 32
    kernel_size = 4
    pool_size = 4

    get_model(filters=filters, kernel_size=kernel_size, pool_size=pool_size).summary()

    ### Cross-validation loop
    result = {}
    start = time()
    for fold, (train_idx, test_idx) in enumerate(cv.split(cnn_data)):
        X_train, y_train, X_val, y_val = get_train_valid_data(
            cnn_data.drop("label", axis=1).sort_index(ascending=False),
            cnn_data.label,
            train_idx,
            test_idx,
        )
        test_date = y_val.index.get_level_values("date").max()
        model = get_model(filters=filters, kernel_size=kernel_size, pool_size=pool_size)

        best_ic = -np.inf
        epoch, p_val, stop = 0, 0, 0
        for epoch in range(20):
            training = model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                epochs=epoch + 1,
                initial_epoch=epoch,
                verbose=0,
                shuffle=True,
            )
            predicted = model.predict(X_val).squeeze()
            ic, p_val_ = spearmanr(predicted, y_val)
            if ic > best_ic:
                best_ic = ic
                p_val = p_val_
                stop = 0
            else:
                stop += 1
            if stop == 10:
                break

        nrounds = epoch + 1 - stop
        result[test_date] = [nrounds, best_ic, p_val]
        df = pd.DataFrame(result, index=["epochs", "IC", "p-value"]).T
        msg = f"{fold + 1:02d} | {format_time(time()-start)} | {nrounds:3.0f} | "
        print(msg + f"{best_ic*100:5.2} ({p_val:7.2%}) | {df.IC.mean()*100:5.2}")

    ### Evaluate Results
    metrics = pd.DataFrame(result, index=["epochs", "IC", "p-value"]).T

    ax = metrics.IC.plot(
        figsize=(12, 4),
        label="Information Coefficient",
        title="Validation Performance",
        ylim=(0, 0.08),
    )
    metrics.IC.expanding().mean().plot(ax=ax, label="Cumulative Average")
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig("images/04_cnn_ts1d_ic.png", dpi=300)
