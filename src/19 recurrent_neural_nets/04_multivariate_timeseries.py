# Multivariate Time Series Regression
# So far, we have limited our modeling efforts to single time series. RNNs are naturally well suited to multivariate
# time series and represent a non-linear alternative to the Vector Autoregressive (VAR) models we covered in Chapter 8,
# Time Series Models.

from pathlib import Path
import numpy as np
import pandas as pd
import pandas_datareader.data as web

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import minmax_scale

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM
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


results_path = Path("../data/ch19", "multivariate_time_series")
if not results_path.exists():
    results_path.mkdir(parents=True)


if __name__ == "__main__":
    # For comparison, we illustrate the application of RNNs to modeling and forecasting several time series using the
    # same dataset we used for the VAR example, monthly data on consumer sentiment, and industrial production from the
    # Federal Reserve's FRED service in Chapter 8, Time Series Models:
    df = web.DataReader(["UMCSENT", "IPGMFN"], "fred", "1980", "2019-12").dropna()
    df.columns = ["sentiment", "ip"]
    df.info()
    print(df.head())

    ## Prepare Data
    ### Stationarity
    # We apply the same transformation—annual difference for both series, prior log-transform for industrial
    # production—to achieve stationarity that we used in Chapter 8 on Time Series Models:
    df_transformed = pd.DataFrame(
        {"ip": np.log(df.ip).diff(12), "sentiment": df.sentiment.diff(12)}
    ).dropna()

    ### Scaling
    # Then we scale the transformed data to the [0,1] interval:
    df_transformed = df_transformed.apply(minmax_scale)

    ### Plot original and transformed series
    fig, axes = plt.subplots(ncols=2, figsize=(14, 4))
    columns = {"ip": "Industrial Production", "sentiment": "Sentiment"}
    df.rename(columns=columns).plot(ax=axes[0], title="Original Series")
    df_transformed.rename(columns=columns).plot(ax=axes[1], title="Tansformed Series")
    sns.despine()
    fig.tight_layout()
    fig.savefig("images/04_multi_rnn.png", dpi=300)

    ### Reshape data into RNN format
    # We can reshape directly to get non-overlapping series, i.e., one sample for each year (works only if the number
    # of samples is divisible by window size):
    print(df.values.reshape(-1, 12, 2).shape)

    # However, we want rolling, not non-overlapping lagged values. The create_multivariate_rnn_data function transforms
    # a dataset of several time series into the shape required by the Keras RNN layers, namely `n_samples` x
    # `window_size` x `n_series`, as follows:
    def create_multivariate_rnn_data(data, window_size):
        y = data[window_size:]
        n = data.shape[0]
        X = np.stack([data[i:j] for i, j in enumerate(range(window_size, n))], axis=0)
        return X, y

    # We will use window_size of 24 months and obtain the desired inputs for our RNN model, as follows:
    window_size = 18

    X, y = create_multivariate_rnn_data(df_transformed, window_size=window_size)
    print(X.shape, y.shape)
    print(df_transformed.head())

    # Finally, we split our data into a train and a test set, using the last 24 months to test the out-of-sample
    # performance, as shown here:
    test_size = 24
    train_size = X.shape[0] - test_size

    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    print(X_train.shape, X_test.shape)

    ## Define Model Architecture
    # We use a similar architecture with two stacked LSTM layers with 12 and 6 units, respectively, followed by a
    # fully-connected layer with 10 units. The output layer has two units, one for each time series. We compile them
    # using mean absolute loss and the recommended RMSProp optimizer, as follows:

    K.clear_session()

    n_features = output_size = 2
    lstm_units = 12
    dense_units = 6

    rnn = Sequential(
        [
            LSTM(
                units=lstm_units,
                dropout=0.1,
                recurrent_dropout=0.1,
                input_shape=(window_size, n_features),
                name="LSTM",
                return_sequences=False,
            ),
            Dense(dense_units, name="FC"),
            Dense(output_size, name="Output"),
        ]
    )

    # The model has 1,268 parameters, as shown here:
    rnn.summary()

    rnn.compile(loss="mae", optimizer="RMSProp")

    ## Train the Model
    # We train for 50 epochs with a batch_size value of 20 using early stopping:
    lstm_path = (results_path / "lstm.h5").as_posix()
    checkpointer = ModelCheckpoint(
        filepath=lstm_path, verbose=1, monitor="val_loss", mode="min", save_best_only=True
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    result = rnn.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=20,
        shuffle=False,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, checkpointer],
        verbose=1,
    )

    ## Evaluate the Results
    # Training stops early after 22 epochs, yielding a test MAE of 1.71, which compares favorably to the test MAE for
    # the VAR model of 1.91.
    # However, the two results are not fully comparable because the RNN model produces 24 one-step-ahead forecasts,
    # whereas the VAR model uses its own predictions as input for its out-of-sample forecast. You may want to tweak
    # the VAR setup to obtain comparable forecasts and compare their performance:
    pd.DataFrame(result.history).plot()
    plt.savefig("images/04-01.png")

    y_pred = pd.DataFrame(rnn.predict(X_test), columns=y_test.columns, index=y_test.index)
    y_pred.info()

    test_mae = mean_absolute_error(y_pred, y_test)
    print(test_mae)
    print(y_test.index)

    fig, axes = plt.subplots(ncols=3, figsize=(17, 4))
    pd.DataFrame(result.history).rename(
        columns={"loss": "Training", "val_loss": "Validation"}
    ).plot(ax=axes[0], title="Train & Validiation Error")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE")
    col_dict = {"ip": "Industrial Production", "sentiment": "Sentiment"}

    for i, col in enumerate(y_test.columns, 1):
        y_train.loc["2010":, col].plot(ax=axes[i], label="training", title=col_dict[col])
        y_test[col].plot(ax=axes[i], label="out-of-sample")
        y_pred[col].plot(ax=axes[i], label="prediction")
        axes[i].set_xlabel("")

    axes[1].set_ylim(0.5, 0.9)
    axes[1].fill_between(x=y_test.index, y1=0.5, y2=0.9, color="grey", alpha=0.5)

    axes[2].set_ylim(0.3, 0.9)
    axes[2].fill_between(x=y_test.index, y1=0.3, y2=0.9, color="grey", alpha=0.5)

    plt.legend()
    fig.suptitle("Multivariate RNN - Results | Test MAE = {:.4f}".format(test_mae), fontsize=14)
    sns.despine()
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig("images/04_multivariate_results.png", dpi=300)
