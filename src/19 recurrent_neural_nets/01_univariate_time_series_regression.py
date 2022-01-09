# Recurrent Neural Networks
## Univariate Time Series Regression
# This notebook demonstrates how to forecast the S&P 500 index using a Recurrent Neural Network.

from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from scipy.stats import spearmanr

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow import keras

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


results_path = Path("../data/ch19", "univariate_time_series")
if not results_path.exists():
    results_path.mkdir(parents=True)


if __name__ == "__main__":
    # We obtain data for 2010-2018 from the Federal Reserve Bank’s Data Service [FRED]
    # (https://fred.stlouisfed.org/) using the [pandas_datareader](https://pandas-datareader.readthedocs.io/)
    # library in introduced in [Chapter 2 on Market and Fundamental Data](../02_market_and_fundamental_data).
    sp500 = web.DataReader("SP500", "fred", start="2010", end="2020").dropna()
    ax = sp500.plot(title="S&P 500", legend=False, figsize=(14, 4), rot=0)
    ax.set_xlabel("")
    sns.despine()
    plt.savefig("images/01-01.png")

    ## Preprocessing
    scaler = MinMaxScaler()
    sp500_scaled = pd.Series(scaler.fit_transform(sp500).squeeze(), index=sp500.index)
    print(sp500_scaled.describe())

    ## Generating recurrent sequences from our time series
    # Our time series is a sequence of numbers indexed by time:
    # $$x_{0},x_{1},x_{2},...,x_{T}$$
    # where $\{x_t\}$ is the numerical value in period $t$ and  $T$ is the total length of the series.
    # To apply an RNN for regression of classification, we use a sliding window to construct a rolling set of
    # input/output pairs for our model to learn from as animated below.
    # <img src="../assets/timeseries_windowing.gif" width=600 height=600/>

    # We will generate sequences of 63 trading days, approximately three months, and use a single LSTM layer with 20
    # hidden units to predict the index value one timestep ahead.
    # The input to every LSTM layer must have three dimensions, namely:
    # - **Samples**: One sequence is one sample. A batch contains one or more samples.
    # - **Time Steps**: One time step is one point of observation in the sample.
    # - **Features**: One feature is one observation at a time step.
    #
    # Our S&P 500 sample has 2,264 observations or time steps. We will create overlapping sequences using a window
    # of 63 observations each.
    # For a simpler window of size T = 5, we obtain input-output pairs as shown in the following table:

    # $$\begin{array}{c|c}
    # \text{Input} & \text{Output}\\
    # \hline  {\langle x_1,x_2,x_3,x_4,x_5\rangle} & { x_6} \\
    # \ {\langle x_{2},x_{3},x_{4},x_{5},x_{6} \rangle } & {x_{7} } \\
    #   {\vdots} & {\vdots}\\
    #  { \langle x_{T-5},x_{T-4},x_{T-3},x_{T-2},x_{T-1} \rangle } &  {x_{T}}
    # \end{array}$$

    # Generally speaking, for window size S, the relationship takes the form
    #
    # $$x_t = f( x_{t-1}, x_{t-2}, ..., x_{t-S}) \quad\forall t=S, S+1, ..., T$$
    #
    # Each of the $T-S$ lagged input sequence or vector is of length S with a corresponding scalar output.

    # We can use the function create_univariate_rnn_data() to stack sequences selected using a rolling windows:
    def create_univariate_rnn_data(data, window_size):
        n = len(data)
        y = data[window_size:]
        data = data.values.reshape(-1, 1)  # make 2D
        X = np.hstack(tuple([data[i : n - j, :] for i, j in enumerate(range(window_size, 0, -1))]))
        return pd.DataFrame(X, index=y.index), y

    # We apply this function to the rescaled stock index for a window_size=63 to obtain a two-dimensional dataset of
    # shape number of samples x number of timesteps:
    window_size = 63
    X, y = create_univariate_rnn_data(sp500_scaled, window_size=window_size)
    print(X.head())
    print(y.head())
    print(X.shape)

    ## Train-test split
    # To respect the time series nature of the data, we set aside the data at the end of the sample as hold-out or test
    # set. More specifically, we'll use the data for 2018.
    ax = sp500_scaled.plot(lw=2, figsize=(14, 4), rot=0)
    ax.set_xlabel("")
    sns.despine()
    plt.savefig("images/01-02.png")

    X_train = X[:"2018"].values.reshape(-1, window_size, 1)
    y_train = y[:"2018"]

    # keep the last year for testing
    X_test = X["2019"].values.reshape(-1, window_size, 1)
    y_test = y["2019"]

    n_obs, window_size, n_features = X_train.shape
    print(y_train.shape)

    ## Keras LSTM Layer
    # Keras has several built-in RNN layers with various configuration options described in detail in the
    # [documentation](https://keras.io/layers/recurrent/).

    # LSTM(units,
    #      activation='tanh',
    #      recurrent_activation='hard_sigmoid',
    #      use_bias=True,
    #      kernel_initializer='glorot_uniform',
    #      recurrent_initializer='orthogonal',
    #      bias_initializer='zeros',
    #      unit_forget_bias=True,
    #      kernel_regularizer=None,
    #      recurrent_regularizer=None,
    #      bias_regularizer=None,
    #      activity_regularizer=None,
    #      kernel_constraint=None,
    #      recurrent_constraint=None,
    #      bias_constraint=None,
    #      dropout=0.0,
    #      recurrent_dropout=0.0,
    #      implementation=1,
    #      return_sequences=False,
    #      return_state=False,
    #      go_backwards=False,
    #      stateful=False,
    #      unroll=False)

    ##  Define the Model Architecture
    # Having created input/output pairs out of our time series and cut this into training/testing sets, we can now begin
    # setting up our RNN.  We use Keras to quickly build a two hidden layer RNN of the following specifications
    # - layer 1 uses an LSTM module with 20 hidden units (note here the input_shape = (window_size,1))
    # - layer 2 uses a fully connected module with one unit
    # - the 'mean_squared_error' loss should be used (remember: we are performing regression here)
    # This can be constructed using just a few lines - see e.g., the [general Keras documentation]
    # (https://keras.io/getting-started/sequential-model-guide/) and the [LSTM documentation in particular]
    # (https://keras.io/layers/recurrent/) for examples of how to quickly use Keras to build neural network models.
    # Make sure you are initializing your optimizer given the [keras-recommended approach for RNNs]
    # (https://keras.io/optimizers/)

    rnn = Sequential(
        [
            LSTM(units=10, input_shape=(window_size, n_features), name="LSTM"),
            Dense(1, name="Output"),
        ]
    )

    # The summary shows that the model has 1,781 parameters:
    rnn.summary()

    ## Train the Model
    # We train the model using the RMSProp optimizer recommended for RNN with default settings and compile the model
    # with mean squared error for this regression problem:
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    rnn.compile(loss="mean_squared_error", optimizer=optimizer)

    # We define an EarlyStopping callback and train the model for up to 100 episodes.
    rnn_path = (results_path / "rnn.h5").as_posix()
    checkpointer = ModelCheckpoint(
        filepath=rnn_path, verbose=1, monitor="val_loss", save_best_only=True
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    lstm_training = rnn.fit(
        X_train,
        y_train,
        epochs=150,
        batch_size=20,
        shuffle=True,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, checkpointer],
        verbose=1,
    )

    # Training stops after 51 epochs; the `early_stopping` callback restores the weights for the best model (after 41 epochs)

    ## Evaluate model performance
    fig, ax = plt.subplots(figsize=(12, 4))

    loss_history = pd.DataFrame(lstm_training.history).pow(0.5)
    loss_history.index += 1
    best_rmse = loss_history.val_loss.min()
    best_epoch = loss_history.val_loss.idxmin()

    title = f"5-Epoch Rolling RMSE (Best Validation RMSE: {best_rmse:.4%})"
    loss_history.columns = ["Training RMSE", "Validation RMSE"]
    loss_history.rolling(5).mean().plot(logy=True, lw=2, title=title, ax=ax)
    ax.axvline(best_epoch, ls="--", lw=1, c="k")
    sns.despine()
    fig.tight_layout()
    fig.savefig("images/01_rnn_sp500_error.png", dpi=300)

    train_rmse_scaled = np.sqrt(rnn.evaluate(X_train, y_train, verbose=0))
    test_rmse_scaled = np.sqrt(rnn.evaluate(X_test, y_test, verbose=0))
    print(f"Train RMSE: {train_rmse_scaled:.4f} | Test RMSE: {test_rmse_scaled:.4f}")

    train_predict_scaled = rnn.predict(X_train)
    test_predict_scaled = rnn.predict(X_test)

    train_ic = spearmanr(y_train, train_predict_scaled)[0]
    test_ic = spearmanr(y_test, test_predict_scaled)[0]
    print(f"Train IC: {train_ic:.4f} | Test IC: {test_ic:.4f}")

    ### Rescale predictions
    train_predict = pd.Series(
        scaler.inverse_transform(train_predict_scaled).squeeze(), index=y_train.index
    )
    test_predict = pd.Series(
        scaler.inverse_transform(test_predict_scaled).squeeze(), index=y_test.index
    )

    y_train_rescaled = scaler.inverse_transform(y_train.to_frame()).squeeze()
    y_test_rescaled = scaler.inverse_transform(y_test.to_frame()).squeeze()

    train_rmse = np.sqrt(mean_squared_error(train_predict, y_train_rescaled))
    test_rmse = np.sqrt(mean_squared_error(test_predict, y_test_rescaled))
    print(f"Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")

    sp500["Train Predictions"] = train_predict
    sp500["Test Predictions"] = test_predict
    sp500 = sp500.join(
        train_predict.to_frame("predictions")
        .assign(data="Train")
        .append(test_predict.to_frame("predictions").assign(data="Test"))
    )

    ### Plot Results
    fig = plt.figure(figsize=(14, 7))
    ax1 = plt.subplot(221)
    sp500.loc["2015":, "SP500"].plot(lw=4, ax=ax1, c="k")
    sp500.loc["2015":, ["Test Predictions", "Train Predictions"]].plot(lw=1, ax=ax1, ls="--")
    ax1.set_title("In- and Out-of-sample Predictions")

    with sns.axes_style("white"):
        ax3 = plt.subplot(223)
        sns.scatterplot(x="SP500", y="predictions", data=sp500, hue="data", ax=ax3)
        ax3.text(x=0.02, y=0.95, s=f"Test IC ={test_ic:.2%}", transform=ax3.transAxes)
        ax3.text(x=0.02, y=0.87, s=f"Train IC={train_ic:.2%}", transform=ax3.transAxes)
        ax3.set_title("Correlation")
        ax3.legend(loc="lower right")

        ax2 = plt.subplot(222)
        ax4 = plt.subplot(224, sharex=ax2, sharey=ax2)
        sns.distplot(train_predict.squeeze() - y_train_rescaled, ax=ax2)
        ax2.set_title("Train Error")
        ax2.text(x=0.03, y=0.92, s=f"Train RMSE ={train_rmse:.4f}", transform=ax2.transAxes)
        sns.distplot(test_predict.squeeze() - y_test_rescaled, ax=ax4)
        ax4.set_title("Test Error")
        ax4.text(x=0.03, y=0.92, s=f"Test RMSE ={test_rmse:.4f}", transform=ax4.transAxes)

    sns.despine()
    fig.tight_layout()
    fig.savefig("images/01_rnn_sp500_regression.png", dpi=300)
