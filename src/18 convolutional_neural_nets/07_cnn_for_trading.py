# CNN for Trading - Part 3: Training and Evaluating a CNN
# To exploit the grid-like structure of time-series data, we can use CNN architectures for univariate and multivariate
# time series. In the latter case, we consider different time series as channels, similar to the different color signals.

## Creating and training a convolutional neural network
# Now we are ready to design, train, and evaluate a CNN following the steps outlined in the previous section.

from time import time
from pathlib import Path
import sys, os

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import MultipleTimeSeriesCV, format_time

idx = pd.IndexSlice
np.random.seed(seed=42)
tf.random.set_seed(seed=42)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
pd.options.display.float_format = "{:,.2f}".format
warnings.filterwarnings("ignore")

DATA_STORE = "../data/assets.h5"

results_path = Path("../data/ch18", "cnn_for_trading")
mnist_path = results_path / "mnist"
if not mnist_path.exists():
    mnist_path.mkdir(parents=True)

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if gpu_devices:
    print("Using GPU")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print("Using CPU")


if __name__ == "__main__":
    size = 15
    lookahead = 1

    ## Load Model Data
    with pd.HDFStore("../data/18_data.h5") as store:
        features = store["img_data"]
        targets = store["targets"]
    features.info()
    targets.info()

    outcome = f"r{lookahead:02}_fwd"
    features = features.join(targets[[outcome]]).dropna()
    target = features[outcome]
    features = features.drop(outcome, axis=1)

    ## Convolutional Neural Network
    # We again closely follow the authors in creating a CNN with 2 convolutional layers with kernel size 3 and 16 and
    # 32 filters, respectively, followed by a max pooling layer of size 2.
    # We flatten the output of the last stack of filters and connect the resulting 1,568 outputs to a dense layer of
    # size 32, applying 25 and 50 percent dropout probability to the incoming and outcoming connections to mitigate
    # overfitting.

    ### Model Architecture
    def make_model(filter1=16, act1="relu", filter2=32, act2="relu", do1=0.25, do2=0.5, dense=32):
        input_shape = (size, size, 1)
        cnn = Sequential(
            [
                Conv2D(
                    filters=filter1,
                    kernel_size=3,
                    padding="same",
                    activation=act1,
                    input_shape=input_shape,
                    name="CONV1",
                ),
                Conv2D(
                    filters=filter2, kernel_size=3, padding="same", activation=act2, name="CONV2"
                ),
                MaxPooling2D(pool_size=2, name="POOL2"),
                Dropout(do1, name="DROP1"),
                Flatten(name="FLAT1"),
                Dense(dense, activation="relu", name="FC1"),
                Dropout(do2, name="DROP2"),
                Dense(1, activation="linear", name="FC2"),
            ]
        )
        cnn.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=0.01, momentum=0.9, nesterov=False, name="SGD"
            ),
            metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )
        return cnn

    cnn = make_model()
    cnn.summary()

    ### Train the Model
    # We cross-validate the model with the MutipleTimeSeriesCV train and validation set index generator introduced
    # in Chapter 7, Linear Models – From Risk Factors to Return Forecasts. We provide 5 years of trading days during
    # the training period in batches of 64 random samples and validate using the subsequent 3 months, covering the years
    # 2014-2017.
    train_period_length = 5 * 12 * 21
    test_period_length = 5 * 21
    n_splits = 16

    cv = MultipleTimeSeriesCV(
        n_splits=n_splits,
        train_period_length=train_period_length,
        test_period_length=test_period_length,
        lookahead=lookahead,
    )

    # We scale the features to the range [-1, 1] and again use NumPy's .reshape() method to create the requisite format:
    def get_train_valid_data(X, y, train_idx, test_idx):
        x_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        x_val, y_val = X.iloc[test_idx, :], y.iloc[test_idx]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        return (
            x_train.reshape(-1, size, size, 1),
            y_train,
            x_val.reshape(-1, size, size, 1),
            y_val,
        )

    batch_size = 64

    checkpoint_path = results_path / f"lookahead_{lookahead:02d}"
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Training and validation follow the process laid out in Chapter 17, Deep Learning for Trading, relying on
    # checkpointing to store weights after each epoch and generate predictions for the best-performing iterations
    # without the need for costly retraining.
    start = time()
    ic = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(features)):
        X_train, y_train, X_val, y_val = get_train_valid_data(features, target, train_idx, test_idx)
        preds = y_val.to_frame("actual")
        r = pd.DataFrame(index=y_val.index.unique(level="date")).sort_index()
        model = make_model(
            filter1=16, act1="relu", filter2=32, act2="relu", do1=0.25, do2=0.5, dense=32
        )
        best_mean = best_median = -np.inf
        for epoch in range(25):
            model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                epochs=epoch + 1,
                initial_epoch=epoch,
                verbose=0,
                shuffle=True,
            )
            model.save_weights((checkpoint_path / f"ckpt_{fold}_{epoch}").as_posix())
            preds[epoch] = model.predict(X_val).squeeze()
            r[epoch] = (
                preds.groupby(level="date")
                .apply(lambda x: spearmanr(x.actual, x[epoch])[0])
                .to_frame(epoch)
            )
            print(
                f"{format_time(time()-start)} | {fold + 1:02d} | {epoch + 1:02d} | {r[epoch].mean():7.4f} | {r[epoch].median():7.4f}"
            )
        ic.append(r.assign(fold=fold))
    ic = pd.concat(ic)
    ic.to_csv(checkpoint_path / "ic.csv")

    ### Evaluate results
    ic.groupby("fold").mean().boxplot()
    plt.savefig("images/07-01.png")

    ic.groupby("fold").mean().mean().sort_index().plot.bar(rot=0)
    plt.savefig("images/07-02.png")

    cmap = sns.diverging_palette(h_neg=20, h_pos=210)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        ic.groupby("fold").mean().mul(100), ax=ax, center=0, cmap=cmap, annot=True, fmt=".1f"
    )
    fig.tight_layout()
    plt.savefig("images/07-03.png")

    ## Make Predictions
    # To evaluate the model's predictive accuracy, we compute the daily information coefficient (IC) for the validation
    # set like so:
    def generate_predictions(epoch):
        predictions = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(features)):
            X_train, y_train, X_val, y_val = get_train_valid_data(
                features, target, train_idx, test_idx
            )
            preds = y_val.to_frame("actual")
            model = make_model(
                filter1=16, act1="relu", filter2=32, act2="relu", do1=0.25, do2=0.5, dense=32
            )
            status = model.load_weights((checkpoint_path / f"ckpt_{fold}_{epoch}").as_posix())
            status.expect_partial()
            predictions.append(pd.Series(model.predict(X_val).squeeze(), index=y_val.index))
        return pd.concat(predictions)

    preds = {}
    for i, epoch in enumerate(ic.drop("fold", axis=1).mean().nlargest(5).index):
        preds[i] = generate_predictions(epoch)

    with pd.HDFStore(results_path / "predictions.h5") as store:
        store.put("predictions", pd.DataFrame(preds).sort_index())
