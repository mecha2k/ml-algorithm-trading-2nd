# Conditional Autoencoder for Asset Pricing - Part 2: The Model
# This notebook uses a dataset created using `yfinance` in the notebook
# [conditional_autoencoder_for_asset_pricing_data](05_conditional_autoencoder_for_asset_pricing_data.ipynb).
# The results will vary depending on which ticker downloads succeeded.

import sys, os
from time import time
from pathlib import Path
from itertools import product
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dot, Reshape, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import quantile_transform
from scipy.stats import spearmanr
import warnings

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import MultipleTimeSeriesCV, format_time

idx = pd.IndexSlice
np.random.seed(seed=42)
tf.random.set_seed(seed=42)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
warnings.filterwarnings("ignore")

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if gpu_devices:
    print("Using GPU")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print("Using CPU")

results_path = Path("../data/ch20", "asset_pricing")
if not results_path.exists():
    results_path.mkdir(parents=True)


if __name__ == "__main__":
    characteristics = [
        "beta",
        "betasq",
        "chmom",
        "dolvol",
        "idiovol",
        "ill",
        "indmom",
        "maxret",
        "mom12m",
        "mom1m",
        "mom36m",
        "mvel",
        "retvol",
        "turn",
        "turn_std",
    ]

    # ## Load Data
    with pd.HDFStore(results_path / "autoencoder.h5") as store:
        print(store.info())

    ### Weekly returns
    data = (
        pd.read_hdf(results_path / "autoencoder.h5", "returns")
        .stack(dropna=False)
        .to_frame("returns")
        .loc[idx["1993":, :], :]
    )

    with pd.HDFStore(results_path / "autoencoder.h5") as store:
        keys = [k[1:] for k in store.keys() if k[1:].startswith("factor")]
        for key in keys:
            data[key.split("/")[-1]] = store[key].squeeze()
    characteristics = data.drop("returns", axis=1).columns.tolist()
    data["returns_fwd"] = data.returns.unstack("ticker").shift(-1).stack()
    data.info(show_counts=True)

    nobs_by_date = data.groupby(level="date").count().max(1)
    nobs_by_characteristic = pd.melt(
        data[characteristics].groupby(level="date").count(),
        value_name="# Observations",
        var_name=["Characteristic"],
    )

    with sns.axes_style("white"):
        fig, axes = plt.subplots(ncols=2, figsize=(14, 4))
        sns.distplot(nobs_by_date, kde=False, ax=axes[0])
        axes[0].set_title("# of Stocks per Week")
        axes[0].set_xlabel("# of Observations")
        sns.boxplot(
            x="Characteristic",
            y="# Observations",
            data=nobs_by_characteristic,
            ax=axes[1],
            palette="Blues",
        )
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=25, ha="right")
        axes[1].set_title("# of Observation per Stock Characteristic")
        fig.tight_layout()
        plt.savefig("images/06-01.png")

    ### Rank-normalize characteristics
    data.loc[:, characteristics] = (
        data.loc[:, characteristics]
        .groupby(level="date")
        .apply(
            lambda x: pd.DataFrame(
                quantile_transform(x, copy=True, n_quantiles=x.shape[0]),
                columns=characteristics,
                index=x.index.get_level_values("ticker"),
            )
        )
        .mul(2)
        .sub(1)
    )
    data.info(show_counts=True)
    print(data.index.names)
    print(data.describe())

    data = data.loc[idx[:"2019", :], :]
    data.loc[:, ["returns", "returns_fwd"]] = data.loc[:, ["returns", "returns_fwd"]].clip(
        lower=-1, upper=1.0
    )
    data = data.fillna(-2)
    data.to_hdf(results_path / "autoencoder.h5", "model_data")

    ## Architecture
    data = pd.read_hdf(results_path / "autoencoder.h5", "model_data")

    ### Key parameters
    n_factors = 3
    n_characteristics = len(characteristics)
    n_tickers = len(data.index.unique("ticker"))
    print(n_tickers)
    print(n_characteristics)

    ### Input Layer
    input_beta = Input((n_tickers, n_characteristics), name="input_beta")
    input_factor = Input((n_tickers,), name="input_factor")

    ### Stock Characteristics Network
    hidden_layer = Dense(units=8, activation="relu", name="hidden_layer")(input_beta)
    batch_norm = BatchNormalization(name="batch_norm")(hidden_layer)
    output_beta = Dense(units=n_factors, name="output_beta")(batch_norm)

    ### Factor Network
    output_factor = Dense(units=n_factors, name="output_factor")(input_factor)

    ### Output Layer
    output = Dot(axes=(2, 1), name="output_layer")([output_beta, output_factor])

    ### Compile Layer
    model = Model(inputs=[input_beta, input_factor], outputs=output)
    model.compile(loss="mse", optimizer="adam")

    ### Automate model generation
    def make_model(hidden_units=8, n_factors=3):
        input_beta = Input((n_tickers, n_characteristics), name="input_beta")
        input_factor = Input((n_tickers,), name="input_factor")

        hidden_layer = Dense(units=hidden_units, activation="relu", name="hidden_layer")(input_beta)
        batch_norm = BatchNormalization(name="batch_norm")(hidden_layer)
        output_beta = Dense(units=n_factors, name="output_beta")(batch_norm)
        output_factor = Dense(units=n_factors, name="output_factor")(input_factor)
        output = Dot(axes=(2, 1), name="output_layer")([output_beta, output_factor])

        model = Model(inputs=[input_beta, input_factor], outputs=output)
        model.compile(loss="mse", optimizer="adam")
        return model

    ### Model Summary
    model.summary()

    ## Train Model
    ### Cross-validation parameters
    YEAR = 52

    cv = MultipleTimeSeriesCV(
        n_splits=5, train_period_length=20 * YEAR, test_period_length=1 * YEAR, lookahead=1
    )

    def get_train_valid_data(data, train_idx, val_idx):
        train, val = data.iloc[train_idx], data.iloc[val_idx]
        X1_train = train.loc[:, characteristics].values.reshape(-1, n_tickers, n_characteristics)
        X1_val = val.loc[:, characteristics].values.reshape(-1, n_tickers, n_characteristics)
        X2_train = train.loc[:, "returns"].unstack("ticker")
        X2_val = val.loc[:, "returns"].unstack("ticker")
        y_train = train.returns_fwd.unstack("ticker")
        y_val = val.returns_fwd.unstack("ticker")
        return X1_train, X2_train, y_train, X1_val, X2_val, y_val

    ### Hyperparameter Options
    factor_opts = [2, 3, 4, 5, 6]
    unit_opts = [8, 16, 32]
    param_grid = list(product(unit_opts, factor_opts))

    ### Run Cross-Validation
    batch_size = 32
    cols = [
        "units",
        "n_factors",
        "fold",
        "epoch",
        "ic_mean",
        "ic_daily_mean",
        "ic_daily_std",
        "ic_daily_median",
    ]

    start = time()
    for units, n_factors in param_grid:
        scores = []
        model = make_model(hidden_units=units, n_factors=n_factors)
        for fold, (train_idx, val_idx) in enumerate(cv.split(data)):
            X1_train, X2_train, y_train, X1_val, X2_val, y_val = get_train_valid_data(
                data, train_idx, val_idx
            )
            for epoch in range(250):
                model.fit(
                    [X1_train, X2_train],
                    y_train,
                    batch_size=batch_size,
                    validation_data=([X1_val, X2_val], y_val),
                    epochs=epoch + 1,
                    initial_epoch=epoch,
                    verbose=0,
                    shuffle=True,
                )
                result = (
                    pd.DataFrame(
                        {
                            "y_pred": model.predict([X1_val, X2_val]).reshape(-1),
                            "y_true": y_val.stack().values,
                        },
                        index=y_val.stack().index,
                    )
                    .replace(-2, np.nan)
                    .dropna()
                )
                r0 = spearmanr(result.y_true, result.y_pred)[0]
                r1 = result.groupby(level="date").apply(lambda x: spearmanr(x.y_pred, x.y_true)[0])

                scores.append([units, n_factors, fold, epoch, r0, r1.mean(), r1.std(), r1.median()])
                if epoch % 50 == 0:
                    print(
                        f"{format_time(time()-start)} | {n_factors} | {units:02} | {fold:02}-{epoch:03} | {r0:6.2%} | "
                        f"{r1.mean():6.2%} | {r1.median():6.2%}"
                    )
        scores = pd.DataFrame(scores, columns=cols)
        scores.to_hdf(results_path / "scores.h5", f"{units}/{n_factors}")

    ### Evaluate Results
    scores = []
    with pd.HDFStore(results_path / "scores.h5") as store:
        for key in store.keys():
            scores.append(store[key])
    scores = pd.concat(scores)
    scores.info()

    avg = (
        scores.groupby(["n_factors", "units", "epoch"])[
            "ic_mean", "ic_daily_mean", "ic_daily_median"
        ]
        .mean()
        .reset_index()
    )
    print(avg.nlargest(n=20, columns=["ic_daily_median"]))

    top = (
        avg.groupby(["n_factors", "units"])
        .apply(lambda x: x.nlargest(n=5, columns=["ic_daily_median"]))
        .reset_index(-1, drop=True)
    )
    print(top.nlargest(n=5, columns=["ic_daily_median"]))

    fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(20, 8), sharey="row", sharex=True)
    for n in range(2, 7):
        df = avg[avg.n_factors == n].pivot(index="epoch", columns="units", values="ic_mean")
        df.rolling(10).mean().loc[:200].plot(ax=axes[0][n - 2], lw=1, title=f"{n} Factors")
        axes[0][n - 2].axhline(0, ls="--", c="k", lw=1)
        axes[0][n - 2].get_legend().remove()
        axes[0][n - 2].set_ylabel("IC (10-epoch rolling mean)")

        df = avg[avg.n_factors == n].pivot(index="epoch", columns="units", values="ic_daily_median")
        df.rolling(10).mean().loc[:200].plot(ax=axes[1][n - 2], lw=1)
        axes[1][n - 2].axhline(0, ls="--", c="k", lw=1)
        axes[1][n - 2].get_legend().remove()
        axes[1][n - 2].set_ylabel("IC, Daily Median (10-epoch rolling mean)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title="# Units")
    fig.suptitle("Cross-Validation Performance (2015-2019)", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig("images/06_cv_performance.png", dpi=300)

    ## Generate Predictions
    # We'll average over a range of epochs that appears to deliver good predictions.
    n_factors = 4
    units = 32
    batch_size = 32
    first_epoch = 50
    last_epoch = 80

    predictions = []
    for epoch in tqdm(list(range(first_epoch, last_epoch))):
        epoch_preds = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(data)):
            X1_train, X2_train, y_train, X1_val, X2_val, y_val = get_train_valid_data(
                data, train_idx, val_idx
            )
            model = make_model(n_factors=n_factors, hidden_units=units)
            model.fit(
                [X1_train, X2_train],
                y_train,
                batch_size=batch_size,
                epochs=epoch,
                verbose=0,
                shuffle=True,
            )
            epoch_preds.append(
                pd.Series(
                    model.predict([X1_val, X2_val]).reshape(-1), index=y_val.stack().index
                ).to_frame(epoch)
            )
        predictions.append(pd.concat(epoch_preds))
    predictions_combined = pd.concat(predictions, axis=1).sort_index()
    predictions_combined.info()
    predictions_combined.to_hdf(results_path / "predictions.h5", "predictions")
