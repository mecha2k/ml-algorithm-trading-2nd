import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
from talib import RSI, BBANDS, MACD, ATR


def compute_bb(close):
    high, mid, low = BBANDS(close, timeperiod=20)
    return pd.DataFrame({"bb_high": high, "bb_low": low}, index=close.index)


def compute_atr(stock_data):
    df = ATR(stock_data.high, stock_data.low, stock_data.close, timeperiod=14)
    return df.sub(df.mean()).div(df.std())


def compute_macd(close):
    macd = MACD(close)[0]
    return (macd - np.mean(macd)) / np.std(macd)


if __name__ == "__main__":
    sns.set_style("whitegrid")
    plt.tight_layout()

    MONTH = 21
    YEAR = 12 * MONTH

    DATA_STORE = "../data/assets.h5"
    ohlcv = ["adj_open", "adj_close", "adj_low", "adj_high", "adj_volume"]
    with pd.HDFStore(DATA_STORE) as store:
        prices = (
            store["quandl/wiki/prices"]
            .loc[pd.IndexSlice["2013-01":"2017-12", :], ohlcv]
            .rename(columns=lambda x: x.replace("adj_", ""))
        )
        prices = prices.swaplevel().sort_index()
        prices.volume /= 1e3
        stocks = store["us_equities/stocks"].loc[:, ["marketcap", "ipoyear", "sector"]]
        prices.to_csv("../data/prepare_model_prices.csv")
        stocks.to_csv("../data/prepare_model_stocks.csv")

    min_obs = int(0.2 * YEAR)
    nobs = prices.groupby(level="ticker").size()
    keep = nobs[nobs > min_obs].index
    prices = prices.loc[pd.IndexSlice[keep, :], :]
    print(prices.index.names)

    stocks = stocks[~stocks.index.duplicated() & stocks.sector.notnull()]
    stocks.sector = stocks.sector.str.lower().str.replace(" ", "_")
    stocks.index.name = "ticker"

    shared = prices.index.get_level_values("ticker").unique()
    shared = shared.intersection(stocks.index)
    stocks = stocks.loc[shared, :]
    prices = prices.loc[pd.IndexSlice[shared, :], :]

    prices.info()
    stocks.info()
    print(stocks.sector.value_counts())

    # compute dollar volume to determine universe
    prices["dollar_vol"] = prices.loc[:, "close"].mul(prices.loc[:, "volume"], axis=0)
    prices["dollar_vol"] = (
        prices.groupby("ticker", group_keys=False, as_index=True)
        .dollar_vol.rolling(window=21)
        .mean()
        .fillna(0)
        .reset_index(level=0, drop=True)
    )
    prices.dollar_vol /= 1e3
    prices["dollar_vol_rank"] = prices.groupby("date").dollar_vol.rank(ascending=False)
    prices["rsi"] = prices.groupby(level="ticker").close.apply(RSI)

    ax = sns.histplot(prices.rsi.dropna())
    ax.axvline(30, ls="--", lw=1, c="k")
    ax.axvline(70, ls="--", lw=1, c="k")
    ax.set_title("RSI Distribution with Signal Threshold")
    plt.savefig("../images/ch07_im04.png", dpi=300, bboxinches="tight")

    prices = prices.join(prices.groupby(level="ticker").close.apply(compute_bb))
    prices["bb_high"] = prices.bb_high.sub(prices.close).div(prices.bb_high).apply(np.log1p)
    prices["bb_low"] = prices.close.sub(prices.bb_low).div(prices.close).apply(np.log1p)

    fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
    sns.histplot(prices.loc[prices.dollar_vol_rank < 100, "bb_low"].dropna(), ax=axes[0])
    sns.histplot(prices.loc[prices.dollar_vol_rank < 100, "bb_high"].dropna(), ax=axes[1])
    plt.savefig("../images/ch07_im05.png", dpi=300, bboxinches="tight")

    prices["atr"] = prices.groupby("ticker", group_keys=False).apply(compute_atr)
    sns.histplot(prices[prices.dollar_vol_rank < 50].atr.dropna())
    plt.savefig("../images/ch07_im06.png", dpi=300, bboxinches="tight")

    prices["macd"] = prices.groupby("ticker", group_keys=False).close.apply(compute_macd)
    print(
        prices.macd.describe(
            percentiles=[0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999]
        ).apply(lambda x: f"{x:,.1f}")
    )
    sns.histplot(prices[prices.dollar_vol_rank < 100].macd.dropna())
    plt.savefig("../images/ch07_im07.png", dpi=300, bboxinches="tight")

    lags = [1, 5, 10, 21, 42, 63]
    returns = prices.groupby(level="ticker").close.pct_change()
    percentiles = [0.0001, 0.001, 0.01]
    percentiles += [1 - p for p in percentiles]
    print(
        returns.describe(percentiles=percentiles)
        .iloc[2:]
        .to_frame("percentiles")
        .style.format(lambda x: f"{x:,.2%}")
    )

    q = 0.0001
    for lag in lags:
        prices[f"return_{lag}d"] = (
            prices.groupby(level="ticker")
            .close.pct_change(lag)
            .pipe(lambda x: x.clip(lower=x.quantile(q), upper=x.quantile(1 - q)))
            .add(1)
            .pow(1 / lag)
            .sub(1)
        )

    for t in [1, 2, 3, 4, 5]:
        for lag in [1, 5, 10, 21]:
            prices[f"return_{lag}d_lag{t}"] = prices.groupby(level="ticker")[
                f"return_{lag}d"
            ].shift(t * lag)

    for t in [1, 5, 10, 21]:
        prices[f"target_{t}d"] = prices.groupby(level="ticker")[f"return_{t}d"].shift(-t)

    prices = prices.join(stocks[["sector"]])
    prices["year"] = prices.index.get_level_values("date").year
    prices["month"] = prices.index.get_level_values("date").month
    prices.info(show_counts=True)
    prices.assign(sector=pd.factorize(prices.sector, sort=True)[0]).to_hdf(
        "../data/lin_models.h5", "model_data/no_dummies"
    )
    prices = pd.get_dummies(
        prices,
        columns=["year", "month", "sector"],
        prefix=["year", "month", ""],
        prefix_sep=["_", "_", ""],
        drop_first=True,
    )
    prices.info(show_counts=True)
    prices.to_hdf("../data/lin_models.h5", "model_data")

    target = "target_5d"
    top100 = prices[prices.dollar_vol_rank < 100].copy()
    top100.loc[:, "rsi_signal"] = pd.cut(top100.rsi, bins=[0, 30, 70, 100])
    top100.groupby("rsi_signal")["target_5d"].describe()

    j = sns.jointplot(x="bb_low", y=target, data=top100)
    x = top100.bb_low
    y = top100.target_5d
    finite = np.isfinite(x) & np.isfinite(y)
    corr, pval = pearsonr(x[finite], y[finite])
    plt.savefig("../images/ch07_im08.png", dpi=300, bboxinches="tight")
    j = sns.jointplot(x="bb_high", y=target, data=top100)
    plt.savefig("../images/ch07_im09.png", dpi=300, bboxinches="tight")
    j = sns.jointplot(x="atr", y=target, data=top100)
    plt.savefig("../images/ch07_im10.png", dpi=300, bboxinches="tight")
    j = sns.jointplot(x="macd", y=target, data=top100)
    plt.savefig("../images/ch07_im11.png", dpi=300, bboxinches="tight")
