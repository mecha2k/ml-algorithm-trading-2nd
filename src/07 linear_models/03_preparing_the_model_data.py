import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from scipy.stats import pearsonr, spearmanr
from talib import RSI, BBANDS, MACD, ATR
from icecream import ic

idx = pd.IndexSlice
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:,.2f}".format


# Preparing Alpha Factors and Features to predict Stock Returns
if __name__ == "__main__":
    MONTH = 21
    YEAR = 12 * MONTH

    START = "2013-01-01"
    END = "2017-12-31"

    # ## Loading Quandl Wiki Stock Prices & Meta Data
    # ohlcv = ["adj_open", "adj_close", "adj_low", "adj_high", "adj_volume"]
    # DATA_STORE = "../data/assets.h5"
    # with pd.HDFStore(DATA_STORE) as store:
    #     prices = (
    #         store["quandl/wiki/prices"]
    #         .loc[idx[START:END, :], ohlcv]
    #         .rename(columns=lambda x: x.replace("adj_", ""))
    #         .assign(volume=lambda x: x.volume.div(1000))
    #         .swaplevel()
    #         .sort_index()
    #     )
    #     stocks = store["us_equities/stocks"].loc[:, ["marketcap", "ipoyear", "sector"]]
    #
    # ## Remove stocks with few observations
    # # want at least 2 years of data
    # min_obs = 2 * YEAR
    #
    # # have this much per ticker
    # nobs = prices.groupby(level="ticker").size()
    #
    # # keep those that exceed the limit
    # keep = nobs[nobs > min_obs].index
    # prices = prices.loc[idx[keep, :], :]
    #
    # ### Align price and meta data
    # stocks = stocks[~stocks.index.duplicated() & stocks.sector.notnull()]
    # stocks.sector = stocks.sector.str.lower().str.replace(" ", "_")
    # stocks.index.name = "ticker"
    #
    # shared = prices.index.get_level_values("ticker").unique().intersection(stocks.index)
    # stocks = stocks.loc[shared, :]
    # prices = prices.loc[idx[shared, :], :]
    # prices.info(show_counts=True)
    # stocks.info(show_counts=True)
    # print(stocks.sector.value_counts())
    #
    # # Optional: persist intermediate results:
    # with pd.HDFStore("../data/lin_models.h5") as store:
    #     store.put("prices", prices)
    #     store.put("stocks", stocks)

    with pd.HDFStore("../data/lin_models.h5") as store:
        prices = store["prices"]
        stocks = store["stocks"]
    ic(prices.head())

    ## Compute Rolling Average Dollar Volume
    # compute dollar volume to determine universe
    prices["dollar_vol"] = prices[["close", "volume"]].prod(axis=1)
    prices["dollar_vol_1m"] = (prices.dollar_vol.groupby("ticker").rolling(window=21).mean()).values
    prices.info(show_counts=True)

    prices["dollar_vol_rank"] = prices.groupby("date").dollar_vol_1m.rank(ascending=False)
    prices.info(show_counts=True)

    ## Add some Basic Factors
    ### Compute the Relative Strength Index
    prices["rsi"] = prices.groupby(level="ticker").close.apply(RSI)

    ax = sns.distplot(prices.rsi.dropna())
    ax.axvline(30, ls="--", lw=1, c="k")
    ax.axvline(70, ls="--", lw=1, c="k")
    ax.set_title("RSI Distribution with Signal Threshold")
    plt.tight_layout()
    plt.savefig("images/03-01.png", bboxinches="tight")

    ### Compute Bollinger Bands
    def compute_bb(close):
        high, mid, low = BBANDS(close, timeperiod=20)
        return pd.DataFrame({"bb_high": high, "bb_low": low}, index=close.index)

    prices = prices.join(prices.groupby(level="ticker").close.apply(compute_bb))
    prices["bb_high"] = prices.bb_high.sub(prices.close).div(prices.bb_high).apply(np.log1p)
    prices["bb_low"] = prices.close.sub(prices.bb_low).div(prices.close).apply(np.log1p)

    fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
    sns.distplot(prices.loc[prices.dollar_vol_rank < 100, "bb_low"].dropna(), ax=axes[0])
    sns.distplot(prices.loc[prices.dollar_vol_rank < 100, "bb_high"].dropna(), ax=axes[1])
    plt.tight_layout()
    plt.savefig("images/03-02.png", bboxinches="tight")

    ### Compute Average True Range
    def compute_atr(stock_data):
        df = ATR(stock_data.high, stock_data.low, stock_data.close, timeperiod=14)
        return df.sub(df.mean()).div(df.std())

    prices["atr"] = prices.groupby("ticker", group_keys=False).apply(compute_atr)

    fig = plt.figure(figsize=(10, 6))
    sns.distplot(prices[prices.dollar_vol_rank < 50].atr.dropna())
    plt.tight_layout()
    plt.savefig("images/03-03.png", bboxinches="tight")

    ### Compute Moving Average Convergence/Divergence
    def compute_macd(close):
        macd = MACD(close)[0]
        return (macd - np.mean(macd)) / np.std(macd)

    prices["macd"] = prices.groupby("ticker", group_keys=False).close.apply(compute_macd)
    print(
        prices.macd.describe(
            percentiles=[0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999]
        ).apply(lambda x: f"{x:,.1f}")
    )

    fig = plt.figure(figsize=(10, 6))
    sns.distplot(prices[prices.dollar_vol_rank < 100].macd.dropna())
    plt.tight_layout()
    plt.savefig("images/03-04.png", bboxinches="tight")

    ## Compute Lagged Returns
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

    ### Winsorize outliers
    for lag in lags:
        prices[f"return_{lag}d"] = (
            prices.groupby(level="ticker")
            .close.pct_change(lag)
            .pipe(lambda x: x.clip(lower=x.quantile(q), upper=x.quantile(1 - q)))
            .add(1)
            .pow(1 / lag)
            .sub(1)
        )

    ### Shift lagged returns
    for t in [1, 2, 3, 4, 5]:
        for lag in [1, 5, 10, 21]:
            prices[f"return_{lag}d_lag{t}"] = prices.groupby(level="ticker")[
                f"return_{lag}d"
            ].shift(t * lag)

    ## Compute Forward Returns
    for t in [1, 5, 10, 21]:
        prices[f"target_{t}d"] = prices.groupby(level="ticker")[f"return_{t}d"].shift(-t)

    ## Combine Price and Meta Data
    prices = prices.join(stocks[["sector"]])

    ## Create time and sector dummy variables
    prices["year"] = prices.index.get_level_values("date").year
    prices["month"] = prices.index.get_level_values("date").month
    prices.info(null_counts=True)
    prices.assign(sector=pd.factorize(prices.sector, sort=True)[0]).to_hdf(
        "../data/data.h5", "model_data/no_dummies"
    )

    prices = pd.get_dummies(
        prices,
        columns=["year", "month", "sector"],
        prefix=["year", "month", ""],
        prefix_sep=["_", "_", ""],
        drop_first=True,
    )
    prices.info(null_counts=True)

    ## Store Model Data
    prices.to_hdf("../data/data.h5", "model_data")

    ## Explore Data
    ### Plot Factors
    target = "target_5d"
    top100 = prices[prices.dollar_vol_rank < 100].copy()

    ### RSI
    top100.loc[:, "rsi_signal"] = pd.cut(top100.rsi, bins=[0, 30, 70, 100])
    print(top100.groupby("rsi_signal")["target_5d"].describe())

    ### Bollinger Bands
    metric = "bb_low"
    fig = plt.figure(figsize=(10, 6))
    j = sns.jointplot(x=metric, y=target, data=top100)
    plt.tight_layout()
    plt.savefig("images/03-05.png", bboxinches="tight")

    df = top100[[metric, target]].dropna()
    r, p = spearmanr(df[metric], df[target])
    print(f"{r:,.2%} ({p:.2%})")

    metric = "bb_high"
    fig = plt.figure(figsize=(10, 6))
    j = sns.jointplot(x=metric, y=target, data=top100)
    plt.tight_layout()
    plt.savefig("images/03-06.png", bboxinches="tight")

    df = top100[[metric, target]].dropna()
    r, p = spearmanr(df[metric], df[target])
    print(f"{r:,.2%} ({p:.2%})")

    ### ATR
    metric = "atr"
    fig = plt.figure(figsize=(10, 6))
    j = sns.jointplot(x=metric, y=target, data=top100)
    plt.tight_layout()
    plt.savefig("images/03-07.png", bboxinches="tight")

    df = top100[[metric, target]].dropna()
    r, p = spearmanr(df[metric], df[target])
    print(f"{r:,.2%} ({p:.2%})")

    ### MACD
    metric = "macd"
    fig = plt.figure(figsize=(10, 6))
    j = sns.jointplot(x=metric, y=target, data=top100)
    plt.tight_layout()
    plt.savefig("images/03-08.png", bboxinches="tight")

    df = top100[[metric, target]].dropna()
    r, p = spearmanr(df[metric], df[target])
    print(f"{r:,.2%} ({p:.2%})")
