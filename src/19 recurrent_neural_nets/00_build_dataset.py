# Create a dataset formatted for RNN examples

from pathlib import Path

import numpy as np
import pandas as pd
import warnings

idx = pd.IndexSlice
np.random.seed(seed=42)
pd.options.display.float_format = "{:,.2f}".format
warnings.filterwarnings("ignore")

DATA_DIR = Path("..", "data")


if __name__ == "__main__":
    prices = pd.read_hdf(DATA_DIR / "assets.h5", "quandl/wiki/prices").loc[
        idx["2010":"2017", :], ["adj_close", "adj_volume"]
    ]
    prices.info()

    ### Select most traded stocks
    n_dates = len(prices.index.unique("date"))
    dollar_vol = (
        prices.adj_close.mul(prices.adj_volume)
        .unstack("ticker")
        .dropna(thresh=int(0.95 * n_dates), axis=1)
        .rank(ascending=False, axis=1)
        .stack("ticker")
    )
    most_traded = dollar_vol.groupby(level="ticker").mean().nsmallest(500).index

    returns = (
        prices.loc[idx[:, most_traded], "adj_close"]
        .unstack("ticker")
        .pct_change()
        .sort_index(ascending=False)
    )
    returns.info()

    ### Stack 21-day time series
    n = len(returns)
    T = 21  # days
    tcols = list(range(T))
    tickers = returns.columns

    data = pd.DataFrame()
    for i in range(n - T - 1):
        df = returns.iloc[i : i + T + 1]
        date = df.index.max()
        data = pd.concat(
            [
                data,
                df.reset_index(drop=True)
                .T.assign(date=date, ticker=tickers)
                .set_index(["ticker", "date"]),
            ]
        )
    data = data.rename(columns={0: "label"}).sort_index().dropna()
    data.loc[:, tcols[1:]] = data.loc[:, tcols[1:]].apply(
        lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
    )
    data.info()
    print(data.shape)

    data.to_hdf("../data/19_data.h5", "returns_daily")

    ## Build weekly dataset
    # We load the Quandl adjusted stock price data:
    prices = (
        pd.read_hdf(DATA_DIR / "assets.h5", "quandl/wiki/prices").adj_close.unstack().loc["2007":]
    )
    prices.info()

    ### Resample to weekly frequency
    # We start by generating weekly returns for close to 2,500 stocks without missing data for the 2008-17 period, as follows:
    returns = (
        prices.resample("W")
        .last()
        .pct_change()
        .loc["2008":"2017"]
        .dropna(axis=1)
        .sort_index(ascending=False)
    )
    returns.info()
    print(returns.head().append(returns.tail()))

    ### Create & stack 52-week sequences
    # We'll use 52-week sequences, which we'll create in a stacked format:
    n = len(returns)
    T = 52  # weeks
    tcols = list(range(T))
    tickers = returns.columns

    data = pd.DataFrame()
    for i in range(n - T - 1):
        df = returns.iloc[i : i + T + 1]
        date = df.index.max()
        data = pd.concat(
            [
                data,
                (
                    df.reset_index(drop=True)
                    .T.assign(date=date, ticker=tickers)
                    .set_index(["ticker", "date"])
                ),
            ]
        )
    data.info()

    data[tcols] = data[tcols].apply(
        lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
    )
    data = data.rename(columns={0: "fwd_returns"})
    data["label"] = (data["fwd_returns"] > 0).astype(int)
    print(data.shape)

    data.sort_index().to_hdf("../data/19_data.h5", "returns_weekly")
