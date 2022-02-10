from pathlib import Path

import numpy as np
import pandas as pd
import talib

import matplotlib.pyplot as plt
import seaborn as sns


idx = pd.IndexSlice
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
pd.options.display.float_format = "{:,.2f}".format

np.random.seed(seed=42)

### Stooq Japanese Equity data 2014-2019
DATA_DIR = Path("..", "data")

if __name__ == "__main__":
    prices = (
        pd.read_hdf(DATA_DIR / "assets.h5", "stooq/jp/tse/stocks/prices")
        .loc[idx[:, "2010":"2019"], :]
        .loc[lambda df: ~df.index.duplicated(), :]
    )
    prices.info(show_counts=True)

    before = len(prices.index.unique("ticker").unique())

    ### Remove symbols with missing values
    prices = (
        prices.unstack("ticker")
        .sort_index()
        .ffill(limit=5)
        .dropna(axis=1)
        .stack("ticker")
        .swaplevel()
    )
    prices.info(show_counts=True)

    after = len(prices.index.unique("ticker").unique())
    print(f"Before: {before:,.0f} after: {after:,.0f}")

    ### Keep most traded symbols
    dv = prices.close.mul(prices.volume)
    keep = dv.groupby("ticker").median().nlargest(1000).index.tolist()

    prices = prices.loc[idx[keep, :], :]
    prices.info(show_counts=True)

    ## Feature Engineering
    ### Compute period returns
    intervals = [1, 5, 10, 21, 63]

    returns = []
    by_ticker = prices.groupby(level="ticker").close
    for t in intervals:
        returns.append(by_ticker.pct_change(t).to_frame(f"ret_{t}"))
    returns = pd.concat(returns, axis=1)
    returns.info(show_counts=True)

    ### Remove outliers
    max_ret_by_sym = returns.groupby(level="ticker").max()

    percentiles = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
    percentiles += [1 - p for p in percentiles]
    max_ret_by_sym.describe(percentiles=sorted(percentiles)[6:])

    quantiles = max_ret_by_sym.quantile(0.95)
    to_drop = []
    for ret, q in quantiles.items():
        to_drop.extend(max_ret_by_sym[max_ret_by_sym[ret] > q].index.tolist())

    to_drop = pd.Series(to_drop).value_counts()
    to_drop = to_drop[to_drop > 1].index.tolist()
    print(len(to_drop))

    prices = prices.drop(to_drop, level="ticker")
    prices.info(show_counts=True)

    ### Calculate relative return percentiles
    returns = []
    by_sym = prices.groupby(level="ticker").close
    for t in intervals:
        ret = by_sym.pct_change(t)
        rel_perc = ret.groupby(level="date").apply(
            lambda x: pd.qcut(x, q=20, labels=False, duplicates="drop")
        )
        returns.extend([ret.to_frame(f"ret_{t}"), rel_perc.to_frame(f"ret_rel_perc_{t}")])
    returns = pd.concat(returns, axis=1)

    ### Technical Indicators
    #### Percentage Price Oscillator
    ppo = prices.groupby(level="ticker").close.apply(talib.PPO).to_frame("PPO")

    #### Normalized Average True Range
    natr = (
        prices.groupby(level="ticker", group_keys=False)
        .apply(lambda x: talib.NATR(x.high, x.low, x.close))
        .to_frame("NATR")
    )

    #### Relative Strength Indicator
    rsi = prices.groupby(level="ticker").close.apply(talib.RSI).to_frame("RSI")

    #### Bollinger Bands
    def get_bollinger(x):
        u, m, l = talib.BBANDS(x)
        return pd.DataFrame({"u": u, "m": m, "l": l})

    bbands = prices.groupby(level="ticker").close.apply(get_bollinger)

    ### Combine Features
    data = pd.concat([prices, returns, ppo, natr, rsi, bbands], axis=1)
    data["bbl"] = data.close.div(data.l)
    data["bbu"] = data.u.div(data.close)
    data = data.drop(["u", "m", "l"], axis=1)
    data.bbu.corr(data.bbl, method="spearman")

    ### Plot Indicators for randomly sample ticker
    indicators = ["close", "bbl", "bbu", "PPO", "NATR", "RSI"]
    ticker = np.random.choice(data.index.get_level_values("ticker"))
    (
        data.loc[idx[ticker, :], indicators]
        .reset_index("ticker", drop=True)
        .plot(lw=1, subplots=True, figsize=(16, 10), title=indicators, layout=(3, 2), legend=False)
    )
    plt.suptitle(ticker, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("images/04-01.png")

    data = data.drop(prices.columns, axis=1)

    ### Create time period indicators
    dates = data.index.get_level_values("date")
    data["weekday"] = dates.weekday
    data["month"] = dates.month
    data["year"] = dates.year

    ## Compute forward returns
    outcomes = []
    by_ticker = data.groupby("ticker")
    for t in intervals:
        k = f"fwd_ret_{t:02}"
        outcomes.append(k)
        data[k] = by_ticker[f"ret_{t}"].shift(-t)
    data.info(show_counts=True)

    data.to_hdf("../data/11_data.h5", "stooq/japan/equities")
