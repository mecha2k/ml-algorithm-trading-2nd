import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

idx = pd.IndexSlice
np.random.seed(seed=42)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
pd.options.display.float_format = "{:,.2f}".format

DATA_STORE = Path("..", "data", "assets.h5")

if __name__ == "__main__":
    deciles = np.arange(0.1, 1, 0.1).round(1)

    with pd.HDFStore(DATA_STORE) as store:
        data = (
            store["quandl/wiki/prices"]
            .loc[
                idx["2007":"2016", :],
                ["adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"],
            ]
            .dropna()
            .swaplevel()
            .sort_index()
            .rename(columns=lambda x: x.replace("adj_", ""))
        )
        metadata = store["us_equities/stocks"].loc[:, ["marketcap", "sector"]]
    data.info(show_counts=True)

    metadata.sector = pd.factorize(metadata.sector)[0]
    metadata.info()

    data = data.join(metadata).dropna(subset=["sector"])
    data.info(show_counts=True)
    print(
        f"# Tickers: {len(data.index.unique('ticker')):,.0f} | # Dates: {len(data.index.unique('date')):,.0f}"
    )

    ## Select 500 most-traded stocks
    dv = data.close.mul(data.volume)
    top500 = (
        dv.groupby(level="date")
        .rank(ascending=False)
        .unstack("ticker")
        .dropna(thresh=8 * 252, axis=1)
        .mean()
        .nsmallest(500)
    )

    ### Visualize the 200 most liquid stocks
    top200 = (
        data.close.mul(data.volume)
        .unstack("ticker")
        .dropna(thresh=8 * 252, axis=1)
        .mean()
        .div(1e6)
        .nlargest(200)
    )
    cutoffs = [0, 50, 100, 150, 200]
    fig, axes = plt.subplots(ncols=4, figsize=(20, 10), sharex=True)
    axes = axes.flatten()
    for i, cutoff in enumerate(cutoffs[1:], 1):
        top200.iloc[cutoffs[i - 1] : cutoffs[i]].sort_values().plot.barh(logx=True, ax=axes[i - 1])
    fig.tight_layout()
    plt.savefig("images/01-01.png")

    to_drop = data.index.unique("ticker").difference(top500.index)
    print(len(to_drop))

    data = data.drop(to_drop, level="ticker")
    data.info(show_counts=True)
    print(
        f"# Tickers: {len(data.index.unique('ticker')):,.0f} | # Dates: {len(data.index.unique('date')):,.0f}"
    )

    ### Remove outlier observations based on daily returns
    before = len(data)
    data["ret"] = data.groupby("ticker").close.pct_change()
    data = data[data.ret.between(-1, 1)].drop("ret", axis=1)
    print(f"Dropped {before-len(data):,.0f}")
    tickers = data.index.unique("ticker")
    print(f"# Tickers: {len(tickers):,.0f} | # Dates: {len(data.index.unique('date')):,.0f}")

    ### Sample price data for illustration
    ticker = "AAPL"
    # alternative
    # ticker = np.random.choice(tickers)
    price_sample = data.loc[idx[ticker, :], :].reset_index("ticker", drop=True)
    price_sample.info()
    price_sample.to_hdf("../data/24_data.h5", "data/sample")

    ## Compute returns
    # Group data by ticker
    by_ticker = data.groupby(level="ticker")

    ### Historical returns
    T = [1, 2, 3, 4, 5, 10, 21, 42, 63, 126, 252]
    for t in T:
        data[f"ret_{t:02}"] = by_ticker.close.pct_change(t)

    ### Forward returns
    data["ret_fwd"] = by_ticker.ret_01.shift(-1)
    data = data.dropna(subset=["ret_fwd"])
    data.info(show_counts=True)
    data.to_hdf("../data/24_data.h5", "data/top500")
