import numpy as np
import pandas as pd
from scipy.stats import spearmanr

pd.set_option("display.expand_frame_repr", False)
np.random.seed(42)


def get_backtest_data():
    """Combine chapter 7 lasso regression predictions
    with adjusted OHLCV Quandl Wiki data"""
    with pd.HDFStore("../data/assets.h5") as store:
        prices = (
            store["quandl/wiki/prices"]
            .filter(like="adj")
            .rename(columns=lambda x: x.replace("adj_", ""))
            .swaplevel(axis=0)
        )

    with pd.HDFStore("../data/data.h5") as store:
        predictions = store["lasso/predictions"]

    best_alpha = (
        predictions.groupby("alpha").apply(lambda x: spearmanr(x.actuals, x.predicted)[0]).idxmax()
    )
    predictions = predictions[predictions.alpha == best_alpha]
    predictions.index.names = ["ticker", "date"]
    tickers = predictions.index.get_level_values("ticker").unique()
    start = predictions.index.get_level_values("date").min().strftime("%Y-%m-%d")
    stop = (predictions.index.get_level_values("date").max() + pd.DateOffset(1)).strftime(
        "%Y-%m-%d"
    )
    idx = pd.IndexSlice
    prices = prices.sort_index().loc[idx[tickers, start:stop], :]
    predictions = predictions.loc[predictions.alpha == best_alpha, ["predicted"]]
    return predictions.join(prices, how="right")


df = get_backtest_data()
df.to_hdf("../data/backtest.h5", "data")
df.info()
