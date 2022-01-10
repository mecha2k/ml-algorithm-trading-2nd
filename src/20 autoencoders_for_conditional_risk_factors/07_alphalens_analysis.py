# Performance Analysis with Alphalens

from pathlib import Path
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from alphalens.tears import (
    create_returns_tear_sheet,
    create_summary_tear_sheet,
    create_full_tear_sheet,
)

from alphalens.performance import mean_return_by_quantile
from alphalens.plotting import plot_quantile_returns_bar
from alphalens.utils import get_clean_factor_and_forward_returns, rate_of_return
import warnings

np.random.seed(seed=42)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
warnings.filterwarnings("ignore")

results_path = Path("../data/ch20", "asset_pricing")
if not results_path.exists():
    results_path.mkdir(parents=True)


if __name__ == "__main__":
    ## Alphalens Analysis
    ### Load predictions
    DATA_STORE = Path(results_path / "data.h5")

    predictions = pd.read_hdf(results_path / "predictions.h5", "predictions")
    factor = (
        predictions.mean(axis=1)
        .unstack("ticker")
        .resample("W-FRI", level="date")
        .last()
        .stack()
        .tz_localize("UTC", level="date")
        .sort_index()
    )
    tickers = factor.index.get_level_values("ticker").unique()

    ### Get trade prices
    def get_trade_prices(tickers):
        prices = pd.read_hdf(DATA_STORE, "stocks/prices/adjusted")
        prices.index.names = ["ticker", "date"]
        prices = prices.loc[idx[tickers, "2014":"2020"], "open"]
        return (
            prices.unstack("ticker")
            .sort_index()
            .shift(-1)
            .resample("W-FRI", level="date")
            .last()
            .tz_localize("UTC")
        )

    trade_prices = get_trade_prices(tickers)
    trade_prices.info()
    trade_prices.to_hdf(results_path / "tmp.h5", "trade_prices")

    ### Generate tearsheet input
    factor_data = get_clean_factor_and_forward_returns(
        factor=factor, prices=trade_prices, quantiles=5, periods=(5, 10, 21)
    ).sort_index()
    factor_data.info()

    ### Create Tearsheet
    create_summary_tear_sheet(factor_data)
    create_full_tear_sheet(factor_data)
