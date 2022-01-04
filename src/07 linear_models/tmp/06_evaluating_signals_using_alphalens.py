import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from alphalens.tears import create_summary_tear_sheet
from alphalens.utils import get_clean_factor_and_forward_returns
from icecream import ic

sns.set_style("darkgrid")
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    idx = pd.IndexSlice
    YEAR = 252

    with pd.HDFStore("../data/data.h5") as store:
        lr_predictions = store["lr/predictions"]
        lasso_predictions = store["lasso/predictions"]
        lasso_scores = store["lasso/scores"]
        ridge_predictions = store["ridge/predictions"]
        ridge_scores = store["ridge/scores"]

    def get_trade_prices(tickers, start, stop):
        prices = pd.read_hdf("../data/assets.h5", "quandl/wiki/prices").swaplevel().sort_index()
        prices.index.names = ["symbol", "date"]
        prices = prices.loc[idx[tickers, str(start) : str(stop)], "adj_open"]
        return prices.unstack("symbol").sort_index().shift(-1).tz_localize("UTC")

    def get_best_alpha(scores):
        return scores.groupby("alpha").ic.mean().idxmax()

    def get_factor(predictions):
        return (
            predictions.unstack("symbol")
            .dropna(how="all")
            .stack()
            .tz_localize("UTC", level="date")
            .sort_index()
        )

    ## Linear Regression
    lr_factor = get_factor(lr_predictions.predicted.swaplevel())
    ic(lr_factor.head())

    tickers = lr_factor.index.get_level_values("symbol").unique()
    trade_prices = get_trade_prices(tickers, 2014, 2017)
    trade_prices.info()

    lr_factor_data = get_clean_factor_and_forward_returns(
        factor=lr_factor, prices=trade_prices, quantiles=5, periods=(1, 5, 10, 21)
    )
    lr_factor_data.info()

    create_summary_tear_sheet(lr_factor_data)

    ## Ridge Regression
    best_ridge_alpha = get_best_alpha(ridge_scores)
    ridge_predictions = ridge_predictions[ridge_predictions.alpha == best_ridge_alpha].drop(
        "alpha", axis=1
    )
    ridge_factor = get_factor(ridge_predictions.predicted.swaplevel())
    ic(ridge_factor.head())

    ridge_factor_data = get_clean_factor_and_forward_returns(
        factor=ridge_factor, prices=trade_prices, quantiles=5, periods=(1, 5, 10, 21)
    )
    ridge_factor_data.info()

    create_summary_tear_sheet(ridge_factor_data)

    ## Lasso Regression
    best_lasso_alpha = get_best_alpha(lasso_scores)
    lasso_predictions = lasso_predictions[lasso_predictions.alpha == best_lasso_alpha].drop(
        "alpha", axis=1
    )
    lasso_factor = get_factor(lasso_predictions.predicted.swaplevel())
    ic(lasso_factor.head())

    lasso_factor_data = get_clean_factor_and_forward_returns(
        factor=lasso_factor, prices=trade_prices, quantiles=5, periods=(1, 5, 10, 21)
    )
    lasso_factor_data.info()

    create_summary_tear_sheet(lasso_factor_data)
