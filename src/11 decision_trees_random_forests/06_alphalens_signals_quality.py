# Testing the signal quality with Alphalens
from pathlib import Path
import pandas as pd
import seaborn as sns

from alphalens.tears import create_summary_tear_sheet, create_full_tear_sheet
from alphalens.utils import get_clean_factor_and_forward_returns

idx = pd.IndexSlice
sns.set_style("whitegrid")
pd.options.display.float_format = "{:,.2f}".format

### Get AlphaLens Input
DATA_DIR = Path("..", "data")
results_path = Path("../data/ch11", "return_predictions")
if not results_path.exists():
    results_path.mkdir(parents=True)


# Using next available prices.
def get_trade_prices(tickers):
    store = DATA_DIR / "assets.h5"
    prices = pd.read_hdf(store, "stooq/jp/tse/stocks/prices")
    return (
        prices.loc[idx[tickers, "2014":"2019"], "open"]
        .unstack("ticker")
        .sort_index()
        .shift(-1)
        .dropna()
        .tz_localize("UTC")
    )


if __name__ == "__main__":
    ## Evaluating the Cross-Validation Results
    lookahead = 1
    cv_store = Path(results_path / "parameter_tuning.h5")

    # Reloading predictions.
    best_predictions = pd.read_hdf(results_path / "predictions.h5", f"test/{lookahead:02}")
    best_predictions.info()

    test_tickers = best_predictions.index.get_level_values("ticker").unique()
    trade_prices = get_trade_prices(test_tickers)
    trade_prices.info()

    factor = (
        best_predictions.iloc[:, :3]
        .mean(1)
        .tz_localize("UTC", level="date")
        .swaplevel()
        .dropna()
        .reset_index()
        .drop_duplicates()
        .set_index(["date", "ticker"])
    )

    factor_data = get_clean_factor_and_forward_returns(
        factor=factor, prices=trade_prices, quantiles=5, periods=(1, 5, 10, 21)
    )
    factor_data.sort_index().info()

    ### Summary Tearsheet
    create_summary_tear_sheet(factor_data)

    ## Evaluating the Out-of-sample predictions
    ### Prepare Factor Data
    t = 1
    predictions = pd.read_hdf(results_path / "predictions.h5", f"test/{t:02}").drop(
        "y_test", axis=1
    )
    predictions.info()

    factor = (
        predictions.iloc[:, :10]
        .mean(1)
        .sort_index()
        .tz_localize("UTC", level="date")
        .swaplevel()
        .dropna()
    )
    factor.head()

    ### Select next available trade prices
    # Using next available prices.
    tickers = factor.index.get_level_values("ticker").unique()
    trade_prices = get_trade_prices(tickers)
    trade_prices.info()

    ### Get AlphaLens Inputs
    factor_data = get_clean_factor_and_forward_returns(
        factor=factor, prices=trade_prices, quantiles=5, periods=(1, 5, 10, 21)
    )
    factor_data.sort_index().info()

    ### Summary Tearsheet
    create_summary_tear_sheet(factor_data)
