# Long-Short Strategy, Part 1: Preparing Alpha Factors and Features
# In this section, we'll start designing, implementing, and evaluating a trading strategy for US equities driven by
# daily return forecasts produced by gradient boosting models.
# As in the previous examples, we'll lay out a framework and build a specific example that you can adapt to run your
# own experiments. There are numerous aspects that you can vary, from the asset class and investment universe to more
# granular aspects like the features, holding period, or trading rules. See, for example, the **Alpha Factor Library**
# in the [Appendix](../24_alpha_factor_library) for numerous additional features.
# We'll keep the trading strategy simple and only use a single ML signal; a real-life application will likely use
# multiple signals from different sources, such as complementary ML models trained on different datasets or with
# different lookahead or lookback periods. It would also use sophisticated risk management, from simple stop-loss to
# value-at-risk analysis.
# **Six notebooks** cover our workflow sequence:
# 1. `preparing_the_model_data` (this noteboook): we'll engineer a few simple features from the Quandl Wiki data
# 2. [trading_signals_with_lightgbm_and_catboost](05_trading_signals_with_lightgbm_and_catboost.ipynb): we tune
#      hyperparameters for LightGBM and CatBoost to select a model, using 2015/16 as our validation period.
# 3. [evaluate_trading_signals](06_evaluate_trading_signals.ipynb): we compare the cross-validation performance using
#      various metrics to select the best model.
# 4. [model_interpretation](07_model_interpretation.ipynb): we take a closer look at the drivers behind
#      the best model's predictions.
# 5. [making_out_of_sample_predictions](08_making_out_of_sample_predictions.ipynb): we generate predictions
#      for our out-of-sample test period 2017.
# 6. [backtesting_with_zipline](09_backtesting_with_zipline.ipynb): evaluate the historical performance of
#      a long-short strategy based on our predictive signals using Zipline.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import talib
from talib import RSI, BBANDS, MACD, ATR
import warnings

np.random.seed(42)
idx = pd.IndexSlice
sns.set_style("darkgrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:,.2f}".format

if __name__ == "__main__":
    MONTH = 21
    YEAR = 12 * MONTH

    START = "2010-01-01"
    END = "2017-12-31"

    percentiles = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05]
    percentiles += [1 - p for p in percentiles[::-1]]

    T = [1, 5, 10, 21, 42, 63]

    ## Loading Quandl Wiki Stock Prices & Meta Data
    DATA_STORE = "../data/assets.h5"
    ohlcv = ["adj_open", "adj_close", "adj_low", "adj_high", "adj_volume"]
    with pd.HDFStore(DATA_STORE) as store:
        prices = (
            store["quandl/wiki/prices"]
            .loc[idx[START:END, :], ohlcv]  # select OHLCV columns from 2010 until 2017
            .rename(columns=lambda x: x.replace("adj_", ""))  # simplify column names
            .swaplevel()
            .sort_index()
        )
        metadata = store["us_equities/stocks"].loc[:, ["marketcap", "sector"]]

    prices.volume /= 1e3  # make vol figures a bit smaller
    prices.index.names = ["symbol", "date"]
    metadata.index.name = "symbol"

    ## Remove stocks with insufficient observations
    # We require at least 7 years of data; we simplify and select using both in- and out-of-sample period; please be
    # aware that it would be more accurate to use only the training period to remove data to avoid lookahead bias.
    min_obs = 7 * YEAR
    nobs = prices.groupby(level="symbol").size()
    keep = nobs[nobs > min_obs].index
    prices = prices.loc[idx[keep, :], :]

    ### Align price and meta data
    metadata = metadata[~metadata.index.duplicated() & metadata.sector.notnull()]
    metadata.sector = metadata.sector.str.lower().str.replace(" ", "_")

    shared = prices.index.get_level_values("symbol").unique().intersection(metadata.index)
    metadata = metadata.loc[shared, :]
    prices = prices.loc[idx[shared, :], :]

    ### Limit universe to 1,000 stocks with the highest market cap
    # Again, we simplify and use the entire sample period, not just the training period, to select our universe.
    universe = metadata.marketcap.nlargest(1000).index
    prices = prices.loc[idx[universe, :], :]
    metadata = metadata.loc[universe]
    print(metadata.sector.value_counts())
    prices.info(show_counts=True)
    metadata.info()

    ### Rank assets by Rolling Average Dollar Volume
    #### Compute dollar volume
    prices["dollar_vol"] = prices[["close", "volume"]].prod(1).div(1e3)

    #### 21-day moving average
    # compute dollar volume to determine universe
    dollar_vol_ma = (
        prices.dollar_vol.unstack("symbol")
        .rolling(window=21, min_periods=1)
        .mean()  # 1 trading month
    )

    #### Rank stocks by moving average
    prices["dollar_vol_rank"] = (
        dollar_vol_ma.rank(axis=1, ascending=False).stack("symbol").swaplevel()
    )
    prices.info(show_counts=True)

    ## Add some Basic Factors
    # See [appendix](../24_alpha_factor_library) for details on the below indicators.
    ### Compute the Relative Strength Index
    prices["rsi"] = prices.groupby(level="symbol").close.apply(RSI)

    ax = sns.distplot(prices.rsi.dropna())
    ax.axvline(30, ls="--", lw=1, c="k")
    ax.axvline(70, ls="--", lw=1, c="k")
    ax.set_title("RSI Distribution with Signal Threshold")
    plt.tight_layout()
    plt.savefig("images/04-01.png")

    ### Compute Bollinger Bands
    def compute_bb(close):
        high, mid, low = BBANDS(close, timeperiod=20)
        return pd.DataFrame({"bb_high": high, "bb_low": low}, index=close.index)

    prices = prices.join(prices.groupby(level="symbol").close.apply(compute_bb))
    prices["bb_high"] = prices.bb_high.sub(prices.close).div(prices.bb_high).apply(np.log1p)
    prices["bb_low"] = prices.close.sub(prices.bb_low).div(prices.close).apply(np.log1p)

    fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
    sns.distplot(prices.loc[prices.dollar_vol_rank < 100, "bb_low"].dropna(), ax=axes[0])
    sns.distplot(prices.loc[prices.dollar_vol_rank < 100, "bb_high"].dropna(), ax=axes[1])
    plt.tight_layout()
    plt.savefig("images/04-02.png")

    ### Compute Average True Range
    prices["NATR"] = prices.groupby(level="symbol", group_keys=False).apply(
        lambda x: talib.NATR(x.high, x.low, x.close)
    )

    def compute_atr(stock_data):
        df = ATR(stock_data.high, stock_data.low, stock_data.close, timeperiod=14)
        return df.sub(df.mean()).div(df.std())

    prices["ATR"] = prices.groupby("symbol", group_keys=False).apply(compute_atr)

    ### Compute Moving Average Convergence/Divergence
    prices["PPO"] = prices.groupby(level="symbol").close.apply(talib.PPO)

    def compute_macd(close):
        macd = MACD(close)[0]
        return (macd - np.mean(macd)) / np.std(macd)

    prices["MACD"] = prices.groupby("symbol", group_keys=False).close.apply(compute_macd)

    ### Combine Price and Meta Data
    metadata.sector = pd.factorize(metadata.sector)[0].astype(int)
    prices = prices.join(metadata[["sector"]])

    ## Compute Returns
    ### Historical Returns
    by_sym = prices.groupby(level="symbol").close
    for t in T:
        prices[f"r{t:02}"] = by_sym.pct_change(t)

    ### Daily historical return deciles
    for t in T:
        prices[f"r{t:02}dec"] = (
            prices[f"r{t:02}"]
            .groupby(level="date")
            .apply(lambda x: pd.qcut(x, q=10, labels=False, duplicates="drop"))
        )

    ### Daily sector return deciles
    for t in T:
        prices[f"r{t:02}q_sector"] = prices.groupby(["date", "sector"])[f"r{t:02}"].transform(
            lambda x: pd.qcut(x, q=5, labels=False, duplicates="drop")
        )

    ### Compute Forward Returns
    for t in [1, 5, 21]:
        prices[f"r{t:02}_fwd"] = prices.groupby(level="symbol")[f"r{t:02}"].shift(-t)

    ## Remove outliers
    print(prices[[f"r{t:02}" for t in T]].describe())

    # We remove daily returns above 100 percent as these are more likely to represent data errors; we are using the
    # 100 percent cutoff here in a somewhat ad-hoc fashion; you would want to apply more careful exploratory and
    # historical analysis to decide which assets are truly not representative of the sample period.
    outliers = prices[prices.r01 > 1].index.get_level_values("symbol").unique()
    prices = prices.drop(outliers, level="symbol")

    ## Create time and sector dummy variables
    prices["year"] = prices.index.get_level_values("date").year
    prices["month"] = prices.index.get_level_values("date").month
    prices["weekday"] = prices.index.get_level_values("date").weekday

    ## Store Model Data
    prices.info(show_counts=True)
    prices.drop(["open", "close", "low", "high", "volume"], axis=1).to_hdf(
        "../data/12_data.h5", "model_data"
    )
