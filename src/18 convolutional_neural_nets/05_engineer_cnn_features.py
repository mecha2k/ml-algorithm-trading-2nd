# Engineer features and convert time series data to images
from talib import (
    RSI,
    BBANDS,
    MACD,
    NATR,
    WILLR,
    WMA,
    EMA,
    SMA,
    CCI,
    CMO,
    MACD,
    PPO,
    ROC,
    ADOSC,
    ADX,
    MOM,
)
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import pandas_datareader.data as web
import pandas as pd
import numpy as np
from pathlib import Path
import warnings


idx = pd.IndexSlice
np.random.seed(seed=42)
tf.random.set_seed(seed=42)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
pd.options.display.float_format = "{:,.2f}".format
warnings.filterwarnings("ignore")

DATA_STORE = "../data/assets.h5"

results_path = Path("../data/ch18", "cnn_for_trading")
mnist_path = results_path / "mnist"
if not mnist_path.exists():
    mnist_path.mkdir(parents=True)

if __name__ == "__main__":
    MONTH = 21
    YEAR = 12 * MONTH

    START = "2000-01-01"
    END = "2017-12-31"

    T = [1, 5, 10, 21, 42, 63]

    ## Loading Quandl Wiki Stock Prices & Meta Data
    adj_ohlcv = ["adj_open", "adj_close", "adj_low", "adj_high", "adj_volume"]
    with pd.HDFStore(DATA_STORE) as store:
        prices = (
            store["quandl/wiki/prices"]
            .loc[idx[START:END, :], adj_ohlcv]
            .rename(columns=lambda x: x.replace("adj_", ""))
            .swaplevel()
            .sort_index()
            .dropna()
        )
        metadata = store["us_equities/stocks"].loc[:, ["marketcap", "sector"]]
    ohlcv = prices.columns.tolist()

    prices.volume /= 1e3
    prices.index.names = ["symbol", "date"]
    metadata.index.name = "symbol"

    ## Rolling universe: pick 500 most-traded stocks
    dollar_vol = prices.close.mul(prices.volume).unstack("symbol").sort_index()
    years = sorted(np.unique([d.year for d in prices.index.get_level_values("date").unique()]))

    train_window = 5  # years
    universe_size = 500

    universe = []
    for i, year in enumerate(years[5:], 5):
        start = str(years[i - 5])
        end = str(years[i])
        most_traded = (
            dollar_vol.loc[start:end, :]
            .dropna(thresh=1000, axis=1)
            .median()
            .nlargest(universe_size)
            .index
        )
        universe.append(prices.loc[idx[most_traded, start:end], :])
    universe = pd.concat(universe)
    universe = universe.loc[~universe.index.duplicated()]
    universe.info(show_counts=True)
    universe.groupby("symbol").size().describe()
    universe.to_hdf("../data/18_data.h5", "universe")

    ## Generate Technical Indicators Factors
    T = list(range(6, 21))

    ### Relative Strength Index
    for t in T:
        universe[f"{t:02}_RSI"] = universe.groupby(level="symbol").close.apply(RSI, timeperiod=t)

    ### Williams %R
    for t in T:
        universe[f"{t:02}_WILLR"] = universe.groupby(level="symbol", group_keys=False).apply(
            lambda x: WILLR(x.high, x.low, x.close, timeperiod=t)
        )

    ### Compute Bollinger Bands
    def compute_bb(close, timeperiod):
        high, mid, low = BBANDS(close, timeperiod=timeperiod)
        return pd.DataFrame(
            {f"{timeperiod:02}_BBH": high, f"{timeperiod:02}_BBL": low}, index=close.index
        )

    for t in T:
        bbh, bbl = f"{t:02}_BBH", f"{t:02}_BBL"
        universe = universe.join(
            universe.groupby(level="symbol").close.apply(compute_bb, timeperiod=t)
        )
        universe[bbh] = universe[bbh].sub(universe.close).div(universe[bbh]).apply(np.log1p)
        universe[bbl] = universe.close.sub(universe[bbl]).div(universe.close).apply(np.log1p)

    ### Normalized Average True Range
    for t in T:
        universe[f"{t:02}_NATR"] = universe.groupby(level="symbol", group_keys=False).apply(
            lambda x: NATR(x.high, x.low, x.close, timeperiod=t)
        )

    ### Percentage Price Oscillator
    for t in T:
        universe[f"{t:02}_PPO"] = universe.groupby(level="symbol").close.apply(
            PPO, fastperiod=t, matype=1
        )

    ### Moving Average Convergence/Divergence
    def compute_macd(close, signalperiod):
        macd = MACD(close, signalperiod=signalperiod)[0]
        return (macd - np.mean(macd)) / np.std(macd)

    for t in T:
        universe[f"{t:02}_MACD"] = universe.groupby("symbol", group_keys=False).close.apply(
            compute_macd, signalperiod=t
        )

    ### Momentum
    for t in T:
        universe[f"{t:02}_MOM"] = universe.groupby(level="symbol").close.apply(MOM, timeperiod=t)

    ### Weighted Moving Average
    for t in T:
        universe[f"{t:02}_WMA"] = universe.groupby(level="symbol").close.apply(WMA, timeperiod=t)

    ### Exponential Moving Average
    for t in T:
        universe[f"{t:02}_EMA"] = universe.groupby(level="symbol").close.apply(EMA, timeperiod=t)

    ### Commodity Channel Index
    for t in T:
        universe[f"{t:02}_CCI"] = universe.groupby(level="symbol", group_keys=False).apply(
            lambda x: CCI(x.high, x.low, x.close, timeperiod=t)
        )

    ### Chande Momentum Oscillator
    for t in T:
        universe[f"{t:02}_CMO"] = universe.groupby(level="symbol").close.apply(CMO, timeperiod=t)

    ### Rate of Change
    # Rate of change is a technical indicator that illustrates the speed of price change over a period of time.
    for t in T:
        universe[f"{t:02}_ROC"] = universe.groupby(level="symbol").close.apply(ROC, timeperiod=t)

    ### Chaikin A/D Oscillator
    for t in T:
        universe[f"{t:02}_ADOSC"] = universe.groupby(level="symbol", group_keys=False).apply(
            lambda x: ADOSC(x.high, x.low, x.close, x.volume, fastperiod=t - 3, slowperiod=4 + t)
        )

    ### Average Directional Movement Index
    for t in T:
        universe[f"{t:02}_ADX"] = universe.groupby(level="symbol", group_keys=False).apply(
            lambda x: ADX(x.high, x.low, x.close, timeperiod=t)
        )

    universe.drop(ohlcv, axis=1).to_hdf("../data/18_data.h5", "features")

    ## Compute Historical Returns
    # ### Historical Returns
    by_sym = universe.groupby(level="symbol").close
    for t in [1, 5]:
        universe[f"r{t:02}"] = by_sym.pct_change(t)

    ### Remove outliers
    universe[[f"r{t:02}" for t in [1, 5]]].describe()
    outliers = universe[universe.r01 > 1].index.get_level_values("symbol").unique()
    print(len(outliers))

    universe = universe.drop(outliers, level="symbol")

    ### Historical return quantiles
    for t in [1, 5]:
        universe[f"r{t:02}dec"] = (
            universe[f"r{t:02}"]
            .groupby(level="date")
            .apply(lambda x: pd.qcut(x, q=10, labels=False, duplicates="drop"))
        )

    ## Rolling Factor Betas
    factor_data = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=START
    )[0].rename(columns={"Mkt-RF": "Market"})
    factor_data.index.names = ["date"]
    factor_data.info()

    windows = list(range(15, 90, 5))
    print(len(windows))

    t = 1
    ret = f"r{t:02}"
    factors = ["Market", "SMB", "HML", "RMW", "CMA"]
    windows = list(range(15, 90, 5))
    for window in windows:
        print(window)
        betas = []
        for symbol, data in universe.groupby(level="symbol"):
            model_data = data[[ret]].merge(factor_data, on="date").dropna()
            model_data[ret] -= model_data.RF

            rolling_ols = RollingOLS(
                endog=model_data[ret], exog=sm.add_constant(model_data[factors]), window=window
            )
            factor_model = rolling_ols.fit(params_only=True).params.drop("const", axis=1)
            result = factor_model.assign(symbol=symbol).set_index("symbol", append=True)
            betas.append(result)
        betas = pd.concat(betas).rename(columns=lambda x: f"{window:02}_{x}")
        universe = universe.join(betas)

    ## Compute Forward Returns
    for t in [1, 5]:
        universe[f"r{t:02}_fwd"] = universe.groupby(level="symbol")[f"r{t:02}"].shift(-t)
        universe[f"r{t:02}dec_fwd"] = universe.groupby(level="symbol")[f"r{t:02}dec"].shift(-t)

    ## Store Model Data
    universe = universe.drop(ohlcv, axis=1)
    universe.info(show_counts=True)

    drop_cols = ["r01", "r01dec", "r05", "r05dec"]
    outcomes = universe.filter(like="_fwd").columns

    universe = universe.sort_index()
    with pd.HDFStore("../data/18_data.h5") as store:
        store.put(
            "features",
            universe.drop(drop_cols, axis=1).drop(outcomes, axis=1).loc[idx[:, "2001":], :],
        )
        store.put("targets", universe.loc[idx[:, "2001":], outcomes])
