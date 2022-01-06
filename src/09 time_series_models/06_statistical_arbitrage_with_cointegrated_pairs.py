# Pairs Selection using Cointegration Tests & Kalman Filter
from collections import Counter
from time import time
from pathlib import Path

import numpy as np
import pandas as pd

from pykalman import KalmanFilter
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

idx = pd.IndexSlice
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
warnings.filterwarnings("ignore")


def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return f"{h:>02.0f}:{m:>02.0f}:{s:>02.0f}"


### Johansen Test Critical Values
critical_values = {
    0: {0.9: 13.4294, 0.95: 15.4943, 0.99: 19.9349},
    1: {0.9: 2.7055, 0.95: 3.8415, 0.99: 6.6349},
}

trace0_cv = critical_values[0][0.95]  # critical value for 0 cointegration relationships
trace1_cv = critical_values[1][0.95]  # critical value for 1 cointegration relationship

DATA_PATH = Path("..", "data")
STORE = DATA_PATH / "assets.h5"

### Get backtest prices
# Combine OHLCV prices for relevant stock and ETF tickers.
def get_backtest_prices():
    with pd.HDFStore("../data/data09.h5") as store:
        tickers = store["tickers"]

    with pd.HDFStore(STORE) as store:
        prices = (
            pd.concat(
                [
                    store["stooq/us/nyse/stocks/prices"],
                    store["stooq/us/nyse/etfs/prices"],
                    store["stooq/us/nasdaq/etfs/prices"],
                    store["stooq/us/nasdaq/stocks/prices"],
                ]
            )
            .sort_index()
            .loc[idx[tickers.index, "2016":"2019"], :]
        )
    print(prices.info(null_counts=True))
    prices.to_hdf("../data/09_backtest.h5", "prices")
    tickers.to_hdf("../data/09_backtest.h5", "tickers")


get_backtest_prices()

### Load Stock Prices
stocks = pd.read_hdf("../data/data09.h5", "stocks/close").loc["2015":]
stocks.info()

### Load ETF Data
etfs = pd.read_hdf("../data/data09.h5", "etfs/close").loc["2015":]
etfs.info()

### Load Ticker Dictionary
names = pd.read_hdf("../data/data09.h5", "tickers").to_dict()
print(pd.Series(names).count())

## Precompute Cointegration
def test_cointegration(etfs, stocks, test_end, lookback=2):
    start = time()
    results = []
    test_start = test_end - pd.DateOffset(years=lookback) + pd.DateOffset(days=1)
    etf_tickers = etfs.columns.tolist()
    etf_data = etfs.loc[str(test_start) : str(test_end)]

    stock_tickers = stocks.columns.tolist()
    stock_data = stocks.loc[str(test_start) : str(test_end)]
    n = len(etf_tickers) * len(stock_tickers)
    j = 0
    for i, s1 in enumerate(etf_tickers, 1):
        for s2 in stock_tickers:
            j += 1
            if j % 1000 == 0:
                print(f"\t{j:5,.0f} ({j/n:3.1%}) | {time() - start:.2f}")
            df = etf_data.loc[:, [s1]].dropna().join(stock_data.loc[:, [s2]].dropna(), how="inner")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                var = VAR(df)
                lags = var.select_order()
                result = [test_end, s1, s2]
                order = lags.selected_orders["aic"]
                result += [coint(df[s1], df[s2], trend="c")[1], coint(df[s2], df[s1], trend="c")[1]]

            cj = coint_johansen(df, det_order=0, k_ar_diff=order)
            result += list(cj.lr1) + list(cj.lr2) + list(cj.evec[:, cj.ind[0]])
            results.append(result)
    return results


### Define Test Periods
dates = stocks.loc["2016-12":"2019-6"].resample("Q").last().index
print(dates)

### Run Tests
test_results = []
columns = ["test_end", "s1", "s2", "eg1", "eg2", "trace0", "trace1", "eig0", "eig1", "w1", "w2"]

for test_end in dates:
    print(test_end)
    result = test_cointegration(etfs, stocks, test_end=test_end)
    test_results.append(pd.DataFrame(result, columns=columns))

pd.concat(test_results).to_hdf("../data/09_backtest.h5", "cointegration_test")

#### Reload  Test Results
test_results = pd.read_hdf("../data/09_backtest.h5", "cointegration_test")
test_results.info()

## Identify Cointegrated Pairs
### Significant Johansen Trace Statistic
test_results["joh_sig"] = (test_results.trace0 > trace0_cv) & (test_results.trace1 > trace1_cv)
print(test_results.joh_sig.value_counts(normalize=True))

### Significant Engle Granger Test
test_results["eg"] = test_results[["eg1", "eg2"]].min(axis=1)
test_results["s1_dep"] = test_results.eg1 < test_results.eg2
test_results["eg_sig"] = test_results.eg < 0.05
print(test_results.eg_sig.value_counts(normalize=True))

### Comparison Engle-Granger vs Johansen
test_results["coint"] = test_results.eg_sig & test_results.joh_sig
print(test_results.coint.value_counts(normalize=True))

test_results = test_results.drop(["eg1", "eg2", "trace0", "trace1", "eig0", "eig1"], axis=1)
test_results.info()

### Comparison
ax = test_results.groupby("test_end").coint.mean().to_frame("# Pairs").plot()
ax.axhline(0.05, lw=1, ls="--", c="k")
plt.tight_layout()
plt.savefig("images/06-01.png", bboxinches="tight")

### Select Candidate Pairs
def select_candidate_pairs(data):
    candidates = data[data.joh_sig | data.eg_sig]
    candidates["y"] = candidates.apply(lambda x: x.s1 if x.s1_dep else x.s2, axis=1)
    candidates["x"] = candidates.apply(lambda x: x.s2 if x.s1_dep else x.s1, axis=1)
    return candidates.drop(["s1_dep", "s1", "s2"], axis=1)


candidates = select_candidate_pairs(test_results)
candidates.to_hdf("../data/09_backtest.h5", "candidates")

candidates = pd.read_hdf("../data/09_backtest.h5", "candidates")
candidates.info()

#### Candidates over Time
fig = plt.figure(figsize=(8, 5))
candidates.groupby("test_end").size().plot()
plt.tight_layout()
plt.savefig("images/06-02.png", bboxinches="tight")

#### Most Common Pairs
with pd.HDFStore("../data/data09.h5") as store:
    print(store.info())
    tickers = store["tickers"]

with pd.HDFStore("../data/09_backtest.h5") as store:
    print(store.info())

counter = Counter()
for s1, s2 in zip(
    candidates[candidates.joh_sig & candidates.eg_sig].y,
    candidates[candidates.joh_sig & candidates.eg_sig].x,
):
    if s1 > s2:
        counter[(s2, s1)] += 1
    else:
        counter[(s1, s2)] += 1

most_common_pairs = pd.DataFrame(counter.most_common(10))
most_common_pairs = pd.DataFrame(most_common_pairs[0].values.tolist(), columns=["s1", "s2"])
print(most_common_pairs)

with pd.HDFStore("../data/09_backtest.h5") as store:
    prices = store["prices"].close.unstack("ticker").ffill(limit=5)
    tickers = store["tickers"].to_dict()

cnt = pd.Series(counter).reset_index()
cnt.columns = ["s1", "s2", "n"]
cnt["name1"] = cnt.s1.map(tickers)
cnt["name2"] = cnt.s2.map(tickers)
cnt.nlargest(10, columns="n")

fig, axes = plt.subplots(ncols=2, figsize=(14, 5))
for i in [0, 1]:
    s1, s2 = most_common_pairs.at[i, "s1"], most_common_pairs.at[i, "s2"]
    prices.loc[:, [s1, s2]].rename(columns=tickers).plot(secondary_y=tickers[s2], ax=axes[i], rot=0)
    axes[i].grid(False)
    axes[i].set_xlabel("")
fig.tight_layout()
plt.savefig("images/06-03.png", bboxinches="tight")

## Get Entry and Exit Dates
### Smooth prices using Kalman filter
def KFSmoother(prices):
    """Estimate rolling mean"""
    kf = KalmanFilter(
        transition_matrices=np.eye(1),
        observation_matrices=np.eye(1),
        initial_state_mean=0,
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.05,
    )

    state_means, _ = kf.filter(prices.values)
    return pd.Series(state_means.flatten(), index=prices.index)


smoothed_prices = prices.apply(KFSmoother)
smoothed_prices.to_hdf("tmp.h5", "smoothed")

smoothed_prices = pd.read_hdf("tmp.h5", "smoothed")


### Compute rolling hedge ratio using Kalman Filter
def KFHedgeRatio(x, y):
    """Estimate Hedge Ratio"""
    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=[0, 0],
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=2,
        transition_covariance=trans_cov,
    )

    state_means, _ = kf.filter(y.values)
    return -state_means


### Estimate mean reversion half life
def estimate_half_life(spread):
    X = spread.shift().iloc[1:].to_frame().assign(const=1)
    y = spread.diff().iloc[1:]
    beta = (np.linalg.inv(X.T @ X) @ X.T @ y).iloc[0]
    halflife = int(round(-np.log(2) / beta, 0))
    return max(halflife, 1)


### Compute Spread & Bollinger Bands
def get_spread(candidates, prices):
    pairs = []
    half_lives = []

    periods = pd.DatetimeIndex(sorted(candidates.test_end.unique()))
    start = time()
    for p, test_end in enumerate(periods, 1):
        start_iteration = time()

        period_candidates = candidates.loc[candidates.test_end == test_end, ["y", "x"]]
        trading_start = test_end + pd.DateOffset(days=1)
        t = trading_start - pd.DateOffset(years=2)
        T = trading_start + pd.DateOffset(months=6) - pd.DateOffset(days=1)
        max_window = len(prices.loc[t:test_end].index)
        print(test_end.date(), len(period_candidates))
        for i, (y, x) in enumerate(zip(period_candidates.y, period_candidates.x), 1):
            if i % 1000 == 0:
                msg = f"{i:5.0f} | {time() - start_iteration:7.1f} | {time() - start:10.1f}"
                print(msg)
            pair = prices.loc[t:T, [y, x]]
            pair["hedge_ratio"] = KFHedgeRatio(
                y=KFSmoother(prices.loc[t:T, y]), x=KFSmoother(prices.loc[t:T, x])
            )[:, 0]
            pair["spread"] = pair[y].add(pair[x].mul(pair.hedge_ratio))
            half_life = estimate_half_life(pair.spread.loc[t:test_end])

            spread = pair.spread.rolling(window=min(2 * half_life, max_window))
            pair["z_score"] = pair.spread.sub(spread.mean()).div(spread.std())
            pairs.append(
                pair.loc[trading_start:T].assign(s1=y, s2=x, period=p, pair=i).drop([x, y], axis=1)
            )

            half_lives.append([test_end, y, x, half_life])
    return pairs, half_lives


candidates = pd.read_hdf("../data/09_backtest.h5", "candidates")
candidates.info()

pairs, half_lives = get_spread(candidates, smoothed_prices)

### Collect Results
#### Half Lives
hl = pd.DataFrame(half_lives, columns=["test_end", "s1", "s2", "half_life"])
hl.info()
print(hl.half_life.describe())
hl.to_hdf("../data/09_backtest.h5", "half_lives")

#### Pair Data
pair_data = pd.concat(pairs)
pair_data.info(show_counts=True)
pair_data.to_hdf("../data/09_backtest.h5", "pair_data")

pair_data = pd.read_hdf("../data/09_backtest.h5", "pair_data")

### Identify Long & Short Entry and Exit Dates
def get_trades(data):
    pair_trades = []
    for i, ((period, s1, s2), pair) in enumerate(data.groupby(["period", "s1", "s2"]), 1):
        if i % 100 == 0:
            print(i)

        first3m = pair.first("3M").index
        last3m = pair.last("3M").index

        entry = pair.z_score.abs() > 2
        entry = (entry.shift() != entry).mul(np.sign(pair.z_score)).fillna(0).astype(int).sub(2)

        exit_t = (
            np.sign(pair.z_score.shift().fillna(method="bfill")) != np.sign(pair.z_score)
        ).astype(int) - 1

        trades = (
            entry[entry != -2]
            .append(exit_t[exit_t == 0])
            .to_frame("side")
            .sort_values(["date", "side"])
            .squeeze()
        )
        if not isinstance(trades, pd.Series):
            continue
        try:
            trades.loc[trades < 0] += 2
        except:
            print(type(trades))
            print(trades)
            print(pair.z_score.describe())
            break

        trades = trades[trades.abs().shift() != trades.abs()]
        window = trades.loc[first3m.min() : first3m.max()]
        extra = trades.loc[last3m.min() : last3m.max()]
        n = len(trades)

        if window.iloc[0] == 0:
            if n > 1:
                print("shift")
                window = window.iloc[1:]
        if window.iloc[-1] != 0:
            extra_exits = extra[extra == 0].head(1)
            if extra_exits.empty:
                continue
            else:
                window = window.append(extra_exits)

        trades = pair[["s1", "s2", "hedge_ratio", "period", "pair"]].join(
            window.to_frame("side"), how="right"
        )
        trades.loc[trades.side == 0, "hedge_ratio"] = np.nan
        trades.hedge_ratio = trades.hedge_ratio.ffill()
        pair_trades.append(trades)
    return pair_trades


pair_trades = get_trades(pair_data)
pair_trade_data = pd.concat(pair_trades)
pair_trade_data.info()
print(pair_trade_data.head())

trades = pair_trade_data["side"].copy()
trades.loc[trades != 0] = 1
trades.loc[trades == 0] = -1
fig = plt.figure(figsize=(14, 4))
trades.sort_index().cumsum().plot()
plt.savefig("images/06-03.png", bboxinches="tight")

pair_trade_data.to_hdf("../data/09_backtest.h5", "pair_trades")
