import warnings
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from pyfolio.utils import extract_rets_pos_txn_from_zipline
from pyfolio.plotting import (
    plot_perf_stats,
    show_perf_stats,
    plot_rolling_beta,
    plot_rolling_returns,
    plot_rolling_sharpe,
    plot_drawdown_periods,
    plot_drawdown_underwater,
)

from pyfolio.timeseries import perf_stats, extract_interesting_date_ranges

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

with pd.HDFStore("backtests.h5") as store:
    backtest = store["backtest/equal_weight"]
backtest.info()

returns, positions, transactions = extract_rets_pos_txn_from_zipline(backtest)

returns.head().append(returns.tail())
positions.info()

positions.columns = [c for c in positions.columns[:-1]] + ["cash"]
positions.index = positions.index.normalize()
positions.info()

transactions.symbol = transactions.symbol.apply(lambda x: x.symbol)
transactions.head().append(transactions.tail())

HDF_PATH = Path("..", "data", "assets.h5")

assets = positions.columns[:-1]
with pd.HDFStore(HDF_PATH) as store:
    df = store.get("us_equities/stocks")["sector"].dropna()
    df = df[~df.index.duplicated()]
sector_map = df.reindex(assets).fillna("Unknown").to_dict()

with pd.HDFStore(HDF_PATH) as store:
    benchmark_rets = store["sp500/fred"].close.pct_change()
benchmark_rets.name = "S&P500"
benchmark_rets = benchmark_rets.tz_localize("UTC").filter(returns.index)
benchmark_rets.tail()

perf_stats(returns=returns, factor_returns=benchmark_rets)

fig, ax = plt.subplots(figsize=(14, 5))
plot_perf_stats(returns=returns, factor_returns=benchmark_rets, ax=ax)
sns.despine()
fig.tight_layout()

oos_date = "2016-01-01"

show_perf_stats(
    returns=returns,
    factor_returns=benchmark_rets,
    positions=positions,
    transactions=transactions,
    live_start_date=oos_date,
)

plot_rolling_returns(
    returns=returns,
    factor_returns=benchmark_rets,
    live_start_date=oos_date,
    cone_std=(1.0, 1.5, 2.0),
)
plt.gcf().set_size_inches(14, 8)
sns.despine()
plt.tight_layout()

plot_rolling_sharpe(returns=returns)
plt.gcf().set_size_inches(14, 8)
sns.despine()
plt.tight_layout()

plot_rolling_beta(returns=returns, factor_returns=benchmark_rets)
plt.gcf().set_size_inches(14, 6)
sns.despine()
plt.tight_layout()

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
axes = ax.flatten()

plot_drawdown_periods(returns=returns, ax=axes[0])
plot_rolling_beta(returns=returns, factor_returns=benchmark_rets, ax=axes[1])
plot_drawdown_underwater(returns=returns, ax=axes[2])
plot_rolling_sharpe(returns=returns)
sns.despine()
plt.tight_layout()

interesting_times = extract_interesting_date_ranges(returns=returns)
(
    interesting_times["Fall2015"]
    .to_frame("momentum_equal_weights")
    .join(benchmark_rets)
    .add(1)
    .cumprod()
    .sub(1)
    .plot(lw=2, figsize=(14, 6), title="Post-Brexit Turmoil")
)
sns.despine()
plt.tight_layout()
