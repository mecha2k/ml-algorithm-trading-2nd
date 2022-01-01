import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web

from scipy.stats import spearmanr
from matplotlib.ticker import FuncFormatter
from icecream import ic

np.random.seed(42)
sns.set_style("whitegrid")


if __name__ == "__main__":
    data = pd.read_hdf("../data/backtest_08.h5", "data")
    sp500 = web.DataReader("SP500", "fred", "2014", "2018").pct_change()
    ic(data.head())
    ic(sp500.head())

    daily_returns = data.open.unstack("ticker").sort_index().pct_change()
    fwd_returns = daily_returns.shift(-1)
    ic(daily_returns.head())

    predictions = data.predicted.unstack("ticker")
    ic(predictions.head())

    N_LONG = N_SHORT = 15
    long_signals = (
        predictions.where(predictions > 0).rank(axis=1, ascending=False) > N_LONG
    ).astype(int)
    short_signals = (predictions.where(predictions < 0).rank(axis=1) > N_SHORT).astype(int)

    long_returns = long_signals.mul(fwd_returns).mean(axis=1)
    short_returns = short_signals.mul(-fwd_returns).mean(axis=1)
    strategy = long_returns.add(short_returns).to_frame("Strategy")

    fig, axes = plt.subplots(ncols=2, figsize=(14, 5))
    strategy.join(sp500).add(1).cumprod().sub(1).plot(ax=axes[0], title="Cumulative Return")
    sns.histplot(strategy.dropna(), ax=axes[1], label="Strategy")
    sns.histplot(sp500, ax=axes[1], label="SP500")
    axes[1].set_title("Daily Standard Deviation")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    fig.tight_layout()
    plt.savefig("images/02_vec_backtest.png", dpi=300, bboxinches="tight")

    res = strategy.join(sp500).dropna()
    ic(res.std())
    ic(res.corr())
