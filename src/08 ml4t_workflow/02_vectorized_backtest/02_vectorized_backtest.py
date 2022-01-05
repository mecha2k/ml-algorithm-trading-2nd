import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from scipy.stats import spearmanr
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from time import time


plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
pd.options.display.float_format = "{:,.2f}".format

np.random.seed(42)
DATA_DIR = Path("..", "..", "data")

if __name__ == "__main__":
    data = pd.read_hdf("../../data/08_backtest.h5", "data")
    data.info()

    ### SP500 Benchmark
    sp500 = web.DataReader("SP500", "fred", "2014", "2018").pct_change()
    sp500.info()

    ## Compute Forward Returns
    daily_returns = data.open.unstack("ticker").sort_index().pct_change()
    daily_returns.info()

    fwd_returns = daily_returns.shift(-1)

    ## Generate Signals
    predictions = data.predicted.unstack("ticker")
    predictions.info()

    N_LONG = N_SHORT = 15

    long_signals = (
        predictions.where(predictions > 0).rank(axis=1, ascending=False) > N_LONG
    ).astype(int)
    short_signals = (predictions.where(predictions < 0).rank(axis=1) > N_SHORT).astype(int)

    ## Compute Portfolio Returns
    long_returns = long_signals.mul(fwd_returns).mean(axis=1)
    short_returns = short_signals.mul(-fwd_returns).mean(axis=1)
    strategy = long_returns.add(short_returns).to_frame("Strategy")

    ## Plot results
    fig, axes = plt.subplots(ncols=2, figsize=(14, 5))
    strategy.join(sp500).add(1).cumprod().sub(1).plot(ax=axes[0], title="Cumulative Return")
    sns.histplot(data=strategy.dropna(), ax=axes[1], label="Strategy")
    sns.histplot(data=sp500, ax=axes[1], label="SP500")
    axes[1].set_title("Daily Standard Deviation")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    fig.tight_layout()
    plt.savefig("../images/02-01.png", bboxinches="tight")

    res = strategy.join(sp500).dropna()
    print(res.std())
    print(res.corr())
