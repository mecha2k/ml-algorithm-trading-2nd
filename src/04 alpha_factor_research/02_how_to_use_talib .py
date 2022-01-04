import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import talib

idx = pd.IndexSlice
sns.set_style("whitegrid")
sns.set_palette("pastel")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16

DATA_STORE = "../data/assets.h5"

if __name__ == "__main__":
    with pd.HDFStore(DATA_STORE) as store:
        data = (
            store["quandl/wiki/prices"]
            .loc[
                pd.IndexSlice["2007":"2010", "AAPL"],
                ["adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"],
            ]
            .unstack("ticker")
            .swaplevel(axis=1)
            .loc[:, "AAPL"]
            .rename(columns=lambda x: x.replace("adj_", ""))
        )
    data.info()

    up, mid, low = talib.BBANDS(data.close, timeperiod=21, nbdevup=2, nbdevdn=2, matype=0)
    rsi = talib.RSI(data.close, timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(
        data.close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    macd_data = pd.DataFrame(
        {"AAPL": data.close, "MACD": macd, "MACD Signal": macdsignal, "MACD History": macdhist}
    )

    fig, axes = plt.subplots(nrows=2, figsize=(15, 8))
    macd_data.AAPL.plot(ax=axes[0])
    macd_data.drop("AAPL", axis=1).plot(ax=axes[1])
    fig.tight_layout()
    plt.savefig("images/02-01.png", bboxinches="tight")

    data = pd.DataFrame(
        {"AAPL": data.close, "BB Up": up, "BB Mid": mid, "BB down": low, "RSI": rsi, "MACD": macd}
    )

    fig, axes = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    data.drop(["RSI", "MACD"], axis=1).plot(ax=axes[0], lw=1, title="Bollinger Bands")
    data["RSI"].plot(ax=axes[1], lw=1, title="Relative Strength Index")
    axes[1].axhline(70, lw=1, ls="--", c="k")
    axes[1].axhline(30, lw=1, ls="--", c="k")
    data.MACD.plot(ax=axes[2], lw=1, title="Moving Average Convergence/Divergence", rot=0)
    axes[2].set_xlabel("")
    fig.tight_layout()
    plt.savefig("images/02-02.png", bboxinches="tight")
