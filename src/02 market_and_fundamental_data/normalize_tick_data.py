import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
from math import pi
from bokeh.plotting import figure, show
from scipy.stats import normaltest
from pathlib import Path


def price_volume(df, price="vwap", vol="vol", suptitle=title, fname=None):
    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(15, 8))
    axes[0].plot(df.index, df[price])
    axes[1].bar(df.index, df[vol], width=1 / (5 * len(df.index)), color="r")

    xfmt = mpl.dates.DateFormatter("%H:%M")
    axes[1].xaxis.set_major_locator(mpl.dates.HourLocator(interval=3))
    axes[1].xaxis.set_major_formatter(xfmt)
    axes[1].get_xaxis().set_tick_params(which="major", pad=25)
    axes[0].set_title("Price", fontsize=14)
    axes[1].set_title("Volume", fontsize=14)
    fig.autofmt_xdate()
    fig.suptitle(suptitle)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)


def get_bar_stats(agg_trades):
    vwap = agg_trades.apply(lambda x: np.average(x.price, weights=x.shares)).to_frame("vwap")
    ohlc = agg_trades.price.ohlc()
    vol = agg_trades.shares.sum().to_frame("vol")
    txn = agg_trades.shares.size().to_frame("txn")
    return pd.concat([ohlc, vwap, vol, txn], axis=1)


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: "%.2f" % x)
    sns.set_style("whitegrid")

    data_path = Path("../data")
    itch_store = str(data_path / "itch.h5")
    order_book_store = str(data_path / "order_book.h5")
    stock = "AAPL"
    date = "20191030"
    title = "{} | {}".format(stock, pd.to_datetime(date).date())

    with pd.HDFStore(itch_store) as store:
        sys_events = store["S"].set_index("event_code").drop_duplicates()
        sys_events.timestamp = sys_events.timestamp.add(pd.to_datetime(date)).dt.time
        market_open = sys_events.loc["Q", "timestamp"]
        market_close = sys_events.loc["M", "timestamp"]

    with pd.HDFStore(itch_store) as store:
        stocks = store["R"]
    print(stocks.info())

    with pd.HDFStore(itch_store) as store:
        stocks = store["R"].loc[:, ["stock_locate", "stock"]]
        trades = (
            store["P"]
            .append(store["Q"].rename(columns={"cross_price": "price"}), sort=False)
            .merge(stocks)
        )

    trades["value"] = trades.shares.mul(trades.price)
    trades["value_share"] = trades.value.div(trades.value.sum())
    trade_summary = trades.groupby("stock").value_share.sum().sort_values(ascending=False)
    trade_summary.iloc[:50].plot.bar(figsize=(14, 6), color="darkblue", title="% of Traded Value")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))

    with pd.HDFStore(order_book_store) as store:
        trades = store["{}/trades".format(stock)]

    trades.price = trades.price.mul(1e-4)  # format price
    trades = trades[trades.cross == 0]
    trades = trades.between_time(market_open, market_close).drop("cross", axis=1)
    print(trades.info())

    tick_bars = trades.copy()
    tick_bars.index = tick_bars.index.time
    tick_bars.price.plot(
        figsize=(10, 5),
        title="Tick Bars | {} | {}".format(stock, pd.to_datetime(date).date()),
        lw=1,
    )
    plt.xlabel("")
    plt.tight_layout()

    normaltest(tick_bars.price.pct_change().dropna())

    resampled = trades.groupby(pd.Grouper(freq="1Min"))
    time_bars = get_bar_stats(resampled)
    normaltest(time_bars.vwap.pct_change().dropna())

    price_volume(
        time_bars,
        suptitle=f"Time Bars | {stock} | {pd.to_datetime(date).date()}",
        fname="time_bars",
    )

    resampled = trades.groupby(pd.Grouper(freq="5Min"))  # 5 Min bars for better print
    df = get_bar_stats(resampled)

    increase = df.close > df.open
    decrease = df.open > df.close
    w = 2.5 * 60 * 1000  # 2.5 min in ms

    WIDGETS = "pan, wheel_zoom, box_zoom, reset, save"

    p = figure(x_axis_type="datetime", tools=WIDGETS, plot_width=1500, title="AAPL Candlestick")
    p.xaxis.major_label_orientation = pi / 4
    p.grid.grid_line_alpha = 0.4

    p.segment(df.index, df.high, df.index, df.low, color="black")
    p.vbar(
        df.index[increase],
        w,
        df.open[increase],
        df.close[increase],
        fill_color="#D5E1DD",
        line_color="black",
    )
    p.vbar(
        df.index[decrease],
        w,
        df.open[decrease],
        df.close[decrease],
        fill_color="#F2583E",
        line_color="black",
    )
    show(p)

    with pd.HDFStore(order_book_store) as store:
        trades = store["{}/trades".format(stock)]

    trades.price = trades.price.mul(1e-4)
    trades = trades[trades.cross == 0]
    trades = trades.between_time(market_open, market_close).drop("cross", axis=1)
    print(trades.info())

    trades_per_min = trades.shares.sum() / (60 * 7.5)  # min per trading day
    trades["cumul_vol"] = trades.shares.cumsum()

    df = trades.reset_index()
    by_vol = df.groupby(df.cumul_vol.div(trades_per_min).round().astype(int))
    vol_bars = pd.concat(
        [by_vol.timestamp.last().to_frame("timestamp"), get_bar_stats(by_vol)], axis=1
    )
    print(vol_bars.head())

    price_volume(
        vol_bars.set_index("timestamp"),
        suptitle=f"Volume Bars | {stock} | {pd.to_datetime(date).date()}",
        fname="volume_bars",
    )

    normaltest(vol_bars.vwap.dropna())

    with pd.HDFStore(order_book_store) as store:
        trades = store["{}/trades".format(stock)]

    trades.price = trades.price.mul(1e-4)
    trades = trades[trades.cross == 0]
    trades = trades.between_time(market_open, market_close).drop("cross", axis=1)
    print(trades.info())

    value_per_min = trades.shares.mul(trades.price).sum() / (60 * 7.5)  # min per trading day
    trades["cumul_val"] = trades.shares.mul(trades.price).cumsum()

    df = trades.reset_index()
    by_value = df.groupby(df.cumul_val.div(value_per_min).round().astype(int))
    dollar_bars = pd.concat(
        [by_value.timestamp.last().to_frame("timestamp"), get_bar_stats(by_value)], axis=1
    )
    print(dollar_bars.head())

    price_volume(
        dollar_bars.set_index("timestamp"),
        suptitle=f"Dollar Bars | {stock} | {pd.to_datetime(date).date()}",
        fname="dollar_bars",
    )
