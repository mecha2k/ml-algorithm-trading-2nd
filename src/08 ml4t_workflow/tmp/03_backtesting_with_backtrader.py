import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
import backtrader as bt
import pyfolio as pf
import csv

from time import time
from pathlib import Path
from backtrader.feeds import PandasData
from scipy.stats import spearmanr
from matplotlib.ticker import FuncFormatter
from icecream import ic

np.random.seed(42)
sns.set_style("darkgrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
pd.set_option("display.expand_frame_repr", False)


def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return f"{h:>02.0f}:{m:>02.0f}:{s:>02.0f}"


class FixedCommisionScheme(bt.CommInfoBase):
    params = (
        ("commission", 0.02),
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_FIXED),
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * self.p.commission


OHLCV = ["open", "high", "low", "close", "volume"]


class SignalData(PandasData):
    cols = OHLCV + ["predicted"]
    lines = tuple(cols)
    params = {c: -1 for c in cols}
    params.update({"datetime": None})
    params = tuple(params.items())


class MLStrategy(bt.Strategy):
    params = (
        ("n_positions", 10),
        ("min_positions", 5),
        ("verbose", False),
        ("log_file", "../data/backtest.csv"),
    )

    def log(self, txt, dt=None):
        """Logger for the strategy"""
        dt = dt or self.datas[0].datetime.datetime(0)
        with Path(self.p.log_file).open("a") as f:
            log_writer = csv.writer(f)
            log_writer.writerow([dt.isoformat()] + txt.split(","))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        # Check if an order has been completed
        # broker could reject order if not enough cash
        if self.p.verbose:
            if order.status in [order.Completed]:
                p = order.executed.price
                if order.isbuy():
                    self.log(f"{order.data._name},BUY executed,{p:.2f}")
                elif order.issell():
                    self.log(f"{order.data._name},SELL executed,{p:.2f}")
            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                self.log(f"{order.data._name},Order Canceled/Margin/Rejected")

    # bt calls prenext instead of next unless
    # all datafeeds have current values
    # => call next to avoid duplicating logic
    def prenext(self):
        self.next()

    def next(self):
        today = self.datas[0].datetime.date()
        #         if today.weekday() not in [0, 3]: # only trade on Mondays;
        #             return
        positions = [d._name for d, pos in self.getpositions().items() if pos]
        up, down = {}, {}
        missing = not_missing = 0
        for data in self.datas:
            if data.datetime.date() == today:
                if data.predicted[0] > 0:
                    up[data._name] = data.predicted[0]
                elif data.predicted[0] < 0:
                    down[data._name] = data.predicted[0]

        # sort dictionaries ascending/descending by value
        # returns list of tuples
        shorts = sorted(down, key=down.get)[: self.p.n_positions]
        longs = sorted(up, key=up.get, reverse=True)[: self.p.n_positions]
        n_shorts, n_longs = len(shorts), len(longs)

        # only take positions if at least min_n longs and shorts
        if n_shorts < self.p.min_positions or n_longs < self.p.min_positions:
            longs, shorts = [], []
        for ticker in positions:
            if ticker not in longs + shorts:
                self.order_target_percent(data=ticker, target=0)
                self.log(f"{ticker},CLOSING ORDER CREATED")

        short_target = -1 / max(self.p.n_positions, n_shorts)
        long_target = 1 / max(self.p.n_positions, n_longs)
        for ticker in shorts:
            self.order_target_percent(data=ticker, target=short_target)
            self.log("{ticker},SHORT ORDER CREATED")
        for ticker in longs:
            self.order_target_percent(data=ticker, target=long_target)
            self.log("{ticker},LONG ORDER CREATED")


if __name__ == "__main__":
    cerebro = bt.Cerebro()
    # comminfo = FixedCommisionScheme()
    # cerebro.broker.addcommissioninfo(comminfo)
    cerebro.broker.setcash(10000)

    idx = pd.IndexSlice
    data = pd.read_hdf("../data/backtest_08.h5", "data").sort_index()
    tickers = data.index.get_level_values(0).unique()

    for ticker in tickers:
        df = data.loc[idx[ticker, :], :].droplevel("ticker", axis=0)
        df.index.name = "datetime"
        bt_data = SignalData(dataname=df)
        cerebro.adddata(bt_data, name=ticker)

    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")
    cerebro.addstrategy(
        MLStrategy, n_positions=25, min_positions=20, verbose=True, log_file="../data/bt_log.csv"
    )
    start = time()
    results = cerebro.run()
    ending_value = cerebro.broker.getvalue()
    duration = time() - start

    print(f"Final Portfolio Value: {ending_value:,.2f}")
    print(f"Duration: {format_time(duration)}")

    # fig = cerebro.plot(iplot=False)[0][0]
    # fig.savefig("images/03_backtrader_01.png", dpi=300, bboxinches="tight")

    # prepare pyfolio inputs
    pyfolio_analyzer = results[0].analyzers.getbyname("pyfolio")
    returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()

    returns.to_hdf("../data/backtrader.h5", "returns")
    positions.to_hdf("../data/backtrader.h5", "positions")
    transactions.to_hdf("../data/backtrader.h5", "transactions/")
    gross_lev.to_hdf("../data/backtrader.h5", "gross_lev")

    returns = pd.read_hdf("../data/backtrader.h5", "returns")
    positions = pd.read_hdf("../data/backtrader.h5", "positions")
    transactions = pd.read_hdf("../data/backtrader.h5", "transactions/")
    gross_lev = pd.read_hdf("../data/backtrader.h5", "gross_lev")
    ic(returns.head())
    ic(positions.head())
    ic(transactions.head())
    ic(gross_lev.head())

    benchmark = web.DataReader("SP500", "fred", "2014", "2018").squeeze()
    benchmark = benchmark.pct_change().tz_localize("UTC")

    daily_tx = transactions.groupby(level=0)
    longs = daily_tx.value.apply(lambda x: x.where(x > 0).sum())
    shorts = daily_tx.value.apply(lambda x: x.where(x < 0).sum())

    fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
    df = returns.to_frame("Strategy").join(benchmark.to_frame("Benchmark (S&P 500)"))
    df.add(1).cumprod().sub(1).plot(ax=axes[0], title="Cumulative Return")
    longs.plot(label="Long", ax=axes[1], title="Positions")
    shorts.plot(ax=axes[1], label="Short")
    positions.cash.plot(ax=axes[1], label="PF Value")
    axes[1].legend()
    fig.tight_layout()
    plt.savefig("images/03_backtrader_02.png", dpi=300, bboxinches="tight")

    # pf.create_full_tear_sheet(
    #     returns, transactions=transactions, positions=positions, benchmark_rets=benchmark.dropna()
    # )
    pf.create_full_tear_sheet(returns, transactions=transactions, benchmark_rets=benchmark.dropna())
    fig = pf.create_returns_tear_sheet(returns, return_fig=True)
    fig.savefig("images/03_pyfolio.png", dpi=300)
