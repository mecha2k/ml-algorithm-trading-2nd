# zipline MeanReversion Backtest
# In the chapter 04, we introduced `Zipline` to simulate the computation of alpha factors from trailing cross-sectional
# market, fundamental, and alternative data. Now we will exploit the alpha factors to derive and act on buy and sell
# signals using the custom MeanReversion factor developed in the last chapter.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings

from pytz import UTC
from logbook import (
    NestedSetup,
    NullHandler,
    Logger,
    StreamHandler,
    StderrHandler,
    INFO,
    WARNING,
    DEBUG,
    ERROR,
)

from zipline import run_algorithm
from zipline.api import (
    attach_pipeline,
    date_rules,
    time_rules,
    get_datetime,
    order_target_percent,
    pipeline_output,
    record,
    schedule_function,
    get_open_orders,
    calendars,
    set_commission,
    set_slippage,
)
from zipline.finance import commission, slippage
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.factors import Returns, AverageDollarVolume
from pyfolio.utils import extract_rets_pos_txn_from_zipline


sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 12
warnings.filterwarnings("ignore")

MONTH = 21
YEAR = 12 * MONTH
N_LONGS = 50
N_SHORTS = 50
VOL_SCREEN = 500


class MeanReversion(CustomFactor):
    """Compute ratio of the latest monthly return to 12m average,
    normalized by std dev of monthly returns"""

    inputs = [Returns(window_length=MONTH)]
    window_length = YEAR

    def compute(self, today, assets, out, monthly_returns):
        df = pd.DataFrame(monthly_returns)
        out[:] = df.iloc[-1].sub(df.mean()).div(df.std())


# The Pipeline created by the `compute_factors()` method returns a table with a long and a short column for the 25
# stocks with the largest negative and positive deviations of their last monthly return from its annual average,
# normalized by the standard deviation. It also limited the universe to the 500 stocks with the highest average trading
# volume over the last 30 trading days.


def compute_factors():
    """Create factor pipeline incl. mean reversion,
    filtered by 30d Dollar Volume; capture factor ranks"""
    mean_reversion = MeanReversion()
    dollar_volume = AverageDollarVolume(window_length=30)
    return Pipeline(
        columns={
            "longs": mean_reversion.bottom(N_LONGS),
            "shorts": mean_reversion.top(N_SHORTS),
            "ranking": mean_reversion.rank(ascending=False),
        },
        screen=dollar_volume.top(VOL_SCREEN),
    )


# Before_trading_start() ensures the daily execution of the pipeline and the recording of the results, including
# the current prices.


def before_trading_start(context, data):
    """Run factor pipeline"""
    context.factor_data = pipeline_output("factor_pipeline")
    record(factor_data=context.factor_data.ranking)
    assets = context.factor_data.index
    record(prices=data.current(assets, "price"))


# The new rebalance() method submits trade orders to the exec_trades() method for the assets flagged for long and
# short positions by the pipeline with equal positive and negative weights. It also divests any current holdings that
# are no longer included in the factor signals:


def rebalance(context, data):
    """Compute long, short and obsolete holdings; place trade orders"""
    factor_data = context.factor_data
    assets = factor_data.index

    longs = assets[factor_data.longs]
    shorts = assets[factor_data.shorts]
    divest = context.portfolio.positions.keys() - longs.union(shorts)
    log.info(
        "{} | Longs: {:2.0f} | Shorts: {:2.0f} | {:,.2f}".format(
            get_datetime().date(), len(longs), len(shorts), context.portfolio.portfolio_value
        )
    )

    exec_trades(data, assets=divest, target_percent=0)
    exec_trades(data, assets=longs, target_percent=1 / N_LONGS if N_LONGS else 0)
    exec_trades(data, assets=shorts, target_percent=-1 / N_SHORTS if N_SHORTS else 0)


def exec_trades(data, assets, target_percent):
    """Place orders for assets using target portfolio percentage"""
    for asset in assets:
        if data.can_trade(asset) and not get_open_orders(asset):
            order_target_percent(asset, target_percent)


# The `rebalance()` method runs according to `date_rules` and `time_rules` set by the `schedule_function()` utility
# at the beginning of the week, right after market_open as stipulated by the built-in US_EQUITIES calendar (see docs
# for details on rules). You can also specify a trade commission both in relative terms and as a minimum amount.
# There is also an option to define slippage, which is the cost of an adverse change in price between trade decision
# and execution


def initialize(context):
    """Setup: register pipeline, schedule rebalancing,
    and set trading params"""
    attach_pipeline(compute_factors(), "factor_pipeline")
    schedule_function(
        rebalance,
        date_rules.week_start(),
        time_rules.market_open(),
        calendar=calendars.US_EQUITIES,
    )
    set_commission(us_equities=commission.PerShare(cost=0.00075, min_trade_cost=0.01))
    set_slippage(us_equities=slippage.VolumeShareSlippage(volume_limit=0.0025, price_impact=0.01))


if __name__ == "__main__":
    format_string = "[{record.time: %H:%M:%S.%f}]: {record.level_name}: {record.message}"
    zipline_logging = NestedSetup(
        [
            NullHandler(level=DEBUG),
            StreamHandler(sys.stdout, format_string=format_string, level=INFO),
            StreamHandler(sys.stderr, level=ERROR),
        ]
    )
    zipline_logging.push_application()
    log = Logger("Algorithm")

    start = pd.Timestamp("2013-01-01", tz=UTC)
    end = pd.Timestamp("2017-01-01", tz=UTC)
    capital_base = 100000

    # The algorithm executes upon calling the run_algorithm() function and returns the backtest performance DataFrame.
    backtest = run_algorithm(
        start=start,
        end=end,
        initialize=initialize,
        before_trading_start=before_trading_start,
        bundle="quandl",
        capital_base=capital_base,
    )

    # The `extract_rets_pos_txn_from_zipline` utility provided by `pyfolio` extracts the data used to compute
    # performance metrics.
    returns, positions, transactions = extract_rets_pos_txn_from_zipline(backtest)

    with pd.HDFStore("../data/backtests.h5") as store:
        store.put("backtest/equal_weight", backtest)
        store.put("returns/equal_weight", returns)
        store.put("positions/equal_weight", positions)
        store.put("transactions/equal_weight", transactions)

    fig, axes = plt.subplots(nrows=2, figsize=(14, 6))
    returns.add(1).cumprod().sub(1).plot(ax=axes[0], title="Cumulative Returns")
    transactions.groupby(transactions.dt.dt.day).txn_dollars.sum().cumsum().plot(
        ax=axes[1], title="Cumulative Transactions"
    )
    fig.tight_layout()
    plt.savefig("images/01-01.png")

    positions.index = positions.index.date
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(
        positions.replace(0, np.nan).dropna(how="all", axis=1).T,
        cmap=sns.diverging_palette(h_neg=20, h_pos=200),
        ax=ax,
        center=0,
    )
    plt.savefig("images/01-02.png")

    print(positions.head())
    transactions.info()
