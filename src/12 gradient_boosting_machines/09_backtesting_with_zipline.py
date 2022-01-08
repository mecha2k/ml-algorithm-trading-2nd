# Long-Short Strategy, Part 6: Backtesting with Zipline

from collections import defaultdict
from time import time
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pandas_datareader.data as web
import seaborn as sns
from logbook import Logger, StderrHandler, INFO, WARNING

from zipline import run_algorithm
from zipline.api import (
    attach_pipeline,
    pipeline_output,
    date_rules,
    time_rules,
    record,
    schedule_function,
    commission,
    slippage,
    set_slippage,
    set_commission,
    set_max_leverage,
    order_target,
    order_target_percent,
    get_open_orders,
    cancel_order,
)
from zipline.data import bundles
from zipline.utils.run_algo import load_extensions
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import Column, DataSet
from zipline.pipeline.domain import US_EQUITIES
from zipline.pipeline.filters import StaticAssets
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.pipeline.loaders.frame import DataFrameLoader
from trading_calendars import get_calendar

import pyfolio as pf
from pyfolio.plotting import plot_rolling_returns, plot_rolling_sharpe
from pyfolio.timeseries import forecast_cone_bootstrap


np.random.seed(42)
idx = pd.IndexSlice
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:,.2f}".format

if __name__ == "__main__":
    ### Load zipline extensions
    # Only need this in notebook to find bundle.
    load_extensions(default=True, extensions=[], strict=True, environ=None)

    log_handler = StderrHandler(
        format_string="[{record.time:%Y-%m-%d %H:%M:%S.%f}]: "
        + "{record.level_name}: {record.func_name}: {record.message}",
        level=WARNING,
    )
    log_handler.push_application()
    log = Logger("Algorithm")

    ## Algo Params
    N_LONGS = 25
    N_SHORTS = 25
    MIN_POSITIONS = 20

    ## Load Data
    ### Quandl Wiki Bundle
    # Requires running `zipline ingest` (see installation instructions and Chapter 8). If you haven't done so yet
    # (but have provided your QUANDL API KEY when launching Docker), uncomment and run the following cell:
    bundle_data = bundles.load("quandl")

    ### ML Predictions
    # If you run into difficulties reading the predictions, run the following to upgrade `tables` ([source]
    # (https://stackoverflow.com/questions/54210073/pd-read-hdf-throws-cannot-set-writable-flag-to-true-of-this-array)).

    def load_predictions(bundle):
        predictions = pd.read_hdf("data/predictions.h5", "lgb/train/01").append(
            pd.read_hdf("data/predictions.h5", "lgb/test/01").drop("y_test", axis=1)
        )
        predictions = (
            predictions.loc[~predictions.index.duplicated()]
            .iloc[:, :10]
            .mean(1)
            .sort_index()
            .dropna()
            .to_frame("prediction")
        )
        tickers = predictions.index.get_level_values("symbol").unique().tolist()

        assets = bundle.asset_finder.lookup_symbols(tickers, as_of_date=None)
        predicted_sids = pd.Int64Index([asset.sid for asset in assets])
        ticker_map = dict(zip(tickers, predicted_sids))

        return (
            predictions.unstack("symbol").rename(columns=ticker_map).prediction.tz_localize("UTC")
        ), assets

    predictions, assets = load_predictions(bundle_data)
    predictions.info()

    ### Define Custom Dataset
    class SignalData(DataSet):
        predictions = Column(dtype=float)
        domain = US_EQUITIES

    ### Define Pipeline Loaders
    signal_loader = {SignalData.predictions: DataFrameLoader(SignalData.predictions, predictions)}

    ## Pipeline Setup
    ### Custom ML Factor
    class MLSignal(CustomFactor):
        """Converting signals to Factor,
        so we can rank and filter in Pipeline"""

        inputs = [SignalData.predictions]
        window_length = 1

        def compute(self, today, assets, out, predictions):
            out[:] = predictions

    ### Create Pipeline
    def compute_signals():
        signals = MLSignal()
        return Pipeline(
            columns={
                "longs": signals.top(N_LONGS, mask=signals > 0),
                "shorts": signals.bottom(N_SHORTS, mask=signals < 0),
            },
            screen=StaticAssets(assets),
        )

    ## Initialize Algorithm
    def initialize(context):
        """
        Called once at the start of the algorithm.
        """
        context.n_longs = N_LONGS
        context.n_shorts = N_SHORTS
        context.min_positions = MIN_POSITIONS
        context.universe = assets
        context.trades = pd.Series()
        context.longs = context.shorts = 0

        set_slippage(slippage.FixedSlippage(spread=0.00))
        set_commission(commission.PerShare(cost=0.001, min_trade_cost=0))

        schedule_function(
            rebalance,
            date_rules.every_day(),
            #                       date_rules.week_start(),
            time_rules.market_open(hours=1, minutes=30),
        )

        schedule_function(record_vars, date_rules.every_day(), time_rules.market_close())

        pipeline = compute_signals()
        attach_pipeline(pipeline, "signals")

    ### Get daily Pipeline results
    def before_trading_start(context, data):
        """
        Called every day before market open.
        """
        output = pipeline_output("signals")
        df = output["longs"].astype(int).append(output["shorts"].astype(int).mul(-1))

        holdings = df[df != 0]
        other = df[df == 0]
        other = other[~other.index.isin(holdings.index) & ~other.index.duplicated()]
        context.trades = holdings.append(other)
        assert len(context.trades.index.unique()) == len(context.trades)

    ## Define Rebalancing Logic
    def rebalance(context, data):
        """
        Execute orders according to schedule_function() date & time rules.
        """
        trades = defaultdict(list)
        for symbol, open_orders in get_open_orders().items():
            for open_order in open_orders:
                cancel_order(open_order)

        positions = context.portfolio.positions
        s = pd.Series({s: v.amount * v.last_sale_price for s, v in positions.items()}).sort_values(
            ascending=False
        )
        for stock, trade in context.trades.items():
            if trade == 0:
                order_target(stock, target=0)
            else:
                trades[trade].append(stock)

        context.longs, context.shorts = len(trades[1]), len(trades[-1])
        #     log.warning('{} {:,.0f}'.format(len(positions), context.portfolio.portfolio_value))
        if context.longs > context.min_positions and context.shorts > context.min_positions:
            for stock in trades[-1]:
                order_target_percent(stock, -1 / context.shorts)
            for stock in trades[1]:
                order_target_percent(stock, 1 / context.longs)
        else:
            for stock in trades[-1] + trades[1]:
                if stock in positions:
                    order_target(stock, 0)

    ## Record Data Points
    def record_vars(context, data):
        """
        Plot variables at the end of each day.
        """
        record(leverage=context.account.leverage, longs=context.longs, shorts=context.shorts)

    ## Run Algorithm
    # We backtest our strategy during the (in-sample) validation and out-of-sample test period:
    dates = predictions.index.get_level_values("date")
    start_date, end_date = dates.min(), dates.max()
    print("Start: {}\nEnd:   {}".format(start_date.date(), end_date.date()))

    start = time()
    results = run_algorithm(
        start=start_date,
        end=end_date,
        initialize=initialize,
        before_trading_start=before_trading_start,
        capital_base=1e5,
        data_frequency="daily",
        bundle="quandl",
        custom_loader=signal_loader,
    )  # need to modify zipline

    print("Duration: {:.2f}s".format(time() - start))

    ## PyFolio Analysis
    # To visualize the out-of-sample performance, we pass '2017-01-01' as start date for the `live_start_date`:
    returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(results)

    benchmark = web.DataReader("SP500", "fred", "2014", "2018").squeeze()
    benchmark = benchmark.pct_change().tz_localize("UTC")

    ### Custom Plots
    fig, axes = plt.subplots(ncols=2, figsize=(16, 5))
    plot_rolling_returns(
        returns,
        factor_returns=benchmark,
        live_start_date="2017-01-01",
        logy=False,
        cone_std=2,
        legend_loc="best",
        volatility_match=False,
        cone_function=forecast_cone_bootstrap,
        ax=axes[0],
    )
    plot_rolling_sharpe(returns, ax=axes[1], rolling_window=63)
    axes[0].set_title("Cumulative Returns - In and Out-of-Sample")
    axes[1].set_title("Rolling Sharpe Ratio (3 Months)")
    fig.tight_layout()
    plt.savefig("images/09-01.png")

    ### Tear Sheets
    pf.create_full_tear_sheet(
        returns,
        positions=positions,
        transactions=transactions,
        benchmark_rets=benchmark,
        live_start_date="2017-01-01",
        round_trips=True,
    )
