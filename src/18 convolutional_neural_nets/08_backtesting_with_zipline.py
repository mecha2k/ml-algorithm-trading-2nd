# Backtesting with zipline - Pipeline API with Custom Data
# > This notebook requires the `conda` environment `backtest`. Please see the [installation instructions]
# (../installation/README.md) for running the latest Docker image or alternative ways to set up your environment.

from pathlib import Path
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_datareader.data as web
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

from alphalens.tears import (
    create_returns_tear_sheet,
    create_summary_tear_sheet,
    create_full_tear_sheet,
)

from alphalens.performance import mean_return_by_quantile
from alphalens.plotting import plot_quantile_returns_bar
from alphalens.utils import get_clean_factor_and_forward_returns, rate_of_return
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


def get_trade_prices(tickers):
    prices = pd.read_hdf(DATA_STORE, "quandl/wiki/prices").swaplevel().sort_index()
    prices.index.names = ["symbol", "date"]
    prices = prices.loc[idx[tickers, "2010":"2018"], "adj_open"]
    return prices.unstack("symbol").sort_index().shift(-1).tz_localize("UTC")


if __name__ == "__main__":
    predictions = (
        pd.read_hdf(results_path / "predictions.h5", "predictions")
        .iloc[:, :4]
        .mean(1)
        .to_frame("prediction")
    )

    factor = (
        predictions.unstack("symbol")
        .asfreq("D")
        .dropna(how="all")
        .stack()
        .tz_localize("UTC", level="date")
        .sort_index()
    )
    tickers = factor.index.get_level_values("symbol").unique()
    factor.info()

    trade_prices = get_trade_prices(tickers)
    trade_prices.info()

    factor_data = get_clean_factor_and_forward_returns(
        factor=factor, prices=trade_prices, quantiles=5, periods=(1, 5, 10, 21)
    ).sort_index()
    factor_data.info()

    create_summary_tear_sheet(factor_data)

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
    MIN_POSITIONS = 10

    ## Load Data
    ### Quandl Wiki Bundel
    bundle_data = bundles.load("quandl")

    ### ML Predictions
    def load_predictions(bundle):
        predictions = (
            pd.read_hdf(results_path / "predictions.h5", "predictions")
            .iloc[:, :4]
            .mean(1)
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
        """Converting signals to Factor
        so we can rank and filter in Pipeline"""

        inputs = [SignalData.predictions]
        window_length = 1

        def compute(self, today, assets, out, predictions):
            out[:] = predictions

    ### Create Pipeline
    def compute_signals():
        signals = MLSignal()
        return Pipeline(
            columns={"longs": signals.top(N_LONGS), "shorts": signals.bottom(N_SHORTS)},
            screen=StaticAssets(assets),
        )

    ## Initialize Algorithm
    def initialize(context):
        """
        Called once at the start of the algorithm.
        """
        context.longs = context.shorts = None
        set_slippage(slippage.FixedSlippage(spread=0.00))
        #     set_commission(commission.PerShare(cost=0.001, min_trade_cost=0))

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
        longs = pipeline_output("signals").longs.astype(int)
        shorts = pipeline_output("signals").shorts.astype(int)
        holdings = context.portfolio.positions.keys()

        if longs.sum() > MIN_POSITIONS and shorts.sum() > MIN_POSITIONS:
            context.longs = longs[longs != 0].index
            context.shorts = shorts[shorts != 0].index
            context.divest = holdings - set(context.longs) - set(context.shorts)
        else:
            context.longs = context.shorts = pd.Index([])
            context.divest = set(holdings)

    ## Define Rebalancing Logic
    def rebalance(context, data):
        """
        Execute orders according to schedule_function() date & time rules.
        """

        for symbol, open_orders in get_open_orders().items():
            for open_order in open_orders:
                cancel_order(open_order)

        for stock in context.divest:
            order_target(stock, target=0)

        #     log.warning('{} {:,.0f}'.format(len(context.portfolio.positions), context.portfolio.portfolio_value))
        if not (context.longs.empty and context.shorts.empty):
            for stock in context.shorts:
                order_target_percent(stock, -1 / len(context.shorts) / 2)
            for stock in context.longs:
                order_target_percent(stock, 1 / len(context.longs))

    ## Record Data Points
    def record_vars(context, data):
        """
        Plot variables at the end of each day.
        """
        record(leverage=context.account.leverage, longs=context.longs, shorts=context.shorts)

    ## Run Algorithm
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
    returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(results)
    benchmark = web.DataReader("SP500", "fred", "2010", "2018").squeeze()
    benchmark = benchmark.pct_change().tz_localize("UTC")

    ### Custom Plots
    LIVE_DATE = "2018-01-01"

    fig, axes = plt.subplots(ncols=2, figsize=(16, 5))
    plot_rolling_returns(
        returns,
        factor_returns=benchmark,
        live_start_date=LIVE_DATE,
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
    fig.savefig((results_path / "pyfolio_out_of_sample").as_posix(), dpi=300)

    ### Tear Sheets
    pf.create_full_tear_sheet(
        returns,
        positions=positions,
        transactions=transactions,
        benchmark_rets=benchmark,
        round_trips=True,
    )
