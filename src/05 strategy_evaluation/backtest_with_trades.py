from zipline.api import order_target, record, symbol


def initialize(context):
    context.i = 0
    context.asset = symbol("AAPL")


def handle_data(context, data):
    # Skip first 300 days to get full windows
    context.i += 1
    if context.i < 300:
        return

    # Compute averages
    # data.history() has to be called with the same params
    # from above and returns a pandas dataframe.
    short_mavg = data.history(context.asset, "price", bar_count=100, frequency="1d").mean()
    long_mavg = data.history(context.asset, "price", bar_count=300, frequency="1d").mean()

    # Trading logic
    if short_mavg > long_mavg:
        # order_target orders as many shares as needed to
        # achieve the desired number of shares.
        order_target(context.asset, 100)
    elif short_mavg < long_mavg:
        order_target(context.asset, 0)

    # Save values for later inspection
    record(AAPL=data.current(context.asset, "price"), short_mavg=short_mavg, long_mavg=long_mavg)


# import zipline.api
# from pytz import UTC
# from zipline import run_algorithm
# from zipline.finance import commission, slippage
# from zipline.pipeline import Pipeline, CustomFactor
# from zipline.pipeline.factors import Returns, AverageDollarVolume
# from pyfolio.utils import extract_rets_pos_txn_from_zipline


# warnings.filterwarnings("ignore")
# sns.set_style("whitegrid")
#
#
# if __name__ == "__main__":
#     zipline_logging = logbook.NestedSetup(
#         [
#             logbook.NullHandler(level=logbook.DEBUG),
#             logbook.StreamHandler(sys.stdout, level=logbook.INFO),
#             logbook.StreamHandler(sys.stderr, level=logbook.ERROR),
#         ]
#     )
#     zipline_logging.push_application()
#
#     MONTH = 21
#     YEAR = 12 * MONTH
#     N_LONGS = 50
#     N_SHORTS = 50
#     VOL_SCREEN = 1000
#
#     start = pd.Timestamp("2013-01-01", tz=UTC)
#     end = pd.Timestamp("2017-01-01", tz=UTC)
#     capital_base = 1e7
#     #
# class MeanReversion(CustomFactor):
#     inputs = [Returns(window_length=MONTH)]
#     window_length = YEAR
#
#     def compute(self, today, assets, out, monthly_returns):
#         df = pd.DataFrame(monthly_returns)
#         out[:] = df.iloc[-1].sub(df.mean()).div(df.std())
#
# def compute_factors():
#     mean_reversion = MeanReversion()
#     dollar_volume = AverageDollarVolume(window_length=30)
#     return Pipeline(
#         columns={
#             "longs": mean_reversion.bottom(N_LONGS),
#             "shorts": mean_reversion.top(N_SHORTS),
#             "ranking": mean_reversion.rank(ascending=False),
#         },
#         screen=dollar_volume.top(VOL_SCREEN),
#     )
#
# # Before_trading_start() ensures the daily execution of the pipeline and the recording of the results, including the current prices.
#
# # In[9]:
#
# def before_trading_start(context, data):
#     """Run factor pipeline"""
#     context.factor_data = pipeline_output("factor_pipeline")
#     record(factor_data=context.factor_data.ranking)
#     assets = context.factor_data.index
#     record(prices=data.current(assets, "price"))
#
# # ## Set up Rebalancing
#
# # The new rebalance() method submits trade orders to the exec_trades() method for the assets flagged for long and short positions by the pipeline with equal positive and negative weights. It also divests any current holdings that are no longer included in the factor signals:
#
# # In[10]:
#
# def rebalance(context, data):
#     """Compute long, short and obsolete holdings; place trade orders"""
#     factor_data = context.factor_data
#     assets = factor_data.index
#
#     longs = assets[factor_data.longs]
#     shorts = assets[factor_data.shorts]
#     divest = context.portfolio.positions.keys() - longs.union(shorts)
#
#     exec_trades(data, assets=divest, target_percent=0)
#     exec_trades(data, assets=longs, target_percent=1 / N_LONGS if N_LONGS else 0)
#     exec_trades(data, assets=shorts, target_percent=-1 / N_SHORTS if N_SHORTS else 0)
#
# # In[11]:
#
# def exec_trades(data, assets, target_percent):
#     """Place orders for assets using target portfolio percentage"""
#     for asset in assets:
#         if data.can_trade(asset) and not get_open_orders(asset):
#             order_target_percent(asset, target_percent)
#
# # ## Initialize Backtest
#
# # The `rebalance()` method runs according to `date_rules` and `time_rules` set by the `schedule_function()` utility at the beginning of the week, right after market_open as stipulated by the built-in US_EQUITIES calendar (see docs for details on rules).
# #
# # You can also specify a trade commission both in relative terms and as a minimum amount. There is also an option to define slippage, which is the cost of an adverse change in price between trade decision and execution
#
# # In[12]:
#
# def initialize(context):
#     """Setup: register pipeline, schedule rebalancing,
#     and set trading params"""
#     attach_pipeline(compute_factors(), "factor_pipeline")
#     schedule_function(
#         rebalance,
#         date_rules.week_start(),
#         time_rules.market_open(),
#         calendar=calendars.US_EQUITIES,
#     )
#
#     set_commission(us_equities=commission.PerShare(cost=0.00075, min_trade_cost=0.01))
#     set_slippage(
#         us_equities=slippage.VolumeShareSlippage(volume_limit=0.0025, price_impact=0.01)
#     )
#
# # ## Run Algorithm
#
# # The algorithm executes upon calling the run_algorithm() function and returns the backtest performance DataFrame.
#
# # In[13]:
#
# backtest = run_algorithm(
#     start=start,
#     end=end,
#     initialize=initialize,
#     before_trading_start=before_trading_start,
#     capital_base=capital_base,
# )
#
# # ## Extract pyfolio Inputs
#
# # The `extract_rets_pos_txn_from_zipline` utility provided by `pyfolio` extracts the data used to compute performance metrics.
#
# # In[17]:
#
# returns, positions, transactions = extract_rets_pos_txn_from_zipline(backtest)
#
# # ## Persist Results for use with `pyfolio`
#
# # In[18]:
#
# with pd.HDFStore("backtests.h5") as store:
#     store.put("backtest/equal_weight", backtest)
#     store.put("returns/equal_weight", returns)
#     store.put("positions/equal_weight", positions)
#     store.put("transactions/equal_weight", transactions)
#
# # ## Plot Results
#
# # In[19]:
#
# fig, axes = plt.subplots(nrows=2, figsize=(14, 6))
# returns.add(1).cumprod().sub(1).plot(ax=axes[0], title="Cumulative Returns")
# transactions.groupby(transactions.dt.dt.day).txn_dollars.sum().cumsum().plot(
#     ax=axes[1], title="Cumulative Transactions"
# )
# fig.tight_layout()
# sns.despine()
#
# # In[ ]:
#
# positions.index = positions.index.date
#
# # In[22]:
#
# fig, ax = plt.subplots(figsize=(15, 8))
# sns.heatmap(
#     positions.replace(0, np.nan).dropna(how="all", axis=1).T,
#     cmap=sns.diverging_palette(h_neg=20, h_pos=200),
#     ax=ax,
#     center=0,
# )
#
# # In[23]:
#
# positions.head()
#
# # In[24]:
#
# transactions.info()
#
# # In[ ]:
