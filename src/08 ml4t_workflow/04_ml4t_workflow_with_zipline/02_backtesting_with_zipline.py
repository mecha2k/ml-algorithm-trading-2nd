#!/usr/bin/env python
# coding: utf-8

# # Backtesting with zipline - Pipeline API with Custom Data

# The [Pipeline API](https://www.quantopian.com/docs/user-guide/tools/pipeline) facilitates the definition and computation of alpha factors for a cross-section of securities from historical data. The Pipeline significantly improves efficiency because it optimizes computations over the entire backtest period rather than tackling each event separately. In other words, it continues to follow an event-driven architecture but vectorizes the computation of factors where possible. 
# 
# A Pipeline uses Factors, Filters, and Classifiers classes to define computations that produce columns in a table with PIT values for a set of securities. Factors take one or more input arrays of historical bar data and produce one or more outputs for each security. There are numerous built-in factors, and you can also design your own `CustomFactor` computations.
# 
# The following figure depicts how loading the data using the `DataFrameLoader`, computing the predictive `MLSignal` using the Pipeline API, and various scheduled activities integrate with the overall trading algorithm executed via the `run_algorithm()` function. We go over the details and the corresponding code in this section.
# 
# ![The Pipeline Workflow](../../assets/zip_pipe_flow.png)
# 
# You need to register your Pipeline with the `initialize()` method and can then execute it at each time step or on a custom schedule. Zipline provides numerous built-in computations such as moving averages or Bollinger Bands that can be used to quickly compute standard factors, but it also allows for the creation of custom factors as we will illustrate next. 
# 
# Most importantly, the Pipeline API renders alpha factor research modular because it separates the alpha factor computation from the remainder of the algorithm, including the placement and execution of trade orders and the bookkeeping of portfolio holdings, values, and so on.

# The goal is to combine the daily return predictions with the OHCLV data from our Quandl bundle and then to go long on up to 10 equities with the highest predicted returns and short on those with the lowest predicted returns, requiring at least five stocks on either side similar to the backtrader example above. See comments in the notebook for implementation details.

# ## Imports & Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from collections import defaultdict
from time import time

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from logbook import Logger, StderrHandler, INFO

import matplotlib.pyplot as plt
import seaborn as sns

from zipline import run_algorithm
from zipline.api import (attach_pipeline,
                         pipeline_output,
                         date_rules,
                         time_rules,
                         record,
                         schedule_function,
                         commission,
                         slippage,
                         set_slippage,
                         set_commission,
                         order_target,
                         order_target_percent)
from zipline.data import bundles
from zipline.utils.run_algo import load_extensions
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import Column, DataSet
from zipline.pipeline.domain import US_EQUITIES
from zipline.pipeline.filters import StaticAssets
from zipline.pipeline.loaders.frame import DataFrameLoader

import pyfolio as pf
from pyfolio.plotting import plot_rolling_returns, plot_rolling_sharpe
from pyfolio.timeseries import forecast_cone_bootstrap


# In[3]:


sns.set_style('whitegrid')
pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)


# ### Load zipline extensions

# Only need this in notebook to find bundle.

# In[4]:


load_extensions(default=True,
                extensions=[],
                strict=True,
                environ=None)


# In[5]:


log_handler = StderrHandler(format_string='[{record.time:%Y-%m-%d %H:%M:%S.%f}]: ' +
                            '{record.level_name}: {record.func_name}: {record.message}',
                            level=INFO)
log_handler.push_application()
log = Logger('Algorithm')


# ## Algo Params

# We plan to hold up to 20 long and 20 short positions whenever there are at least 10 on either side that meet the criteria (positive/negative prediction for long/short position).

# In[6]:


N_LONGS = 20
N_SHORTS = 20
MIN_POSITIONS = 10


# ## Load Data

# ### Quandl Wiki Bundle

# Load the Wiki Quandl `bundle` data that we ingested earlier using `zipline ingest`. This gives us access to the security SID values, among other things.

# In[7]:


bundle_data = bundles.load('quandl')


# ### ML Predictions

# We load our predictions for the 2015-17 period and extract the Zipline IDs for the ~250 stocks in our universe during this period using the `bundle.asset_finder.lookup_symbols()` method:

# In[8]:


def load_predictions(bundle):
    predictions = pd.read_hdf('../00_data/backtest.h5', 'data')[['predicted']].dropna()
    tickers = predictions.index.get_level_values(0).unique().tolist()

    assets = bundle.asset_finder.lookup_symbols(tickers, as_of_date=None)
    predicted_sids = pd.Int64Index([asset.sid for asset in assets])
    ticker_map = dict(zip(tickers, predicted_sids))
    return (predictions
            .unstack('ticker')
            .rename(columns=ticker_map)
            .predicted
            .tz_localize('UTC')), assets


# In[9]:


predictions, assets = load_predictions(bundle_data)


# ### Define Custom Dataset

# To merge additional columns with our bundle, we define a custom `SignalData` class that inherits from `zipline.pipeline.DataSset` and contains a single `zipline.pipeline.Column` of type `float` and has the domain `US_EQUITIES`:

# In[10]:


class SignalData(DataSet):
    predictions = Column(dtype=float)
    domain = US_EQUITIES


# ### Define Pipeline Loaders

# While the bundle’s OHLCV data can rely on the built-in `USEquityPricingLoader`, we need to define our own `zipline.pipeline.loaders.frame.DataFrameLoader`:

# In[11]:


signal_loader = {SignalData.predictions: DataFrameLoader(SignalData.predictions, 
                                                         predictions)}


# In fact, we need to slightly modify the Zipline library’s source code to bypass the assumption that we will only load price data. To this end, we will add a `custom_loader` parameter to the `run_algorithm` and ensure that this loader is used when the `Pipeline` needs one of `SignalData`’s `Column` instances.

# ## Pipeline Setup

# Our Pipeline is going to have two Boolean columns that identify the assets we would like to trade as long and short positions. 
# 
# To get there, we first define a `CustomFactor` called `MLSignal` that just receives the current `SignalData.predictions`. The motivation is to allow us to use some of the convenient `Factor` methods designed to rank and filter securities.

# ### Custom ML Factor

# In[12]:


class MLSignal(CustomFactor):
    """Converting signals to Factor
        so we can rank and filter in Pipeline"""
    inputs = [SignalData.predictions]
    window_length = 1

    def compute(self, today, assets, out, preds):
        out[:] = preds


# ### Create Pipeline

# Now we create a `compute_signals()` that returns a `zipline.pipeline.Pipeline` which filters the assets that meet our long/short criteria. We will call ths function periodically while executing the backtest.

# More specifically, we set up our Pipeline by instantiating the `CustomFactor` that requires no arguments other than the defaults. We combine its `top()` and `bottom()` methods with a filter to select the highest positive and lowest negative predictions:

# In[13]:


def compute_signals():
    signals = MLSignal()
#     predictions = SignalData.predictions.latest
    return Pipeline(columns={
        'longs' : signals.top(N_LONGS, mask=signals > 0),
        'shorts': signals.bottom(N_SHORTS, mask=signals < 0)},
            screen=StaticAssets(assets)
    )


# ## Initialize Algorithm

# The `initialize()` function is part of the Algorithm API. It permits us to add entries to the `context` dictionary available to all backtest components, set parameters like commission and slippage, and schedule functions. We also attach our Pipeline to the algorithm:

# In[14]:


def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    context.n_longs = N_LONGS
    context.n_shorts = N_SHORTS
    context.min_positions = MIN_POSITIONS
    context.universe = assets

    set_slippage(slippage.FixedSlippage(spread=0.00))
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))

    schedule_function(rebalance,
                      date_rules.every_day(),
                      time_rules.market_open(hours=1, minutes=30))

    schedule_function(record_vars,
                      date_rules.every_day(),
                      time_rules.market_close())

    pipeline = compute_signals()
    attach_pipeline(pipeline, 'signals')


# ### Get daily Pipeline results

# The algorithm calls the `before_trading_start()` function every day before market opens and we use it to obtain the current pipeline values, i.e., the assets suggested for long and short positions based on the ML model predictions:

# In[15]:


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    output = pipeline_output('signals')
    context.trades = (output['longs'].astype(int)
                      .append(output['shorts'].astype(int).mul(-1))
                      .reset_index()
                      .drop_duplicates()
                      .set_index('index')
                      .squeeze())


# ## Define Rebalancing Logic

# The `rebalance()` function takes care of adjusting the portfolio positions to reflect the target long and short positions implied by the model forecasets:

# In[16]:


def rebalance(context, data):
    """
    Execute orders according to schedule_function() date & time rules.
    """
    trades = defaultdict(list)

    for stock, trade in context.trades.items():
        if not trade:
            order_target(stock, 0)
        else:
            trades[trade].append(stock)
    context.longs, context.shorts = len(trades[1]), len(trades[-1])
    if context.longs > context.min_positions and context.shorts > context.min_positions:
        for stock in trades[-1]:
            order_target_percent(stock, -1 / context.shorts)
        for stock in trades[1]:
            order_target_percent(stock, 1 / context.longs)


# ## Record Data Points

# The `record_vars()` logs information to the `pd.DataFrame` returned by `run_algorithm()` as scheduled.

# In[17]:


def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    record(leverage=context.account.leverage,
           longs=context.longs,
           shorts=context.shorts)


# ## Run Algorithm

# At this point, we have defined all ingredients for the algorithm and are ready to call `run_algorithm()` with the desired `start` and `end` dates, references to the various functions we just created, and the `custom_loader` to ensure our model predictions are available to the backtest. 

# In[18]:


dates = predictions.index.get_level_values('date')
start_date = dates.min()
end_date = (dates.max() + pd.DateOffset(1))


# In[19]:


start_date, end_date


# In[20]:


start = time()
results = run_algorithm(start=start_date,
                       end=end_date,
                       initialize=initialize,
                       before_trading_start=before_trading_start,
                       capital_base=1e6,
                       data_frequency='daily',
                       bundle='quandl',
                       custom_loader=signal_loader) # need to modify zipline

print('Duration: {:.2f}s'.format(time() - start))


# ## Performance Analysis with PyFolio

# Now we can evaluate the results using `pyfolio` tearsheets or its various `pyfolio.plotting` functions.

# In[21]:


returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(results)


# In[22]:


benchmark = web.DataReader('SP500', 'fred', '2014', '2018').squeeze()
benchmark = benchmark.pct_change().tz_localize('UTC')


# In[23]:


LIVE_DATE = '2017-01-01'


# In[24]:


fig, axes = plt.subplots(ncols=2, figsize=(16, 5))
plot_rolling_returns(returns,
                     factor_returns=benchmark,
                     live_start_date=LIVE_DATE,
                     logy=False,
                     cone_std=2,
                     legend_loc='best',
                     volatility_match=False,
                     cone_function=forecast_cone_bootstrap,
                    ax=axes[0])
plot_rolling_sharpe(returns, ax=axes[1], rolling_window=63)
axes[0].set_title('Cumulative Returns - In and Out-of-Sample')
axes[1].set_title('Rolling Sharpe Ratio (3 Months)')
sns.despine()
fig.tight_layout();


# In[25]:


pf.create_full_tear_sheet(returns, 
                          positions=positions, 
                          transactions=transactions,
                          benchmark_rets=benchmark,
                          live_start_date=LIVE_DATE, 
                          round_trips=True)

