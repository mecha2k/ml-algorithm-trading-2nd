#!/usr/bin/env python
# coding: utf-8

# # ML4T Workflow with zipline

# We can also integrate the model training into our backtest. The goal is to replicate the daily return predictions we used in [backtesting_with_zipline](02_backtesting_with_zipline.ipynb) and generated in [Chapter 7](../../07_linear_models). 
# 
# We will, however, use a few additional Pipeline factors to illustrate their usage. The principal new element is a CustomFactor that receives features and returns as inputs to train a model and produce predictions. We follow the workflow displayed in the following figure:
# 
# ![ML4T with Zipline](../../assets/zip_pipe_model_flow.png "ML4T Workflow with Zipline")

# ## Imports and Settings

# ### Imports

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from collections import OrderedDict
from time import time

import numpy as np
import pandas as pd
import pandas_datareader.data as web

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr

from logbook import Logger, StderrHandler, INFO

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import AverageDollarVolume, EWMA, Returns
from zipline.pipeline.factors.technical import RSI, MACDSignal, TrueRange
from zipline import run_algorithm
from zipline.api import (attach_pipeline, pipeline_output,
                         date_rules, time_rules,
                         schedule_function, commission, slippage,
                         set_slippage, set_commission,
                         record,
                         order_target, order_target_percent)

import pyfolio as pf
from pyfolio.plotting import plot_rolling_returns, plot_rolling_sharpe
from pyfolio.timeseries import forecast_cone_bootstrap


# In[3]:


sns.set_style('whitegrid')


# ### Logging Setup

# In[4]:


log_handler = StderrHandler(
        format_string='[{record.time:%Y-%m-%d %H:%M:%S.%f}]: ' +
                      '{record.level_name}: {record.func_name}: {record.message}',
        level=INFO
)
log_handler.push_application()
log = Logger('Algorithm')


# ### Algo Parameters

# In[5]:


# Target long/short positions
N_LONGS = 25
N_SHORTS = 25
MIN_POSITIONS = 15

UNIVERSE = 250

# Length of the training period (memory-intensive due to pre-processing)
TRAINING_PERIOD = 252 * 2

# train to predict N day forward returns
N_FORWARD_DAYS = 1

# How often to trade; align with prediction horizon
# for weekly, set to date_rules.week_start(days_offset=1)
TRADE_FREQ = date_rules.every_day()  


# In[6]:


start = pd.Timestamp('2015-01-01', tz='UTC')
end = pd.Timestamp('2017-12-31', tz='UTC')


# ## Factor Engineering

# To create a Pipeline Factor, we need one or more input variables, a window_length that indicates the number of most recent data points for each input and security, and the computation we want to conduct.

# We will use ten custom and built-in factors as features for our model to capture risk factors like momentum and volatility. Next, we’ll come up with a CustomFactor that trains our model.

# ### Momentum

# In[7]:


def Price_Momentum_3M():
    return Returns(window_length=63)


# In[8]:


def Returns_39W():
    return Returns(window_length=215)


# ### Volatility

# In[9]:


class Vol_3M(CustomFactor):
    # 3 months volatility
    inputs = [Returns(window_length=2)]
    window_length = 63

    def compute(self, today, assets, out, rets):
        out[:] = np.nanstd(rets, axis=0)


# ### Mean Reversion

# In[10]:


class Mean_Reversion_1M(CustomFactor):
    # standardized difference between latest monthly return
    # and their annual average 
    inputs = [Returns(window_length=21)]
    window_length = 252

    def compute(self, today, assets, out, monthly_rets):
        out[:] = (monthly_rets[-1] - np.nanmean(monthly_rets, axis=0)) /                  np.nanstd(monthly_rets, axis=0)


# ### Money Flow Volume

# In[11]:


class Moneyflow_Volume_5d(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.volume]
    window_length = 5

    def compute(self, today, assets, out, close, volume):

        mfvs = []

        for col_c, col_v in zip(close.T, volume.T):

            # denominator
            denominator = np.dot(col_c, col_v)

            # numerator
            numerator = 0.
            for n, price in enumerate(col_c.tolist()):
                if price > col_c[n - 1]:
                    numerator += price * col_v[n]
                else:
                    numerator -= price * col_v[n]

            mfvs.append(numerator / denominator)
        out[:] = mfvs


# ### Price Trend

# A linear price trend that we estimate using linear regression (see [Chapter 7](../../07_linear_models) works as follows: we use the 252 latest close prices to compute the regression coefficient on a linear time trend:

# In[12]:


class Trendline(CustomFactor):
    # linear 12 month price trend regression
    inputs = [USEquityPricing.close]
    window_length = 252

    def compute(self, today, assets, out, close):
        X = np.arange(self.window_length).reshape(-1, 1).astype(float)
        X -= X.mean()
        Y = close - np.nanmean(close, axis=0)
        out[:] = (X.T @ Y / np.var(X)) / self.window_length


# ### Price Oscillator

# In[13]:


class Price_Oscillator(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 252

    def compute(self, today, assets, out, close):
        four_week_period = close[-20:]
        out[:] = (np.nanmean(four_week_period, axis=0) /
                  np.nanmean(close, axis=0)) - 1.


# ### Combine Features

# In[14]:


vol_3M = Vol_3M()
mean_reversion_1M = Mean_Reversion_1M()
macd_signal_10d = MACDSignal()
moneyflow_volume_5d = Moneyflow_Volume_5d()
trendline = Trendline()
price_oscillator = Price_Oscillator()
price_momentum_3M = Price_Momentum_3M()
returns_39W = Returns_39W()
true_range = TrueRange()


# In[15]:


features = {
    'Vol 3M'             : vol_3M,
    'Mean Reversion 1M'  : mean_reversion_1M,
    'MACD Signal 10d'    : macd_signal_10d,
    'Moneyflow Volume 5D': moneyflow_volume_5d,
    'Trendline'          : trendline,
    'Price Oscillator'   : price_oscillator,
    'Price Momentum 3M'  : price_momentum_3M,
    '39 Week Returns'    : returns_39W,
    'True Range'         : true_range
}


# ## ML CustomFactor

# Our `CustomFactor` called LinearModel will have the `StandardScaler` and a stochastic gradient descent (SGD) implementation of ridge regression as instance attributes, and we will train the model on three days a week.
# 
# - The `compute` method generates predictions (addressing potential missing values), but first checks if the model should be trained.
# - The `_train_model` method is the centerpiece of the puzzle. It shifts the returns and aligns the resulting forward returns with the Factor features, removing missing values in the process. It scales the remaining data points and trains the linear SGDRegressor.

# In[16]:


class LinearModel(CustomFactor):
    """Obtain model predictions"""
    train_on_weekday = [0, 2, 4]

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        self._scaler = StandardScaler()
        self._model = SGDRegressor(penalty='L2')
        self._trained = False

    def _train_model(self, today, returns, inputs):

        scaler = self._scaler
        model = self._model

        shift_by = N_FORWARD_DAYS + 1
        outcome = returns[shift_by:].flatten()
        features = np.dstack(inputs)[:-shift_by]
        n_days, n_stocks, n_features = features.shape
        features = features.reshape(-1, n_features)
        features = features[~np.isnan(outcome)]
        outcome = outcome[~np.isnan(outcome)]
        outcome = outcome[np.all(~np.isnan(features), axis=1)]
        features = features[np.all(~np.isnan(features), axis=1)]
        features = scaler.fit_transform(features)

        start = time()
        model.fit(X=features, y=outcome)
#         log.info('{} | {:.2f}s'.format(today.date(), time() - start))
        self._trained = True

    def _maybe_train_model(self, today, returns, inputs):
        if (today.weekday() in self.train_on_weekday) or not self._trained:
            self._train_model(today, returns, inputs)

    def compute(self, today, assets, out, returns, *inputs):
        self._maybe_train_model(today, returns, inputs)

        # Predict most recent feature values
        X = np.dstack(inputs)[-1]
        missing = np.any(np.isnan(X), axis=1)
        X[missing, :] = 0
        X = self._scaler.transform(X)
        preds = self._model.predict(X)
        out[:] = np.where(missing, np.nan, preds)


# ## Pipeline

# The `make_ml_pipeline()` function preprocesses and combines the outcome, feature, and model parts into a Pipeline with a column for predictions.

# In[17]:


def make_ml_pipeline(universe, window_length=21, n_forward_days=5):
    pipeline_columns = OrderedDict()

    # ensure that returns is the first input
    pipeline_columns['Returns'] = Returns(inputs=[USEquityPricing.open],
                                          mask=universe,
                                          window_length=n_forward_days + 1)

    # convert factors to ranks; append to pipeline
    pipeline_columns.update({k: v.rank(mask=universe)
                             for k, v in features.items()})

    # Create ML pipeline factor.
    # window_length = length of the training period
    pipeline_columns['predictions'] = LinearModel(inputs=pipeline_columns.values(),
                                         window_length=window_length + n_forward_days,
                                         mask=universe)

    return Pipeline(screen=universe, columns=pipeline_columns)


# ## Universe

# In[18]:


def make_universe():
    # Set screen
    dollar_volume = AverageDollarVolume(window_length=90)
    return dollar_volume.top(UNIVERSE)


# In[19]:


universe = make_universe()


# ## Initialize Algorithm

# In[20]:


def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    
    set_slippage(slippage.FixedSlippage(spread=0.00))
    set_commission(commission.PerShare(cost=0, min_trade_cost=0))

    schedule_function(rebalance, TRADE_FREQ,
                      date_rules.every_day(),
                      time_rules.market_open(hours=1, minutes=30),
    )

    schedule_function(record_vars, date_rules.every_day(),
                      time_rules.market_close())

    ml_pipeline = make_ml_pipeline(universe,
                                   n_forward_days=N_FORWARD_DAYS,
                                   window_length=TRAINING_PERIOD)

    # Create our dynamic stock selector.
    attach_pipeline(ml_pipeline, 'ml_model')

    context.past_predictions = {}
    context.ic = 0
    context.rmse = 0
    context.mae = 0
    context.returns_spread_bps = 0


# ## Evaluate Predictive Accuracy

# The `evaluate_predictions()` function does exactly this: it tracks the past predictions of our model and evaluates them once returns for the relevant time horizon materialize (in our example the next day):

# In[21]:


def evaluate_predictions(output, context):
    # Look at past predictions to evaluate model performance
    # A day has passed, shift days and drop old ones
    context.past_predictions = {
        k - 1: v for k, v in context.past_predictions.items() if k > 0
    }

    if 0 in context.past_predictions:
        
        # Use today's n-day returns to evaluate predictions
        returns, predictions = (output['Returns'].dropna()
                                .align(context.past_predictions[0].dropna(),
                                       join='inner'))
        if len(returns) > 0 and len(predictions) > 0:
            context.ic = spearmanr(returns, predictions)[0]
            context.rmse = np.sqrt(
                mean_squared_error(returns, predictions))
            context.mae = mean_absolute_error(returns, predictions)

            long_rets = returns[predictions > 0].mean()
            short_rets = returns[predictions < 0].mean()
            context.returns_spread_bps = (long_rets - short_rets) * 10000

    # Store current predictions
    context.past_predictions[N_FORWARD_DAYS] = context.predicted_returns


# ## Get Pipeline Output

# We obtain new predictions using the `before_trading_start()` function that runs every morning before market open:

# In[38]:


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    output = pipeline_output('ml_model')
    context.predicted_returns = output['predictions']
    context.predicted_returns.index.set_names(['equity'], inplace=True)

    evaluate_predictions(output, context)

    # These are the securities that we are interested in trading each day.
    context.security_list = context.predicted_returns.index


# ## Rebalance

# In[39]:


def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    predictions = context.predicted_returns

    # Drop stocks that can not be traded
    predictions = predictions.loc[data.can_trade(predictions.index)]
    longs = (predictions[predictions > 0]
             .sort_values(ascending=False)[:N_LONGS]
             .index
             .tolist())
    shorts = (predictions[predictions < 0]
              .sort_values()[:N_SHORTS]
              .index
              .tolist())
    targets = set(longs + shorts)
    for position in context.portfolio.positions:
        if position not in targets:
            order_target(position, 0)
    
    n_longs, n_shorts = len(longs), len(shorts)
    if n_longs > MIN_POSITIONS and n_shorts > MIN_POSITIONS:
        for stock in longs:
            order_target_percent(stock, target=1/n_longs)
        for stock in shorts:
            order_target_percent(stock, target=-1/n_shorts)
    else:
        for stock in targets:
            if stock in context.portfolio.positions:
                order_target(stock, 0)


# ## Logging

# In[40]:


def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    record(
            leverage=context.account.leverage,
            ic=context.ic,
            rmse=context.rmse,
            mae=context.mae,
            returns_spread_bps=context.returns_spread_bps
    )


# ## Run Algo

# In[41]:


go = time()
results = run_algorithm(start=start,
                        end=end,
                        initialize=initialize,
                        before_trading_start=before_trading_start,
                        capital_base=1e6,
                        data_frequency='daily',
                        benchmark_returns=None,
                        bundle='quandl')

print('{:.2f}'.format(time()-go))


# ## Analyze using PyFolio

# In[42]:


axes = (results[['ic', 'returns_spread_bps']]
        .dropna()
        .rolling(21)
        .mean()
        .plot(subplots=True, 
              layout=(2,1), 
              figsize=(14, 6), 
              title=['Informmation Coefficient (21-day Rolling Avg.)', 'Returns Spread (bps, 21-day Rolling Avg.)'],
              legend=False))
axes = axes.flatten()
axes[0].set_ylabel('IC')
axes[1].set_ylabel('bps')
plt.suptitle('Model Performance', fontsize=14)
sns.despine()
plt.tight_layout()
plt.subplots_adjust(top=.9);


# ### Get PyFolio Input

# In[43]:


returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(results)


# ### Get Benchmark Data

# In[44]:


benchmark = web.DataReader('SP500', 'fred', '2014', '2018').squeeze()
benchmark = benchmark.pct_change().tz_localize('UTC')


# ### Custom Performance Plots

# In[45]:


fig, axes = plt.subplots(ncols=2, figsize=(16, 5))
plot_rolling_returns(returns,
                     factor_returns=benchmark,
                     live_start_date='2017-01-01',
                     logy=False,
                     cone_std=2,
                     legend_loc='best',
                     volatility_match=False,
                     cone_function=forecast_cone_bootstrap,
                    ax=axes[0])
plot_rolling_sharpe(returns, ax=axes[1])
axes[0].set_title('Cumulative Returns - In and Out-of-Sample')
sns.despine()
fig.tight_layout();


# ### Full Tearsheet

# In[46]:


pf.create_full_tear_sheet(returns, 
                          positions=positions, 
                          transactions=transactions,
                          benchmark_rets=benchmark,
                          live_start_date='2017-01-01', 
                          round_trips=True)


# In[ ]:




