#!/usr/bin/env python
# coding: utf-8

# # How to transform data into factors

# Based on a conceptual understanding of key factor categories, their rationale and popular metrics, a key task is to identify new factors that may better capture the risks embodied by the return drivers laid out previously, or to find new ones. 
# 
# In either case, it will be important to compare the performance of innovative factors to that of known factors to identify incremental signal gains.

# We create the dataset here and store it in our [data](../../data) folder to facilitate reuse in later chapters.

# ## Imports & Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import pandas_datareader.data as web

# from pyfinance.ols import PandasRollingOLS
# replaces pyfinance.ols.PandasRollingOLS (no longer maintained)
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from talib import RSI, BBANDS, MACD, NATR, ATR

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


sns.set_style('whitegrid')
idx = pd.IndexSlice


# ## Load US equity OHLCV data

# The `assets.h5` store can be generated using the the notebook [create_datasets](../../data/create_datasets.ipynb) in the [data](../../data) directory in the root directory of this repo for instruction to download the following dataset.

# We load the Quandl stock price datasets covering the US equity markets 2000-18 using `pd.IndexSlice` to perform a slice operation on the `pd.MultiIndex`, select the adjusted close price and unpivot the column to convert the DataFrame to wide format with tickers in the columns and timestamps in the rows:

# Set data store location:

# In[4]:


DATA_STORE = '../data/assets.h5'


# In[5]:


YEAR = 12


# In[6]:


START = 1995
END = 2017


# In[7]:


with pd.HDFStore(DATA_STORE) as store:
    prices = (store['quandl/wiki/prices']
              .loc[idx[str(START):str(END), :], :]
              .filter(like='adj_')
              .dropna()
              .swaplevel()
              .rename(columns=lambda x: x.replace('adj_', ''))
              .join(store['us_equities/stocks']
                    .loc[:, ['sector']])
              .dropna())


# In[8]:


prices.info(null_counts=True)


# In[9]:


len(prices.index.unique('ticker'))


# ## Remove stocks with less than ten years of data

# In[10]:


min_obs = 10 * 252
nobs = prices.groupby(level='ticker').size()
to_drop = nobs[nobs < min_obs].index
prices = prices.drop(to_drop, level='ticker')


# In[11]:


prices.info(null_counts=True)


# In[12]:


len(prices.index.unique('ticker'))


# ## Add some Basic Factors

# ### Compute the Relative Strength Index

# In[13]:


prices['rsi'] = prices.groupby(level='ticker').close.apply(RSI)


# In[14]:


sns.distplot(prices.rsi);


# ### Compute Bollinger Bands

# In[15]:


def compute_bb(close):
    high, mid, low = BBANDS(np.log1p(close), timeperiod=20)
    return pd.DataFrame({'bb_high': high,
                         'bb_mid': mid,
                         'bb_low': low}, index=close.index)


# In[16]:


prices = (prices.join(prices
                      .groupby(level='ticker')
                      .close
                      .apply(compute_bb)))


# In[17]:


prices.info(null_counts=True)


# In[18]:


prices.filter(like='bb_').describe()


# In[19]:


fig, axes = plt.subplots(ncols=3, figsize=(15,4))
for i, col in enumerate(['bb_low', 'bb_mid', 'bb_low']):
    sns.distplot(prices[col], ax=axes[i])
    axes[i].set_title(col);
fig.tight_layout();


# In[20]:


prices['bb_up'] = prices.bb_high.sub(np.log1p(prices.close))
prices['bb_down'] = np.log1p(prices.close).sub(prices.bb_low)


# In[21]:


fig, axes = plt.subplots(ncols=2, figsize=(10,4))
for i, col in enumerate(['bb_down', 'bb_up']):
    sns.boxenplot(prices[col], ax=axes[i])
    axes[i].set_title(col);
fig.tight_layout();


# ### Compute Average True Range

# Helper for indicators with multiple inputs:

# In[22]:


by_ticker = prices.groupby('ticker', group_keys=False)


# In[23]:


def compute_atr(stock_data):
    atr = ATR(stock_data.high, 
              stock_data.low, 
              stock_data.close, 
              timeperiod=14)
    return atr.sub(atr.mean()).div(atr.std())


# In[24]:


prices['atr'] = by_ticker.apply(compute_atr)


# In[25]:


sns.distplot(prices.atr);


# In[26]:


prices['natr'] = by_ticker.apply(lambda x: NATR(high=x.high, low=x.low, close=x.close))


# In[27]:


sns.distplot(prices.natr[prices.natr<10]);


# ### Compute Moving Average Convergence/Divergence

# In[28]:


def compute_macd(close):
    macd = MACD(close)[0]
    return macd.sub(macd.mean()).div(macd.std())

prices['macd'] = prices.groupby(level='ticker').close.apply(compute_macd)


# In[29]:


sns.distplot(prices.macd);


# ## Compute dollar volume to determine universe

# In[30]:


prices['dollar_volume'] = (prices.loc[:, 'close']
                           .mul(prices.loc[:, 'volume'], axis=0))

prices.dollar_volume /= 1e6


# In[31]:


prices.to_hdf('data.h5', 'us/equities/prices')


# In[32]:


prices = pd.read_hdf('data.h5', 'us/equities/prices')
prices.info(null_counts=True)


# ## Resample OHLCV prices to monthly frequency

# To reduce training time and experiment with strategies for longer time horizons, we convert the business-daily data to month-end frequency using the available adjusted close price:

# In[33]:


last_cols = [c for c in prices.columns.unique(0) if c not in ['dollar_volume', 'volume',
                                                              'open', 'high', 'low']]


# In[34]:


prices = prices.unstack('ticker')


# In[35]:


data = (pd.concat([prices.dollar_volume.resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                   prices[last_cols].resample('M').last().stack('ticker')],
                  axis=1)
        .swaplevel()
        .dropna())


# In[36]:


data.info()


# ## Select 500 most-traded equities

# Select the 500 most-traded stocks based on a 5-year rolling average of dollar volume.

# In[37]:


data['dollar_volume'] = (data.loc[:, 'dollar_volume']
                         .unstack('ticker')
                         .rolling(window=5*12, min_periods=12)
                         .mean()
                         .stack()
                         .swaplevel())


# In[38]:


data['dollar_vol_rank'] = (data
                           .groupby('date')
                           .dollar_volume
                           .rank(ascending=False))

data = data[data.dollar_vol_rank < 500].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)


# In[39]:


len(data.index.unique('ticker'))


# ## Create monthly return series

# To capture time series dynamics that reflect, for example, momentum patterns, we compute historical returns using the method `.pct_change(n_periods)`, that is, returns over various monthly periods as identified by lags.
# 
# We then convert the wide result back to long format with the `.stack()` method, use `.pipe()` to apply the `.clip()` method to the resulting `DataFrame`, and winsorize returns at the [1%, 99%] levels; that is, we cap outliers at these percentiles.
# 
# Finally, we normalize returns using the geometric average. After using `.swaplevel()` to change the order of the `MultiIndex` levels, we obtain compounded monthly returns for six periods ranging from 1 to 12 months:

# In[40]:


outlier_cutoff = 0.01
lags = [1, 3, 6, 12]
returns = []


# In[41]:


for lag in lags:
    returns.append(data
                   .close
                   .unstack('ticker')
                   .sort_index()
                   .pct_change(lag)
                   .stack('ticker')
                   .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                          upper=x.quantile(1-outlier_cutoff)))
                   .add(1)
                   .pow(1/lag)
                   .sub(1)
                   .to_frame(f'return_{lag}m')
                   )
    
returns = pd.concat(returns, axis=1).swaplevel()
returns.info(null_counts=True)


# In[42]:


returns.describe()


# In[43]:


cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.clustermap(returns.corr('spearman'), annot=True, center=0, cmap=cmap);


# In[44]:


data = data.join(returns).drop('close', axis=1).dropna()
data.info(null_counts=True)


# In[45]:


min_obs = 5*12
nobs = data.groupby(level='ticker').size()
to_drop = nobs[nobs < min_obs].index
data = data.drop(to_drop, level='ticker')


# In[46]:


len(data.index.unique('ticker'))


# We are left with 613 tickers.

# ## Rolling Factor Betas

# We will introduce the Fama—French data to estimate the exposure of assets to common risk factors using linear regression in [Chapter 8, Time Series Models]([](../../08_time_series_models)).

# The five Fama—French factors, namely market risk, size, value, operating profitability, and investment have been shown empirically to explain asset returns and are commonly used to assess the risk/return profile of portfolios. Hence, it is natural to include past factor exposures as financial features in models that aim to predict future returns.

# We can access the historical factor returns using the `pandas-datareader` and estimate historical exposures using the `PandasRollingOLS` rolling linear regression functionality in the `pyfinance` library as follows:

# Use Fama-French research factors to estimate the factor exposures of the stock in the dataset to the 5 factors market risk, size, value, operating profitability and investment.

# In[47]:


factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 
                             'famafrench', 
                             start=START)[0].drop('RF', axis=1)
factor_data.index = factor_data.index.to_timestamp()
factor_data = factor_data.resample('M').last().div(100)
factor_data.index.name = 'date'
factor_data.info()


# In[48]:


factor_data = factor_data.join(data['return_1m']).dropna().sort_index()
factor_data['return_1m'] -= factor_data['Mkt-RF']
factor_data.info()


# In[49]:


factor_data.describe()


# In[54]:


T = 60
# betas = (factor_data
#          .groupby(level='ticker', group_keys=False)
#          .apply(lambda x: PandasRollingOLS(window=min(T, x.shape[0]-1), 
#                                            y=x.return_1m, 
#                                            x=x.drop('return_1m', axis=1)).beta)
#         .rename(columns={'Mkt-RF': 'beta'}))
betas = (factor_data.groupby(level='ticker',
                             group_keys=False)
         .apply(lambda x: RollingOLS(endog=x.return_1m,
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                     window=min(T, x.shape[0]-1))
                .fit(params_only=True)
                .params
                .rename(columns={'Mkt-RF': 'beta'})
                .drop('const', axis=1)))


# In[55]:


betas.describe().join(betas.sum(1).describe().to_frame('total'))


# In[57]:


betas.describe().join(betas.sum(1).describe().to_frame('total'))


# In[58]:


cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.clustermap(betas.corr(), annot=True, cmap=cmap, center=0);


# In[59]:


data = (data
        .join(betas
              .groupby(level='ticker')
              .shift())
       .dropna()
       .sort_index())


# In[60]:


data.info()


# ## Momentum factors

# We can use these results to compute momentum factors based on the difference between returns over longer periods and the most recent monthly return, as well as for the difference between 3 and 12 month returns as follows:

# In[61]:


for lag in [3, 6, 12]:
    data[f'momentum_{lag}'] = data[f'return_{lag}m'].sub(data.return_1m)
    if lag > 3:
        data[f'momentum_3_{lag}'] = data[f'return_{lag}m'].sub(data.return_3m)    


# ## Date Indicators

# In[62]:


dates = data.index.get_level_values('date')
data['year'] = dates.year
data['month'] = dates.month


# ## Target: Holding Period Returns

# To compute returns for our one-month target holding period, we use the returns computed previously and shift them back to align them with the current financial features.

# In[63]:


data['target'] = data.groupby(level='ticker')[f'return_1m'].shift(-1)


# In[64]:


data = data.dropna()


# In[65]:


data.sort_index().info(null_counts=True)


# ## Sector Breakdown

# In[66]:


ax = data.reset_index().groupby('sector').ticker.nunique().sort_values().plot.barh(title='Sector Breakdown')
ax.set_ylabel('')
ax.set_xlabel('# Tickers')
sns.despine()
plt.tight_layout();


# ## Store data

# In[67]:


with pd.HDFStore('data.h5') as store:
    store.put('us/equities/monthly', data)


# ## Evaluate mutual information

# In[68]:


X = data.drop('target', axis=1)
X.sector = pd.factorize(X.sector)[0]


# In[69]:


mi = mutual_info_regression(X=X, y=data.target)


# In[70]:


mi_reg = pd.Series(mi, index=X.columns)
mi_reg.nlargest(10)


# In[71]:


mi = mutual_info_classif(X=X, y=(data.target>0).astype(int))


# In[72]:


mi_class = pd.Series(mi, index=X.columns)
mi_class.nlargest(10)


# In[73]:


mi = mi_reg.to_frame('Regression').join(mi_class.to_frame('Classification'))


# In[74]:


mi.index = [' '.join(c.upper().split('_')) for c in mi.index]


# In[75]:


fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
for i, t in enumerate(['Regression', 'Classification']):
    mi[t].nlargest(20).sort_values().plot.barh(title=t, ax=axes[i])
    axes[i].set_xlabel('Mutual Information')
fig.suptitle('Mutual Information', fontsize=14)
sns.despine()
fig.tight_layout()
fig.subplots_adjust(top=.9)

