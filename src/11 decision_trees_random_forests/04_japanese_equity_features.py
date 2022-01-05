#!/usr/bin/env python
# coding: utf-8

# # Japanese Equity Data - Feature Engineering

# ## Imports & Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path

import numpy as np
import pandas as pd
import talib

import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


sns.set_style('white')


# In[4]:


idx = pd.IndexSlice


# ## Get Data

# ### Stooq Japanese Equity data 2014-2019

# In[5]:


DATA_DIR = Path('..', 'data')


# In[6]:


prices = (pd.read_hdf(DATA_DIR / 'assets.h5', 'stooq/jp/tse/stocks/prices')
          .loc[idx[:, '2010': '2019'], :]
          .loc[lambda df: ~df.index.duplicated(), :])


# In[7]:


prices.info(show_counts=True)


# In[8]:


before = len(prices.index.unique('ticker').unique())


# ### Remove symbols with missing values

# In[9]:


prices = (prices.unstack('ticker')
        .sort_index()
        .ffill(limit=5)
        .dropna(axis=1)
        .stack('ticker')
        .swaplevel())
prices.info(show_counts=True)


# In[10]:


after = len(prices.index.unique('ticker').unique())
print(f'Before: {before:,.0f} after: {after:,.0f}')


# ### Keep most traded symbols

# In[11]:


dv = prices.close.mul(prices.volume)
keep = dv.groupby('ticker').median().nlargest(1000).index.tolist()


# In[13]:


prices = prices.loc[idx[keep, :], :]
prices.info(show_counts=True)


# ## Feature Engineering

# ### Compute period returns

# In[14]:


intervals = [1, 5, 10, 21, 63]


# In[15]:


returns = []
by_ticker = prices.groupby(level='ticker').close
for t in intervals:
    returns.append(by_ticker.pct_change(t).to_frame(f'ret_{t}'))
returns = pd.concat(returns, axis=1)


# In[16]:


returns.info(show_counts=True)


# ### Remove outliers

# In[20]:


max_ret_by_sym = returns.groupby(level='ticker').max()


# In[21]:


percentiles = [0.001, .005, .01, .025, .05, .1]
percentiles += [1-p for p in percentiles]
max_ret_by_sym.describe(percentiles=sorted(percentiles)[6:])


# In[22]:


quantiles = max_ret_by_sym.quantile(.95)
to_drop = []
for ret, q in quantiles.items():
    to_drop.extend(max_ret_by_sym[max_ret_by_sym[ret]>q].index.tolist()) 


# In[23]:


to_drop = pd.Series(to_drop).value_counts()
to_drop = to_drop[to_drop > 1].index.tolist()
len(to_drop)


# In[24]:


prices = prices.drop(to_drop, level='ticker')
prices.info(show_counts=True)


# ### Calculate relative return percentiles

# In[25]:


returns = []
by_sym = prices.groupby(level='ticker').close
for t in intervals:
    ret = by_sym.pct_change(t)
    rel_perc = (ret.groupby(level='date')
             .apply(lambda x: pd.qcut(x, q=20, labels=False, duplicates='drop')))
    returns.extend([ret.to_frame(f'ret_{t}'), rel_perc.to_frame(f'ret_rel_perc_{t}')])
returns = pd.concat(returns, axis=1)


# ### Technical Indicators

# #### Percentage Price Oscillator

# In[26]:


ppo = prices.groupby(level='ticker').close.apply(talib.PPO).to_frame('PPO')


# #### Normalized Average True Range

# In[27]:


natr = prices.groupby(level='ticker', group_keys=False).apply(lambda x: talib.NATR(x.high, x.low, x.close)).to_frame('NATR')


# #### Relative Strength Indicator

# In[28]:


rsi = prices.groupby(level='ticker').close.apply(talib.RSI).to_frame('RSI')


# #### Bollinger Bands

# In[29]:


def get_bollinger(x):
    u, m, l = talib.BBANDS(x)
    return pd.DataFrame({'u': u, 'm': m, 'l': l})


# In[30]:


bbands = prices.groupby(level='ticker').close.apply(get_bollinger)


# ### Combine Features

# In[31]:


data = pd.concat([prices, returns, ppo, natr, rsi, bbands], axis=1)


# In[32]:


data['bbl'] = data.close.div(data.l)
data['bbu'] = data.u.div(data.close)
data = data.drop(['u', 'm', 'l'], axis=1)


# In[33]:


data.bbu.corr(data.bbl, method='spearman')


# ### Plot Indicators for randomly sample ticker

# In[34]:


indicators = ['close', 'bbl', 'bbu', 'PPO', 'NATR', 'RSI']
ticker = np.random.choice(data.index.get_level_values('ticker'))
(data.loc[idx[ticker, :], indicators].reset_index('ticker', drop=True)
 .plot(lw=1, subplots=True, figsize=(16, 10), title=indicators, layout=(3, 2), legend=False))
plt.suptitle(ticker, fontsize=14)
sns.despine()
plt.tight_layout()
plt.subplots_adjust(top=.95)


# In[35]:


data = data.drop(prices.columns, axis=1)


# ### Create time period indicators

# In[36]:


dates = data.index.get_level_values('date')
data['weekday'] = dates.weekday
data['month'] = dates.month
data['year'] = dates.year


# ## Compute forward returns

# In[37]:


outcomes = []
by_ticker = data.groupby('ticker')
for t in intervals:
    k = f'fwd_ret_{t:02}'
    outcomes.append(k)
    data[k] = by_ticker[f'ret_{t}'].shift(-t)


# In[38]:


data.info(null_counts=True)


# In[39]:


data.to_hdf('data.h5', 'stooq/japan/equities')

