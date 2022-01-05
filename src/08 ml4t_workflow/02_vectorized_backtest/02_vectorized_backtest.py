#!/usr/bin/env python
# coding: utf-8

# # Vectorized Backtest

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from pathlib import Path
from time import time
import datetime

import numpy as np
import pandas as pd
import pandas_datareader.data as web

from scipy.stats import spearmanr

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


# In[3]:


sns.set_style('whitegrid')
np.random.seed(42)


# ## Load Data

# ### Return Predictions

# In[4]:


DATA_DIR = Path('..', 'data')


# In[5]:


data = pd.read_hdf('00_data/backtest.h5', 'data')
data.info()


# ### SP500 Benchmark

# In[6]:


sp500 = web.DataReader('SP500', 'fred', '2014', '2018').pct_change()


# In[7]:


sp500.info()


# ## Compute Forward Returns

# In[8]:


daily_returns = data.open.unstack('ticker').sort_index().pct_change()
daily_returns.info()


# In[9]:


fwd_returns = daily_returns.shift(-1)


# ## Generate Signals

# In[10]:


predictions = data.predicted.unstack('ticker')
predictions.info()


# In[11]:


N_LONG = N_SHORT = 15


# In[12]:


long_signals = ((predictions
                .where(predictions > 0)
                .rank(axis=1, ascending=False) > N_LONG)
                .astype(int))
short_signals = ((predictions
                  .where(predictions < 0)
                  .rank(axis=1) > N_SHORT)
                 .astype(int))


# ## Compute Portfolio Returns

# In[13]:


long_returns = long_signals.mul(fwd_returns).mean(axis=1)
short_returns = short_signals.mul(-fwd_returns).mean(axis=1)
strategy = long_returns.add(short_returns).to_frame('Strategy')


# ## Plot results

# In[14]:


fig, axes = plt.subplots(ncols=2, figsize=(14,5))
strategy.join(sp500).add(1).cumprod().sub(1).plot(ax=axes[0], title='Cumulative Return')
sns.distplot(strategy.dropna(), ax=axes[1], hist=False, label='Strategy')
sns.distplot(sp500, ax=axes[1], hist=False, label='SP500')
axes[1].set_title('Daily Standard Deviation')
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
axes[1].xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
sns.despine()
fig.tight_layout();


# In[15]:


res = strategy.join(sp500).dropna()


# In[16]:


res.std()


# In[17]:


res.corr()

