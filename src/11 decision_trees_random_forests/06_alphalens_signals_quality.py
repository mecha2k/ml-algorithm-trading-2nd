#!/usr/bin/env python
# coding: utf-8

# # Testing the signal quality with Alphalens

# ## Imports & Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path
import pandas as pd
import seaborn as sns

from alphalens.tears import (create_summary_tear_sheet,
                             create_full_tear_sheet)

from alphalens.utils import get_clean_factor_and_forward_returns


# In[3]:


sns.set_style('whitegrid')


# In[4]:


idx = pd.IndexSlice


# In[5]:


results_path = Path('results', 'return_predictions')
if not results_path.exists():
    results_path.mkdir(parents=True)


# ## Evaluating the Cross-Validation Results

# In[6]:


lookahead = 1


# In[7]:


cv_store = Path(results_path / 'parameter_tuning.h5')


# ### Get AlphaLens Input

# In[8]:


DATA_DIR = Path('..', 'data')


# Using next available prices.

# In[9]:


def get_trade_prices(tickers):
    store = DATA_DIR / 'assets.h5'
    prices = pd.read_hdf(store, 'stooq/jp/tse/stocks/prices')
    return (prices.loc[idx[tickers, '2014': '2019'], 'open']
            .unstack('ticker')
            .sort_index()
            .shift(-1)
            .dropna()
            .tz_localize('UTC'))


# Reloading predictions.

# In[10]:


best_predictions = pd.read_hdf(results_path / 'predictions.h5', f'test/{lookahead:02}')
best_predictions.info()


# In[11]:


test_tickers = best_predictions.index.get_level_values('ticker').unique()


# In[12]:


trade_prices = get_trade_prices(test_tickers)
trade_prices.info()


# In[13]:


factor = (best_predictions
          .iloc[:, :3]
          .mean(1)
          .tz_localize('UTC', level='date')
          .swaplevel()
          .dropna()
          .reset_index()
          .drop_duplicates()
          .set_index(['date', 'ticker']))


# In[14]:


factor_data = get_clean_factor_and_forward_returns(factor=factor,
                                                   prices=trade_prices,
                                                   quantiles=5,
                                                   periods=(1, 5, 10, 21))
factor_data.sort_index().info()


# ### Summary Tearsheet

# In[15]:


create_summary_tear_sheet(factor_data)


# ## Evaluating the Out-of-sample predictions

# ### Prepare Factor Data

# In[16]:


t = 1
predictions = pd.read_hdf(results_path / 'predictions.h5',
                          f'test/{t:02}').drop('y_test', axis=1)


# In[17]:


predictions.info()


# In[18]:


factor = (predictions.iloc[:, :10]
                   .mean(1)
                   .sort_index().tz_localize('UTC', level='date').swaplevel().dropna())
factor.head()


# ### Select next available trade prices

# Using next available prices.

# In[19]:


tickers = factor.index.get_level_values('ticker').unique()
trade_prices = get_trade_prices(tickers)
trade_prices.info()


# ### Get AlphaLens Inputs

# In[20]:


factor_data = get_clean_factor_and_forward_returns(factor=factor,
                                                   prices=trade_prices,
                                                   quantiles=5,
                                                   periods=(1, 5, 10, 21))
factor_data.sort_index().info()


# ### Summary Tearsheet

# In[21]:


create_summary_tear_sheet(factor_data)

