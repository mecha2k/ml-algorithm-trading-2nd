#!/usr/bin/env python
# coding: utf-8

# # Alphalens Analysis

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from pathlib import Path
import pandas as pd
from alphalens.tears import create_summary_tear_sheet
from alphalens.utils import get_clean_factor_and_forward_returns


# In[3]:


idx = pd.IndexSlice


# ## Load Data

# In[4]:


with pd.HDFStore('data.h5') as store:
    lr_predictions = store['lr/predictions']
    lasso_predictions = store['lasso/predictions']
    lasso_scores = store['lasso/scores']
    ridge_predictions = store['ridge/predictions']
    ridge_scores = store['ridge/scores']


# In[5]:


DATA_STORE = Path('..', 'data', 'assets.h5')


# In[6]:


def get_trade_prices(tickers, start, stop):
    prices = (pd.read_hdf(DATA_STORE, 'quandl/wiki/prices').swaplevel().sort_index())
    prices.index.names = ['symbol', 'date']
    prices = prices.loc[idx[tickers, str(start):str(stop)], 'adj_open']
    return (prices
            .unstack('symbol')
            .sort_index()
            .shift(-1)
            .tz_localize('UTC'))


# In[7]:


def get_best_alpha(scores):
    return scores.groupby('alpha').ic.mean().idxmax()


# In[8]:


def get_factor(predictions):
    return (predictions.unstack('symbol')
            .dropna(how='all')
            .stack()
            .tz_localize('UTC', level='date')
            .sort_index())    


# ## Linear Regression

# In[9]:


lr_factor = get_factor(lr_predictions.predicted.swaplevel())
lr_factor.head()


# In[10]:


tickers = lr_factor.index.get_level_values('symbol').unique()


# In[11]:


trade_prices = get_trade_prices(tickers, 2014, 2017)
trade_prices.info()


# In[12]:


lr_factor_data = get_clean_factor_and_forward_returns(factor=lr_factor,
                                                      prices=trade_prices,
                                                      quantiles=5,
                                                      periods=(1, 5, 10, 21))
lr_factor_data.info()


# In[13]:


create_summary_tear_sheet(lr_factor_data);


# ## Ridge Regression

# In[14]:


best_ridge_alpha = get_best_alpha(ridge_scores)
ridge_predictions = ridge_predictions[ridge_predictions.alpha==best_ridge_alpha].drop('alpha', axis=1)


# In[15]:


ridge_factor = get_factor(ridge_predictions.predicted.swaplevel())
ridge_factor.head()


# In[16]:


ridge_factor_data = get_clean_factor_and_forward_returns(factor=ridge_factor,
                                                         prices=trade_prices,
                                                         quantiles=5,
                                                         periods=(1, 5, 10, 21))
ridge_factor_data.info()


# In[17]:


create_summary_tear_sheet(ridge_factor_data);


# ## Lasso Regression

# In[18]:


best_lasso_alpha = get_best_alpha(lasso_scores)
lasso_predictions = lasso_predictions[lasso_predictions.alpha==best_lasso_alpha].drop('alpha', axis=1)


# In[19]:


lasso_factor = get_factor(lasso_predictions.predicted.swaplevel())
lasso_factor.head()


# In[20]:


lasso_factor_data = get_clean_factor_and_forward_returns(factor=lasso_factor,
                                                      prices=trade_prices,
                                                      quantiles=5,
                                                      periods=(1, 5, 10, 21))
lasso_factor_data.info()


# In[21]:


create_summary_tear_sheet(lasso_factor_data);

