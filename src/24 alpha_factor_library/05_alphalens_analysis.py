#!/usr/bin/env python
# coding: utf-8

# # Performance Analysis with Alphalens

# ## Imports & Settings

# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[85]:


from pathlib import Path
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from alphalens.tears import (create_returns_tear_sheet,
                             create_summary_tear_sheet,
                             create_full_tear_sheet)

from alphalens import plotting
from alphalens import performance as perf
from alphalens import utils


# In[3]:


sns.set_style('whitegrid')
np.random.seed(42)
idx = pd.IndexSlice


# In[12]:


DATA_STORE = Path('..', 'data', 'assets.h5')


# ## Alphalens Analysis

# ### Get trade prices

# In[53]:


def get_trade_prices(tickers):
    return (pd.read_hdf(DATA_STORE, 'quandl/wiki/prices')
              .loc[idx['2006':'2017', tickers], 'adj_open']
              .unstack('ticker')
              .sort_index()
            .shift(-1)
            .tz_localize('UTC'))


# In[54]:


trade_prices = get_trade_prices(tickers)


# In[55]:


trade_prices.info()


# ### Load factors

# In[50]:


factors = (pd.concat([pd.read_hdf('data.h5', 'factors/common'),
                      pd.read_hdf('data.h5', 'factors/formulaic')
                      .rename(columns=lambda x: f'alpha_{int(x):03}')],
                     axis=1)
           .dropna(axis=1, thresh=100000)
           .sort_index())


# In[51]:


factors.info()


# In[52]:


tickers = factors.index.get_level_values('ticker').unique()


# In[71]:


alpha = 'alpha_054'


# In[72]:


factor = (factors[alpha]
          .unstack('ticker')
          .stack()
          .tz_localize('UTC', level='date')
          .sort_index())


# ### Generate Alphalens input data

# In[74]:


factor_data = utils.get_clean_factor_and_forward_returns(factor=factor,
                                                   prices=trade_prices,
                                                   quantiles=5,
                                                   max_loss=0.35,
                                                   periods=(1, 5, 10)).sort_index()
factor_data.info()


# ### Compute Metrics

# In[98]:


mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
    factor_data,
    by_date=True,
    by_group=False,
    demeaned=True,
    group_adjust=False,
)

mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
    rate_of_return,
    base_period=mean_quant_ret_bydate.columns[0],
)

compstd_quant_daily = std_quant_daily.apply(std_conversion,
                                            base_period=std_quant_daily.columns[0])

alpha_beta = perf.factor_alpha_beta(factor_data,
                                    demeaned=True)

mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
    mean_quant_rateret_bydate,
    factor_data["factor_quantile"].max(),
    factor_data["factor_quantile"].min(),
    std_err=compstd_quant_daily,
)


# In[100]:


mean_ret_spread_quant.mean().mul(10000).to_frame('Mean Period Wise Spread (bps)').join(alpha_beta.T).T


# ### Plot spread and cumulative returns

# In[95]:


fig, axes = plt.subplots(ncols=3, figsize=(20, 5))

mean_quant_ret, std_quantile = mean_return_by_quantile(factor_data,
                                                       by_group=False,
                                                       demeaned=True)

mean_quant_rateret = mean_quant_ret.apply(rate_of_return, axis=0,
                                          base_period=mean_quant_ret.columns[0])

plot_quantile_returns_bar(mean_quant_rateret, ax=axes[0])


factor_returns = perf.factor_returns(factor_data)

title = "Factor Weighted Long/Short Portfolio Cumulative Return (1D Period)"
plotting.plot_cumulative_returns(factor_returns['1D'],
                                 period='1D',
                                 freq=pd.tseries.offsets.BDay(),
                                 title=title,
                                 ax=axes[1])

plotting.plot_cumulative_returns_by_quantile(mean_quant_ret_bydate['1D'],
                                             freq=pd.tseries.offsets.BDay(),
                                             period='1D',
                                             ax=axes[2])
fig.tight_layout();


# ### Create Tearsheet

# In[77]:


create_summary_tear_sheet(factor_data)


# In[ ]:




