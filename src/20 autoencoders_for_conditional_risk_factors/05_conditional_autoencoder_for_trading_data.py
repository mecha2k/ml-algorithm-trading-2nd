#!/usr/bin/env python
# coding: utf-8

# # Conditional Autoencoder for Asset Pricing - Data Preparation

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


idx = pd.IndexSlice
sns.set_style('whitegrid')


# In[3]:


results_path = Path('results', 'asset_pricing')
if not results_path.exists():
    results_path.mkdir(parents=True)


# ## Load Data

# ### Prices

# In[4]:


prices = pd.read_hdf(results_path / 'data.h5', 'stocks/prices/adjusted')


# In[5]:


prices.info(null_counts=True)


# ### Metadata

# In[6]:


metadata = pd.read_hdf(results_path / 'data.h5', 'stocks/info').rename(columns=str.lower)


# In[7]:


metadata.info()


# ### Select tickers with metadata

# In[8]:


sectors = (metadata.sector.value_counts() > 50).index


# In[9]:


tickers_with_errors = ['FTAI', 'AIRT', 'CYBR', 'GRAF', 'KTB']


# In[10]:


tickers_with_metadata = metadata[metadata.sector.isin(sectors) & 
                                 metadata.marketcap.notnull() &
                                 metadata.sharesoutstanding.notnull() & 
                                (metadata.sharesoutstanding > 0)].index.drop(tickers_with_errors)


# In[11]:


metadata = metadata.loc[tickers_with_metadata, ['sector', 'sharesoutstanding', 'marketcap']]
metadata.index.name = 'ticker'


# In[12]:


prices = prices.loc[idx[tickers_with_metadata, :], :]


# In[13]:


prices.info(null_counts=True)


# In[14]:


metadata.info()


# In[15]:


close = prices.close.unstack('ticker').sort_index()
close.info()


# In[16]:


volume = prices.volume.unstack('ticker').sort_index()
volume.info()


# ### Create weekly returns

# In[17]:


returns = (prices.close
           .unstack('ticker')
           .resample('W-FRI').last()
           .sort_index().pct_change().iloc[1:])
returns.info()


# In[18]:


dates = returns.index


# In[19]:


sns.distplot(returns.count(1), kde=False);


# In[20]:


with pd.HDFStore(results_path / 'autoencoder.h5') as store:
    store.put('close', close)
    store.put('volume', volume)
    store.put('returns', returns)
    store.put('metadata', metadata)


# ## Factor Engineering

# In[21]:


MONTH = 21


# ### Price Trend

# #### Short-Term Reversal

# 1-month cumulative return

# In[22]:


dates[:5]


# In[23]:


mom1m = close.pct_change(periods=MONTH).resample('W-FRI').last().stack().to_frame('mom1m')
mom1m.info()


# In[24]:


mom1m.squeeze().to_hdf(results_path / 'autoencoder.h5', 'factor/mom1m')


# #### Stock Momentum

# 11-month cumulative returns ending 1-month before month end

# In[25]:


mom12m = (close
            .pct_change(periods=11 * MONTH)
            .shift(MONTH)
            .resample('W-FRI')
            .last()
            .stack()
            .to_frame('mom12m'))


# In[26]:


mom12m.info(null_counts=True)


# In[27]:


mom12m.to_hdf(results_path / 'autoencoder.h5', 'factor/mom12m')


# #### Momentum Change

# Cumulative return from months t-6 to t-1 minus months t-12 to t-7.

# In[28]:


chmom = (close
         .pct_change(periods=6 * MONTH)
         .sub(close.pct_change(periods=6 * MONTH).shift(6 * MONTH))
         .resample('W-FRI')
         .last()
         .stack()
         .to_frame('chmom'))


# In[29]:


chmom.info(null_counts=True)


# In[30]:


chmom.to_hdf(results_path / 'autoencoder.h5', 'factor/chmom')


# #### Industry Momentum

# Equal-weighted avg. industry 12-month returns

# In[31]:


indmom = (close.pct_change(12*MONTH)
          .resample('W-FRI')
          .last()
          .stack()
          .to_frame('close')
          .join(metadata[['sector']]).groupby(['date', 'sector'])
          .close.mean()
          .to_frame('indmom')
          .reset_index())


# In[32]:


indmom.info(null_counts=True)


# In[33]:


indmom = (returns
          .stack()
          .to_frame('ret')
          .join(metadata[['sector']])
          .reset_index()
          .merge(indmom)
          .set_index(['date', 'ticker'])
          .loc[:, ['indmom']])


# In[34]:


indmom.info(null_counts=True)


# In[35]:


indmom.to_hdf(results_path / 'autoencoder.h5', 'factor/indmom')


# #### Recent Max Return

# Max daily returns from calendar month t-1

# In[36]:


maxret = (close
           .pct_change(periods=MONTH)
           .rolling(21)
           .max()
           .resample('W-FRI')
           .last()
           .stack()
           .to_frame('maxret'))


# In[37]:


maxret.info(null_counts=True)


# In[38]:


maxret.to_hdf(results_path / 'autoencoder.h5', 'factor/maxret')


# #### Long-Term Reversal

# Cumulative returns months t-36 to t-13.

# In[39]:


mom36m = (close
           .pct_change(periods=24*MONTH)
           .shift(12*MONTH)
           .resample('W-FRI')
           .last()
           .stack()
           .to_frame('mom36m'))


# In[40]:


mom36m.info(null_counts=True)


# In[41]:


mom36m.to_hdf(results_path / 'autoencoder.h5', 'factor/mom36m')


# ### Liquidity Metrics

# #### Turnover

# Avg. monthly trading volume for most recent three months scaled by number of shares; we are using the most recent no of shares from yahoo finance

# In[42]:


turn = (volume
        .rolling(3*MONTH)
        .mean()
        .resample('W-FRI')
        .last()
        .div(metadata.sharesoutstanding)
        .stack('ticker')
        .to_frame('turn'))


# In[43]:


turn.info(null_counts=True)


# In[44]:


turn.to_hdf(results_path / 'autoencoder.h5', 'factor/turn')


# #### Turnover Volatility

# Monthly std dev of daily share turnover

# In[45]:


turn_std = (prices
            .volume
            .unstack('ticker')
            .div(metadata.sharesoutstanding)
            .rolling(MONTH)
            .std()
            .resample('W-FRI')
            .last()
            .stack('ticker')
            .to_frame('turn_std'))


# In[46]:


turn_std.to_hdf(results_path / 'autoencoder.h5', 'factor/turn_std')


# #### Log Market Equity

# Natural log of market cap at end of month t-1

# In[47]:


last_price = close.ffill()
factor = close.div(last_price.iloc[-1])
mvel = np.log1p(factor.mul(metadata.marketcap).resample('W-FRI').last()).stack().to_frame('mvel')


# In[48]:


mvel.info(null_counts=True)


# In[49]:


mvel.to_hdf(results_path / 'autoencoder.h5', 'factor/mvel')


# #### Dollar Volume

# Natural log of trading volume time price per share from month t-2

# In[50]:


dv = close.mul(volume)


# In[51]:


dolvol = (np.log1p(dv.rolling(21)
                  .mean()
                  .shift(21)
                  .resample('W-FRI')
                  .last())
          .stack()
          .to_frame('dolvol'))


# In[52]:


dolvol.to_hdf(results_path / 'autoencoder.h5', 'factor/dolvol')


# #### Amihud Illiquidity

# Average of daily (absolute return / dollar volume)

# In[53]:


ill = (close.pct_change().abs()
       .div(dv)
       .rolling(21)
       .mean()
       .resample('W-FRI').last()
       .stack()
       .to_frame('ill'))


# In[54]:


ill.info(null_counts=True)


# In[55]:


ill.to_hdf(results_path / 'autoencoder.h5', 'factor/ill')


# ### Risk Measures

# #### Return Volatility

# Standard dev of daily returns from month t-1.

# In[56]:


retvol = (close.pct_change()
          .rolling(21)
          .std()
          .resample('W-FRI')
          .last()
          .stack()
          .to_frame('retvol'))


# In[57]:


retvol.info(null_counts=True)


# In[58]:


retvol.to_hdf(results_path / 'autoencoder.h5', 'factor/retvol')


# #### Market Beta

# Estimated market beta from weekly returns and equal weighted market returns for 3 years ending month t-1 with at least 52 weeks of returns.

# In[59]:


index = close.resample('W-FRI').last().pct_change().mean(1).to_frame('x')


# In[60]:


def get_market_beta(y, x=index):
    df = x.join(y.to_frame('y')).dropna()
    model = RollingOLS(endog=df.y, 
                       exog=sm.add_constant(df[['x']]),
                      window=3*52)

    return model.fit(params_only=True).params['x']


# In[61]:


beta = (returns.dropna(thresh=3*52, axis=1)
        .apply(get_market_beta).stack().to_frame('beta'))


# In[62]:


beta.info(null_counts=True)


# In[63]:


beta.to_hdf(results_path / 'autoencoder.h5', 'factor/beta')


# #### Beta Squared

# Market beta squared

# In[64]:


betasq = beta.beta.pow(2).to_frame('betasq')


# In[65]:


betasq.info(null_counts=True)


# In[66]:


betasq.to_hdf(results_path / 'autoencoder.h5', 'factor/betasq')


# #### Idiosyncratic return volatility

# Standard dev of a regression of residuals of weekly returns on the returns of an equal weighted market index returns for the prior three years.

# This takes a while!

# In[67]:


def get_ols_residuals(y, x=index):
    df = x.join(y.to_frame('y')).dropna()
    model = sm.OLS(endog=df.y, exog=sm.add_constant(df[['x']]))
    result = model.fit()
    return result.resid.std()


# In[68]:


idiovol = (returns.apply(lambda x: x.rolling(3 * 52)
                         .apply(get_ols_residuals)))


# In[69]:


idiovol = idiovol.stack().to_frame('idiovol')


# In[70]:


idiovol.info(null_counts=True)


# In[71]:


idiovol.to_hdf(results_path / 'autoencoder.h5', 'factor/idiovol')

