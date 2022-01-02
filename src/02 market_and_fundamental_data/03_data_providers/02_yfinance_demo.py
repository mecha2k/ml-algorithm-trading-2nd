#!/usr/bin/env python
# coding: utf-8

# # Downloading Market and Fundamental Data with `yfinance`

# ## Imports & Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import yfinance as yf


# ## How to work with a Ticker object

# In[3]:


symbol = 'FB'
ticker = yf.Ticker(symbol)


# ### Show ticker info

# In[4]:


pd.Series(ticker.info).head(20)


# ### Get market data

# In[5]:


data = ticker.history(period='5d',
                      interval='1m',
                      start=None,
                      end=None,
                      actions=True,
                      auto_adjust=True,
                      back_adjust=False)
data.info()


# ### View company actions

# In[6]:


# show actions (dividends, splits)
ticker.actions


# In[7]:


ticker.dividends


# In[8]:


ticker.splits


# ### Annual and Quarterly Financial Statement Summary

# In[9]:


ticker.financials


# In[10]:


ticker.quarterly_financials


# ### Annual and Quarterly Balance Sheet

# In[11]:


ticker.balance_sheet


# In[12]:


ticker.quarterly_balance_sheet


# ### Annual and Quarterly Cashflow Statement

# In[13]:


ticker.cashflow


# In[14]:


ticker.quarterly_cashflow


# In[15]:


ticker.earnings


# In[16]:


ticker.quarterly_earnings


# ### Sustainability: Environmental, Social and Governance (ESG)

# In[17]:


ticker.sustainability


# ### Analyst Recommendations

# In[18]:


ticker.recommendations.info()


# In[19]:


ticker.recommendations.tail(10)


# ### Upcoming Events

# In[20]:


ticker.calendar


# ### Option Expiration Dates

# In[21]:


ticker.options


# In[22]:


expiration = ticker.options[0]


# In[23]:


options = ticker.option_chain(expiration)


# In[24]:


options.calls.info()


# In[25]:


options.calls.head()


# In[26]:


options.puts.info()


# ## Data Download with proxy server

# You can use a proxy server to avoid having your IP blacklisted as illustrated below (but need an actual PROXY_SERVER).

# In[27]:


PROXY_SERVER = 'PROXY_SERVER'


# The following will only work with proper PROXY_SERVER...

# In[28]:


# msft = yf.Ticker("MSFT")

# msft.history(proxy=PROXY_SERVER)
# msft.get_actions(proxy=PROXY_SERVER)
# msft.get_dividends(proxy=PROXY_SERVER)
# msft.get_splits(proxy=PROXY_SERVER)
# msft.get_balance_sheet(proxy=PROXY_SERVER)
# msft.get_cashflow(proxy=PROXY_SERVER)
# msgt.option_chain(proxy=PROXY_SERVER)


# ## Downloading multiple symbols

# In[29]:


tickers = yf.Tickers('msft aapl goog')


# In[30]:


tickers


# In[31]:


pd.Series(tickers.tickers['MSFT'].info)


# In[32]:


tickers.tickers['AAPL'].history(period="1mo")


# In[33]:


tickers.history(period='1mo').stack(-1)


# In[34]:


data = yf.download("SPY AAPL", start="2020-01-01", end="2020-01-05")


# In[35]:


data.info()


# In[36]:


data = yf.download(
        tickers = "SPY AAPL MSFT", # list or string

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        period = "5d",

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval = "1m",

        # group by ticker (to access via data['SPY'])
        # (optional, default is 'column')
        group_by = 'ticker',

        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust = True,

        # download pre/post regular market hours data
        # (optional, default is False)
        prepost = True,

        # use threads for mass downloading? (True/False/Integer)
        # (optional, default is True)
        threads = True,

        # proxy URL scheme use use when downloading?
        # (optional, default is None)
        proxy = None
    )


# In[37]:


data.info()


# In[38]:


from pandas_datareader import data as pdr

import yfinance as yf
yf.pdr_override()

# download dataframe
data = pdr.get_data_yahoo('SPY',
                          start='2017-01-01',
                          end='2019-04-30',
                          auto_adjust=False)


# In[39]:


# auto_adjust = True
data.tail()


# In[40]:


# auto_adjust = False
data.tail()


# In[ ]:




