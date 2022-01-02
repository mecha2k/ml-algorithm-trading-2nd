#!/usr/bin/env python
# coding: utf-8

# # Remote data access using pandas

# The pandas library enables access to data displayed on websites using the `read_html()` function and access to the API endpoints of various data providers through the related `pandas-datareader` library.

# ## Imports & Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
from datetime import datetime
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns


# ## Download html table with SP500 constituents

# The download of the content of one or more html tables works as follows, for instance for the constituents of the S&P500 index from Wikipedia

# In[3]:


sp_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_constituents = pd.read_html(sp_url, header=0)[0]


# In[4]:


sp500_constituents.info()


# In[5]:


sp500_constituents.head()


# ## pandas-datareader for Market Data

# `pandas` used to facilitate access to data providers' APIs directly, but this functionality has moved to the related pandas-datareader library. The stability of the APIs varies with provider policies, and as of June 2o18 at version 0.7, the following sources are available

# See [documentation](https://pandas-datareader.readthedocs.io/en/latest/); functionality frequently changes as underlying provider APIs evolve.

# ### Yahoo Finance

# In[6]:


start = '2014'
end = datetime(2017, 5, 24)

yahoo= web.DataReader('FB', 'yahoo', start=start, end=end)
yahoo.info()


# In[7]:


mpf.plot(yahoo.drop('Adj Close', axis=1), type='candle')
plt.tight_layout()


# ### IEX

# IEX is an alternative exchange started in response to the HFT controversy and portrayed in Michael Lewis' controversial Flash Boys. It aims to slow down the speed of trading to create a more level playing field and has been growing rapidly since launch in 2016 while still small with a market share of around 2.5% in June 2018.

# > **Note:** IEX now requires an [API](https://iexcloud.io/) key after registration for (free) account that you can store as environment variable and retrieve as illustrated below, or pass directly via keyword argument to `pandas_datareader`.

# In[8]:


IEX_API_KEY=os.getenv('IEX_API_KEY')


# In[9]:


start = datetime(2015, 2, 9)
# end = datetime(2017, 5, 24)

iex = web.DataReader('FB', 'iex', start, api_key=IEX_API_KEY)
iex.info()


# In[10]:


iex.tail()


# In[11]:


sns.set_style('whitegrid')
iex.close.plot(figsize=(14, 5))
sns.despine()


# #### Book Data
# 
# In addition to historical EOD price and volume data, IEX provides real-time depth of book quotations that offer an aggregated size of orders by price and side. This service also includes last trade price and size information.
# 
# DEEP is used to receive real-time depth of book quotations direct from IEX. The depth of book quotations received via DEEP provide an aggregated size of resting displayed orders at a price and side, and do not indicate the size or number of individual orders at any price level. Non-displayed orders and non-displayed portions of reserve orders are not represented in DEEP.
# 
# DEEP also provides last trade price and size information. Trades resulting from either displayed or non-displayed orders matching on IEX will be reported. Routed executions will not be reported.

# Only works on trading days.

# In[12]:


book = web.get_iex_book('AAPL')


# In[13]:


list(book.keys())


# In[14]:


orders = pd.concat([pd.DataFrame(book[side]).assign(side=side) for side in ['bids', 'asks']])
orders.head()


# In[15]:


for key in book.keys():
    try:
        print(f'\n{key}')
        print(pd.DataFrame(book[key]))
    except:
        print(book[key])


# In[16]:


pd.DataFrame(book['trades']).head()


# ### Quandl

# Obtain Quandl [API Key](https://www.quandl.com/tools/api) and store in environment variable as `QUANDL_API_KEY`.

# In[17]:


symbol = 'FB.US'

quandl = web.DataReader(symbol, 'quandl', '2015-01-01')
quandl.info()


# ### FRED

# In[18]:


start = datetime(2010, 1, 1)

end = datetime(2013, 1, 27)

gdp = web.DataReader('GDP', 'fred', start, end)

gdp.info()


# In[19]:


inflation = web.DataReader(['CPIAUCSL', 'CPILFESL'], 'fred', start, end)
inflation.info()


# ### Fama/French

# In[20]:


from pandas_datareader.famafrench import get_available_datasets
get_available_datasets()


# In[21]:


ds = web.DataReader('5_Industry_Portfolios', 'famafrench')
print(ds['DESCR'])


# ### World Bank

# In[22]:


from pandas_datareader import wb
gdp_variables = wb.search('gdp.*capita.*const')
gdp_variables.head()


# In[23]:


wb_data = wb.download(indicator='NY.GDP.PCAP.KD', 
                      country=['US', 'CA', 'MX'], 
                      start=1990, 
                      end=2019)
wb_data.head()


# ### OECD

# In[24]:


df = web.DataReader('TUD', 'oecd', start='2010', end='2019')
df[['Japan', 'United States']]


# 
# 
# ### Stooq

# Google finance stopped providing common index data download. The Stooq site had this data for download for a while but is currently broken, awaiting release of [fix](https://github.com/pydata/pandas-datareader/issues/594)

# In[25]:


index_url = 'https://stooq.com/t/'
ix = pd.read_html(index_url)
len(ix)


# In[26]:


sp500_stooq = web.DataReader('^SPX', 'stooq')
sp500_stooq.info()


# In[27]:


sp500_stooq.head()


# In[28]:


sp500_stooq.Close.plot(figsize=(14,4))
sns.despine()
plt.tight_layout()


# ### NASDAQ Symbols

# In[29]:


from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
symbols = get_nasdaq_symbols()
symbols.info()


# ### Tiingo

# Requires [signing up](https://api.tiingo.com/) and storing API key in environment

# In[30]:


df = web.get_data_tiingo('GOOG', api_key=os.getenv('TIINGO_API_KEY'))


# In[31]:


df.info()

