#!/usr/bin/env python
# coding: utf-8

# # 101 Formulaic Alphas

# Based on [101 Formulaic Alphas](https://arxiv.org/pdf/1601.00991.pdf), Zura Kakushadze, arxiv, 2015

# ## Imports & Settings

# In[4]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from talib import WMA


# In[6]:


idx= pd.IndexSlice
sns.set_style('whitegrid')


# >“An alpha is a combination of mathematical expressions, computer source code, and configuration parameters
# > that can be used, in combination with historical data, to make predictions about future movements of various
# > financial instruments"
# 
# [Finding Alphas: A Quantitative Approach to Building Trading Strategies](https://books.google.com/books?hl=en&lr=&id=ntuuDwAAQBAJ&oi=fnd&pg=PR11&dq=Finding+Alphas:+A+Quantitative+Approach+to+Building+Trading+Strategies&ots=nQrqbJlQu1&sig=FWfLI0_AIJWiNJ3D6tE0twHjj5I#v=onepage&q=Finding%20Alphas%3A%20A%20Quantitative%20Approach%20to%20Building%20Trading%20Strategies&f=false), Igor Tulchinsky, 2019

# ## Functions

# The expressions below that define the 101 formulaic alphas contain functions for both time-series and cross-sectional computations.

# ### Cross-section

# | Function| Definition |
# |:---|:---|
# |rank(x) | Cross-sectional rank|
# |scale(x, a) | Rescaled x such that sum(abs(x)) = a (the default is a = 1)|
# |indneutralize(x, g) | x cross-sectionally demeaned within groups g (subindustries, industries, etc.)

# In[7]:


def rank(df):
    """Return the cross-sectional percentile rank

     Args:
         :param df: tickers in columns, sorted dates in rows.

     Returns:
         pd.DataFrame: the ranked values
     """
    return df.rank(axis=1, pct=True)


# In[8]:


def scale(df):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return df.div(df.abs().sum(axis=1), axis=0)


# ### Operators

# - abs(x), log(x), sign(x), power(x, a) = standard definitions
# - same for the operators “+”, “-”, “*”, “/”, “>”, “<”, “==”, “||”, “x ? y : z”

# In[9]:


def log(df):
    return np.log1p(df)


# In[10]:


def sign(df):
    return np.sign(df)


# In[11]:


def power(df, exp):
    return df.pow(exp)


# ### Time Series

# | Function| Definition |
# |:---|:---|
# |ts_{O}(x, d) | Operator O applied to the time-series for the past d days; non-integer number of days d is converted to floor(d)
# |ts_lag(x, d) | Value of x d days ago|
# |ts_delta(x, d) | Difference between the value of x today and d days ago|
# |ts_weighted_mean(x, d) | Weighted moving average over the past d days with linearly decaying weights d, d – 1, …, 1 (rescaled to sum up to 1)
# | ts_sum(x, d) | Rolling sum over the past d days|
# | ts_product(x, d) | Rolling product over the past d days|
# | ts_stddev(x, d) | Moving standard deviation over the past d days| 
# |ts_rank(x, d) | Rank over the past d days|
# |ts_min(x, d) | Rolling min over the past d days \[alias: min(x, d)\]|
# |ts_max(x, d) | Rolling max over the past d days \[alias: max(x, d)\]|
# |ts_argmax(x, d) | Day of ts_max(x, d)|
# |ts_argmin(x, d) | Day of ts_min(x, d)|
# |ts_correlation(x, y, d) | Correlation of x and y for the past d days|
# |ts_covariance(x, y, d) | Covariance of x and y for the past d days|

# #### Pandas Implementation

# In[12]:


def ts_lag(df: pd.DataFrame, t: int = 1) -> pd.DataFrame:
    """Return the lagged values t periods ago.

    Args:
        :param df: tickers in columns, sorted dates in rows.
        :param t: lag

    Returns:
        pd.DataFrame: the lagged values
    """
    return df.shift(t)


# In[13]:


def ts_delta(df, period=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)


# In[14]:


def ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Computes the rolling ts_sum for the given window size.

    Args:
        df (pd.DataFrame): tickers in columns, dates in rows.
        window      (int): size of rolling window.

    Returns:
        pd.DataFrame: the ts_sum over the last 'window' days.
    """
    return df.rolling(window).sum()


# In[15]:


def ts_mean(df, window=10):
    """Computes the rolling mean for the given window size.

    Args:
        df (pd.DataFrame): tickers in columns, dates in rows.
        window      (int): size of rolling window.

    Returns:
        pd.DataFrame: the mean over the last 'window' days.
    """
    return df.rolling(window).mean()


# In[16]:


def ts_weighted_mean(df, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    return (df.apply(lambda x: WMA(x, timeperiod=period)))


# In[17]:


def ts_std(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return (df
            .rolling(window)
            .std())


# In[18]:


def ts_rank(df, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return (df
            .rolling(window)
            .apply(lambda x: x.rank().iloc[-1]))


# In[19]:


def ts_product(df, window=10):
    """
    Wrapper function to estimate rolling ts_product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series ts_product over the past 'window' days.
    """
    return (df
            .rolling(window)
            .apply(np.prod))


# In[20]:


def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()


# In[21]:


def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()


# In[22]:


def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax).add(1)


# In[23]:


def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return (df.rolling(window)
            .apply(np.argmin)
            .add(1))


# In[24]:


def ts_corr(x, y, window=10):
    """
    Wrapper function to estimate rolling correlations.
    :param x, y: pandas DataFrames.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y)


# In[25]:


def ts_cov(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)


# ## Load Data

# ### 500 most-traded stocks

# In[26]:


ohlcv = ['open', 'high', 'low', 'close', 'volume']
data = (pd.read_hdf('data.h5', 'data/top500')
        .loc[:, ohlcv + ['ret_01', 'sector', 'ret_fwd']]
        .rename(columns={'ret_01': 'returns'})
        .sort_index())


# In[27]:


adv20 = data.groupby('ticker').rolling(20).volume.mean().reset_index(0, drop=True)


# In[28]:


data = data.assign(adv20=adv20)


# In[29]:


data = data.join(data.groupby('date')[ohlcv].rank(axis=1, pct=True), rsuffix='_rank')


# In[30]:


data.info(null_counts=True)


# In[31]:


# data.to_hdf('factors.h5', 'data')


# ### Input Data

# |Variable|Description|
# |:---|:---|
# |returns | daily close-to-close returns|
# |open, close, high, low, volume | standard definitions for daily price and volume data|
# |vwap | daily volume-weighted average price|
# |cap | market cap|
# |adv{d} | average daily dollar volume for the past d days|
# |IndClass | a generic placeholder for a binary industry classification such as GICS, BICS, NAICS, SIC, etc., in indneutralize(x, IndClass.level), where level = sector, industry, subindustry, etc. Multiple IndClass in the same alpha need not correspond to the same industry classification. |

# In[32]:


o = data.open.unstack('ticker')
h = data.high.unstack('ticker')
l = data.low.unstack('ticker')
c = data.close.unstack('ticker')
v = data.volume.unstack('ticker')
vwap = o.add(h).add(l).add(c).div(4)
adv20 = v.rolling(20).mean()
r = data.returns.unstack('ticker')


# ## Evaluate Alphas

# In[33]:


alphas = data[['returns', 'ret_fwd']].copy()
mi,ic = {}, {}


# In[34]:


def get_mutual_info_score(returns, alpha, n=100000):
    df = pd.DataFrame({'y': returns, 'alpha': alpha}).dropna().sample(n=n)
    return mutual_info_regression(y=df.y, X=df[['alpha']])[0]


# ## Alpha 001

# ```
# rank(ts_argmax(power(((returns < 0) ? ts_std(returns, 20) : close), 2.), 5))
# ```

# In[32]:


def alpha001(c, r):
    """(rank(ts_argmax(power(((returns < 0)
        ? ts_std(returns, 20)
        : close), 2.), 5)) -0.5)"""
    c[r < 0] = ts_std(r, 20)
    return (rank(ts_argmax(power(c, 2), 5)).mul(-.5)
            .stack().swaplevel())


# In[33]:


alpha = 1


# In[34]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha001(c, r)")


# In[35]:


alphas.info()


# In[36]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[37]:


sns.distplot(alphas[f'{alpha:03}']);


# In[38]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas)


# In[39]:


mi[1] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[1]


# ## Alpha 002

# ```
# correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
# ```

# In[40]:


def alpha002(o, c, v):
    """(-1 * ts_corr(rank(ts_delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
    s1 = rank(ts_delta(log(v), 2))
    s2 = rank((c / o) - 1)
    alpha = -ts_corr(s1, s2, 6)
    return alpha.stack('ticker').swaplevel().replace([-np.inf, np.inf], np.nan)


# In[41]:


alpha = 2


# In[42]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha002(o, c, v)")


# In[43]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[44]:


sns.distplot(alphas[f'{alpha:03}']);


# In[45]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas)


# In[46]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[2]


# ## Alpha 003

# ```
# (-1 * correlation(rank(open), rank(volume), 10))
# ```

# In[47]:


def alpha003(o, v):
    """(-1 * ts_corr(rank(open), rank(volume), 10))"""

    return (-ts_corr(rank(o), rank(v), 10)
            .stack('ticker')
            .swaplevel()
            .replace([-np.inf, np.inf], np.nan))


# In[48]:


alpha = 3


# In[49]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha003(o, v)")


# In[50]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[51]:


sns.distplot(alphas[f'{alpha:03}'].clip(lower=-1));


# In[52]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[53]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 004

# ```
# (-1 * Ts_Rank(rank(low), 9))
# ```

# In[54]:


def alpha004(l):
    """(-1 * Ts_Rank(rank(low), 9))"""
    return (-ts_rank(rank(l), 9)
            .stack('ticker')
            .swaplevel())


# In[55]:


alpha = 4


# In[56]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha004(l)")


# In[57]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[58]:


sns.distplot(alphas[f'{alpha:03}']);


# In[59]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[60]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 005

# Very roughly approximating wvap as average of OHLC.

# ```
# (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
# ```

# In[61]:


def alpha005(o, vwap, c):
    """(rank((open - ts_mean(vwap, 10))) * (-1 * abs(rank((close - vwap)))))"""
    return (rank(o.sub(ts_mean(vwap, 10)))
            .mul(rank(c.sub(vwap)).mul(-1).abs())
            .stack('ticker')
            .swaplevel())


# In[62]:


alpha = 5


# In[63]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha005(o, vwap, c)")


# In[64]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[65]:


sns.distplot(alphas[f'{alpha:03}']);


# In[66]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[67]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 006

# ```
# -ts_corr(open, volume, 10)
# ```

# In[68]:


def alpha006(o, v):
    """(-ts_corr(open, volume, 10))"""
    return (-ts_corr(o, v, 10)
            .stack('ticker')
            .swaplevel())


# In[69]:


alpha = 6


# In[70]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha006(o, v)")


# In[71]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[72]:


sns.distplot(alphas[f'{alpha:03}']);


# In[73]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[74]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[75]:


mi[alpha]


# ## Alpha 007

# ```
# (adv20 < volume) 
# ? ((-1 * ts_rank(abs(ts_delta(close, 7)), 60)) * sign(ts_delta(close, 7))) 
# : -1
# ```

# In[76]:


def alpha007(c, v, adv20):
    """(adv20 < volume) 
        ? ((-ts_rank(abs(ts_delta(close, 7)), 60)) * sign(ts_delta(close, 7))) 
        : -1
    """
    
    delta7 = ts_delta(c, 7)
    return (-ts_rank(abs(delta7), 60)
            .mul(sign(delta7))
            .where(adv20<v, -1)
            .stack('ticker')
            .swaplevel())


# In[77]:


alpha = 7


# In[78]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha007(c, v, adv20)")


# In[79]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[80]:


ax = sns.distplot(alphas[f'{alpha:03}'], kde=False)
ax.set_yscale('log')
ax.set_ylabel('Frequency (log scale)')
plt.tight_layout();


# In[81]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[82]:


# mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[83]:


# mi[alpha]


# ## Alpha 008

# ```
# -rank(((ts_sum(open, 5) * ts_sum(returns, 5)) - ts_lag((ts_sum(open, 5) * ts_sum(returns, 5)),10)))
# ```

# In[84]:


def alpha008(o, r):
    """-rank(((ts_sum(open, 5) * ts_sum(returns, 5)) - 
        ts_lag((ts_sum(open, 5) * ts_sum(returns, 5)),10)))
    """
    return (-(rank(((ts_sum(o, 5) * ts_sum(r, 5)) -
                       ts_lag((ts_sum(o, 5) * ts_sum(r, 5)), 10))))
           .stack('ticker')
            .swaplevel())


# In[85]:


alpha = 8


# In[86]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha008(o, r)")


# In[87]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[88]:


sns.distplot(alphas[f'{alpha:03}']);


# In[89]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[90]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[91]:


mi[alpha]


# ## Alpha 009

# ```
# (0 < ts_min(ts_delta(close, 1), 5)) ? ts_delta(close, 1) 
# : ((ts_max(ts_delta(close, 1), 5) < 0) 
# ? ts_delta(close, 1) : (-1 * ts_delta(close, 1)))
# ```

# In[92]:


def alpha009(c):
    """(0 < ts_min(ts_delta(close, 1), 5)) ? ts_delta(close, 1) 
    : ((ts_max(ts_delta(close, 1), 5) < 0) 
    ? ts_delta(close, 1) : (-1 * ts_delta(close, 1)))
    """
    close_diff = ts_delta(c, 1)
    alpha = close_diff.where(ts_min(close_diff, 5) > 0,
                             close_diff.where(ts_max(close_diff, 5) < 0,
                                              -close_diff))
    return (alpha
            .stack('ticker')
            .swaplevel())


# In[93]:


alpha = 9


# In[94]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha009(c)")


# In[95]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[96]:


q = 0.01
sns.distplot(alphas[f'{alpha:03}'].clip(lower=alphas[f'{alpha:03}'].quantile(q),
                                        upper=alphas[f'{alpha:03}'].quantile(1-q)));


# In[97]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[98]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[99]:


mi[alpha]


# In[100]:


pd.Series(mi)


# ## Alpha 010

# ```
# rank(((0 < ts_min(ts_delta(close, 1), 4)) 
# ? ts_delta(close, 1) 
# : ((ts_max(ts_delta(close, 1), 4) < 0)
#     ? ts_delta(close, 1) 
#     : (-1 * ts_delta(close, 1)))))
# ```

# In[101]:


def alpha010(c):
    """rank(((0 < ts_min(ts_delta(close, 1), 4)) 
        ? ts_delta(close, 1) 
        : ((ts_max(ts_delta(close, 1), 4) < 0)
            ? ts_delta(close, 1) 
            : (-1 * ts_delta(close, 1)))))
    """
    close_diff = ts_delta(c, 1)
    alpha = close_diff.where(ts_min(close_diff, 4) > 0,
                             close_diff.where(ts_min(close_diff, 4) > 0,
                                              -close_diff))

    return (rank(alpha)
            .stack('ticker')
            .swaplevel())


# In[102]:


alpha = 10


# In[103]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha010(c)")


# In[104]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[105]:


sns.distplot(alphas[f'{alpha:03}']);


# In[106]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[107]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[108]:


mi[alpha]


# In[109]:


pd.Series(mi).to_csv('mi.csv')


# ## Alpha 011

# ```
# ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *rank(ts_delta(volume, 3)))
# ```

# In[110]:


def alpha011(c, vwap, v):
    """(rank(ts_max((vwap - close), 3)) + 
        rank(ts_min(vwap - close), 3)) * 
        rank(ts_delta(volume, 3))
        """
    return (rank(ts_max(vwap.sub(c), 3))
            .add(rank(ts_min(vwap.sub(c), 3)))
            .mul(rank(ts_delta(v, 3)))
            .stack('ticker')
            .swaplevel())


# In[111]:


alpha = 11


# In[112]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha011(c, vwap, v)")


# In[113]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[114]:


sns.distplot(alphas[f'{alpha:03}']);


# In[115]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[116]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[117]:


mi[alpha]


# ## Alpha 012

# ```
# sign(ts_delta(volume, 1)) * -ts_delta(close, 1)
# ```

# In[118]:


def alpha012(v, c):
    """(sign(ts_delta(volume, 1)) * 
            (-1 * ts_delta(close, 1)))
        """
    return (sign(ts_delta(v, 1)).mul(-ts_delta(c, 1))
            .stack('ticker')
            .swaplevel())


# In[119]:


alpha = 12


# In[120]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha012(v, c)")


# In[121]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[122]:


q = 0.01
sns.distplot(alphas[f'{alpha:03}'].clip(lower=alphas[f'{alpha:03}'].quantile(q),
                                        upper=alphas[f'{alpha:03}'].quantile(1-q)));


# In[123]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[124]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[125]:


mi[alpha]


# ## Alpha 013

# ```
# -rank(ts_cov(rank(close), rank(volume), 5))
# ```

# In[126]:


def alpha013(c, v):
    """-rank(ts_cov(rank(close), rank(volume), 5))"""
    return (-rank(ts_cov(rank(c), rank(v), 5))
            .stack('ticker')
            .swaplevel())


# In[127]:


alpha = 13


# In[128]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha013(c, v)")


# In[129]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[130]:


sns.distplot(alphas[f'{alpha:03}']);


# In[131]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[132]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[133]:


mi[alpha]


# In[134]:


pd.Series(mi).to_csv('mi.csv')


# ## Alpha 014

# ```
# (-rank(ts_delta(returns, 3))) * ts_corr(open, volume, 10))
# ```

# In[135]:


def alpha014(o, v, r):
    """
    (-rank(ts_delta(returns, 3))) * ts_corr(open, volume, 10))
    """

    alpha = -rank(ts_delta(r, 3)).mul(ts_corr(o, v, 10)
                                      .replace([-np.inf,
                                                np.inf],
                                               np.nan))
    return (alpha
            .stack('ticker')
            .swaplevel())


# In[136]:


alpha = 14


# In[137]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha014(o, v, r)")


# In[138]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[139]:


sns.distplot(alphas[f'{alpha:03}']);


# In[140]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[141]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[142]:


mi[alpha]


# ## Alpha 015

# ```
# (-1 * ts_sum(rank(ts_corr(rank(high), rank(volume), 3)), 3))
# ```

# In[143]:


def alpha015(h, v):
    """(-1 * ts_sum(rank(ts_corr(rank(high), rank(volume), 3)), 3))"""
    alpha = (-ts_sum(rank(ts_corr(rank(h), rank(v), 3)
                          .replace([-np.inf, np.inf], np.nan)), 3))
    return (alpha
            .stack('ticker')
            .swaplevel())


# In[144]:


alpha = 15


# In[145]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha015(h, v)")


# In[146]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[147]:


sns.distplot(alphas[f'{alpha:03}']);


# In[148]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[149]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[150]:


mi[alpha]


# ## Alpha 016

# ```
# (-1 * rank(ts_cov(rank(high), rank(volume), 5)))```

# In[151]:


def alpha016(h, v):
    """(-1 * rank(ts_cov(rank(high), rank(volume), 5)))"""
    return (-rank(ts_cov(rank(h), rank(v), 5))
            .stack('ticker')
            .swaplevel())


# In[152]:


alpha = 16


# In[153]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha016(h, v)")


# In[154]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[155]:


sns.distplot(alphas[f'{alpha:03}']);


# In[156]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[157]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[158]:


mi[alpha]


# In[159]:


pd.Series(mi).to_csv('mi.csv')


# ## Alpha 017

# ```
# rank(((0 < ts_min(ts_delta(close, 1), 4)) 
# ? ts_delta(close, 1) 
# : ((ts_max(ts_delta(close, 1), 4) < 0)
#     ? ts_delta(close, 1) 
#     : (-1 * ts_delta(close, 1)))))
# ```

# In[160]:


def alpha017(c, v):
    """(((-1 * rank(ts_rank(close, 10))) * rank(ts_delta(ts_delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))
        """
    adv20 = ts_mean(v, 20)
    return (-rank(ts_rank(c, 10))
            .mul(rank(ts_delta(ts_delta(c, 1), 1)))
            .mul(rank(ts_rank(v.div(adv20), 5)))
            .stack('ticker')
            .swaplevel())


# In[161]:


alpha = 17


# In[162]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha017(c, v)")


# In[163]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[164]:


sns.distplot(alphas[f'{alpha:03}']);


# In[165]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[166]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[167]:


mi[alpha]


# ## Alpha 018

# ```
# -rank((ts_std(abs((close - open)), 5) + (close - open)) +
#             ts_corr(close, open,10))
# ```

# In[168]:


def alpha018(o, c):
    """-rank((ts_std(abs((close - open)), 5) + (close - open)) +
            ts_corr(close, open,10))
    """
    return (-rank(ts_std(c.sub(o).abs(), 5)
                  .add(c.sub(o))
                  .add(ts_corr(c, o, 10)
                       .replace([-np.inf,
                                 np.inf],
                                np.nan)))
            .stack('ticker')
            .swaplevel())


# In[169]:


alpha = 18


# In[170]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha018(o, c)")


# In[171]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[172]:


sns.distplot(alphas[f'{alpha:03}']);


# In[173]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[174]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[175]:


mi[alpha]


# ## Alpha 019

# ```
# rank(((0 < ts_min(ts_delta(close, 1), 4)) 
# ? ts_delta(close, 1) 
# : ((ts_max(ts_delta(close, 1), 4) < 0)
#     ? ts_delta(close, 1) 
#     : (-1 * ts_delta(close, 1)))))
# ```

# In[176]:


def alpha019(c, r):
    """((-1 * sign(((close - ts_lag(close, 7)) + ts_delta(close, 7)))) * 
    (1 + rank((1 + ts_sum(returns,250)))))
    """
    return (-sign(ts_delta(c, 7) + ts_delta(c, 7))
            .mul(1 + rank(1 + ts_sum(r, 250)))
            .stack('ticker')
            .swaplevel())


# In[177]:


alpha = 19


# In[178]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha019(c, r)")


# In[179]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[180]:


sns.distplot(alphas[f'{alpha:03}']);


# In[181]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[182]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[183]:


mi[alpha]


# In[184]:


pd.Series(mi).to_csv('mi.csv')


# ## Alpha 020

# ```
# -rank(open - ts_lag(high, 1)) * 
#  rank(open - ts_lag(close, 1)) * 
#  rank(open -ts_lag(low, 1))
# ```

# In[185]:


def alpha020(o, h, l, c):
    """-rank(open - ts_lag(high, 1)) * 
        rank(open - ts_lag(close, 1)) * 
        rank(open -ts_lag(low, 1))"""
    return (rank(o - ts_lag(h, 1))
            .mul(rank(o - ts_lag(c, 1)))
            .mul(rank(o - ts_lag(l, 1)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[186]:


alpha = 20


# In[187]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha020(o, h, l, c)")


# In[188]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[189]:


sns.distplot(alphas[f'{alpha:03}']);


# In[190]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[191]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[192]:


mi[alpha]


# ## Alpha 021

# ```
# ts_mean(close, 8) + ts_std(close, 8) < ts_mean(close, 2)
#         ? -1 
#         : (ts_mean(close,2) < ts_mean(close, 8) - ts_std(close, 8)
#             ? 1 
#             : (volume / adv20 < 1
#                 ? -1 
#                 : 1))
# ```

# In[193]:


def alpha021(c, v):
    """ts_mean(close, 8) + ts_std(close, 8) < ts_mean(close, 2)
        ? -1
        : (ts_mean(close,2) < ts_mean(close, 8) - ts_std(close, 8)
            ? 1
            : (volume / adv20 < 1
                ? -1
                : 1))
    """
    sma2 = ts_mean(c, 2)
    sma8 = ts_mean(c, 8)
    std8 = ts_std(c, 8)

    cond_1 = sma8.add(std8) < sma2
    cond_2 = sma8.add(std8) > sma2
    cond_3 = v.div(ts_mean(v, 20)) < 1

    val = np.ones_like(c)
    alpha = pd.DataFrame(np.select(condlist=[cond_1, cond_2, cond_3],
                                   choicelist=[-1, 1, -1], default=1),
                         index=c.index,
                         columns=c.columns)

    return (alpha
            .stack('ticker')
            .swaplevel())


# In[194]:


alpha = 21


# In[195]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha021(c, v)")


# In[196]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[197]:


alphas[f'{alpha:03}'].value_counts()


# In[198]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[199]:


# mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[200]:


# mi[alpha]


# ## Alpha 022

# ```
# -(ts_delta(ts_corr(high, volume, 5), 5) * 
#         rank(ts_std(close, 20)))
# ```

# In[201]:


def alpha022(h, c, v):
    """-(ts_delta(ts_corr(high, volume, 5), 5) * 
        rank(ts_std(close, 20)))
    """

    return (ts_delta(ts_corr(h, v, 5)
                     .replace([-np.inf,
                               np.inf],
                              np.nan), 5)
            .mul(rank(ts_std(c, 20)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[202]:


alpha = 22


# In[203]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha022(h, c, v)")


# In[204]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[205]:


sns.distplot(alphas[f'{alpha:03}']);


# In[206]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[207]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[208]:


mi[alpha]


# In[209]:


pd.Series(mi).to_csv('mi.csv')


# ## Alpha 023

# ```
# ((ts_sum(high, 20) / 20) < high)
#             ? (-1 * ts_delta(high, 2))
#             : 0
# ```

# In[210]:


def alpha023(h, c):
    """((ts_mean(high, 20) < high)
            ? (-1 * ts_delta(high, 2))
            : 0
        """

    return (ts_delta(h, 2)
            .mul(-1)
            .where(ts_mean(h, 20) < h, 0)
            .stack('ticker')
            .swaplevel())


# In[211]:


alpha = 23


# In[212]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha023(h, c)")


# In[213]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[214]:


q = 0.025
sns.distplot(alphas[f'{alpha:03}'].clip(lower=alphas[f'{alpha:03}'].quantile(q),
                                        upper=alphas[f'{alpha:03}'].quantile(1-q)));


# In[215]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[216]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[217]:


mi[alpha]


# ## Alpha 024

# ```
# ((((ts_delta((ts_mean(close, 100)), 100) / ts_lag(close, 100)) <= 0.05)  
#         ? (-1 * (close - ts_min(close, 100))) 
#         : (-1 * ts_delta(close, 3)))
# ```

# In[218]:


def alpha024(c):
    """((((ts_delta((ts_mean(close, 100)), 100) / ts_lag(close, 100)) <= 0.05)  
        ? (-1 * (close - ts_min(close, 100))) 
        : (-1 * ts_delta(close, 3)))
    """
    cond = ts_delta(ts_mean(c, 100), 100) / ts_lag(c, 100) <= 0.05

    return (c.sub(ts_min(c, 100)).mul(-1).where(cond, -ts_delta(c, 3))
            .stack('ticker')
            .swaplevel())


# In[219]:


alpha = 24


# In[220]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha024(c)")


# In[221]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[222]:


q = 0.01
sns.distplot(alphas[f'{alpha:03}'].clip(lower=alphas[f'{alpha:03}'].quantile(q),
                                        upper=alphas[f'{alpha:03}'].quantile(1-q)));


# In[223]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[224]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[225]:


mi[alpha]


# ## Alpha 025

# ```
# rank((-1 * returns) * adv20 * vwap * (high - close))
# ```

# In[226]:


def alpha025(h, c, r, vwap, adv20):
    """rank((-1 * returns) * adv20 * vwap * (high - close))"""
    return (rank(-r.mul(adv20)
                 .mul(vwap)
                 .mul(h.sub(c)))
            .stack('ticker')
            .swaplevel())


# In[227]:


alpha = 25


# In[228]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha025(h, c, r, vwap, adv20)")


# In[229]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[230]:


sns.distplot(alphas[f'{alpha:03}']);


# In[231]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[232]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[233]:


mi[alpha]


# In[234]:


pd.Series(mi).to_csv('mi.csv')


# ## Alpha 026

# ```
# (-1 * rank(ts_cov(rank(high), rank(volume), 5)))```

# In[235]:


def alpha026(h, v):
    """(-1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))"""
    return (ts_max(ts_corr(ts_rank(v, 5), 
                           ts_rank(h, 5), 5)
                   .replace([-np.inf, np.inf], np.nan), 3)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[236]:


alpha = 26


# In[237]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha026(h, v)")


# In[238]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[239]:


sns.distplot(alphas[f'{alpha:03}']);


# In[240]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[241]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[242]:


mi[alpha]


# ## Alpha 027

# ```
# rank(((0 < ts_min(ts_delta(close, 1), 4)) 
# ? ts_delta(close, 1) 
# : ((ts_max(ts_delta(close, 1), 4) < 0)
#     ? ts_delta(close, 1) 
#     : (-1 * ts_delta(close, 1)))))
# ```

# In[243]:


def alpha027(v, vwap):
    """((0.5 < rank(ts_mean(ts_corr(rank(volume), rank(vwap), 6), 2))) 
            ? -1
            : 1)"""
    cond = rank(ts_mean(ts_corr(rank(v),
                                rank(vwap), 6), 2))
    alpha = cond.notnull().astype(float)
    return (alpha.where(cond <= 0.5, -alpha)
            .stack('ticker')
            .swaplevel())


# In[244]:


alpha = 27


# In[245]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha027(v, vwap)")


# In[246]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[247]:


sns.distplot(alphas[f'{alpha:03}']);


# In[248]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[249]:


# mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[250]:


# mi[alpha]


# ## Alpha 028

# ```
# -rank((ts_std(abs((close - open)), 5) + (close - open)) +
#             ts_corr(close, open,10))
# ```

# In[251]:


def alpha028(h, l, c, v, adv20):
    """scale(((ts_corr(adv20, low, 5) + (high + low) / 2) - close))"""
    return (scale(ts_corr(adv20, l, 5)
                  .replace([-np.inf, np.inf], 0)
                  .add(h.add(l).div(2).sub(c)))
            .stack('ticker')
            .swaplevel())


# In[252]:


alpha = 28


# In[253]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha028(h, l, c, v, adv20)")


# In[254]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[255]:


sns.distplot(alphas[f'{alpha:03}']);


# In[256]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[257]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[258]:


mi[alpha]


# In[259]:


pd.Series(mi).to_csv('mi.csv')


# ## Alpha 029

# ```
# rank(((0 < ts_min(ts_delta(close, 1), 4)) 
# ? ts_delta(close, 1) 
# : ((ts_max(ts_delta(close, 1), 4) < 0)
#     ? ts_delta(close, 1) 
#     : (-1 * ts_delta(close, 1)))))
# ```

# In[260]:


def alpha029(c, r):
    """(ts_min(ts_product(rank(rank(scale(log(ts_sum(ts_min(rank(rank((-1 * 
            rank(ts_delta((close - 1),5))))), 2), 1))))), 1), 5)
        + ts_rank(ts_lag((-1 * returns), 6), 5))
    """
    return (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-rank(ts_delta((c - 1), 5)))), 2))))), 5)
            .add(ts_rank(ts_lag((-1 * r), 6), 5))
            .stack('ticker')
            .swaplevel())


# In[261]:


alpha = 29


# In[262]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha029(c, r)")


# In[263]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[264]:


sns.distplot(alphas[f'{alpha:03}']);


# In[265]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[266]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[267]:


mi[alpha]


# ## Alpha 030

# ```
# -rank(open - ts_lag(high, 1)) * 
#  rank(open - ts_lag(close, 1)) * 
#  rank(open -ts_lag(low, 1))
# ```

# In[268]:


def alpha030(c, v):
    """(((1.0 - rank(((sign((close - ts_lag(close, 1))) +
            sign((ts_lag(close, 1) - ts_lag(close, 2)))) +
            sign((ts_lag(close, 2) - ts_lag(close, 3)))))) *
            ts_sum(volume, 5)) / ts_sum(volume, 20))"""
    close_diff = ts_delta(c, 1)
    return (rank(sign(close_diff)
                 .add(sign(ts_lag(close_diff, 1)))
                 .add(sign(ts_lag(close_diff, 2))))
            .mul(-1).add(1)
            .mul(ts_sum(v, 5))
            .div(ts_sum(v, 20))
            .stack('ticker')
            .swaplevel())


# In[269]:


alpha = 30


# In[270]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha030(c, v)")


# In[271]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[272]:


sns.distplot(alphas[f'{alpha:03}']);


# In[273]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[274]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[275]:


mi[alpha]


# ## Alpha 031

# ```
# ts_mean(close, 8) + ts_std(close, 8) < ts_mean(close, 2)
#         ? -1 
#         : (ts_mean(close,2) < ts_mean(close, 8) - ts_std(close, 8)
#             ? 1 
#             : (volume / adv20 < 1
#                 ? -1 
#                 : 1))
# ```

# In[276]:


def alpha031(l, c, adv20):
    """((rank(rank(rank(ts_weighted_mean((-1 * rank(rank(ts_delta(close, 10)))), 10)))) +
        rank((-1 * ts_delta(close, 3)))) + sign(scale(ts_corr(adv20, low, 12))))
    """
    return (rank(rank(rank(ts_weighted_mean(rank(rank(ts_delta(c, 10))).mul(-1), 10))))
            .add(rank(ts_delta(c, 3).mul(-1)))
            .add(sign(scale(ts_corr(adv20, l, 12)
                            .replace([-np.inf, np.inf],
                                     np.nan))))
            .stack('ticker')
            .swaplevel())


# In[277]:


alpha = 31


# In[278]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha031(l, c, adv20)")


# In[279]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[280]:


sns.distplot(alphas[f'{alpha:03}']);


# In[281]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# ## Alpha 032

# ```
# scale(ts_mean(close, 7) - close) + 
#         (20 * scale(ts_corr(vwap, ts_lag(close, 5),230)))```

# In[282]:


def alpha032(c, vwap):
    """scale(ts_mean(close, 7) - close) + 
        (20 * scale(ts_corr(vwap, ts_lag(close, 5),230)))"""
    return (scale(ts_mean(c, 7).sub(c))
            .add(20 * scale(ts_corr(vwap,
                                    ts_lag(c, 5), 230)))
            .stack('ticker')
            .swaplevel())


# In[283]:


alpha = 32


# In[284]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha032(c, vwap)")


# In[285]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[286]:


sns.distplot(alphas[f'{alpha:03}']);


# In[287]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[288]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, 
                                  alphas[f'{alpha:03}'])


# In[289]:


mi[alpha]


# ## Alpha 033

# ```
# ((ts_sum(high, 20) / 20) < high)
#             ? (-1 * ts_delta(high, 2))
#             : 0
# ```

# In[290]:


def alpha033(o, c):
    """rank(-(1 - (open / close)))"""
    return (rank(o.div(c).mul(-1).add(1).mul(-1))
            .stack('ticker')
            .swaplevel())


# In[291]:


alpha = 33


# In[292]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha033(o, c)")


# In[293]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[294]:


sns.distplot(alphas[f'{alpha:03}']);


# In[295]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[296]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[297]:


mi[alpha]


# ## Alpha 034

# ```
# ((((ts_delta((ts_mean(close, 100)), 100) / ts_lag(close, 100)) <= 0.05)  
#         ? (-1 * (close - ts_min(close, 100))) 
#         : (-1 * ts_delta(close, 3)))
# ```

# In[298]:


def alpha034(c, r):
    """rank(((1 - rank((ts_std(returns, 2) / ts_std(returns, 5)))) + (1 - rank(ts_delta(close, 1)))))"""

    return (rank(rank(ts_std(r, 2).div(ts_std(r, 5))
                      .replace([-np.inf, np.inf],
                               np.nan))
                 .mul(-1)
                 .sub(rank(ts_delta(c, 1)))
                 .add(2))
            .stack('ticker')
            .swaplevel())


# In[299]:


alpha = 34


# In[300]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha034(c, r)")


# In[301]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[302]:


sns.distplot(alphas[f'{alpha:03}']);


# In[303]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[304]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[305]:


mi[alpha]


# In[306]:


pd.Series(mi).to_csv('mi.csv')


# ## Alpha 035

# ```
# rank((-1 * returns) * adv20 * vwap * (high - close))
# ```

# In[307]:


def alpha035(h, l, c, v, r):
    """((ts_Rank(volume, 32) *
        (1 - ts_Rank(((close + high) - low), 16))) *
        (1 -ts_Rank(returns, 32)))
    """
    return (ts_rank(v, 32)
            .mul(1 - ts_rank(c.add(h).sub(l), 16))
            .mul(1 - ts_rank(r, 32))
            .stack('ticker')
            .swaplevel())


# In[308]:


alpha = 35


# In[309]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha035(h, l, c, v, r)")


# In[310]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[311]:


sns.distplot(alphas[f'{alpha:03}']);


# In[312]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[313]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[314]:


mi[alpha]


# ## Alpha 036

# ```
# 2.21 * rank(ts_corr((close - open), ts_lag(volume, 1), 15)) +
# 0.7 * rank((open- close)) +
# 0.73 * rank(ts_Rank(ts_lag(-1 * returns, 6), 5)) +
# rank(abs(ts_corr(vwap,adv20, 6))) +
# 0.6 * rank(((ts_mean(close, 200) - open) * (close - open)))
# ```

# In[315]:


def alpha036(o, c, v, r, adv20):
    """2.21 * rank(ts_corr((close - open), ts_lag(volume, 1), 15)) +
        0.7 * rank((open- close)) +
        0.73 * rank(ts_Rank(ts_lag(-1 * returns, 6), 5)) +
        rank(abs(ts_corr(vwap,adv20, 6))) +
        0.6 * rank(((ts_mean(close, 200) - open) * (close - open)))
    """

    return (rank(ts_corr(c.sub(o), ts_lag(v, 1), 15)).mul(2.21)
            .add(rank(o.sub(c)).mul(.7))
            .add(rank(ts_rank(ts_lag(-r, 6), 5)).mul(0.73))
            .add(rank(abs(ts_corr(vwap, adv20, 6))))
            .add(rank(ts_mean(c, 200).sub(o).mul(c.sub(o))).mul(0.6))
            .stack('ticker')
            .swaplevel())


# In[316]:


alpha = 36


# In[317]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha036(o, c, v, r, adv20)")


# In[318]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[319]:


sns.distplot(alphas[f'{alpha:03}']);


# In[320]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[321]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[322]:


mi[alpha]


# ## Alpha 037

# ```
# rank(ts_corr(ts_lag(open - close, 1), close, 200)) + 
#         rank(open - close)
# ```

# In[323]:


def alpha037(o, c):
    """(rank(ts_corr(ts_lag((open - close), 1), close, 200)) + rank((open - close)))"""
    return (rank(ts_corr(ts_lag(o.sub(c), 1), c, 200))
            .add(rank(o.sub(c)))
            .stack('ticker')
            .swaplevel())


# In[324]:


alpha = 37


# In[325]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha037(o, c)")


# In[326]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[327]:


sns.distplot(alphas[f'{alpha:03}']);


# In[328]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[329]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[330]:


mi[alpha]


# In[331]:


pd.Series(mi).to_csv('mi.csv')


# ## Alpha 038

# ```
# 1 * rank(ts_rank(close, 10)) * rank(close / open)
# ```

# In[332]:


def alpha038(o, c):
    """"-1 * rank(ts_rank(close, 10)) * rank(close / open)"""
    return (rank(ts_rank(o, 10))
            .mul(rank(c.div(o).replace([-np.inf, np.inf], np.nan)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[333]:


alpha = 38


# In[334]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha038(o, c)")


# In[335]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[336]:


sns.distplot(alphas[f'{alpha:03}']);


# In[337]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[338]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[339]:


mi[alpha]


# ## Alpha 039

# ```
# -rank(ts_delta(close, 7) * (1 - rank(ts_weighted_mean(volume / adv20, 9)))) * 
#     (1 + rank(ts_sum(returns, 250)))
# ```

# In[340]:


def alpha039(c, v, r, adv20):
    """-rank(ts_delta(close, 7) * (1 - rank(ts_weighted_mean(volume / adv20, 9)))) * 
            (1 + rank(ts_sum(returns, 250)))"""
    return (rank(ts_delta(c, 7).mul(rank(ts_weighted_mean(v.div(adv20), 9)).mul(-1).add(1))).mul(-1)
            .mul(rank(ts_mean(r, 250).add(1)))
            .stack('ticker')
            .swaplevel())


# In[341]:


alpha = 39


# In[342]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha039(c, v, r, adv20)")


# In[343]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[344]:


sns.distplot(alphas[f'{alpha:03}']);


# In[345]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[346]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])


# In[347]:


mi[alpha]


# ## Alpha 040

# ```
# -rank(open - ts_lag(high, 1)) * 
#  rank(open - ts_lag(close, 1)) * 
#  rank(open -ts_lag(low, 1))
# ```

# In[348]:


def alpha040(h, v):
    """((-1 * rank(ts_std(high, 10))) * ts_corr(high, volume, 10))
    """
    return (rank(ts_std(h, 10))
            .mul(ts_corr(h, v, 10))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[349]:


alpha = 40


# In[350]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha040(h, v)")


# In[351]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[352]:


sns.distplot(alphas[f'{alpha:03}']);


# In[353]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[354]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 041

# ```
# power(high * low, 0.5) - vwap```

# In[355]:


def alpha041(h, l, vwap):
    """power(high * low, 0.5 - vwap"""
    return (power(h.mul(l), 0.5)
            .sub(vwap)
            .stack('ticker')
            .swaplevel())


# In[356]:


alpha = 41


# In[357]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha041(h, l, vwap)")


# In[358]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[359]:


sns.distplot(alphas[f'{alpha:03}']);


# In[360]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[361]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 042

# ```
# rank(vwap - close) / rank(vwap + close)
# ```

# In[362]:


def alpha042(c, vwap):
    """rank(vwap - close) / rank(vwap + close)"""
    return (rank(vwap.sub(c))
            .div(rank(vwap.add(c)))
            .stack('ticker')
            .swaplevel())


# In[363]:


alpha = 42


# In[364]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha042(c, vwap)")


# In[365]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[366]:


sns.distplot(alphas[f'{alpha:03}']);


# In[367]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[368]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 043

# ```
# ((ts_sum(high, 20) / 20) < high)
#             ? (-1 * ts_delta(high, 2))
#             : 0
# ```

# In[369]:


def alpha043(c, adv20):
    """(ts_rank((volume / adv20), 20) * ts_rank((-1 * ts_delta(close, 7)), 8))"""

    return (ts_rank(v.div(adv20), 20)
            .mul(ts_rank(ts_delta(c, 7).mul(-1), 8))
            .stack('ticker')
            .swaplevel())


# In[370]:


alpha = 43


# In[371]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha043(c, adv20)")


# In[372]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[373]:


sns.distplot(alphas[f'{alpha:03}']);


# In[374]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[375]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 044

# ```
# -ts_corr(high, rank(volume), 5)
# ```

# In[376]:


def alpha044(h, v):
    """-ts_corr(high, rank(volume), 5)"""

    return (ts_corr(h, rank(v), 5)
            .replace([-np.inf, np.inf], np.nan)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[377]:


alpha = 44


# In[378]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha044(h, v)")


# In[379]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[380]:


sns.distplot(alphas[f'{alpha:03}']);


# In[381]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[382]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 045

# ```
# -(rank((ts_mean(ts_lag(close, 5), 20)) * 
#         ts_corr(close, volume, 2)) *
#         rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2)))
# ```

# In[383]:


def alpha045(c, v):
    """-(rank((ts_mean(ts_lag(close, 5), 20)) * 
        ts_corr(close, volume, 2)) *
        rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2)))"""

    return (rank(ts_mean(ts_lag(c, 5), 20))
            .mul(ts_corr(c, v, 2)
                 .replace([-np.inf, np.inf], np.nan))
            .mul(rank(ts_corr(ts_sum(c, 5),
                              ts_sum(c, 20), 2)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[384]:


alpha = 45


# In[385]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha045(c, v)")


# In[386]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[387]:


sns.distplot(alphas[f'{alpha:03}']);


# In[388]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[389]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 046

# ```
# 0.25 < ts_lag(ts_delta(close, 10), 10) / 10 - ts_delta(close, 10) / 10
#     ? -1
#     : ((ts_lag(ts_delta(close, 10), 10) / 10 - ts_delta(close, 10) / 10 < 0) 
#         ? 1 
#         : -ts_delta(close, 1))
# ```

# In[390]:


def alpha046(c):
    """0.25 < ts_lag(ts_delta(close, 10), 10) / 10 - ts_delta(close, 10) / 10
            ? -1
            : ((ts_lag(ts_delta(close, 10), 10) / 10 - ts_delta(close, 10) / 10 < 0) 
                ? 1 
                : -ts_delta(close, 1))
    """

    cond = ts_lag(ts_delta(c, 10), 10).div(10).sub(ts_delta(c, 10).div(10))
    alpha = pd.DataFrame(-np.ones_like(cond),
                         index=c.index,
                         columns=c.columns)
    alpha[cond.isnull()] = np.nan
    return (cond.where(cond > 0.25,
                       -alpha.where(cond < 0,
                       -ts_delta(c, 1)))
            .stack('ticker')
            .swaplevel())


# In[391]:


alpha = 46


# In[392]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha046(c)")


# In[393]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[394]:


sns.distplot(alphas[f'{alpha:03}']);


# In[395]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[396]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 047

# ```
# rank(ts_corr(ts_lag(open - close, 1), close, 200)) + 
#         rank(open - close)
# ```

# In[397]:


def alpha047(h, c, v, vwap, adv20):
    """((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / 
        (ts_sum(high, 5) /5))) - rank((vwap - ts_lag(vwap, 5))))"""

    return (rank(c.pow(-1)).mul(v).div(adv20)
            .mul(h.mul(rank(h.sub(c))
                       .div(ts_mean(h, 5)))
                 .sub(rank(ts_delta(vwap, 5))))
            .stack('ticker')
            .swaplevel())


# In[398]:


alpha = 47


# In[399]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha047(h, c, v, vwap, adv20)")


# In[400]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[401]:


sns.distplot(alphas[f'{alpha:03}']);


# In[402]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[403]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 048

# ```
# (indneutralize(((ts_corr(ts_delta(close, 1), ts_delta(ts_lag(close, 1), 1), 250) *ts_delta(close, 1)) / close), IndClass.subindustry) / ts_sum(((ts_delta(close, 1) / ts_lag(close, 1))^2), 250))
# ```

# In[404]:


def alpha48(c, industry):
    """(indneutralize(((ts_corr(ts_delta(close, 1), ts_delta(ts_lag(close, 1), 1), 250) * 
        ts_delta(close, 1)) / close), IndClass.subindustry) / 
        ts_sum(((ts_delta(close, 1) / ts_lag(close, 1))^2), 250))"""
    pass


# In[405]:


alpha = 48


# In[406]:


# %%time
# alphas[f'{alpha:03}'] = alpha48(o, c)


# In[407]:


# alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[408]:


# sns.distplot(alphas[f'{alpha:03}']);


# In[409]:


# g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);
# 


# In[410]:


# mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
# mi[alpha]


# ## Alpha 049

# ```
# ts_delta(ts_lag(close, 10), 10).div(10).sub(ts_delta(close, 10).div(10)) < -0.1 * c
#         ? 1 
#         : -ts_delta(close, 1)
# ```

# In[411]:


def alpha049(c):
    """ts_delta(ts_lag(close, 10), 10).div(10).sub(ts_delta(close, 10).div(10)) < -0.1 * c
        ? 1 
        : -ts_delta(close, 1)"""
    cond = (ts_delta(ts_lag(c, 10), 10).div(10)
            .sub(ts_delta(c, 10).div(10)) >= -0.1 * c)
    return (-ts_delta(c, 1)
            .where(cond, 1)
            .stack('ticker')
            .swaplevel())


# In[412]:


alpha = 49


# In[413]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha049(c)")


# In[414]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[415]:


sns.distplot(alphas[f'{alpha:03}'], kde=False);


# In[416]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# ## Alpha 050

# ```
# -ts_max(rank(ts_corr(rank(volume), rank(vwap), 5)), 5)
# ```

# In[417]:


def alpha050(v, vwap):
    """-ts_max(rank(ts_corr(rank(volume), rank(vwap), 5)), 5)"""
    return (ts_max(rank(ts_corr(rank(v),
                                rank(vwap), 5)), 5)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[418]:


alpha = 50


# In[419]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha050(v, vwap)")


# In[420]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[421]:


sns.distplot(alphas[f'{alpha:03}']);


# In[422]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[423]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 051

# ```
# ts_delta(ts_lag(close, 10), 10).div(10).sub(ts_delta(close, 10).div(10)) < -0.05 * c
#         ? 1 
#         : -ts_delta(close, 1)
# ```

# In[424]:


def alpha051(c):
    """ts_delta(ts_lag(close, 10), 10).div(10).sub(ts_delta(close, 10).div(10)) < -0.05 * c
        ? 1 
        : -ts_delta(close, 1)"""
    cond = (ts_delta(ts_lag(c, 10), 10).div(10)
            .sub(ts_delta(c, 10).div(10)) >= -0.05 * c)
    return (-ts_delta(c, 1)
            .where(cond, 1)
            .stack('ticker')
            .swaplevel())


# In[425]:


alpha = 51


# In[426]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha051(c)")


# In[427]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[428]:


sns.distplot(alphas[f'{alpha:03}']);


# In[429]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[430]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 052

# ```
# (ts_lag(ts_min(low, 5), 5) - ts_min(low, 5)) * 
#         rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220) * 
#         ts_rank(volume, 5)
# ```

# In[431]:


def alpha052(l, v, r):
    """(ts_lag(ts_min(low, 5), 5) - ts_min(low, 5)) * 
        rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220) * 
        ts_rank(volume, 5)
    """
    return (ts_delta(ts_min(l, 5), 5)
            .mul(rank(ts_sum(r, 240)
                      .sub(ts_sum(r, 20))
                      .div(220)))
            .mul(ts_rank(v, 5))
            .stack('ticker')
            .swaplevel())


# In[432]:


alpha = 52


# In[433]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha052(l, v, r)")


# In[434]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[435]:


sns.distplot(alphas[f'{alpha:03}']);


# In[436]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[437]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 053

# ```
# ((ts_sum(high, 20) / 20) < high)
#             ? (-1 * ts_delta(high, 2))
#             : 0
# ```

# In[438]:


def alpha053(h, l, c):
    """-1 * ts_delta(1 - (high - close) / (close - low), 9)"""
    inner = (c.sub(l)).add(1e-6)
    return (ts_delta(h.sub(c)
                     .mul(-1).add(1)
                     .div(c.sub(l)
                          .add(1e-6)), 9)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[439]:


alpha = 53


# In[440]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha053(h, l, c)")


# In[441]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[442]:


sns.distplot(alphas[f'{alpha:03}']);


# In[ ]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[ ]:


# mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
# mi[alpha]


# ## Alpha 054

# ```
# -(low - close) * power(open, 5) / ((low - high) * power(close, 5))
# ```

# In[35]:


def alpha054(o, h, l, c):
    """-(low - close) * power(open, 5) / ((low - high) * power(close, 5))"""
    return (l.sub(c).mul(o.pow(5)).mul(-1)
            .div(l.sub(h).replace(0, -0.0001).mul(c ** 5))
            .stack('ticker')
            .swaplevel())


# In[36]:


alpha = 54


# In[37]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha054(o, h, l, c)")


# In[38]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[39]:


sns.distplot(alphas[f'{alpha:03}']);


# In[40]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[41]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# In[42]:


pd.Series(mi).tail()


# ## Alpha 055

# ```
# (-1 * ts_corr(rank(((close - ts_min(low, 12)) / 
#                             (ts_max(high, 12) - ts_min(low,12)))), 
#                     rank(volume), 6))
# ```

# In[43]:


def alpha055(h, l, c):
    """(-1 * ts_corr(rank(((close - ts_min(low, 12)) / 
                            (ts_max(high, 12) - ts_min(low,12)))), 
                    rank(volume), 6))"""

    return (ts_corr(rank(c.sub(ts_min(l, 12))
                         .div(ts_max(h, 12).sub(ts_min(l, 12))
                              .replace(0, 1e-6))),
                    rank(v), 6)
            .replace([-np.inf, np.inf], np.nan)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[44]:


alpha = 55


# In[45]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha055(h, l, c)")


# In[46]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[47]:


sns.distplot(alphas[f'{alpha:03}']);


# In[48]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[49]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 056

# ```
# -rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) * 
#         rank((returns * cap))
# ```

# In[50]:


def alpha056(r, cap):
    """-rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) * 
        rank((returns * cap))
    """
    pass


# ## Alpha 057

# ```
# rank(ts_corr(ts_lag(open - close, 1), close, 200)) + 
#         rank(open - close)
# ```

# In[51]:


def alpha057(c, vwap):
    """-(close - vwap) / ts_weighted_mean(rank(ts_argmax(close, 30)), 2)"""
    return (c.sub(vwap.add(1e-5))
            .div(ts_weighted_mean(rank(ts_argmax(c, 30)))).mul(-1)
            .stack('ticker')
            .swaplevel())


# In[52]:


alpha = 57


# In[53]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha057(c, vwap)")


# In[54]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[55]:


sns.distplot(alphas[f'{alpha:03}']);


# In[56]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[57]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 058

# ```
# (indneutralize(((ts_corr(ts_delta(close, 1), ts_delta(ts_lag(close, 1), 1), 250) *ts_delta(close, 1)) / close), IndClass.subindustry) / ts_sum(((ts_delta(close, 1) / ts_lag(close, 1))^2), 250))
# ```

# In[58]:


def alpha58(v, wvap, sector):
    """(-1 * ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(vwap, IndClass.sector), volume, 3), 7), 5))"""
    pass


# ## Alpha 059

# ```
# (indneutralize(((ts_corr(ts_delta(close, 1), ts_delta(ts_lag(close, 1), 1), 250) *ts_delta(close, 1)) / close), IndClass.subindustry) / ts_sum(((ts_delta(close, 1) / ts_lag(close, 1))^2), 250))
# ```

# In[59]:


def alpha59(v, wvap, industry):
    """-ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(vwap, IndClass.industry), volume, 4), 16), 8)"""
    pass


# ## Alpha 060

# ```
# -ts_max(rank(ts_corr(rank(volume), rank(vwap), 5)), 5)
# ```

# In[60]:


def alpha060(l, h, c, v):
    """-((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))"""
    return (scale(rank(c.mul(2).sub(l).sub(h)
                       .div(h.sub(l).replace(0, 1e-5))
                       .mul(v))).mul(2)
            .sub(scale(rank(ts_argmax(c, 10)))).mul(-1)
            .stack('ticker')
            .swaplevel())


# In[61]:


alpha = 60


# In[62]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha060(l, h, c, v)")


# In[63]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[64]:


sns.distplot(alphas[f'{alpha:03}']);


# In[65]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[66]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 061

# ```
# (rank((vwap - ts_min(vwap, 16.1219))) < rank(ts_corr(vwap, adv180, 17.9282)))
# ```

# In[67]:


def alpha061(v, vwap):
    """rank((vwap - ts_min(vwap, 16))) < rank(ts_corr(vwap, adv180, 17))"""

    return (rank(vwap.sub(ts_min(vwap, 16)))
            .lt(rank(ts_corr(vwap, ts_mean(v, 180), 18)))
            .astype(int)
            .stack('ticker')
            .swaplevel())


# In[68]:


alpha = 61


# In[69]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha061(v, vwap)")


# In[70]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[71]:


sns.distplot(alphas[f'{alpha:03}']);


# In[72]:


alphas.groupby(alphas[f'{alpha:03}']).ret_fwd.describe()


# In[73]:


g = sns.boxenplot(x=f'{alpha:03}', y='ret_fwd', data=alphas[alphas.ret_fwd.between(-.1, .1)]);


# ## Alpha 062

# ```
# ((rank(ts_corr(vwap, ts_sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) +rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
# ```

# In[74]:


def alpha062(o, h, l, vwap, adv20):
    """((rank(ts_corr(vwap, ts_sum(adv20, 22.4101), 9.91009)) < 
    rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)"""
    return (rank(ts_corr(vwap, ts_sum(adv20, 22), 9))
            .lt(rank(
                rank(o).mul(2))
                .lt(rank(h.add(l).div(2))
                    .add(rank(h))))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[75]:


alpha = 62


# In[76]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha062(o, h, l, vwap, adv20)")


# In[77]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[78]:


sns.distplot(alphas[f'{alpha:03}'], kde=False);


# In[79]:


alphas.groupby(alphas[f'{alpha:03}']).ret_fwd.describe()


# In[80]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[81]:


# mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
# mi[alpha]


# ## Alpha 063

# ```
# ((rank(ts_weighted_mean(ts_delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))- rank(ts_weighted_mean(ts_corr(((vwap * 0.318108) + (open * (1 - 0.318108))), ts_sum(adv180,37.2467), 13.557), 12.2883))) * -1)
# ```

# In[82]:


def alpha63(v, wvap, industry):
    """((rank(ts_weighted_mean(ts_delta(IndNeutralize(close, IndClass.industry), 2), 8)) - 
        rank(ts_weighted_mean(ts_corr(((vwap * 0.318108) + (open * (1 - 0.318108))), 
                                        ts_sum(adv180, 37), 13), 12))) * -1)
    """
    pass


# In[83]:


alpha = 63


# In[84]:


# %%time
# alphas[f'{alpha:03}'] = alpha48(o, c)


# In[85]:


# alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[86]:


# sns.distplot(alphas[f'{alpha:03}']);


# In[87]:


# g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);
# 


# In[88]:


# mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
# mi[alpha]


# ## Alpha 064

# ```
# -ts_max(rank(ts_corr(rank(volume), rank(vwap), 5)), 5)
# ```

# In[89]:


def alpha064(o, h, l, v, vwap):
    """((rank(ts_corr(ts_sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),ts_sum(adv120, 12.7054), 16.6208)) <
        rank(ts_delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))) * -1)"""
    w = 0.178404
    return (rank(ts_corr(ts_sum(o.mul(w).add(l.mul(1 - w)), 12),
                         ts_sum(ts_mean(v, 120), 12), 16))
            .lt(rank(ts_delta(h.add(l).div(2).mul(w)
                               .add(vwap.mul(1 - w)), 3)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[90]:


alpha = 64


# In[91]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha064(o, h, l, v, vwap)")


# In[92]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[93]:


alphas.groupby(alphas[f'{alpha:03}']).ret_fwd.describe()


# In[94]:


sns.distplot(alphas[f'{alpha:03}']);


# In[95]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# ## Alpha 065

# ```
# ((rank(ts_corr(((open * 0.00817205) + (vwap * (1 - 0.00817205))), 
#                         ts_sum(adv60,8.6911), 6.40374)) < 
#         rank((open - ts_min(open, 13.635)))) * -1)
# ```

# In[96]:


def alpha065(o, v, vwap):
    """((rank(ts_corr(((open * 0.00817205) + (vwap * (1 - 0.00817205))), 
                        ts_sum(adv60,8.6911), 6.40374)) < 
        rank((open - ts_min(open, 13.635)))) * -1)
    """
    w = 0.00817205
    return (rank(ts_corr(o.mul(w).add(vwap.mul(1 - w)),
                         ts_mean(ts_mean(v, 60), 9), 6))
            .lt(rank(o.sub(ts_min(o, 13))))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[97]:


alpha = 65


# In[98]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha065(o, v, vwap)")


# In[99]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[100]:


sns.distplot(alphas[f'{alpha:03}']);


# In[101]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# ## Alpha 066

# ```
# ((rank(ts_weighted_mean(ts_delta(vwap, 3.51013), 7.23052)) + 
#         ts_rank(ts_weighted_mean(((((low* 0.96633) + (low * 
#                                     (1 - 0.96633))) - vwap) / 
#                                     (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
# ```

# In[102]:


def alpha066(l, h, vwap):
    """((rank(ts_weighted_mean(ts_delta(vwap, 3.51013), 7.23052)) +
        ts_rank(ts_weighted_mean(((((low* 0.96633) + (low *
                                    (1 - 0.96633))) - vwap) /
                                    (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    """
    w = 0.96633
    return (rank(ts_weighted_mean(ts_delta(vwap, 4), 7))
            .add(ts_rank(ts_weighted_mean(l.mul(w).add(l.mul(1 - w))
                                           .sub(vwap)
                                           .div(o.sub(h.add(l).div(2)).add(1e-3)), 11), 7))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[103]:


alpha = 66


# In[104]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha066(l, h, vwap)")


# In[105]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[106]:


sns.distplot(alphas[f'{alpha:03}']);


# In[107]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[108]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 067

# ```
# (rank(ts_delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),IndClass.sector), 1.23438)) < 
#         rank(ts_corr(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150,9.18637), 14.6644)))
# ```

# In[109]:


def alpha067(h, v, sector, subindustry):
    """(power(rank((high - ts_min(high, 2.14593))),
        rank(ts_corr(IndNeutralize(vwap,IndClass.sector), 
                IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
    """
    pass


# In[110]:


alpha = 67


# In[111]:


# %%time
# alphas[f'{alpha:03}'] = alpha056(r, cap)


# In[112]:


# alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[113]:


# sns.distplot(alphas[f'{alpha:03}']);


# In[114]:


# g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);
# 


# In[115]:


# mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
# mi[alpha]


# ## Alpha 068

# ```
# ((ts_rank(ts_corr(rank(high), rank(adv15), 8.91644), 13.9333) <
#         rank(ts_delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
# ```

# In[116]:


def alpha068(h, c, v):
    """((ts_rank(ts_corr(rank(high), rank(adv15), 8.91644), 13.9333) <
        rank(ts_delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    """
    w = 0.518371
    return (ts_rank(ts_corr(rank(h), rank(ts_mean(v, 15)), 9), 14)
            .lt(rank(ts_delta(c.mul(w).add(l.mul(1 - w)), 1)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[117]:


alpha = 68


# In[118]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha068(h, c, v)")


# In[119]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[120]:


sns.distplot(alphas[f'{alpha:03}']);


# In[121]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# ## Alpha 069

# ```
# ((power(rank(ts_max(ts_delta(IndNeutralize(vwap, IndClass.industry), 2.72412),4.79344)),
#         Ts_Rank(ts_corr(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),9.0615))) * -1)
# ```

# In[122]:


def alpha069(c, vwap, industry):
    """((power(rank(ts_max(ts_delta(IndNeutralize(vwap, IndClass.industry), 2.72412),4.79344)),
    Ts_Rank(ts_corr(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),9.0615))) * -1)
    """
    pass


# ## Alpha 070

# ```
# ((power(rank(ts_delta(vwap, 1.29456)),
#         ts_rank(ts_corr(IndNeutralize(close,IndClass.industry), adv50, 17.8256), 17.9171))) * -1)
# ```

# In[123]:


def alpha076(c, v, vwap, industry):
    """((power(rank(ts_delta(vwap, 1.29456)),
        ts_rank(ts_corr(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171))) * -1)
    """
    pass


# In[124]:


alpha = 70


# ## Alpha 071

# ```
# -ts_max(rank(ts_corr(rank(volume), rank(vwap), 5)), 5)
# ```

# In[125]:


def alpha071(o, c, v, vwap):
    """max(ts_rank(ts_weighted_mean(ts_corr(ts_rank(close, 3.43976), ts_rank(adv180,12.0647), 18.0175), 4.20501), 15.6948), 
            ts_rank(ts_weighted_mean((rank(((low + open) - (vwap +vwap)))^2), 16.4662), 4.4388))"""

    s1 = (ts_rank(ts_weighted_mean(ts_corr(ts_rank(c, 3),
                                           ts_rank(ts_mean(v, 180), 12), 18), 4), 16))
    s2 = (ts_rank(ts_weighted_mean(rank(l.add(o).
                                        sub(vwap.mul(2)))
                                   .pow(2), 16), 4))
    return (s1.where(s1 > s2, s2)
            .stack('ticker')
            .swaplevel())


# In[126]:


alpha = 71


# In[127]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha071(o, c, v, vwap)")


# In[128]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[129]:


sns.distplot(alphas[f'{alpha:03}']);


# In[130]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[131]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 072

# ```
# (rank(ts_weighted_mean(ts_corr(((high + low) / 2), adv40, 8.93345), 10.1519)) /
#         rank(ts_weighted_mean(ts_corr(ts_rank(vwap, 3.72469), ts_rank(volume, 18.5188), 6.86671), 2.95011)))
# ```

# In[132]:


def alpha072(h, l, v, vwap):
    """(rank(ts_weighted_mean(ts_corr(((high + low) / 2), adv40, 8.93345), 10.1519)) /
        rank(ts_weighted_mean(ts_corr(ts_rank(vwap, 3.72469), ts_rank(volume, 18.5188), 6.86671), 2.95011)))
    """
    return (rank(ts_weighted_mean(ts_corr(h.add(l).div(2), ts_mean(v, 40), 9), 10))
            .div(rank(ts_weighted_mean(ts_corr(ts_rank(vwap, 3), ts_rank(v, 18), 6), 2)))
            .stack('ticker')
            .swaplevel())


# In[133]:


alpha = 72


# In[134]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha072(h, l, v, vwap)")


# In[135]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[136]:


sns.distplot(alphas[f'{alpha:03}']);


# In[137]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# ## Alpha 073

# ```
# (max(rank(ts_weighted_mean(ts_delta(vwap, 4.72775), 2.91864)),
#         ts_rank(ts_weighted_mean(((ts_delta(((open * 0.147155) + 
#             (low * (1 - 0.147155))), 2.03608) / 
#             ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
# ```

# In[138]:


def alpha073(l, vwap):
    """(max(rank(ts_weighted_mean(ts_delta(vwap, 4.72775), 2.91864)),
        ts_rank(ts_weighted_mean(((ts_delta(((open * 0.147155) + 
            (low * (1 - 0.147155))), 2.03608) / 
            ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
        """
    w = 0.147155
    s1 = rank(ts_weighted_mean(ts_delta(vwap, 5), 3))
    s2 = (ts_rank(ts_weighted_mean(ts_delta(o.mul(w).add(l.mul(1 - w)), 2)
                                   .div(o.mul(w).add(l.mul(1 - w)).mul(-1)), 3), 16))

    print(s2)
    return (s1.where(s1 > s2, s2)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[139]:


alpha = 73


# In[140]:


# %%time
alphas[f'{alpha:03}'] = alpha073(l, vwap)


# In[141]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[142]:


sns.distplot(alphas[f'{alpha:03}']);


# In[143]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[144]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 074

# ```
# ((rank(ts_corr(close, ts_sum(adv30, 37.4843), 15.1365)) <
#         rank(ts_corr(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)
# ```

# In[145]:


def alpha074(v, vwap):
    """((rank(ts_corr(close, ts_sum(adv30, 37.4843), 15.1365)) <
        rank(ts_corr(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)"""

    w = 0.0261661
    return (rank(ts_corr(c, ts_mean(ts_mean(v, 30), 37), 15))
            .lt(rank(ts_corr(rank(h.mul(w).add(vwap.mul(1 - w))), rank(v), 11)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[146]:


alpha = 74


# In[147]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha074(v, vwap)")


# In[148]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[149]:


sns.distplot(alphas[f'{alpha:03}']);


# In[150]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[151]:


alphas.groupby(alphas[f'{alpha:03}']).ret_fwd.describe()


# ## Alpha 075

# ```
# (rank(ts_corr(vwap, volume, 4.24304)) < 
#         rank(ts_corr(rank(low), rank(adv50),12.4413)))
# ```

# In[152]:


def alpha075(l, v, vwap):
    """(rank(ts_corr(vwap, volume, 4.24304)) < 
        rank(ts_corr(rank(low), rank(adv50),12.4413)))
    """

    return (rank(ts_corr(vwap, v, 4))
            .lt(rank(ts_corr(rank(l), rank(ts_mean(v, 50)), 12)))
            .astype(int)
            .stack('ticker')
            .swaplevel())


# In[153]:


alpha = 75


# In[154]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha075(l, v, vwap)")


# In[155]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[156]:


sns.distplot(alphas[f'{alpha:03}']);


# In[157]:


g = sns.boxenplot(x=f'{alpha:03}', y='ret_fwd', data=alphas[alphas.ret_fwd.between(-.025, .025)]);


# In[158]:


alphas.groupby(alphas[f'{alpha:03}']).ret_fwd.describe()


# ## Alpha 076

# ```
# (rank(ts_delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),IndClass.sector), 1.23438)) < 
#         rank(ts_corr(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150,9.18637), 14.6644)))
# ```

# In[159]:


def alpha076(l, vwap, sector):
    """(max(rank(ts_weighted_mean(ts_delta(vwap, 1.24383), 11.8259)),
            ts_rank(ts_weighted_mean(ts_rank(ts_corr(IndNeutralize(low, IndClass.sector), adv81,8.14941), 19.569), 17.1543), 19.383)) * -1)
    """
    pass


# In[160]:


alpha = 76


# ## Alpha 077

# ```
# min(rank(ts_weighted_mean(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),
#             rank(ts_weighted_mean(ts_corr(((high + low) / 2), adv40, 3.1614), 5.64125)))
# ```

# In[161]:


def alpha077(l, h, vwap):
    """min(rank(ts_weighted_mean(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),
            rank(ts_weighted_mean(ts_corr(((high + low) / 2), adv40, 3.1614), 5.64125)))
    """

    s1 = rank(ts_weighted_mean(h.add(l).div(2).sub(vwap), 20))
    s2 = rank(ts_weighted_mean(ts_corr(h.add(l).div(2), ts_mean(v, 40), 3), 5))
    return (s1.where(s1 < s2, s2)
            .stack('ticker')
            .swaplevel())


# In[162]:


alpha = 77


# In[163]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha077(l, h, vwap)")


# In[164]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[165]:


sns.distplot(alphas[f'{alpha:03}']);


# In[166]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[167]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 078

# ```
# (rank(ts_corr(ts_sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),
#         ts_sum(adv40, 19.7428), 6.83313))^rank(ts_corr(rank(vwap), rank(volume), 5.77492)))
# ```

# In[168]:


def alpha078(l, v, vwap):
    """(rank(ts_corr(ts_sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),
        ts_sum(adv40, 19.7428), 6.83313))^rank(ts_corr(rank(vwap), rank(volume), 5.77492)))"""

    w = 0.352233
    return (rank(ts_corr(ts_sum((l.mul(w).add(vwap.mul(1 - w))), 19),
                         ts_sum(ts_mean(v, 40), 19), 6))
            .pow(rank(ts_corr(rank(vwap), rank(v), 5)))
            .stack('ticker')
            .swaplevel())


# In[169]:


alpha = 78


# In[170]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha078(l, v, vwap)")


# In[171]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[172]:


sns.distplot(alphas[f'{alpha:03}']);


# In[173]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[174]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 079

# ```
# (rank(ts_delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),IndClass.sector), 1.23438)) < 
#         rank(ts_corr(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150,9.18637), 14.6644)))
# ```

# In[175]:


def alpha079(o, v, sector):
    """(rank(ts_delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),IndClass.sector), 1.23438)) < 
        rank(ts_corr(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150,9.18637), 14.6644)))
    """
    pass


# ## Alpha 080

# ```
# ((power(rank(sign(ts_delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),IndClass.industry), 4.04545))),
#         ts_rank(ts_corr(high, adv10, 5.11456), 5.53756)) * -1)
# ```

# In[176]:


def alpha080(h, industry):
    """((power(rank(sign(ts_delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),IndClass.industry), 4.04545))),
        ts_rank(ts_corr(high, adv10, 5.11456), 5.53756)) * -1)
    """
    pass


# ## Alpha 081

# ```
# -(rank(log(ts_product(rank((rank(ts_corr(vwap, ts_sum(adv10, 49.6054),8.47743))^4)), 14.9655))) <
#         rank(ts_corr(rank(vwap), rank(volume), 5.07914)))
# ```

# In[177]:


def alpha081(v, vwap):
    """-(rank(log(ts_product(rank((rank(ts_corr(vwap, ts_sum(adv10, 49.6054),8.47743))^4)), 14.9655))) <
        rank(ts_corr(rank(vwap), rank(volume), 5.07914)))"""

    return (rank(log(ts_product(rank(rank(ts_corr(vwap,
                                                  ts_sum(ts_mean(v, 10), 50), 8))
                                     .pow(4)), 15)))
            .lt(rank(ts_corr(rank(vwap), rank(v), 5)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[178]:


alpha = 81


# In[179]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha081(v, vwap)")


# In[180]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[181]:


sns.distplot(alphas[f'{alpha:03}']);


# In[182]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[183]:


# mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
# mi[alpha]


# ## Alpha 082

# ```
# -rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) * 
#         rank((returns * cap))
# ```

# In[184]:


def alpha082(o, v, sector):
    """(min(rank(ts_weighted_mean(ts_delta(open, 1.46063), 14.8717)),
        ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(volume, IndClass.sector), 
        ((open * 0.634196) +(open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
    
    """
    pass


# ## Alpha 083

# ```
# (rank(ts_lag((high - low) / ts_mean(close, 5), 2)) * rank(rank(volume)) / 
#     (((high - low) / ts_mean(close, 5) / (vwap - close)))
# ```

# In[185]:


def alpha083(h, l, c):
    """(rank(ts_lag((high - low) / ts_mean(close, 5), 2)) * rank(rank(volume)) / 
            (((high - low) / ts_mean(close, 5) / (vwap - close)))
    """
    s = h.sub(l).div(ts_mean(c, 5))

    return (rank(rank(ts_lag(s, 2))
                 .mul(rank(rank(v)))
                 .div(s).div(vwap.sub(c).add(1e-3)))
            .stack('ticker')
            .swaplevel()
            .replace((np.inf, -np.inf), np.nan))


# In[186]:


alpha = 83


# In[187]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha083(h, l, c)")


# In[188]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[189]:


sns.distplot(alphas[f'{alpha:03}']);


# In[190]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[191]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 084

# ```
# power(ts_rank((vwap - ts_max(vwap, 15.3217)), 20.7127), 
#             ts_delta(close,4.96796))
# ```

# In[192]:


def alpha084(c, vwap):
    """power(ts_rank((vwap - ts_max(vwap, 15.3217)), 20.7127), 
        ts_delta(close,4.96796))"""
    return (rank(power(ts_rank(vwap.sub(ts_max(vwap, 15)), 20),
                       ts_delta(c, 6)))
            .stack('ticker')
            .swaplevel())


# In[193]:


alpha = 84


# In[194]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha084(c, vwap)")


# In[195]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[196]:


sns.distplot(alphas[f'{alpha:03}']);


# In[197]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[198]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 085

# ```
# power(rank(ts_corr(((high * 0.876703) + (close * (1 - 0.876703))), adv30,9.61331)),
#         rank(ts_corr(ts_rank(((high + low) / 2), 3.70596), 
#                      ts_rank(volume, 10.1595),7.11408)))
# ```

# In[199]:


def alpha085(l, v):
    """power(rank(ts_corr(((high * 0.876703) + (close * (1 - 0.876703))), adv30,9.61331)),
        rank(ts_corr(ts_rank(((high + low) / 2), 3.70596), 
                     ts_rank(volume, 10.1595),7.11408)))
                     """
    w = 0.876703
    return (rank(ts_corr(h.mul(w).add(c.mul(1 - w)), ts_mean(v, 30), 10))
            .pow(rank(ts_corr(ts_rank(h.add(l).div(2), 4),
                              ts_rank(v, 10), 7)))
            .stack('ticker')
            .swaplevel())


# In[200]:


alpha = 85


# In[201]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha085(l, v)")


# In[202]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[203]:


sns.distplot(alphas[f'{alpha:03}']);


# In[204]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[205]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 086

# ```
# ((ts_rank(ts_corr(close, ts_sum(adv20, 14.7444), 6.00049), 20.4195) < 
#         rank(((open + close) - (vwap + open)))) * -1)
# ```

# In[206]:


def alpha086(c, v, vwap):
    """((ts_rank(ts_corr(close, ts_sum(adv20, 14.7444), 6.00049), 20.4195) < 
        rank(((open + close) - (vwap + open)))) * -1)
    """
    return (ts_rank(ts_corr(c, ts_mean(ts_mean(v, 20), 15), 6), 20)
            .lt(rank(c.sub(vwap)))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[207]:


alpha = 86


# In[208]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha086(c, v, vwap)")


# In[209]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[210]:


sns.distplot(alphas[f'{alpha:03}']);


# In[211]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[212]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 087

# ```
# (max(rank(ts_weighted_mean(ts_delta(((close * 0.369701) + (vwap * (1 - 0.369701))),1.91233), 2.65461)), 
#             ts_rank(ts_weighted_mean(abs(ts_corr(IndNeutralize(adv81,IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
#             ```

# In[213]:


def alpha087(c, vwap, industry):
    """(max(rank(ts_weighted_mean(ts_delta(((close * 0.369701) + (vwap * (1 - 0.369701))),1.91233), 2.65461)), 
            ts_rank(ts_weighted_mean(abs(ts_corr(IndNeutralize(adv81,IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
    """
    pass


# ## Alpha 088

# ```
# min(rank(ts_weighted_mean(((rank(open) + rank(low)) - (rank(high) + rank(close))),8.06882)), 
#         ts_rank(ts_weighted_mean(ts_corr(ts_rank(close, 8.44728), 
#                 ts_rank(adv60,20.6966), 8.01266), 6.65053), 2.61957))
# ```

# In[214]:


def alpha088(o, h, l, c, v):
    """min(rank(ts_weighted_mean(((rank(open) + rank(low)) - (rank(high) + rank(close))),8.06882)), 
        ts_rank(ts_weighted_mean(ts_corr(ts_rank(close, 8.44728), 
                ts_rank(adv60,20.6966), 8.01266), 6.65053), 2.61957))"""

    s1 = (rank(ts_weighted_mean(rank(o)
                                .add(rank(l))
                                .sub(rank(h))
                                .add(rank(c)), 8)))
    s2 = ts_rank(ts_weighted_mean(ts_corr(ts_rank(c, 8),
                                          ts_rank(ts_mean(v, 60), 20), 8), 6), 2)

    return (s1.where(s1 < s2, s2)
            .stack('ticker')
            .swaplevel())


# In[215]:


alpha = 88


# In[216]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha088(o, h, l, c, v)")


# In[217]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[218]:


sns.distplot(alphas[f'{alpha:03}']);


# In[219]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[220]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'], n=30000)
mi[alpha]


# ## Alpha 089

# ```
# -rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) * 
#         rank((returns * cap))
# ```

# In[221]:


def alpha089(l, v, vwap, industry):
    """(ts_rank(ts_weighted_mean(ts_corr(((low * 0.967285) + 
        (low * (1 - 0.967285))), adv10,6.94279), 5.51607), 3.79744) - 
        ts_rank(ts_weighted_mean(ts_delta(IndNeutralize(vwap,IndClass.industry), 3.48158), 10.1466), 15.3012))
    """
    pass


# ## Alpha 090

# ```
# -rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) * 
#         rank((returns * cap))
# ```

# In[222]:


def alpha090(c, l, subindustry):
    """((rank((close - ts_max(close, 4.66719)))
        ^ts_rank(ts_corr(IndNeutralize(adv40,IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
    """
    pass


# ## Alpha 091

# ```
# ((ts_rank(ts_weighted_mean(ts_weighted_mean(ts_corr(IndNeutralize(close,IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) -
#         rank(ts_weighted_mean(ts_corr(vwap, adv30, 4.01303), 2.6809))) * -1)
# ```

# In[223]:


def alpha091(v, vwap, industry):
    """((ts_rank(ts_weighted_mean(ts_weighted_mean(ts_corr(IndNeutralize(close,IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) -
        rank(ts_weighted_mean(ts_corr(vwap, adv30, 4.01303), 2.6809))) * -1)
    """
    pass


# ## Alpha 092

# ```
# min(ts_rank(ts_weighted_mean(((((high + low) / 2) + close) < (low + open)), 14.7221),18.8683), 
#             ts_rank(ts_weighted_mean(ts_corr(rank(low), rank(adv30), 7.58555), 6.94024),6.80584))
# ```

# In[224]:


def alpha092(o, l, c, v):
    """min(ts_rank(ts_weighted_mean(((((high + low) / 2) + close) < (low + open)), 14.7221),18.8683), 
            ts_rank(ts_weighted_mean(ts_corr(rank(low), rank(adv30), 7.58555), 6.94024),6.80584))
    """
    p1 = ts_rank(ts_weighted_mean(h.add(l).div(2).add(c).lt(l.add(o)), 15), 18)
    p2 = ts_rank(ts_weighted_mean(ts_corr(rank(l), rank(ts_mean(v, 30)), 7), 6), 6)

    return (p1.where(p1<p2, p2)
            .stack('ticker')
            .swaplevel())


# In[225]:


alpha = 92


# In[226]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha092(o, l, c, v)")


# In[227]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[228]:


sns.distplot(alphas[f'{alpha:03}']);


# In[229]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[230]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 093

# ```
# (ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(vwap, IndClass.industry), adv81,17.4193), 19.848), 7.54455) / 
#         rank(ts_weighted_mean(ts_delta(((close * 0.524434) + (vwap * (1 -0.524434))), 2.77377), 16.2664)))
# ```

# In[231]:


def alpha093(c, v, vwap, industry):
    """(ts_rank(ts_weighted_mean(ts_corr(IndNeutralize(vwap, IndClass.industry), adv81,17.4193), 19.848), 7.54455) / 
        rank(ts_weighted_mean(ts_delta(((close * 0.524434) + (vwap * (1 -0.524434))), 2.77377), 16.2664)))
    """
    pass


# ## Alpha 094

# ```
# ((rank((vwap - ts_min(vwap, 11.5783)))^ts_rank(ts_corr(ts_rank(vwap,19.6462), 
#         ts_rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
# ```

# In[232]:


def alpha094(v, vwap):
    """((rank((vwap - ts_min(vwap, 11.5783)))^ts_rank(ts_corr(ts_rank(vwap,19.6462), 
        ts_rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    """

    return (rank(vwap.sub(ts_min(vwap, 11)))
            .pow(ts_rank(ts_corr(ts_rank(vwap, 20),
                                 ts_rank(ts_mean(v, 60), 4), 18), 2))
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[233]:


alpha = 94


# In[234]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha094(v, vwap)")


# In[235]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[236]:


sns.distplot(alphas[f'{alpha:03}']);


# In[237]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[238]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 095

# ```
# (rank((open - ts_min(open, 12.4105))) < 
#     ts_rank((rank(ts_corr(ts_sum(((high + low)/ 2), 19.1351), 
#     ts_sum(adv40, 19.1351), 12.8742))^5), 11.7584))
# ```

# In[239]:


def alpha095(o, l, v):
    """(rank((open - ts_min(open, 12.4105))) < 
        ts_rank((rank(ts_corr(ts_sum(((high + low)/ 2), 19.1351), ts_sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    """
    
    return (rank(o.sub(ts_min(o, 12)))
            .lt(ts_rank(rank(ts_corr(ts_mean(h.add(l).div(2), 19),
                                     ts_sum(ts_mean(v, 40), 19), 13).pow(5)), 12))
            .astype(int)
            .stack('ticker')
            .swaplevel())


# In[240]:


alpha = 95


# In[241]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha095(o, l, v)")


# In[242]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[243]:


sns.distplot(alphas[f'{alpha:03}'], kde=False);


# In[244]:


g = sns.boxenplot(x=f'{alpha:03}', y='ret_fwd', data=alphas[alphas.ret_fwd.between(-.025, .025)]);


# In[245]:


alphas.groupby(alphas[f'{alpha:03}']).ret_fwd.describe()


# ## Alpha 096

# ```
# (max(ts_rank(ts_weighted_mean(ts_corr(rank(vwap), rank(volume), 5.83878),4.16783), 8.38151), 
#         ts_rank(ts_weighted_mean(ts_argmax(ts_corr(ts_rank(close, 7.45404), ts_rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
# ```

# In[246]:


def alpha096(c, v, vwap):
    """(max(ts_rank(ts_weighted_mean(ts_corr(rank(vwap), rank(volume), 5.83878),4.16783), 8.38151), 
        ts_rank(ts_weighted_mean(ts_argmax(ts_corr(ts_rank(close, 7.45404), ts_rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)"""
    
    s1 = ts_rank(ts_weighted_mean(ts_corr(rank(vwap), rank(v), 10), 4), 8)
    s2 = ts_rank(ts_weighted_mean(ts_argmax(ts_corr(ts_rank(c, 7),
                                                    ts_rank(ts_mean(v, 60), 10), 10), 12), 14), 13)
    return (s1.where(s1 > s2, s2)
            .mul(-1)
            .stack('ticker')
            .swaplevel())


# In[247]:


alpha = 96


# In[248]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha096(c, v, vwap)")


# In[249]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[250]:


sns.distplot(alphas[f'{alpha:03}']);


# In[251]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas)


# ## Alpha 097

# ```
# -rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) * 
#         rank((returns * cap))
# ```

# In[253]:


def alpha097(l):
    """((rank(ts_weighted_mean(ts_delta(IndNeutralize(((low * 0.721001) + 
        (vwap * (1 - 0.721001))),IndClass.industry), 3.3705), 20.4523)) - 
        ts_rank(ts_weighted_mean(ts_rank(ts_corr(Ts_Rank(low,7.87871), 
        ts_rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
    """
    pass


# ## Alpha 098

# ```
# (rank(ts_weighted_mean(ts_corr(vwap, ts_sum(adv5, 26.4719), 4.58418), 7.18088)) -
#         rank(ts_weighted_mean(ts_tank(ts_argmin(ts_corr(rank(open), 
#         rank(adv15), 20.8187), 8.62571),6.95668), 8.07206)))
# ```

# In[254]:


def alpha098(o, v, vwap):
    """(rank(ts_weighted_mean(ts_corr(vwap, ts_sum(adv5, 26.4719), 4.58418), 7.18088)) -
        rank(ts_weighted_mean(ts_tank(ts_argmin(ts_corr(rank(open), 
        rank(adv15), 20.8187), 8.62571),6.95668), 8.07206)))
    """
    adv5 = ts_mean(v, 5)
    adv15 = ts_mean(v, 15)
    return (rank(ts_weighted_mean(ts_corr(vwap, ts_mean(adv5, 26), 4), 7))
            .sub(rank(ts_weighted_mean(ts_rank(ts_argmin(ts_corr(rank(o),
                                                                 rank(adv15), 20), 8), 6))))
            .stack('ticker')
            .swaplevel())


# In[255]:


alpha = 98


# In[256]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha098(o, v, vwap)")


# In[257]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[258]:


sns.distplot(alphas[f'{alpha:03}']);


# In[259]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[260]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Alpha 099

# ```
# ((rank(ts_corr(ts_sum(((high + low) / 2), 19.8975), 
#                     ts_sum(adv60, 19.8975), 8.8136)) <
#                     rank(ts_corr(low, volume, 6.28259))) * -1)
# ```

# In[261]:


def alpha099(l, v):
    """((rank(ts_corr(ts_sum(((high + low) / 2), 19.8975), 
                    ts_sum(adv60, 19.8975), 8.8136)) <
                    rank(ts_corr(low, volume, 6.28259))) * -1)"""

    return ((rank(ts_corr(ts_sum((h.add(l).div(2)), 19),
                          ts_sum(ts_mean(v, 60), 19), 8))
             .lt(rank(ts_corr(l, v, 6)))
             .mul(-1))
            .stack('ticker')
            .swaplevel())


# In[262]:


alpha = 99


# In[263]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha099(l, v)")


# In[264]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[265]:


sns.distplot(alphas[f'{alpha:03}']);


# In[266]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[267]:


alphas.groupby(alphas[f'{alpha:03}']).ret_fwd.describe()


# ## Alpha 100

# ```
# -rank(ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3)) * 
#         rank((returns * cap))
# ```

# In[268]:


def alpha100(r, cap):
    """(0 - (1 * (((1.5 * scale(indneutralize(
                indneutralize(rank(((((close - low) - (high -close)) / (high - low)) * volume)), 
                                IndClass.subindustry), IndClass.subindustry))) - 
    scale(indneutralize((ts_corr(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))
    """
    pass


# ## Alpha 101

# ```
# -ts_max(rank(ts_corr(rank(volume), rank(vwap), 5)), 5)
# ```

# In[269]:


def alpha101(o, h, l, c):
    """((close - open) / ((high - low) + .001))"""
    return (c.sub(o).div(h.sub(l).add(1e-3))
            .stack('ticker')
            .swaplevel())


# In[270]:


alpha = 101


# In[271]:


get_ipython().run_cell_magic('time', '', "alphas[f'{alpha:03}'] = alpha101(o, h, l, c)")


# In[272]:


alphas[f'{alpha:03}'].to_hdf('alphas.h5', f'alphas/{alpha:03}')


# In[273]:


sns.distplot(alphas[f'{alpha:03}']);


# In[274]:


g = sns.jointplot(x=f'{alpha:03}', y='ret_fwd', data=alphas);


# In[275]:


mi[alpha] = get_mutual_info_score(alphas.ret_fwd, alphas[f'{alpha:03}'])
mi[alpha]


# ## Store results

# In[276]:


alphas = []
with pd.HDFStore('alphas.h5') as store:
    keys = [k[1:] for k in store.keys()]
    for key in keys:
        i = int(key.split('/')[-1])
        alphas.append(store[key].to_frame(i))
alphas = pd.concat(alphas, axis=1)


# In[277]:


alphas.info(null_counts=True)


# In[278]:


alphas.to_hdf('data.h5', 'factors/formulaic')

