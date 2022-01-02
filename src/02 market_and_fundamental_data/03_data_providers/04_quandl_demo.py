#!/usr/bin/env python
# coding: utf-8

# # Quandl - API Demo

# Quandl uses a very straightforward API to make its free and premium data available. Currently, 50 anonymous calls are allowed, then a (free) API key is required. See [documentation](https://www.quandl.com/tools/api) for more details.

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import quandl

import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


sns.set_style('whitegrid')


# In[6]:


api_key = os.environ['QUANDL_API_KEY']
oil = quandl.get('EIA/PET_RWTC_D', api_key=api_key).squeeze()


# In[7]:


oil.plot(lw=2, title='WTI Crude Oil Price', figsize=(12, 4))
sns.despine()
plt.tight_layout();

