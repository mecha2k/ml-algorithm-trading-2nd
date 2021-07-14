#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression with Macro Data

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


sns.set_style('whitegrid')


# ## Data Set

# | Variable   | Description                                  | Transformation     |
# |------------|----------------------------------------------|--------------------|
# | realgdp    | Real gross domestic product                  | Annual Growth Rate |
# | realcons   | Real personal consumption expenditures       | Annual Growth Rate |
# | realinv    | Real gross private domestic investment       | Annual Growth Rate |
# | realgovt   | Real federal expenditures & gross investment | Annual Growth Rate |
# | realdpi    | Real private disposable income               | Annual Growth Rate |
# | m1         | M1 nominal money stock                       | Annual Growth Rate |
# | tbilrate   | Monthly treasury bill rate                 | Level              |
# | unemp      | Seasonally adjusted unemployment rate (%)    | Level              |
# | infl       | Inflation rate                               | Level              |
# | realint    |  Real interest rate                          | Level              |

# In[3]:


data = pd.DataFrame(sm.datasets.macrodata.load().data)
data.info()


# In[4]:


data.head()


# ## Data Prep

# To obtain a binary target variable, we compute the 20-quarter rolling average of the annual growth rate of quarterly real GDP. We then assign 1 if current growth exceeds the moving average and 0 otherwise. Finally, we shift the indicator variables to align next quarter's outcome with the current quarter.

# In[5]:


data['growth_rate'] = data.realgdp.pct_change(4)
data['target'] = (data.growth_rate > data.growth_rate.rolling(20).mean()).astype(int).shift(-1)
data.quarter = data.quarter.astype(int)


# In[6]:


data.target.value_counts()


# In[7]:


data.tail()


# In[8]:


pct_cols = ['realcons', 'realinv', 'realgovt', 'realdpi', 'm1']
drop_cols = ['year', 'realgdp', 'pop', 'cpi', 'growth_rate']
data.loc[:, pct_cols] = data.loc[:, pct_cols].pct_change(4)


# In[9]:


data = pd.get_dummies(data.drop(drop_cols, axis=1), columns=['quarter'], drop_first=True).dropna()


# In[10]:


data.head()


# In[11]:


data.info()


# We use an intercept and convert the quarter values to dummy variables and train the logistic regression model as follows:

# This produces the following summary for our model with 198 observations and 13 variables, including intercept:
# The summary indicates that the model has been trained using maximum likelihood and provides the maximized value of the log-likelihood function at -67.9.

# In[12]:


model = sm.Logit(data.target, sm.add_constant(data.drop('target', axis=1)))
result = model.fit()
result.summary()


# The LL-Null value of -136.42 is the result of the maximized log-likelihood function when only an intercept is included. It forms the basis for the pseudo-R2 statistic and the Log-Likelihood Ratio (LLR) test. 
# The pseudo-R2 statistic is a substitute for the familiar R2 available under least squares. It is computed based on the ratio of the maximized log-likelihood function for the null model m0 and the full model m1 as follows:
# The values vary from 0 (when the model does not improve the likelihood) to 1 where the model fits perfectly and the log-likelihood is maximized at 0. Consequently, higher values indicate a better fit.
# 

# In[13]:


plt.rc('figure', figsize=(12, 7))
plt.text(0.01, 0.05, str(result.summary()), {'fontsize': 14}, fontproperties = 'monospace')
plt.axis('off')
plt.tight_layout()
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.1)
plt.savefig('logistic_example.png', bbox_inches='tight', dpi=300);

