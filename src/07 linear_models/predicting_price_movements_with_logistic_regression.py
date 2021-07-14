#!/usr/bin/env python
# coding: utf-8

# # Predicting stock price moves with Logistic Regression

# ## Imports & Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from pathlib import Path
import sys, os
from time import time

import pandas as pd
import numpy as np

from scipy.stats import spearmanr

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import MultipleTimeSeriesCV


# In[4]:


sns.set_style('darkgrid')
idx = pd.IndexSlice


# In[5]:


YEAR = 252


# ## Load Data

# In[6]:


with pd.HDFStore('data.h5') as store:
    data = (store['model_data']
            .dropna()
            .drop(['open', 'close', 'low', 'high'], axis=1))
data = data.drop([c for c in data.columns if 'year' in c or 'lag' in c], axis=1)


# ### Select Investment Universe

# In[7]:


data = data[data.dollar_vol_rank<100]


# ### Create Model Data

# In[8]:


y = data.filter(like='target')
X = data.drop(y.columns, axis=1)
X = X.drop(['dollar_vol', 'dollar_vol_rank', 'volume', 'consumer_durables'], axis=1)


# ## Logistic Regression

# ### Define cross-validation parameters

# In[9]:


train_period_length = 63
test_period_length = 10
lookahead =1
n_splits = int(3 * YEAR/test_period_length)

cv = MultipleTimeSeriesCV(n_splits=n_splits,
                          test_period_length=test_period_length,
                          lookahead=lookahead,
                          train_period_length=train_period_length)


# In[10]:


target = f'target_{lookahead}d'


# In[11]:


y.loc[:, 'label'] = (y[target] > 0).astype(int)
y.label.value_counts()


# In[12]:


Cs = np.logspace(-5, 5, 11)


# In[13]:


cols = ['C', 'date', 'auc', 'ic', 'pval']


# ### Run cross-validation

# In[14]:


get_ipython().run_cell_magic('time', '', "log_coeffs, log_scores, log_predictions = {}, [], []\nfor C in Cs:\n    print(C)\n    model = LogisticRegression(C=C,\n                               fit_intercept=True,\n                               random_state=42,\n                               n_jobs=-1)\n\n    pipe = Pipeline([\n        ('scaler', StandardScaler()),\n        ('model', model)])\n    ics = aucs = 0\n    start = time()\n    coeffs = []\n    for i, (train_idx, test_idx) in enumerate(cv.split(X), 1):\n        X_train, y_train, = X.iloc[train_idx], y.label.iloc[train_idx]\n        pipe.fit(X=X_train, y=y_train)\n        X_test, y_test = X.iloc[test_idx], y.label.iloc[test_idx]\n        actuals = y[target].iloc[test_idx]\n        if len(y_test) < 10 or len(np.unique(y_test)) < 2:\n            continue\n        y_score = pipe.predict_proba(X_test)[:, 1]\n       \n        auc = roc_auc_score(y_score=y_score, y_true=y_test)\n        actuals = y[target].iloc[test_idx]\n        ic, pval = spearmanr(y_score, actuals)\n\n        log_predictions.append(y_test.to_frame('labels').assign(\n            predicted=y_score, C=C, actuals=actuals))\n        date = y_test.index.get_level_values('date').min()\n        log_scores.append([C, date, auc, ic * 100, pval])\n        coeffs.append(pipe.named_steps['model'].coef_)\n        ics += ic\n        aucs += auc\n        if i % 10 == 0:\n            print(f'\\t{time()-start:5.1f} | {i:03} | {ics/i:>7.2%} | {aucs/i:>7.2%}')\n\n    log_coeffs[C] = np.mean(coeffs, axis=0).squeeze()")


# ### Evaluate Results

# In[15]:


log_scores = pd.DataFrame(log_scores, columns=cols)
log_scores.to_hdf('data.h5', 'logistic/scores')

log_coeffs = pd.DataFrame(log_coeffs, index=X.columns).T
log_coeffs.to_hdf('data.h5', 'logistic/coeffs')

log_predictions = pd.concat(log_predictions)
log_predictions.to_hdf('data.h5', 'logistic/predictions')


# In[16]:


log_scores = pd.read_hdf('data.h5', 'logistic/scores')


# In[17]:


log_scores.info()


# In[18]:


log_scores.groupby('C').auc.describe()


# ### Plot Validation Scores

# In[19]:


def plot_ic_distribution(df, ax=None):
    if ax is not None:
        sns.distplot(df.ic, ax=ax)    
    else:
        ax = sns.distplot(df.ic)
    mean, median = df.ic.mean(), df.ic.median()
    ax.axvline(0, lw=1, ls='--', c='k')
    ax.text(x=.05, y=.9, s=f'Mean: {mean:8.2f}\nMedian: {median:5.2f}',
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlabel('Information Coefficient')
    sns.despine()
    plt.tight_layout()


# In[21]:


fig, axes= plt.subplots(ncols=2, figsize=(15, 5))

sns.lineplot(x='C', y='auc', data=log_scores, estimator=np.mean, label='Mean', ax=axes[0])
by_alpha = log_scores.groupby('C').auc.agg(['mean', 'median'])
best_auc = by_alpha['mean'].idxmax()
by_alpha['median'].plot(logx=True, ax=axes[0], label='Median', xlim=(10e-6, 10e5))
axes[0].axvline(best_auc, ls='--', c='k', lw=1, label='Max. Mean')
axes[0].axvline(by_alpha['median'].idxmax(), ls='-.', c='k', lw=1, label='Max. Median')
axes[0].legend()
axes[0].set_ylabel('AUC')
axes[0].set_xscale('log')
axes[0].set_title('Area Under the Curve')

plot_ic_distribution(log_scores[log_scores.C==best_auc], ax=axes[1])
axes[1].set_title('Information Coefficient')

fig.suptitle('Logistic Regression', fontsize=14)
sns.despine()
fig.tight_layout()
fig.subplots_adjust(top=.9);

