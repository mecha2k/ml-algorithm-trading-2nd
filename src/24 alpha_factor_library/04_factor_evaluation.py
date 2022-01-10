#!/usr/bin/env python
# coding: utf-8

# 
# # Alpha Factor Evaluation

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[43]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os, sys
from time import time

from pathlib import Path
import numpy as np
import pandas as pd
import pandas_datareader.data as web

import statsmodels.api as sm
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import scale
import lightgbm as lgb
from scipy.stats import spearmanr
from tqdm import tqdm
import shap

import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import MultipleTimeSeriesCV


# In[4]:


sns.set_style('whitegrid')
idx = pd.IndexSlice
deciles = np.arange(.1, 1, .1).round(1)


# In[19]:


results_path = Path('results')
if not results_path.exists():
    results_path.mkdir()


# ## Load Data

# In[5]:


factors = (pd.concat([pd.read_hdf('data.h5', 'factors/common'),
                      pd.read_hdf('data.h5', 'factors/formulaic')
                      .rename(columns=lambda x: f'alpha_{int(x):03}')],
                     axis=1)
           .dropna(axis=1, thresh=100000)
           .sort_index())


# In[6]:


factors.info()


# In[7]:


fwd_returns = factors.filter(like='fwd').columns
features = factors.columns.difference(fwd_returns).tolist()
alphas = pd.Index([f for f in features if f.startswith('alpha')])


# In[8]:


features


# In[9]:


len(alphas)


# ## Factor Correlation

# ### 'Classic' Factors

# In[10]:


corr_common = factors.drop(fwd_returns.union(alphas), axis=1).corr(method='spearman')


# In[11]:


corr_common.to_hdf('data.h5', 'correlation/common')


# In[20]:


mask = np.triu(np.ones_like(corr_common, dtype=np.bool))
fig, ax = plt.subplots(figsize=(22, 18))
cmap = sns.diverging_palette(10, 220, as_cmap=True)

sns.heatmap(corr_common, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
fig.tight_layout()
fig.savefig(results_path / 'factor_corr_common', dpi=300);


# In[21]:


g = sns.clustermap(corr_common, cmap=cmap, figsize=(15, 15))
g.savefig(results_path / 'factor_corr_common_cluster', dpi=300);


# In[16]:


corr_ = corr_common.stack().reset_index()
corr_.columns = ['x1', 'x2', 'rho']
corr_ = corr_[corr_.x1!=corr_.x2].drop_duplicates('rho')


# In[17]:


corr_.nlargest(5, columns='rho').append(corr_.nsmallest(5, columns='rho'))


# ### Formulaic Alphas

# In[22]:


get_ipython().run_cell_magic('time', '', "corr_formula = factors[alphas].sort_index().corr(method='spearman').dropna(how='all', axis=1)\ncorr_formula.to_hdf('data.h5', 'correlation/formula')")


# In[23]:


corr_formula = corr_formula.dropna(how='all').dropna(how='all', axis=1)


# In[24]:


mask = np.triu(np.ones_like(corr_formula, dtype=np.bool))
fig, ax = plt.subplots(figsize=(22, 18))
cmap = sns.diverging_palette(10, 220, as_cmap=True)

sns.heatmap(corr_formula, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
fig.tight_layout()
fig.savefig(results_path / 'factor_correlation_formula', dpi=300);


# In[25]:


g = sns.clustermap(corr_formula.replace((np.inf, -np.inf), np.nan), cmap=cmap, figsize=(15, 15))
g.savefig(results_path / 'factor_correlation_formula_cluster', dpi=300);


# In[26]:


corr_formula_ = corr_formula.stack().reset_index()
corr_formula_.columns = ['x1', 'x2', 'rho']
corr_formula_ = corr_formula_[corr_formula_.x1!=corr_formula_.x2].drop_duplicates('rho')


# In[27]:


corr_formula_.nlargest(5, columns='rho').append(corr_formula_.nsmallest(5, columns='rho'))


# ### All Factors

# In[28]:


corr = factors.drop(['ret_fwd', 'alpha_051'], axis=1).corr()


# In[29]:


corr = corr.dropna(how='all').dropna(how='all', axis=1)


# In[30]:


corr.to_hdf('data.h5', 'correlation/all')


# In[31]:


corr.info()


# In[32]:


corr.shape


# In[33]:


sns.set(font_scale=1.2)

mask = np.zeros_like(corr)
np.fill_diagonal(mask, 1)

g = sns.clustermap(corr, 
                   cmap=cmap, 
                   figsize=(20, 20), 
                   dendrogram_ratio=.05,
                   mask=mask,
                   cbar_pos=(0.01, 0.05, 0.01, 0.2));

g.savefig(results_path / 'factor_correlation_all', dpi=300);


# ## Forward return correlation

# In[34]:


fwd_corr = factors.drop(['ret_fwd', 'alpha_051'], axis=1).corrwith(factors.ret_fwd, method='spearman')


# In[35]:


fwd_corr = fwd_corr.dropna()


# In[36]:


fwd_corr.to_hdf('data.h5', 'correlation/fwd_ret')


# In[37]:


top50 = fwd_corr.abs().nlargest(50).index
fwd_corr.loc[top50].sort_values().plot.barh(figsize=(10, 15),
                                            legend=False);


# ## Mutual Information

# In[45]:


mi = {}
for feature in tqdm(features):
    df = (factors
          .loc[:, ['ret_fwd', feature]]
          .dropna().sample(n=100000))
    discrete_features = df[feature].nunique() < 10
    mi[feature] = mutual_info_regression(X=df[[feature]],
                                         y=df.ret_fwd,
                                         discrete_features=discrete_features)[0]
mi = pd.Series(mi)


# In[55]:


mi.nlargest(50).sort_values().plot.barh(figsize=(8, 14));


# In[49]:


mi.to_hdf('data.h5', 'mutual_information')


# ## LightGBM Feature Importance

# In[56]:


def get_fi(model):
    fi = model.feature_importance(importance_type='gain')
    return (pd.Series(fi / fi.sum(),
                      index=model.feature_name()))


# In[57]:


def ic_lgbm(preds, train_data):
    """Custom IC eval metric for lightgbm"""
    is_higher_better = True
    return 'ic', spearmanr(preds, train_data.get_label())[0], is_higher_better


# In[58]:


uniques = factors.nunique()


# In[59]:


categoricals = uniques[uniques < 20].index.tolist()


# In[60]:


categoricals


# In[61]:


features = factors.columns.difference(fwd_returns).tolist()


# In[62]:


label = 'ret_fwd'


# In[63]:


train_length = int(8.5 * 252)
test_length = 252
n_splits = 1


# In[66]:


params = dict(boosting='gbdt',
              objective='regression',
              verbose=-1,
              metric='None')
num_boost_round = 5000


# In[67]:


lgb_data = lgb.Dataset(data=factors.loc[:, features],
                       label=factors.loc[:, label],
                       categorical_feature=categoricals,
                       free_raw_data=False)

cv = MultipleTimeSeriesCV(n_splits=n_splits,
                          lookahead=1,
                          test_period_length=test_length,
                          train_period_length=train_length)

feature_importance, ic, daily_ic = [], [], []

for i, (train_idx, test_idx) in enumerate(cv.split(X=factors)):
    start = time()
    lgb_train = lgb_data.subset(used_indices=train_idx.tolist(),
                               params=params).construct()
    lgb_test = lgb_data.subset(used_indices=test_idx.tolist(),
                               params=params).construct()
    evals_result = {}
    model = lgb.train(params=params,
                      train_set=lgb_train,
                      num_boost_round=num_boost_round,
                      valid_sets=[lgb_train, lgb_test],
                      valid_names=['train', 'valid'],
                      feval=ic_lgbm,
                      evals_result=evals_result,
                      early_stopping_rounds=500,
                      verbose_eval=100)
    model.save_model(f'models/lgb_model.txt')
    fi = get_fi(model)
    fi.to_hdf('data.h5', f'fi/{i:02}')
    test_set = factors.iloc[test_idx, :]
    X_test = test_set.loc[:, model.feature_name()]
    y_test = test_set.loc[:, label]
    y_pred = model.predict(X_test)
    cv_preds = y_test.to_frame('y_test').assign(y_pred=y_pred)
    cv_preds.to_hdf('preds.h5', f'preds/{i:02}')

    by_day = cv_preds.groupby(level='date')
    ic_by_day = by_day.apply(lambda x: spearmanr(x.y_test, x.y_pred)[0])
    daily_ic_mean = ic_by_day.mean()
    daily_ic_median = ic_by_day.median()
    ic = spearmanr(cv_preds.y_test, cv_preds.y_pred)[0]
    print(f'\n{time()-start:6.1f} | {ic:6.2%} | {daily_ic_mean: 6.2%} | {daily_ic_median: 6.2%}')


# In[68]:


cv_result = pd.DataFrame({'Train Set': evals_result['train']['ic'], 
                          'Validation Set': evals_result['valid']['ic']})

ax = cv_result.loc[:300].plot(figsize=(12, 4))
ax.axvline(cv_result['Validation Set'].idxmax(), c='k', ls='--', lw=1);


# ## SHAP Values

# In[69]:


shap.initjs()


# In[70]:


# model = lgb.Booster(model_file='models/lgb_model.txt')


# In[71]:


explainer = shap.TreeExplainer(model)


# In[72]:


# workaround for SHAP version 0.30: https://github.com/slundberg/shap/issues/794
model.params['objective'] = 'regression'


# In[73]:


shap_values = explainer.shap_values(factors.iloc[train_idx, :].loc[:, model.feature_name()])


# In[74]:


np.save(models / 'shap_values.npy', shap_values)


# In[75]:


shap_values = np.load(models / 'shap_values.npy')


# In[76]:


shap.summary_plot(shap_values,
                  factors
                  .iloc[train_idx, :]
                  .loc[:, model.feature_name()],
                  show=False)
plt.gcf().suptitle('SHAP Values')
plt.gcf().tight_layout()
plt.gcf().savefig(results_path / 'shap_summary_dot', dpi=300)


# In[77]:


shap_values = pd.DataFrame(shap_values, columns = features)


# ## Summary

# In[78]:


mi = pd.read_hdf('data.h5', 'mutual_information')
fwd_corr = pd.read_hdf('data.h5', 'correlation/fwd_ret')


# In[79]:


shap_summary = shap_values.abs().mean()
shap_summary /= shap_summary.sum()


# In[80]:


stats = (mi.to_frame('Mutual Information')
         .join(fwd_corr.to_frame('Information Coefficient'))
         .join(fi.to_frame('Feature Importance'))
         .join(shap_summary.to_frame('SHAP Values')))


# In[81]:


cols = {'Information Coefficient': stats['Information Coefficient'].abs()}
corr = stats.assign(**cols).corr('spearman')
mask = np.triu(np.ones_like(corr, dtype=np.bool))
corr = corr.iloc[1:, :-1]
mask = mask[1:, :-1]

fig, ax = plt.subplots(figsize=(8, 5))

cmap = sns.diverging_palette(10, 220, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f')
plt.xticks(rotation=0)
fig.suptitle('Rank Correlation of Feature Metrics', fontsize=12)
fig.tight_layout()
fig.subplots_adjust(top=.92)
fig.savefig(results_path / 'metrics_correlation', dpi=300);


# In[82]:


top_n = 25
fig, axes = plt.subplots(ncols=4, figsize=(16, 8))

shap_summary.nlargest(top_n).sort_values().plot.barh(ax=axes[0], title='SHAP Values')

fi.nlargest(top_n).sort_values().plot.barh(ax=axes[1], title='Feature Importance')

mi.nlargest(top_n).sort_values().plot.barh(ax=axes[2], title='Mutual Information')

top_corr = fwd_corr.abs().nlargest(top_n).index
fwd_corr.loc[top_corr].sort_values().plot.barh(ax=axes[3], title='Information Coefficient')

fig.suptitle('Univariate and Multivariate Feature Evaluation Metrics', fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=.91)
fig.savefig(results_path / 'all_feature_metrics');

