#!/usr/bin/env python
# coding: utf-8

# # Evaluation of the GBM GridSearchCV results

# This file illustrates how to evaluate the [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) results for sklearn's [GradientBoostingClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) obtained after first running `sklearn_gbm_tuning.py` in this directory to test various hyperparameter combinations and store the result.

# ## Imports & Settings

# In[27]:


import warnings
warnings.filterwarnings('ignore')


# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path
import os
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

from statsmodels.api import OLS, add_constant
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import roc_auc_score
import joblib


# In[29]:


sns.set_style("white")
np.random.seed(42)
pd.options.display.float_format = '{:,.4f}'.format


# In[30]:


with pd.HDFStore('data/tuning_sklearn_gbm.h5') as store:
    test_feature_data = store['holdout/features']
    test_features = test_feature_data.columns
    test_target = store['holdout/target']


# ## GBM GridsearchCV with sklearn

# Need OneStepTimeSeriesSplit because stored GridSearchCV result expects it

# In[31]:


class OneStepTimeSeriesSplit:
    pass


# ### Load Result

# Need to first run `sklearn_gbm_tuning.py` to perform gridsearch and store result (not included due to file size).

# In[32]:


gridsearch_result = joblib.load('results/sklearn_gbm_gridsearch.joblib')


# The GridSearchCV object has several additional attributes after completion that we can access after loading the pickled result to learn which hyperparameter combination performed best and its average cross-validation AUC score, which results in a modest improvement over the default values. This is shown in the following code:

# ### Best Parameters & AUC Score

# In[33]:


pd.Series(gridsearch_result.best_params_)


# In[34]:


f'{gridsearch_result.best_score_:.4f}'


# ### Evaluate best model

# #### Test on hold-out set

# In[35]:


best_model = gridsearch_result.best_estimator_


# In[36]:


idx = pd.IndexSlice
test_dates = sorted(test_feature_data.index.get_level_values('date').unique())


# In[37]:


auc = {}
for i, test_date in enumerate(test_dates):
    test_data = test_feature_data.loc[idx[:, test_date], :]
    preds = best_model.predict(test_data)
    auc[i] = roc_auc_score(y_true=test_target.loc[test_data.index], y_score=preds)


# In[38]:


auc = pd.Series(auc)


# In[39]:


auc.head()


# In[40]:


ax = auc.sort_index(ascending=False).plot.barh(xlim=(.45, .55),
                                               title=f'Test AUC: {auc.mean():.2%}',
                                               figsize=(8, 4))
ax.axvline(auc.mean(), ls='--', lw=1, c='k')
sns.despine()
plt.tight_layout()


# #### Inspect global feature importance

# In[41]:


(pd.Series(best_model.feature_importances_,
           index=test_features)
 .sort_values()
 .tail(25)
 .plot.barh(figsize=(8, 5)))
sns.despine()
plt.tight_layout()


# ### CV Train-Test Scores

# In[42]:


results = pd.DataFrame(gridsearch_result.cv_results_).drop('params', axis=1)
results.info()


# In[43]:


results.head()


# ### Get parameter values & mean test scores

# In[44]:


test_scores = results.filter(like='param').join(results[['mean_test_score']])
test_scores = test_scores.rename(columns={c: '_'.join(c.split('_')[1:]) for c in test_scores.columns})
test_scores.info()


# In[45]:


params = test_scores.columns[:-1].tolist()


# In[46]:


test_scores = test_scores.set_index('test_score').stack().reset_index()
test_scores.columns= ['test_score', 'parameter', 'value']
test_scores.head()


# In[47]:


test_scores.info()


# In[48]:


def get_test_scores(df):
    """Select parameter values and test scores"""
    data = df.filter(like='param').join(results[['mean_test_score']])
    return data.rename(columns={c: '_'.join(c.split('_')[1:]) for c in data.columns})


# ### Plot Test Scores vs Parameter Settings

# The GridSearchCV result stores the average cross-validation scores so that we can analyze how different hyperparameter settings affect the outcome.
# 
# The six seaborn swarm plots below show the distribution of AUC test scores for all parameter values. In this case, the highest AUC  test scores required a low learning_rate and a large value for max_features. Some parameter settings, such as a low learning_rate, produce a wide range of outcomes that depend on the complementary settings of other parameters. Other parameters are compatible with high scores for all settings use in the experiment:

# In[49]:


plot_data = get_test_scores(results).drop('min_impurity_decrease', axis=1)
plot_params = plot_data.columns[:-1].tolist()
plot_data.info()


# In[50]:


fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))
axes = axes.flatten()

for i, param in enumerate(plot_params):
    sns.swarmplot(x=param, y='test_score', data=plot_data, ax=axes[i])
    
fig.suptitle('Mean Test Score Distribution by Hyperparameter', fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=.94)
fig.savefig('sklearn_cv_scores_by_param', dpi=300);


# ### Dummy-encode parameters

# In[51]:


data = get_test_scores(results)
params = data.columns[:-1].tolist()
data = pd.get_dummies(data,columns=params, drop_first=False)
data.info()


# ### Build Regression Tree

# We will now explore how hyperparameter settings jointly affect the mean cross-validation score. To gain insight into how parameter settings interact, we can train a DecisionTreeRegressor with the mean test score as the outcome and the parameter settings, encoded as categorical variables in one-hot or dummy format. 
# 
# The tree structure highlights that using all features (max_features_1), a low learning_rate, and a max_depth over three led to the best results, as shown in the following diagram:

# In[52]:


reg_tree = DecisionTreeRegressor(criterion='mse',
                                 splitter='best',
                                 max_depth=4,
                                 min_samples_split=5,
                                 min_samples_leaf=10,
                                 min_weight_fraction_leaf=0.0,
                                 max_features=None,
                                 random_state=42,
                                 max_leaf_nodes=None,
                                 min_impurity_decrease=0.0,
                                 min_impurity_split=None)


# In[53]:


gbm_features = data.drop('test_score', axis=1).columns
reg_tree.fit(X=data[gbm_features], y=data.test_score)


# In[54]:


reg_tree.feature_importances_


# #### Visualize Tree

# In[55]:


out_file = 'results/gbm_sklearn_tree.dot'
dot_data = export_graphviz(reg_tree,
                          out_file=out_file,
                          feature_names=gbm_features,
                          max_depth=4,
                          filled=True,
                          rounded=True,
                          special_characters=True)
if out_file is not None:
    dot_data = Path(out_file).read_text()

graphviz.Source(dot_data)


# #### Compute Feature Importance

# Overfit regression tree to learn detailed rules that classify all samples

# In[57]:


reg_tree = DecisionTreeRegressor(criterion='mse',
                                 splitter='best',
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 min_weight_fraction_leaf=0.0,
                                 max_features=None,
                                 random_state=42,
                                 max_leaf_nodes=None,
                                 min_impurity_decrease=0.0,
                                 min_impurity_split=None)

gbm_features = data.drop('test_score', axis=1).columns
reg_tree.fit(X=data[gbm_features], y=data.test_score)


# The bar chart below displays the influence of the hyperparameter settings in producing different outcomes, measured by their feature importance for a decision tree that is grown to its maximum depth. Naturally, the features that appear near the top of the tree also accumulate the highest importance scores.

# In[58]:


gbm_fi = (pd.Series(reg_tree.feature_importances_, 
                    index=gbm_features)
          .sort_values(ascending=False))
gbm_fi = gbm_fi[gbm_fi > 0]
idx = [p.split('_') for p in gbm_fi.index]
gbm_fi.index = ['_'.join(p[:-1]) + '=' + p[-1] for p in idx]
gbm_fi.sort_values().plot.barh(figsize=(5,5))
plt.title('Hyperparameter Importance')
sns.despine()
plt.tight_layout();


# ### Run linear regression

# Alternatively, we can use a linear regression to gain insights into the statistical significance of the linear relationship between hyperparameters and test scores.

# In[59]:


data = get_test_scores(results)
params = data.columns[:-1].tolist()
data = pd.get_dummies(data,columns=params, drop_first=True)

model = OLS(endog=data.test_score, exog=add_constant(data.drop('test_score', axis=1))).fit(cov_type='HC3')
print(model.summary())


# In[ ]:




