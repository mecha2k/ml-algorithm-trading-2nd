#!/usr/bin/env python
# coding: utf-8

# # Text classification and sentiment analysis: Twitter

# Once text data has been converted into numerical features using the natural language processing techniques discussed in the previous sections, text classification works just like any other classification task.
# 
# In this notebook, we will apply these preprocessing technique to news articles, product reviews, and Twitter data and teach various classifiers to predict discrete news categories, review scores, and sentiment polarity.

# ## Imports

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# spacy, textblob and nltk for language processing
from textblob import TextBlob

# sklearn for feature extraction & modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score


# In[3]:


sns.set_style('white')


# ## Twitter Sentiment

# ### Download the data

# We use a dataset that contains 1.6 million training and 350 test tweets from 2009 with algorithmically assigned binary positive and negative sentiment scores that are fairly evenly split.

# Follow the [instructions](../data/twitter_sentiment.ipynb) to create the dataset.

# - 0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive); training data has no neutral tweets
# - 1 - the id of the tweet (2087)
# - 2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
# - 3 - the query (lyx). If there is no query, then this value is NO_QUERY. (only test data uses query)
# - 4 - the user that tweeted (robotickilldozr)
# - 5 - the text of the tweet (Lyx is cool)

# ### Read and preprocess train/test data

# In[4]:


data_path = Path('..', 'data', 'sentiment140')
if not data_path.exists():
    data_path.mkdir(parents=True)


# In[5]:


names = ['polarity', 'id', 'date', 'query', 'user', 'text']


# Take a few preprocessing steps:
# - remove tweets above the legal (at the time) length of 140 characters,
# - binarize polarity, and 
# - move the data to the faster parquet format.

# In[6]:


def load_train_data():
    parquet_file = data_path / 'train.parquet'
    if not parquet_file.exists():
        df = (pd.read_csv(data_path / 'train.csv',
                          low_memory=False,
                          encoding='latin1',
                          header=None,
                          names=names,
                          parse_dates=['date'])
              .drop(['id', 'query'], axis=1)
              .drop_duplicates(subset=['polarity', 'text']))
        df = df[df.text.str.len() <= 140]
        df.polarity = (df.polarity > 0).astype(int)
        df.to_parquet(parquet_file)
        return df
    else:
        return pd.read_parquet(parquet_file)


# In[7]:


train = load_train_data()
train.info(null_counts=True)


# In[8]:


def load_test_data():
    parquet_file = data_path / 'test.parquet'
    if not parquet_file.exists():
        df = (pd.read_csv('data/sentiment140/test.csv',
                          low_memory=False,
                          encoding='latin1',
                          header=None,
                          names=names,
                          parse_dates=['date'])
              .drop(['id', 'query'], axis=1)
              .drop_duplicates(subset=['polarity', 'text']))
        df = df[(df.text.str.len() <= 140) &
                (df.polarity.isin([0, 4]))]
        df.to_parquet(parquet_file)
        return df
    else:
        return pd.read_parquet(parquet_file)


# In[9]:


test = load_test_data()
test.info(null_counts=True)


# ### Explore data

# In[10]:


train.head()


# In[11]:


train.polarity = (train.polarity>0).astype(int)
train.polarity.value_counts()


# In[12]:


test.polarity = (test.polarity>0).astype(int)
test.polarity.value_counts()


# In[13]:


sns.distplot(train.text.str.len(), kde=False)
sns.despine();


# In[14]:


train.date.describe()


# In[15]:


train.user.nunique()


# In[16]:


train.user.value_counts().head(10)


# ### Create text vectorizer

# We create a document-term matrix with 934 tokens as follows:

# In[17]:


vectorizer = CountVectorizer(min_df=.001, max_df=.8, stop_words='english')
train_dtm = vectorizer.fit_transform(train.text)


# In[18]:


train_dtm


# In[19]:


test_dtm = vectorizer.transform(test.text)


# ### Train Naive Bayes Classifier

# In[20]:


nb = MultinomialNB()
nb.fit(train_dtm, train.polarity)


# ### Predict Test Polarity

# In[21]:


predicted_polarity = nb.predict(test_dtm)


# ### Evaluate Results

# In[22]:


accuracy_score(test.polarity, predicted_polarity)


# ### TextBlob for Sentiment Analysis

# In[23]:


sample_positive = train.text.loc[256332]
print(sample_positive)
parsed_positive = TextBlob(sample_positive)
parsed_positive.polarity


# In[24]:


sample_negative = train.text.loc[636079]
print(sample_negative)
parsed_negative = TextBlob(sample_negative)
parsed_negative.polarity


# In[25]:


def estimate_polarity(text):
    return TextBlob(text).sentiment.polarity


# In[26]:


train[['text']].sample(10).assign(sentiment=lambda x: x.text.apply(estimate_polarity)).sort_values('sentiment')


# ### Compare with TextBlob Polarity Score

# We also obtain TextBlob sentiment scores for the tweets and note (see left panel in below figure) that positive test tweets receive a significantly higher sentiment estimate. We then use the MultinomialNB ‘s model .predict_proba() method to compute predicted probabilities and compare both models using the respective Area Under the Curve (see right panel below).

# In[27]:


test['sentiment'] = test.text.apply(estimate_polarity)


# In[28]:


accuracy_score(test.polarity, (test.sentiment>0).astype(int))


# #### ROC AUC Scores

# In[29]:


roc_auc_score(y_true=test.polarity, y_score=test.sentiment)


# In[30]:


roc_auc_score(y_true=test.polarity, y_score=nb.predict_proba(test_dtm)[:, 1])


# In[31]:


fpr_tb, tpr_tb, _ = roc_curve(y_true=test.polarity, y_score=test.sentiment)
roc_tb = pd.Series(tpr_tb, index=fpr_tb)
fpr_nb, tpr_nb, _ = roc_curve(y_true=test.polarity, y_score=nb.predict_proba(test_dtm)[:, 1])
roc_nb = pd.Series(tpr_nb, index=fpr_nb)


# The Naive Bayes model outperforms TextBlob in this case.

# In[32]:


fig, axes = plt.subplots(ncols=2, figsize=(14, 6))
sns.boxplot(x='polarity', y='sentiment', data=test, ax=axes[0])
axes[0].set_title('TextBlob Sentiment Scores')
roc_nb.plot(ax=axes[1], label='Naive Bayes', legend=True, lw=1, title='ROC Curves')
roc_tb.plot(ax=axes[1], label='TextBlob', legend=True, lw=1)
sns.despine()
fig.tight_layout();

