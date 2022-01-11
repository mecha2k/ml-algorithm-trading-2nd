#!/usr/bin/env python
# coding: utf-8

# # Word vectors from SEC filings using Gensim: word2vec model 

# ## Imports & Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from pathlib import Path
from time import time
import logging

import numpy as np
import pandas as pd

from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


# In[3]:


sns.set_style('white')
np.random.seed(42)


# In[4]:


def format_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:02.0f}:{m:02.0f}:{s:02.0f}'


# ### Paths

# In[5]:


sec_path = Path('..', 'data', 'sec-filings')
ngram_path = sec_path / 'ngrams'


# In[6]:


results_path = Path('results', 'sec-filings')

model_path = results_path / 'models'
if not model_path.exists():
    model_path.mkdir(parents=True)

log_path = results_path / 'logs'
if not log_path.exists():
    log_path.mkdir(parents=True)


# ### Logging Setup

# In[7]:


logging.basicConfig(
    filename=log_path / 'word2vec.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S')


# ## word2vec

# In[22]:


analogies_path = Path('data', 'analogies-en.txt')


# ### Set up Sentence Generator

# In[9]:


NGRAMS = 2


# To facilitate memory-efficient text ingestion, the LineSentence class creates a generator from individual sentences contained in the provided text file:

# In[10]:


sentence_path = ngram_path  / f'ngrams_{NGRAMS}.txt'
sentences = LineSentence(sentence_path)


# ### Train word2vec Model

# The [gensim.models.word2vec](https://radimrehurek.com/gensim/models/word2vec.html) class implements the skipgram and CBOW architectures.

# In[11]:


start = time()
model = Word2Vec(sentences,
                 sg=1,          # 1 for skip-gram; otherwise CBOW
                 hs=0,          # hierarchical softmax if 1, negative sampling if 0
                 size=300,      # Vector dimensionality
                 window=5,      # Max distance betw. current and predicted word
                 min_count=50,  # Ignore words with lower frequency
                 negative=15,    # noise word count for negative sampling
                 workers=4,     # no threads 
                 iter=1,        # no epochs = iterations over corpus
                 alpha=0.05,   # initial learning rate
                 min_alpha=0.0001 # final learning rate
                ) 
print('Duration:', format_time(time() - start))


# ### Persist model & vectors

# In[12]:


model.save((model_path / 'word2vec_0.model').as_posix())
model.wv.save((model_path / 'word_vectors_0.bin').as_posix())


# ### Load model and vectors

# In[13]:


model = Word2Vec.load((model_path / 'word2vec_0.model').as_posix())


# In[14]:


wv = KeyedVectors.load((model_path / 'word_vectors_0.bin').as_posix())


# ### Get vocabulary

# In[15]:


vocab = []
for k, _ in model.wv.vocab.items():
    v_ = model.wv.vocab[k]
    vocab.append([k, v_.index, v_.count])


# In[16]:


vocab = (pd.DataFrame(vocab, 
                     columns=['token', 'idx', 'count'])
         .sort_values('count', ascending=False))


# In[17]:


vocab.info()


# In[18]:


vocab.head(10)


# In[19]:


vocab['count'].describe(percentiles=np.arange(.1, 1, .1)).astype(int)


# ### Evaluate Analogies

# In[23]:


def accuracy_by_category(acc, detail=True):
    results = [[c['section'], len(c['correct']), len(c['incorrect'])] for c in acc]
    results = pd.DataFrame(results, columns=['category', 'correct', 'incorrect'])
    results['average'] = results.correct.div(results[['correct', 'incorrect']].sum(1))
    if detail:
        print(results.sort_values('average', ascending=False))
    return results.loc[results.category=='total', ['correct', 'incorrect', 'average']].squeeze().tolist()


# In[24]:


detailed_accuracy = model.wv.accuracy(analogies_path.as_posix(), case_insensitive=True)


# In[25]:


summary = accuracy_by_category(detailed_accuracy)


# In[26]:


def eval_analogies(w2v, max_vocab=15000):
    accuracy = w2v.wv.accuracy(analogies_path,
                               restrict_vocab=15000,
                               case_insensitive=True)
    return (pd.DataFrame([[c['section'],
                        len(c['correct']),
                        len(c['incorrect'])] for c in accuracy],
                      columns=['category', 'correct', 'incorrect'])
          .assign(average=lambda x: 
                  x.correct.div(x.correct.add(x.incorrect))))  


# In[27]:


def total_accuracy(w2v):
    df = eval_analogies(w2v)
    return df.loc[df.category == 'total', ['correct', 'incorrect', 'average']].squeeze().tolist()


# In[28]:


accuracy = eval_analogies(model)
accuracy


# ### Validate Vector Arithmetic

# In[29]:


sims=model.wv.most_similar(positive=['iphone'], restrict_vocab=15000)
print(pd.DataFrame(sims, columns=['term', 'similarity']))


# In[30]:


analogy = model.wv.most_similar(positive=['france', 'london'], 
                                negative=['paris'], 
                                restrict_vocab=15000)
print(pd.DataFrame(analogy, columns=['term', 'similarity']))


# ### Check similarity for random words

# In[31]:


VALID_SET = 5  # Random set of words to get nearest neighbors for
VALID_WINDOW = 100  # Most frequent words to draw validation set from
valid_examples = np.random.choice(VALID_WINDOW, size=VALID_SET, replace=False)
similars = pd.DataFrame()

for id in sorted(valid_examples):
    word = vocab.loc[id, 'token']
    similars[word] = [s[0] for s in model.wv.most_similar(word)]
similars


# ## Continue Training

# In[ ]:


accuracies = [summary]
best_accuracy = summary[-1]
for i in range(1, 15):
    start = time()
    model.train(sentences, epochs=1, total_examples=model.corpus_count)
    detailed_accuracy = model.wv.accuracy(analogies_path)
    accuracies.append(accuracy_by_category(detailed_accuracy, detail=False))
    print(f'{i:02} | Duration: {format_time(time() - start)} | Accuracy: {accuracies[-1][-1]:.2%} ')
    if accuracies[-1][-1] > best_accuracy:
        model.save((model_path / f'word2vec_{i:02}.model').as_posix())
        model.wv.save((model_path / f'word_vectors_{i:02}.bin').as_posix())
        best_accuracy = accuracies[-1][-1]
    (pd.DataFrame(accuracies, 
                 columns=['correct', 'wrong', 'average'])
     .to_csv(model_path / 'accuracies.csv', index=False))
model.wv.save((model_path / 'word_vectors_final.bin').as_posix())


# ### Sample Output

# 
# |Epoch|Duration| Accuracy|
# |---|---|---|
# 01 | 00:14:00 | 31.64% | 
# 02 | 00:14:21 | 31.72% | 
# 03 | 00:14:34 | 33.65% | 
# 04 | 00:16:11 | 34.03% | 
# 05 | 00:13:51 | 33.04% | 
# 06 | 00:13:46 | 33.28% | 
# 07 | 00:13:51 | 33.10% | 
# 08 | 00:13:54 | 34.11% | 
# 09 | 00:13:54 | 33.70% | 
# 10 | 00:13:55 | 34.09% | 
# 11 | 00:13:57 | 35.06% | 
# 12 | 00:13:38 | 33.79% | 
# 13 | 00:13:26 | 32.40% | 

# In[ ]:


(pd.DataFrame(accuracies, 
             columns=['correct', 'wrong', 'average'])
 .to_csv(results_path / 'accuracies.csv', index=False))


# In[ ]:


best_model = Word2Vec.load((results_path / 'word2vec_11.model').as_posix())


# In[ ]:


detailed_accuracy = best_model.wv.accuracy(analogies_path.as_posix(), case_insensitive=True)


# In[ ]:


summary = accuracy_by_category(detailed_accuracy)
print('Base Accuracy: Correct {:,.0f} | Wrong {:,.0f} | Avg {:,.2%}\n'.format(*summary))


# In[ ]:


cat_dict = {'capital-common-countries':'Capitals',
            'capital-world':'Capitals RoW',
            'city-in-state':'City-State',
            'currency':'Currency',
            'family':'Famliy',
            'gram1-adjective-to-adverb':'Adj-Adverb',
            'gram2-opposite':'Opposite',
            'gram3-comparative':'Comparative',
            'gram4-superlative':'Superlative',
            'gram5-present-participle':'Pres. Part.',
            'gram6-nationality-adjective':'Nationality',
            'gram7-past-tense':'Past Tense',
            'gram8-plural':'Plural',
            'gram9-plural-verbs':'Plural Verbs',
            'total':'Total'}


# In[ ]:


results = [[c['section'], len(c['correct']), len(c['incorrect'])] for c in detailed_accuracy]
results = pd.DataFrame(results, columns=['category', 'correct', 'incorrect'])
results['category'] = results.category.map(cat_dict)
results['average'] = results.correct.div(results[['correct', 'incorrect']].sum(1))
results = results.rename(columns=str.capitalize).set_index('Category')
total = results.loc['Total']
results = results.drop('Total')


# In[ ]:


most_sim = best_model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=20)
pd.DataFrame(most_sim, columns=['token', 'similarity'])


# In[ ]:


fig, axes = plt.subplots(figsize=(16, 5), ncols=2)

axes[0] = results.loc[:, ['Correct', 'Incorrect']].plot.bar(stacked=True, ax=axes[0]
                                                           , title='Analogy Accuracy')
ax1 = results.loc[:, ['Average']].plot(ax=axes[0], secondary_y=True, lw=1, c='k', rot=35)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

(pd.DataFrame(most_sim, columns=['token', 'similarity'])
 .set_index('token').similarity
 .sort_values().tail(10).plot.barh(xlim=(.3, .37), ax=axes[1], title='Closest matches for Woman + King - Man'))
fig.tight_layout();

