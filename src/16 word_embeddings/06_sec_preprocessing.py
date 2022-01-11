#!/usr/bin/env python
# coding: utf-8

# # Word vectors from SEC filings using Gensim: Preprocessing

# In this section, we will learn word and phrase vectors from annual SEC filings using gensim to illustrate the potential value of word embeddings for algorithmic trading. In the following sections, we will combine these vectors as features with price returns to train neural networks to predict equity prices from the content of security filings.
# 
# In particular, we use a dataset containing over 22,000 10-K annual reports from the period 2013-2016 that are filed by listed companies and contain both financial information and management commentary (see chapter 3 on Alternative Data). For about half of 11K filings for companies that we have stock prices to label the data for predictive modeling

# ## Imports & Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from dateutil.relativedelta import relativedelta
from pathlib import Path
import numpy as np
import pandas as pd
from time import time
from collections import Counter
import logging
import spacy

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phrases, Phraser


# In[3]:


np.random.seed(42)


# In[4]:


def format_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:02.0f}:{m:02.0f}:{s:02.0f}'


# ### Logging Setup

# In[5]:


logging.basicConfig(
        filename='preprocessing.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S')


# ## Data Download

# The data can be downloaded from [here](https://drive.google.com/uc?id=0B4NK0q0tDtLFendmeHNsYzNVZ2M&export=download). Unzip and move into the `data` folder in the repository's root directory and rename to `filings`.

# ### Paths

# Each filing is a separate text file and a master index contains filing metadata. We extract the most informative sections, namely
# - Item 1 and 1A: Business and Risk Factors
# - Item 7 and 7A: Management's Discussion and Disclosures about Market Risks
# 
# The notebook preprocessing shows how to parse and tokenize the text using spaCy, similar to the approach in chapter 14. We do not lemmatize the tokens to preserve nuances of word usage.
# 
# We use gensim to detect phrases. The Phrases module scores the tokens and the Phraser class transforms the text data accordingly. The notebook shows how to repeat the process to create longer phrases.

# In[6]:


sec_path = Path('..', 'data', 'sec-filings')
filing_path = sec_path / 'filings'
sections_path = sec_path / 'sections'


# In[7]:


if not sections_path.exists():
    sections_path.mkdir(exist_ok=True, parents=True)


# ## Identify Sections

# In[8]:


for i, filing in enumerate(filing_path.glob('*.txt'), 1):
    if i % 500 == 0:
        print(i, end=' ', flush=True)
    filing_id = int(filing.stem)
    items = {}
    for section in filing.read_text().lower().split('°'):
        if section.startswith('item '):
            if len(section.split()) > 1:
                item = section.split()[1].replace('.', '').replace(':', '').replace(',', '')
                text = ' '.join([t for t in section.split()[2:]])
                if items.get(item) is None or len(items.get(item)) < len(text):
                    items[item] = text

    txt = pd.Series(items).reset_index()
    txt.columns = ['item', 'text']
    txt.to_csv(sections_path / (filing.stem + '.csv'), index=False)


# ## Parse Sections

# Select the following sections:

# In[9]:


sections = ['1', '1a', '7', '7a']


# In[9]:


clean_path = sec_path / 'selected_sections'
if not clean_path.exists():
    clean_path.mkdir(exist_ok=True)


# In[14]:


nlp = spacy.load('en', disable=['ner'])
nlp.max_length = 6000000


# In[16]:


vocab = Counter()
t = total_tokens = 0
stats = []

start = time()
to_do = len(list(sections_path.glob('*.csv')))
done = len(list(clean_path.glob('*.csv'))) + 1
for text_file in sections_path.glob('*.csv'):
    file_id = int(text_file.stem)
    clean_file = clean_path / f'{file_id}.csv'
    if clean_file.exists():
        continue
    items = pd.read_csv(text_file).dropna()
    items.item = items.item.astype(str)
    items = items[items.item.isin(sections)]
    if done % 100 == 0:
        duration = time() - start
        to_go = (to_do - done) * duration / done
        print(f'{done:>5}\t{format_time(duration)}\t{total_tokens / duration:,.0f}\t{format_time(to_go)}')
    
    clean_doc = []
    for _, (item, text) in items.iterrows():
        doc = nlp(text)
        for s, sentence in enumerate(doc.sents):
            clean_sentence = []
            if sentence is not None:
                for t, token in enumerate(sentence, 1):
                    if not any([token.is_stop,
                                token.is_digit,
                                not token.is_alpha,
                                token.is_punct,
                                token.is_space,
                                token.lemma_ == '-PRON-',
                                token.pos_ in ['PUNCT', 'SYM', 'X']]):
                        clean_sentence.append(token.text.lower())
                total_tokens += t
                if len(clean_sentence) > 0:
                    clean_doc.append([item, s, ' '.join(clean_sentence)])
    (pd.DataFrame(clean_doc,
                  columns=['item', 'sentence', 'text'])
     .dropna()
     .to_csv(clean_file, index=False))
    done += 1


# ## Create ngrams

# In[10]:


ngram_path = sec_path / 'ngrams'
stats_path = sec_path / 'corpus_stats'
for path in [ngram_path, stats_path]:
    if not path.exists():
        path.mkdir(parents=True)


# In[19]:


unigrams = ngram_path / 'ngrams_1.txt'


# In[20]:


def create_unigrams(min_length=3):
    texts = []
    sentence_counter = Counter()
    vocab = Counter()
    for i, f in enumerate(clean_path.glob('*.csv')):
        if i % 1000 == 0:
            print(i, end=' ', flush=True)
        df = pd.read_csv(f)
        df.item = df.item.astype(str)
        df = df[df.item.isin(sections)]
        sentence_counter.update(df.groupby('item').size().to_dict())
        for sentence in df.text.dropna().str.split().tolist():
            if len(sentence) >= min_length:
                vocab.update(sentence)
                texts.append(' '.join(sentence))
    
    (pd.DataFrame(sentence_counter.most_common(), 
                  columns=['item', 'sentences'])
     .to_csv(stats_path / 'selected_sentences.csv', index=False))
    (pd.DataFrame(vocab.most_common(), columns=['token', 'n'])
     .to_csv(stats_path / 'sections_vocab.csv', index=False))
    
    unigrams.write_text('\n'.join(texts))
    return [l.split() for l in texts]


# In[21]:


start = time()
if not unigrams.exists():
    texts = create_unigrams()
else:
    texts = [l.split() for l in unigrams.open()]
print('\nReading: ', format_time(time() - start))


# In[22]:


def create_ngrams(max_length=3):
    """Using gensim to create ngrams"""

    n_grams = pd.DataFrame()
    start = time()
    for n in range(2, max_length + 1):
        print(n, end=' ', flush=True)

        sentences = LineSentence(ngram_path / f'ngrams_{n - 1}.txt')
        phrases = Phrases(sentences=sentences,
                          min_count=25,  # ignore terms with a lower count
                          threshold=0.5,  # accept phrases with higher score
                          max_vocab_size=40000000,  # prune of less common words to limit memory use
                          delimiter=b'_',  # how to join ngram tokens
                          progress_per=50000,  # log progress every
                          scoring='npmi')

        s = pd.DataFrame([[k.decode('utf-8'), v] for k, v in phrases.export_phrases(sentences)], 
                         columns=['phrase', 'score']).assign(length=n)

        n_grams = pd.concat([n_grams, s])
        grams = Phraser(phrases)
        sentences = grams[sentences]
        (ngram_path / f'ngrams_{n}.txt').write_text('\n'.join([' '.join(s) for s in sentences]))

    n_grams = n_grams.sort_values('score', ascending=False)
    n_grams.phrase = n_grams.phrase.str.replace('_', ' ')
    n_grams['ngram'] = n_grams.phrase.str.replace(' ', '_')

    n_grams.to_parquet(sec_path / 'ngrams.parquet')

    print('\n\tDuration: ', format_time(time() - start))
    print('\tngrams: {:,d}\n'.format(len(n_grams)))
    print(n_grams.groupby('length').size())


# In[ ]:


create_ngrams()


# ## Inspect Corpus

# In[18]:


percentiles=np.arange(.1, 1, .1).round(2)


# In[11]:


nsents, ntokens = Counter(), Counter()
for f in clean_path.glob('*.csv'):
    df = pd.read_csv(f)
    nsents.update({str(k): v for k, v in df.item.value_counts().to_dict().items()})
    df['ntokens'] = df.text.str.split().str.len()
    ntokens.update({str(k): v for k, v in df.groupby('item').ntokens.sum().to_dict().items()})


# In[12]:


ntokens = pd.DataFrame(ntokens.most_common(), columns=['Item', '# Tokens'])
nsents = pd.DataFrame(nsents.most_common(), columns=['Item', '# Sentences'])


# In[13]:


nsents.set_index('Item').join(ntokens.set_index('Item')).plot.bar(secondary_y='# Tokens', rot=0);


# In[ ]:


ngrams = pd.read_parquet(sec_path / 'ngrams.parquet')


# In[ ]:


ngrams.info()


# In[ ]:


ngrams.head()


# In[ ]:


ngrams.score.describe(percentiles=percentiles)


# In[ ]:


ngrams[ngrams.score>.7].sort_values(['length', 'score']).head(10)


# In[15]:


vocab = pd.read_csv(stats_path / 'sections_vocab.csv').dropna()


# In[16]:


vocab.info()


# In[19]:


vocab.n.describe(percentiles).astype(int)


# In[20]:


tokens = Counter()
for l in (ngram_path / 'ngrams_2.txt').open():
    tokens.update(l.split())


# In[21]:


tokens = pd.DataFrame(tokens.most_common(),
                     columns=['token', 'count'])


# In[22]:


tokens.info()


# In[23]:


tokens.head()


# In[24]:


tokens.loc[tokens.token.str.contains('_'), 'count'].describe(percentiles).astype(int)


# In[25]:


tokens[tokens.token.str.contains('_')].head(20).to_csv(sec_path / 'ngram_examples.csv', index=False)


# In[26]:


tokens[tokens.token.str.contains('_')].head(20)


# ## Get returns

# In[27]:


DATA_FOLDER = Path('..', 'data')


# In[28]:


with pd.HDFStore(DATA_FOLDER / 'assets.h5') as store:
    prices = store['quandl/wiki/prices'].adj_close


# In[29]:


sec = pd.read_csv(sec_path / 'filing_index.csv').rename(columns=str.lower)
sec.date_filed = pd.to_datetime(sec.date_filed)


# In[30]:


sec.info()


# In[31]:


idx = pd.IndexSlice


# In[32]:


first = sec.date_filed.min() + relativedelta(months=-1)
last = sec.date_filed.max() + relativedelta(months=1)
prices = (prices
          .loc[idx[first:last, :]]
          .unstack().resample('D')
          .ffill()
          .dropna(how='all', axis=1)
          .filter(sec.ticker.unique()))


# In[33]:


sec = sec.loc[sec.ticker.isin(prices.columns), ['ticker', 'date_filed']]

price_data = []
for ticker, date in sec.values.tolist():
    target = date + relativedelta(months=1)
    s = prices.loc[date: target, ticker]
    price_data.append(s.iloc[-1] / s.iloc[0] - 1)

df = pd.DataFrame(price_data,
                  columns=['returns'],
                  index=sec.index)


# In[34]:


df.returns.describe()       


# In[35]:


sec['returns'] = price_data
sec.info()


# In[36]:


sec.dropna().to_csv(sec_path / 'sec_returns.csv', index=False)

