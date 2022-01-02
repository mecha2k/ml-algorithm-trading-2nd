#!/usr/bin/env python
# coding: utf-8

# # Create Yelp Reviews data for Sentiment Analysis and Word Embeddings

# ## Imports & Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from pathlib import Path
import pandas as pd
from pandas.io.json import json_normalize


# ## About the Data

# The data consists of several files with information on the business, the user, the review and other aspects that Yelp provides to encourage data science innovation.
# 
# The data consists of several files with information on the business, the user, the review and other aspects that Yelp provides to encourage data science innovation. 
# 
# We will use around six million reviews produced over the 2010-2019 period to extract text features. In addition, we will use other information submitted with the review about the user. 

# ## Getting the Data

# You can download the data from [here](https://www.yelp.com/dataset) in json format after accepting the license. The 2020 version has 4.7GB (compressed) and around 10.5GB (uncompressed) of text data.
# 
# After download, extract the following two of the five `.json` files into to `./yelp/json`:
# - the `yelp_academic_dataset_user.json`
# - the `yelp_academic_dataset_reviews.json`
# 
# Rename both files by stripping out the `yelp_academic_dataset_` prefix so you have the following directory structure:
# ```
# data
# |-create_yelp_review_data.ipynb
# |-yelp
#     |-json
#         |-user.json
#         |-review.json
# ```
# 

# In[3]:


yelp_dir = Path('yelp')

if not yelp_dir.exists():
    yelp_dir.mkdir(exist_ok=True)


# ## Parse json and store as parquet files

# Convert json to faster parquet format:

# In[4]:


for fname in ['review', 'user']:
    print(fname)
    
    json_file = yelp_dir / 'json' / f'{fname}.json'
    parquet_file = yelp_dir / f'{fname}.parquet'
    if parquet_file.exists():
        print('\talready exists')
        continue

    data = json_file.read_text(encoding='utf-8')
    json_data = '[' + ','.join([l.strip()
                                for l in data.split('\n') if l.strip()]) + ']\n'
    data = json.loads(json_data)
    df = json_normalize(data)
    if fname == 'review':
        df.date = pd.to_datetime(df.date)
        latest = df.date.max()
        df['year'] = df.date.dt.year
        df['month'] = df.date.dt.month
        df = df.drop(['date', 'business_id', 'review_id'], axis=1)
    if fname == 'user':
        df.yelping_since = pd.to_datetime(df.yelping_since)
        df = (df.assign(member_yrs=lambda x: (latest - x.yelping_since)
                        .dt.days.div(365).astype(int))
              .drop(['elite', 'friends', 'name', 'yelping_since'], axis=1))
    df.dropna(how='all', axis=1).to_parquet(parquet_file)


# Now you can remove the json files.

# In[8]:


def merge_files(remove=False):
    combined_file = yelp_dir / 'user_reviews.parquet'
    if not combined_file.exists():
        user = pd.read_parquet(yelp_dir / 'user.parquet')
        print(user.info(null_counts=True))

        review = pd.read_parquet(yelp_dir / 'review.parquet')
        print(review.info(null_counts=True))

        combined = (review.merge(user, on='user_id',
                                 how='left', suffixes=['', '_user'])
                    .drop('user_id', axis=1))
        combined = combined[combined.stars > 0]
        print(combined.info(null_counts=True))
        combined.to_parquet(yelp_dir / 'user_reviews.parquet')
    else:
        print('already merged')
    if remove:
        for fname in ['user', 'review']:
            f = yelp_dir / (fname + '.parquet')
            if f.exists():
                f.unlink()


# In[9]:


merge_files(remove=True)

