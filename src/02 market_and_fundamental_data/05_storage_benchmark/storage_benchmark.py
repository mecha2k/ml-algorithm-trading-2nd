#!/usr/bin/env python
# coding: utf-8

# # Storage Benchmark

# In this notebook, we'll compare the following storage formats:
# - CSV: Comma-separated, standard flat text file format.
# - HDF5: Hierarchical data format, developed initially at the National Center for Supercomputing Applications. It is a fast and scalable storage format for numerical data, available in pandas using the PyTables library.
# - Parquet: Part of the Apache Hadoop ecosystem, a binary, columnar storage format that provides efficient data compression and encoding and has been developed by Cloudera and Twitter. It is available for pandas through the `pyarrow` library, led by Wes McKinney, the original author of pandas.
# 
# This notebook compares the performance of the preceding libraries using a test DataFrame that can be configured to contain numerical or text data, or both. For the HDF5 library, we test both the fixed and table formats. The table format allows for queries and can be appended to.
# 
# ## Usage
# 
# To recreate the charts used in the book, you need to run this notebook twice up to section 'Store Result' using different settings for `data_type` and arguments for `generate_test_data` as follows:
# 1. `data_type='Numeric`: `numerical_cols=2000`, `text_cols=0` (default)
# 2. `data_type='Mixed`: `numerical_cols=1000`, `text_cols=1000`

# ## Imports & Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import string


# In[3]:


sns.set_style('whitegrid')


# In[4]:


results = {}


# ## Generate Test Data

# The test `DataFrame` that can be configured to contain numerical or text data, or both. For the HDF5 library, we test both the fixed and table format. 

# In[5]:


def generate_test_data(nrows=100000, numerical_cols=2000, text_cols=0, text_length=10):
    s = "".join([random.choice(string.ascii_letters)
                 for _ in range(text_length)])
    data = pd.concat([pd.DataFrame(np.random.random(size=(nrows, numerical_cols))),
                      pd.DataFrame(np.full(shape=(nrows, text_cols), fill_value=s))],
                     axis=1, ignore_index=True)
    data.columns = [str(i) for i in data.columns]
    return data


# In[6]:


data_type = 'Numeric'


# In[7]:


df = generate_test_data(numerical_cols=1000, text_cols=1000)
df.info()


# ## Parquet

# ### Size

# In[8]:


parquet_file = Path('test.parquet')


# In[9]:


df.to_parquet(parquet_file)
size = parquet_file.stat().st_size


# ### Read

# In[10]:


get_ipython().run_cell_magic('timeit', '-o', 'df = pd.read_parquet(parquet_file)')


# In[11]:


read = _


# In[12]:


parquet_file.unlink()


# ### Write

# In[13]:


get_ipython().run_cell_magic('timeit', '-o', 'df.to_parquet(parquet_file)\nparquet_file.unlink()')


# In[14]:


write = _


# ### Results

# In[15]:


results['Parquet'] = {'read': np.mean(read.all_runs), 'write': np.mean(write.all_runs), 'size': size}


# ## HDF5

# In[16]:


test_store = Path('index.h5')


# ### Fixed Format

# #### Size

# In[17]:


with pd.HDFStore(test_store) as store:
    store.put('file', df)
size = test_store.stat().st_size


# #### Read

# In[18]:


get_ipython().run_cell_magic('timeit', '-o', "with pd.HDFStore(test_store) as store:\n    store.get('file')")


# In[19]:


read = _


# In[20]:


test_store.unlink()


# #### Write

# In[21]:


get_ipython().run_cell_magic('timeit', '-o', "with pd.HDFStore(test_store) as store:\n    store.put('file', df)\ntest_store.unlink()")


# In[22]:


write = _


# #### Results

# In[23]:


results['HDF Fixed'] = {'read': np.mean(read.all_runs), 'write': np.mean(write.all_runs), 'size': size}


# ### Table Format

# #### Size

# In[24]:


with pd.HDFStore(test_store) as store:
    store.append('file', df, format='t')
size = test_store.stat().st_size    


# #### Read

# In[ ]:


get_ipython().run_cell_magic('timeit', '-o', "with pd.HDFStore(test_store) as store:\n    df = store.get('file')")


# In[ ]:


read = _


# In[ ]:


test_store.unlink()


# #### Write

# Note that `write` in table format does not work with text data.

# In[ ]:


get_ipython().run_cell_magic('timeit', '-o', "with pd.HDFStore(test_store) as store:\n    store.append('file', df, format='t')\ntest_store.unlink()    ")


# In[ ]:


write = _


# #### Results

# In[ ]:


results['HDF Table'] = {'read': np.mean(read.all_runs), 'write': np.mean(write.all_runs), 'size': size}


# ### Table Select

# #### Size

# In[ ]:


with pd.HDFStore(test_store) as store:
    store.append('file', df, format='t', data_columns=['company', 'form'])
size = test_store.stat().st_size 


# #### Read

# In[ ]:


company = 'APPLE INC'


# In[ ]:


get_ipython().run_cell_magic('timeit', '', "with pd.HDFStore(test_store) as store:\n    s = store.get('file')")


# In[ ]:


read = _


# In[ ]:


test_store.unlink()


# #### Write

# In[ ]:


get_ipython().run_cell_magic('timeit', '', "with pd.HDFStore(test_store) as store:\n    store.append('file', df, format='t', data_columns=['company', 'form'])\ntest_store.unlink() ")


# In[ ]:


write = _


# #### Results

# In[ ]:


results['HDF Select'] = {'read': np.mean(read.all_runs), 'write': np.mean(write.all_runs), 'size': size}


# ## CSV

# In[ ]:


test_csv = Path('test.csv')


# ### Size

# In[ ]:


df.to_csv(test_csv)
test_csv.stat().st_size


# ### Read

# In[ ]:


get_ipython().run_cell_magic('timeit', '-o', 'df = pd.read_csv(test_csv)')


# In[ ]:


read = _


# In[ ]:


test_csv.unlink()  


# ### Write

# In[ ]:


get_ipython().run_cell_magic('timeit', '-o', 'df.to_csv(test_csv)\ntest_csv.unlink()')


# In[ ]:


write = _


# ### Results

# In[ ]:


results['CSV'] = {'read': np.mean(read.all_runs), 'write': np.mean(write.all_runs), 'size': size}


# ## Store Results

# In[ ]:


pd.DataFrame(results).assign(Data=data_type).to_csv(f'{data_type}.csv')


# ## Display Results

# Please run the notebook twice as described above under `Usage` to create the two `csv` files with results for different test data.

# In[ ]:


df = (pd.read_csv('Numeric.csv', index_col=0)
      .append(pd.read_csv('Mixed.csv', index_col=0))
      .rename(columns=str.capitalize))
df.index.name='Storage'
df = df.set_index('Data', append=True).unstack()
df.Size /= 1e9


# In[ ]:


fig, axes = plt.subplots(ncols=3, figsize=(16, 4))
for i, op in enumerate(['Read', 'Write', 'Size']):
    flag= op in ['Read', 'Write']
    df.loc[:, op].plot.barh(title=op, ax=axes[i], logx=flag)
    if flag:
        axes[i].set_xlabel('seconds (log scale)')
    else:
        axes[i].set_xlabel('GB')
fig.tight_layout()
fig.savefig('storage', dpi=300);

