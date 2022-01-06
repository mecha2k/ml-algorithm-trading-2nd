# In this notebook, we'll compare the following storage formats:
# - CSV: Comma-separated, standard flat text file format.
# - HDF5: Hierarchical data format, developed initially at the National Center for Supercomputing Applications.
#   It is a fast and scalable storage format for numerical data, available in pandas using the PyTables library.
# - Parquet: Part of the Apache Hadoop ecosystem, a binary, columnar storage format that provides efficient data
#   compression and encoding and has been developed by Cloudera and Twitter. It is available for pandas through
#   the `pyarrow` library, led by Wes McKinney, the original author of pandas.
#
# This notebook compares the performance of the preceding libraries using a test DataFrame that can be configured to
# contain numerical or text data, or both. For the HDF5 library, we test both the fixed and table formats. The table
# format allows for queries and can be appended to.
#
# ## Usage
#
# To recreate the charts used in the book, you need to run this notebook twice up to section 'Store Result' using
# different settings for `data_type` and arguments for `generate_test_data` as follows:
# 1. `data_type='Numeric`: `numerical_cols=2000`, `text_cols=0` (default)
# 2. `data_type='Mixed`: `numerical_cols=1000`, `text_cols=1000`

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import string
import warnings
import time


sns.set_style("whitegrid")
sns.set_palette("pastel")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
warnings.filterwarnings("ignore")


## Generate Test Data
# The test `DataFrame` that can be configured to contain numerical or text data, or both. For the HDF5 library,
# we test both the fixed and table format.
def generate_test_data(nrows=100000, numerical_cols=2000, text_cols=0, text_length=10):
    s = "".join([random.choice(string.ascii_letters) for _ in range(text_length)])
    data = pd.concat(
        [
            pd.DataFrame(np.random.random(size=(nrows, numerical_cols))),
            pd.DataFrame(np.full(shape=(nrows, text_cols), fill_value=s)),
        ],
        axis=1,
        ignore_index=True,
    )
    data.columns = [str(i) for i in data.columns]
    return data


results = dict()

data_type = "Numeric"
df = generate_test_data(numerical_cols=100, text_cols=100)
df.info()

## Parquet
parquet_file = Path("../../data/storage/test.parquet")
df.to_parquet(parquet_file)
size = parquet_file.stat().st_size

start = time.time()
df = pd.read_parquet(parquet_file)
print("elapsed time(read) : ", time.time() - start)

start = time.time()
df.to_parquet(parquet_file)
parquet_file.unlink()
print("elapsed time(read) : ", time.time() - start)

test_store = Path("../../data/storage/index.h5")
with pd.HDFStore(test_store) as store:
    store.put("file", df)
size = test_store.stat().st_size

with pd.HDFStore(test_store) as store:
    store.get("file")
test_store.unlink()

with pd.HDFStore(test_store) as store:
    store.append("file", df, format="t")
size = test_store.stat().st_size

with pd.HDFStore(test_store) as store:
    df = store.get("file")
test_store.unlink()

# results["HDF Select"] = {
#     "read": np.mean(read.all_runs),
#     "write": np.mean(write.all_runs),
#     "size": size,
# }
#
# test_csv = Path("../../data/storage/test.csv")
#
# df.to_csv(test_csv)
# print(test_csv.stat().st_size)

# results["CSV"] = {"read": np.mean(read.all_runs), "write": np.mean(write.all_runs), "size": size}
# pd.DataFrame(results).assign(Data=data_type).to_csv(f"{data_type}.csv")

# ## Display Results
# # Please run the notebook twice as described above under `Usage` to create the two `csv` files with results for
# # different test data.
# df = (
#     pd.read_csv("Numeric.csv", index_col=0)
#     .append(pd.read_csv("Mixed.csv", index_col=0))
#     .rename(columns=str.capitalize)
# )
# df.index.name = "Storage"
# df = df.set_index("Data", append=True).unstack()
# df.Size /= 1e9
#
# fig, axes = plt.subplots(ncols=3, figsize=(16, 4))
# for i, op in enumerate(["Read", "Write", "Size"]):
#     flag = op in ["Read", "Write"]
#     df.loc[:, op].plot.barh(title=op, ax=axes[i], logx=flag)
#     if flag:
#         axes[i].set_xlabel("seconds (log scale)")
#     else:
#         axes[i].set_xlabel("GB")
# fig.tight_layout()
# fig.savefig("images/storage.png", bboxinches="tight")
