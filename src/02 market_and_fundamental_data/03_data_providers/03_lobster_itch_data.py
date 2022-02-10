# Tick Data from LOBSTER (Limit Order Book System - The Efficient Reconstructor) is an [online]
# (https://lobsterdata.com/info/WhatIsLOBSTER.php) limit order book data tool to provide easy-to-use, high-quality
# limit order book data. Since 2013 LOBSTER acts as a data provider for the academic community, giving access to
# reconstructed limit order book data for the entire universe of NASDAQ traded stocks.
# More recently, it has started to make the data available on a commercial basis.
from pathlib import Path
from datetime import datetime, timedelta
from itertools import chain

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

sns.set_style("whitegrid")
sns.set_palette("pastel")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 16
warnings.filterwarnings("ignore")

## Load Orderbook Data
# We will illustrate the functionality using a free sample.
# Obtain data here: https://lobsterdata.com/info/DataSamples.php; [this]
# (https://lobsterdata.com/info/sample/LOBSTER_SampleFile_AMZN_2012-06-21_10.zip) is the link to the 10-level file
# The code assumes the file has been extracted into a `data` subfolder of the current directory.

path = Path("../../data/lobster")

# We use the following to label the table columns:
print(
    list(
        chain(
            *[
                ("Ask Price {0},Ask Size {0},Bid Price {0},Bid Size {0}".format(i)).split(",")
                for i in range(10)
            ]
        )
    )
)

price = list(chain(*[("Ask Price {0},Bid Price {0}".format(i)).split(",") for i in range(10)]))
size = list(chain(*[("Ask Size {0},Bid Size {0}".format(i)).split(",") for i in range(10)]))
cols = list(chain(*zip(price, size)))

order_data = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
orders = pd.read_csv(path / order_data, header=None, names=cols)
orders.info()
print(orders.head())


## Parse Message Data
# Message Type Codes:
#     1: Submission of a new limit order
#     2: Cancellation (Partial deletion
#        of a limit order)
#     3: Deletion (Total deletion of a limit order)
#     4: Execution of a visible limit order
#     5: Execution of a hidden limit order
#     7: Trading halt indicator
#        (Detailed information below)
types = {
    1: "submission",
    2: "cancellation",
    3: "deletion",
    4: "execution_visible",
    5: "execution_hidden",
    7: "trading_halt",
}

trading_date = "2012-06-21"
levels = 10

message_data = "AMZN_{}_34200000_57600000_message_{}.csv".format(trading_date, levels)
messages = pd.read_csv(
    path / message_data,
    header=None,
    names=["time", "type", "order_id", "size", "price", "direction"],
)
messages.info()
print(messages.head())

# Around 80% of executions were visible, the remaining 20% were not:
print(messages.type.map(types).value_counts())

messages.time = pd.to_timedelta(messages.time, unit="s")
messages["trading_date"] = pd.to_datetime(trading_date)
messages.time = messages.trading_date.add(messages.time)
messages.drop("trading_date", axis=1, inplace=True)
print(messages.head())

## Combine message and price data
data = pd.concat([messages, orders], axis=1)
data.info()

ex = data[data.type.isin([4, 5])]
print(ex.head())

## Plot limit order prices for messages with visible or hidden execution
cmaps = {"Bid": "Blues", "Ask": "Reds"}
fig, ax = plt.subplots(figsize=(14, 8))
time = ex["time"].dt.to_pydatetime()
for i in range(10):
    for t in ["Bid", "Ask"]:
        y, c = ex["{} Price {}".format(t, i)], ex["{} Size {}".format(t, i)]
        ax.scatter(x=time, y=y, c=c, cmap=cmaps[t], s=1, vmin=1, vmax=c.quantile(0.95))
ax.set_xlim(datetime(2012, 6, 21, 9, 30), datetime(2012, 6, 21, 16, 0))
fig.tight_layout()
plt.savefig("images/03-01.png")
