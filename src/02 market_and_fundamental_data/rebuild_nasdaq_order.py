import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from pathlib import Path
from collections import Counter
from datetime import timedelta
from datetime import datetime
from time import time


def format_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f"{h:0>2.0f}:{m:0>2.0f}:{s:0>5.2f}"


def get_messages(date, stock=stock):
    """Collect trading messages for given stock"""
    with pd.HDFStore(itch_store) as store:
        stock_locate = store.select("R", where="stock = stock").stock_locate.iloc[0]
        target = "stock_locate = stock_locate"

        data = {}
        # trading message types
        messages = ["A", "F", "E", "C", "X", "D", "U", "P", "Q"]
        for m in messages:
            data[m] = store.select(m, where=target).drop("stock_locate", axis=1).assign(type=m)

    order_cols = ["order_reference_number", "buy_sell_indicator", "shares", "price"]
    orders = pd.concat([data["A"], data["F"]], sort=False, ignore_index=True).loc[:, order_cols]

    for m in messages[2:-3]:
        data[m] = data[m].merge(orders, how="left")

    data["U"] = data["U"].merge(
        orders,
        how="left",
        right_on="order_reference_number",
        left_on="original_order_reference_number",
        suffixes=["", "_replaced"],
    )

    data["Q"].rename(columns={"cross_price": "price"}, inplace=True)
    data["X"]["shares"] = data["X"]["cancelled_shares"]
    data["X"] = data["X"].dropna(subset=["price"])

    data = pd.concat([data[m] for m in messages], ignore_index=True, sort=False)
    data["date"] = pd.to_datetime(date, format="%m%d%Y")
    data.timestamp = data["date"].add(data.timestamp)
    data = data[data.printable != 0]

    drop_cols = [
        "tracking_number",
        "order_reference_number",
        "original_order_reference_number",
        "cross_type",
        "new_order_reference_number",
        "attribution",
        "match_number",
        "printable",
        "date",
        "cancelled_shares",
    ]

    return data.drop(drop_cols, axis=1).sort_values("timestamp").reset_index(drop=True)


def get_trades(m):
    trade_dict = {"executed_shares": "shares", "execution_price": "price"}
    cols = ["timestamp", "executed_shares"]
    trades = (
        pd.concat(
            [
                m.loc[m.type == "E", cols + ["price"]].rename(columns=trade_dict),
                m.loc[m.type == "C", cols + ["execution_price"]].rename(columns=trade_dict),
                m.loc[m.type == "P", ["timestamp", "price", "shares"]],
                m.loc[m.type == "Q", ["timestamp", "price", "shares"]].assign(cross=1),
            ],
            sort=False,
        )
        .dropna(subset=["price"])
        .fillna(0)
    )
    return trades.set_index("timestamp").sort_index().astype(int)


def add_orders(orders, buysell, nlevels):
    """Add orders up to desired depth given by nlevels;
    sell in ascending, buy in descending order
    """
    new_order = []
    items = sorted(orders.copy().items())
    if buysell == 1:
        items = reversed(items)
    for i, (p, s) in enumerate(items, 1):
        new_order.append((p, s))
        if i == nlevels:
            break
    return orders, new_order


def save_orders(orders, append=False):
    cols = ["price", "shares"]
    for buysell, book in orders.items():
        df = pd.concat(
            [pd.DataFrame(data=data, columns=cols).assign(timestamp=t) for t, data in book.items()]
        )
        key = f"{stock}/{order_dict[buysell]}"
        df.loc[:, ["price", "shares"]] = df.loc[:, ["price", "shares"]].astype(int)
        with pd.HDFStore(order_book_store) as store:
            if append:
                store.append(key, df.set_index("timestamp"), format="t")
            else:
                store.put(key, df.set_index("timestamp"))


if __name__ == "__main__":
    sns.set_style("whitegrid")

    data_path = Path("../data")
    itch_store = str(data_path / "itch.h5")
    order_book_store = data_path / "order_book.h5"
    date = "10302019"
    stock = "AAPL"
    order_dict = {-1: "sell", 1: "buy"}

    messages = get_messages(date=date)
    print(messages.info(null_counts=True))

    with pd.HDFStore(order_book_store) as store:
        key = f"{stock}/messages"
        store.put(key, messages)
        print(store.info())

    trades = get_trades(messages)
    print(trades.info())

    with pd.HDFStore(order_book_store) as store:
        store.put(f"{stock}/trades", trades)

    order_book = {-1: {}, 1: {}}
    current_orders = {-1: Counter(), 1: Counter()}
    message_counter = Counter()
    nlevels = 100

    start = time()
    for message in messages.itertuples():
        i = message[0]
        if i % 1e5 == 0 and i > 0:
            print(f"{i:,.0f}\t\t{format_time(time() - start)}")
            save_orders(order_book, append=True)
            order_book = {-1: {}, 1: {}}
            start = time()
        if np.isnan(message.buy_sell_indicator):
            continue
        message_counter.update(message.type)

        buysell = message.buy_sell_indicator
        price, shares = None, None

        if message.type in ["A", "F", "U"]:
            price = int(message.price)
            shares = int(message.shares)

            current_orders[buysell].update({price: shares})
            current_orders[buysell], new_order = add_orders(
                current_orders[buysell], buysell, nlevels
            )
            order_book[buysell][message.timestamp] = new_order

        if message.type in ["E", "C", "X", "D", "U"]:
            if message.type == "U":
                if not np.isnan(message.shares_replaced):
                    price = int(message.price_replaced)
                    shares = -int(message.shares_replaced)
            else:
                if not np.isnan(message.price):
                    price = int(message.price)
                    shares = -int(message.shares)

            if price is not None:
                current_orders[buysell].update({price: shares})
                if current_orders[buysell][price] <= 0:
                    current_orders[buysell].pop(price)
                current_orders[buysell], new_order = add_orders(
                    current_orders[buysell], buysell, nlevels
                )
                order_book[buysell][message.timestamp] = new_order

    message_counter = pd.Series(message_counter)
    print(message_counter)

    with pd.HDFStore(order_book_store) as store:
        print(store.info())

    with pd.HDFStore(order_book_store) as store:
        buy = store[f"{stock}/buy"].reset_index().drop_duplicates()
        sell = store[f"{stock}/sell"].reset_index().drop_duplicates()

    buy.price = buy.price.mul(1e-4)
    sell.price = sell.price.mul(1e-4)

    percentiles = [0.01, 0.02, 0.1, 0.25, 0.75, 0.9, 0.98, 0.99]
    pd.concat(
        [
            buy.price.describe(percentiles=percentiles).to_frame("buy"),
            sell.price.describe(percentiles=percentiles).to_frame("sell"),
        ],
        axis=1,
    )

    buy = buy[buy.price > buy.price.quantile(0.01)]
    sell = sell[sell.price < sell.price.quantile(0.99)]

    market_open = "0930"
    market_close = "1600"

    fig, ax = plt.subplots(figsize=(7, 5))
    hist_kws = {"linewidth": 1, "alpha": 0.5}
    sns.distplot(
        buy[buy.price.between(240, 250)]
        .set_index("timestamp")
        .between_time(market_open, market_close)
        .price,
        ax=ax,
        label="Buy",
        kde=False,
        hist_kws=hist_kws,
    )
    sns.distplot(
        sell[sell.price.between(240, 250)]
        .set_index("timestamp")
        .between_time(market_open, market_close)
        .price,
        ax=ax,
        label="Sell",
        kde=False,
        hist_kws=hist_kws,
    )

    ax.legend(fontsize=10)
    ax.set_title("Limit Order Price Distribution")
    ax.set_yticklabels([f"{int(y / 1000):,}" for y in ax.get_yticks().tolist()])
    ax.set_xticklabels([f"${int(x):,}" for x in ax.get_xticks().tolist()])
    ax.set_xlabel("Price")
    ax.set_ylabel("Shares ('000)")
    sns.despine()
    fig.tight_layout()

    utc_offset = timedelta(hours=4)
    depth = 100

    buy_per_min = (
        buy.groupby([pd.Grouper(key="timestamp", freq="Min"), "price"])
        .shares.sum()
        .apply(np.log)
        .to_frame("shares")
        .reset_index("price")
        .between_time(market_open, market_close)
        .groupby(level="timestamp", as_index=False, group_keys=False)
        .apply(lambda x: x.nlargest(columns="price", n=depth))
        .reset_index()
    )
    buy_per_min.timestamp = buy_per_min.timestamp.add(utc_offset).astype(int)
    print(buy_per_min.info())

    sell_per_min = (
        sell.groupby([pd.Grouper(key="timestamp", freq="Min"), "price"])
        .shares.sum()
        .apply(np.log)
        .to_frame("shares")
        .reset_index("price")
        .between_time(market_open, market_close)
        .groupby(level="timestamp", as_index=False, group_keys=False)
        .apply(lambda x: x.nsmallest(columns="price", n=depth))
        .reset_index()
    )

    sell_per_min.timestamp = sell_per_min.timestamp.add(utc_offset).astype(int)
    print(sell_per_min.info())

    with pd.HDFStore(order_book_store) as store:
        trades = store[f"{stock}/trades"]
    trades.price = trades.price.mul(1e-4)
    trades = trades[trades.cross == 0].between_time(market_open, market_close)

    trades_per_min = trades.resample("Min").agg({"price": "mean", "shares": "sum"})
    trades_per_min.index = trades_per_min.index.to_series().add(utc_offset).astype(int)
    trades_per_min.info()

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(14, 6))

    buy_per_min.plot.scatter(
        x="timestamp", y="price", c="shares", ax=ax, colormap="Blues", colorbar=False, alpha=0.25
    )

    sell_per_min.plot.scatter(
        x="timestamp", y="price", c="shares", ax=ax, colormap="Reds", colorbar=False, alpha=0.25
    )

    title = f"AAPL | {date} | Buy & Sell Limit Order Book | Depth = {depth}"
    trades_per_min.price.plot(figsize=(14, 8), c="k", ax=ax, lw=2, title=title)

    xticks = [datetime.fromtimestamp(ts / 1e9).strftime("%H:%M") for ts in ax.get_xticks()]
    ax.set_xticklabels(xticks)

    ax.set_xlabel("")
    ax.set_ylabel("Price", fontsize=12)

    red_patch = mpatches.Patch(color="red", label="Sell")
    blue_patch = mpatches.Patch(color="royalblue", label="Buy")

    plt.legend(handles=[red_patch, blue_patch])
    sns.despine()
    fig.tight_layout()
