import gzip
import shutil
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from struct import unpack
from collections import namedtuple, Counter, defaultdict
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urljoin
from datetime import timedelta
from time import time
from matplotlib.ticker import FuncFormatter


def format_time(time):
    m, s = divmod(time, 60)
    h, m = divmod(m, 60)
    return f"{h:0>2.0f}:{m:0>2.0f}:{s:0>5.2f}"


def may_be_download(url):
    if not data_path.exists():
        print("Creating directory")
        data_path.mkdir()
    else:
        print("Directory exists")

    filename = data_path / url.split("/")[-1]
    if not filename.exists():
        print("Downloading...", url)
        urlretrieve(url, filename)
    else:
        print("File exists")

    unzipped = data_path / (filename.stem + ".bin")
    if not unzipped.exists():
        print("Unzipping to", unzipped)
        with gzip.open(str(filename), "rb") as f_in:
            with open(unzipped, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print("File already unpacked")

    return unzipped


def clean_message_types(df):
    df.columns = [c.lower().strip() for c in df.columns]
    df.value = df.value.str.strip()
    df.name = (
        df.name.str.strip()  # remove whitespace
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )
    df.notes = df.notes.str.strip()
    df["message_type"] = df.loc[df.name == "message_type", "value"]

    return df


def format_alpha(mtype, data):
    for col in alpha_formats.get(mtype).keys():
        if mtype != "R" and col == "stock":
            data = data.drop(col, axis=1)
            continue
        data.loc[:, col] = data.loc[:, col].str.decode("utf-8").str.strip()
        if encoding.get(col):
            data.loc[:, col] = data.loc[:, col].map(encoding.get(col))
    return data


def store_messages(m):
    with pd.HDFStore(itch_store) as store:
        for mtype, data in m.items():
            # convert to DataFrame
            data = pd.DataFrame(data)
            # parse timestamp info
            data.timestamp = data.timestamp.apply(int.from_bytes, byteorder="big")
            data.timestamp = pd.to_timedelta(data.timestamp)
            # apply alpha formatting
            if mtype in alpha_formats.keys():
                data = format_alpha(mtype, data)

            s = alpha_length.get(mtype)
            if s:
                s = {c: s.get(c) for c in data.columns}
            dc = ["stock_locate"]
            if m == "R":
                dc.append("stock")
            try:
                store.append(mtype, data, format="t", min_itemsize=s, data_columns=dc)
            except Exception as e:
                print(e)
                print(mtype)
                print(data.info())
                print(pd.Series(list(m.keys())).value_counts())
                data.to_csv("data.csv", index=False)
                return 1
    return 0


if __name__ == "__main__":
    sns.set_style("whitegrid")

    data_path = Path("../data")
    itch_store = str(data_path / "itch.h5")
    order_book_store = data_path / "order_book.h5"

    FTP_URL = "ftp://emi.nasdaq.com/ITCH/"
    SOURCE_FILE = "10302019.NASDAQ_ITCH50.gz"

    file_name = may_be_download(urljoin(FTP_URL, SOURCE_FILE))
    date = file_name.name.split(".")[0]

    event_codes = {
        "O": "Start of Messages",
        "S": "Start of System Hours",
        "Q": "Start of Market Hours",
        "M": "End of Market Hours",
        "E": "End of System Hours",
        "C": "End of Messages",
    }

    encoding = {
        "primary_market_maker": {"Y": 1, "N": 0},
        "printable": {"Y": 1, "N": 0},
        "buy_sell_indicator": {"B": 1, "S": -1},
        "cross_type": {"O": 0, "C": 1, "H": 2},
        "imbalance_direction": {"B": 0, "S": 1, "N": 0, "O": -1},
    }

    formats = {
        ("integer", 2): "H",  # int of length 2 => format string 'H'
        ("integer", 4): "I",
        ("integer", 6): "6s",  # int of length 6 => parse as string, convert later
        ("integer", 8): "Q",
        ("alpha", 1): "s",
        ("alpha", 2): "2s",
        ("alpha", 4): "4s",
        ("alpha", 8): "8s",
        ("price_4", 4): "I",
        ("price_8", 8): "Q",
    }

    message_data = (
        pd.read_excel("../data/message_types.xlsx", engine="openpyxl", sheet_name="messages")
        .sort_values("id")
        .drop("id", axis=1)
    )
    print(message_data.head())

    message_types = clean_message_types(message_data)

    message_labels = (
        message_types.loc[:, ["message_type", "notes"]].dropna().rename(columns={"notes": "name"})
    )
    message_labels.name = (
        message_labels.name.str.lower()
        .str.replace("message", "")
        .str.replace(".", "")
        .str.strip()
        .str.replace(" ", "_")
    )
    message_labels.to_csv("../data/message_labels.csv", index=False)
    print(message_labels.head())

    message_types.message_type = message_types.message_type.ffill()
    message_types = message_types[message_types.name != "message_type"]
    message_types.value = (
        message_types.value.str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )
    print(message_types.info())
    print(message_types.head())

    message_types.to_csv("../data/message_types.csv", index=False)
    message_types = pd.read_csv("../data/message_types.csv")
    message_types.loc[:, "formats"] = (
        message_types[["value", "length"]].apply(tuple, axis=1).map(formats)
    )

    alpha_fields = message_types[message_types.value == "alpha"].set_index("name")
    alpha_msgs = alpha_fields.groupby("message_type")
    alpha_formats = {k: v.to_dict() for k, v in alpha_msgs.formats}
    alpha_length = {k: v.add(5).to_dict() for k, v in alpha_msgs.length}

    message_fields, fstring = {}, {}
    for t, message in message_types.groupby("message_type"):
        message_fields[t] = namedtuple(typename=t, field_names=message.name.tolist())
        fstring[t] = ">" + "".join(message.formats.tolist())
    print(alpha_fields.info())
    print(alpha_fields.head())

    messages = defaultdict(list)
    message_count = 0
    message_type_counter = Counter()

    start = time()
    with file_name.open("rb") as data:
        while True:
            # determine message size in bytes
            message_size = int.from_bytes(data.read(2), byteorder="big", signed=False)
            # get message type by reading first byte
            message_type = data.read(1).decode("ascii")
            message_type_counter.update([message_type])
            # read & store message
            record = data.read(message_size - 1)
            message = message_fields[message_type]._make(unpack(fstring[message_type], record))
            messages[message_type].append(message)
            # deal with system events
            if message_type == "S":
                seconds = int.from_bytes(message.timestamp, byteorder="big") * 1e-9
                print("\n", event_codes.get(message.event_code.decode("ascii"), "Error"))
                print(f"\t{format_time(seconds)}\t{message_count:12,.0f}")
                if message.event_code.decode("ascii") == "C":
                    store_messages(messages)
                    break
            message_count += 1
            if message_count % 2.5e7 == 0:
                seconds = int.from_bytes(message.timestamp, byteorder="big") * 1e-9
                d = format_time(time() - start)
                print(f"\t{format_time(seconds)}\t{message_count:12,.0f}\t{d}")
                res = store_messages(messages)
                if res == 1:
                    print(pd.Series(dict(message_type_counter)).sort_values())
                    break
                messages.clear()
    print("Duration:", format_time(time() - start))

    counter = pd.Series(message_type_counter).to_frame("# Trades")
    counter["Message Type"] = counter.index.map(
        message_labels.set_index("message_type").name.to_dict()
    )
    counter = counter[["Message Type", "# Trades"]].sort_values("# Trades", ascending=False)
    print(counter)

    with pd.HDFStore(itch_store) as store:
        store.put("summary", counter)

    with pd.HDFStore(itch_store) as store:
        stocks = store["R"].loc[:, ["stock_locate", "stock"]]
        trades = (
            store["P"]
            .append(store["Q"].rename(columns={"cross_price": "price"}), sort=False)
            .merge(stocks)
        )

    trades["value"] = trades.shares.mul(trades.price)
    trades["value_share"] = trades.value.div(trades.value.sum())

    trade_summary = trades.groupby("stock").value_share.sum().sort_values(ascending=False)
    trade_summary.iloc[:50].plot.bar(
        figsize=(14, 6), color="darkblue", title="Share of Traded Value"
    )

    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    sns.despine()
    plt.tight_layout()

    with pd.HDFStore(itch_store) as store:
        stocks = store["R"].loc[:, ["stock_locate", "stock"]]
        trades = (
            store["P"]
            .append(store["Q"].rename(columns={"cross_price": "price"}), sort=False)
            .merge(stocks)
        )

    trades["value"] = trades.shares.mul(trades.price)
    trades["value_share"] = trades.value.div(trades.value.sum())

    trade_summary = trades.groupby("stock").value_share.sum().sort_values(ascending=False)
    trade_summary.iloc[:50].plot.bar(
        figsize=(14, 6), color="darkblue", title="Share of Traded Value"
    )

    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    sns.despine()
    plt.tight_layout()
