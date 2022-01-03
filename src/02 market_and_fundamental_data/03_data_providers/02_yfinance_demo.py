import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import warnings

warnings.filterwarnings("ignore")

symbol = "FB"
ticker = yf.Ticker(symbol)
print(pd.Series(ticker.info).head(20))

data = ticker.history(
    period="5d",
    interval="1m",
    start=None,
    end=None,
    actions=True,
    auto_adjust=True,
    back_adjust=False,
)
data.info()

### View company actions
# show actions (dividends, splits)
print(ticker.actions)
print(ticker.dividends)
print(ticker.splits)

### Annual and Quarterly Financial Statement Summary
print(ticker.financials)
print(ticker.quarterly_financials)

### Annual and Quarterly Balance Sheet
print(ticker.balance_sheet)
print(ticker.quarterly_balance_sheet)

### Annual and Quarterly Cashflow Statement
print(ticker.cashflow)
print(ticker.quarterly_cashflow)
print(ticker.earnings)
print(ticker.quarterly_earnings)

### Sustainability: Environmental, Social and Governance (ESG)
print(ticker.sustainability)

### Analyst Recommendations
ticker.recommendations.info()
print(ticker.recommendations.tail(10))

### Upcoming Events
print(ticker.calendar)

### Option Expiration Dates
print(ticker.options)

expiration = ticker.options[0]
options = ticker.option_chain(expiration)
options.calls.info()
print(options.calls.head())
options.puts.info()

## Data Download with proxy server
# You can use a proxy server to avoid having your IP blacklisted as illustrated below (but need an actual PROXY_SERVER).
# PROXY_SERVER = "PROXY_SERVER"

# The following will only work with proper PROXY_SERVER...
# msft = yf.Ticker("MSFT")
# msft.history(proxy=PROXY_SERVER)
# msft.get_actions(proxy=PROXY_SERVER)
# msft.get_dividends(proxy=PROXY_SERVER)
# msft.get_splits(proxy=PROXY_SERVER)
# msft.get_balance_sheet(proxy=PROXY_SERVER)
# msft.get_cashflow(proxy=PROXY_SERVER)
# msgt.option_chain(proxy=PROXY_SERVER)

## Downloading multiple symbols
tickers = yf.Tickers("msft aapl goog")
print(tickers)

pd.Series(tickers.tickers["MSFT"].info)
print(tickers.tickers["AAPL"].history(period="1mo"))
print(tickers.history(period="1mo").stack(-1))

data = yf.download("SPY AAPL", start="2020-01-01", end="2020-01-05")
data.info()
data = yf.download(
    tickers="SPY AAPL MSFT",  # list or string
    # use "period" instead of start/end
    # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    # (optional, default is '1mo')
    period="5d",
    # fetch data by interval (including intraday if period < 60 days)
    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    # (optional, default is '1d')
    interval="1m",
    # group by ticker (to access via data['SPY'])
    # (optional, default is 'column')
    group_by="ticker",
    # adjust all OHLC automatically
    # (optional, default is False)
    auto_adjust=True,
    # download pre/post regular market hours data
    # (optional, default is False)
    prepost=True,
    # use threads for mass downloading? (True/False/Integer)
    # (optional, default is True)
    threads=True,
    # proxy URL scheme use use when downloading?
    # (optional, default is None)
    proxy=None,
)
data.info()


yf.pdr_override()
# download dataframe
data = pdr.get_data_yahoo("SPY", start="2017-01-01", end="2019-04-30", auto_adjust=False)
auto_adjust = True
print(data.tail())

auto_adjust = False
print(data.tail())
