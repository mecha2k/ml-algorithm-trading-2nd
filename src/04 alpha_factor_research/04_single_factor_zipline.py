from zipline import run_algorithm
from zipline.api import (
    attach_pipeline,
    date_rules,
    time_rules,
    order_target_percent,
    pipeline_output,
    record,
    schedule_function,
    get_open_orders,
    calendars,
)
from zipline.finance import commission, slippage
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.factors import Returns, AverageDollarVolume
import pandas as pd

MONTH = 21
YEAR = 12 * MONTH
N_LONGS = N_SHORTS = 25
VOL_SCREEN = 1000


class MeanReversion(CustomFactor):
    """Compute ratio of the latest monthly return to 12m average,
    normalized by std dev of monthly returns"""

    inputs = [Returns(window_length=MONTH)]
    window_length = YEAR

    def compute(self, today, assets, out, monthly_returns):
        df = pd.DataFrame(monthly_returns)
        out[:] = df.iloc[-1].sub(df.mean()).div(df.std())


def compute_factors():
    """Create factor pipeline incl. mean reversion,
    filtered by 30d Dollar Volume; capture factor ranks"""
    mean_reversion = MeanReversion()
    dollar_volume = AverageDollarVolume(window_length=30)
    return Pipeline(
        columns={
            "longs": mean_reversion.bottom(N_LONGS),
            "shorts": mean_reversion.top(N_SHORTS),
            "ranking": mean_reversion.rank(ascending=False),
        },
        screen=dollar_volume.top(VOL_SCREEN),
    )


def exec_trades(data, assets, target_percent):
    # Place orders for assets using target portfolio percentage
    for asset in assets:
        if data.can_trade(asset) and not get_open_orders(asset):
            order_target_percent(asset, target_percent)


def rebalance(context, data):
    # Compute long, short and obsolete holdings; place trade orders
    factor_data = context.factor_data
    record(factor_data=factor_data.ranking)

    assets = factor_data.index
    record(prices=data.current(assets, "price"))

    longs = assets[factor_data.longs]
    shorts = assets[factor_data.shorts]
    divest = set(context.portfolio.positions.keys()) - set(longs.union(shorts))

    exec_trades(data, assets=divest, target_percent=0)
    exec_trades(data, assets=longs, target_percent=1 / N_LONGS)
    exec_trades(data, assets=shorts, target_percent=-1 / N_SHORTS)


def initialize(context):
    # Setup: register pipeline, schedule rebalancing, and set trading params
    attach_pipeline(compute_factors(), "factor_pipeline")
    schedule_function(
        rebalance, date_rules.week_start(), time_rules.market_open(), calendar=calendars.US_EQUITIES
    )
    context.set_commission(commission.PerShare(cost=0.01, min_trade_cost=0))
    context.set_slippage(slippage.VolumeShareSlippage())


def before_trading_start(context, data):
    # Run factor pipeline
    context.factor_data = pipeline_output("factor_pipeline")


if __name__ == "__main__":
    result = run_algorithm(
        start=pd.Timestamp("2015-1-1", tz="UTC"),
        end=pd.Timestamp("2018-1-1", tz="UTC"),
        initialize=initialize,
        capital_base=10_000_000,
        before_trading_start=before_trading_start,
        bundle="quandl",
    )

    print(result)
    result.to_pickle("../data/single_factor.pkl")
