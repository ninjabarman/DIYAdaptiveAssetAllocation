from datetime import timedelta
from os.path import exists

import numpy as np
import pandas as pd
from numba import jit
from ffn import calc_stats
from pyrb import RiskBudgeting

from yahoo_data import ASSET_PRICES_FILE, DATA_DIR, get_tickers

#  [Risk weight for stocks, bonds, and commodities]
POLICY_RISK_BUDGET = [.8, .1, .1]

TARGET_WEIGHTS_FILE = f"{DATA_DIR}/target_weights.csv"


def get_returns():
    prices = pd.read_csv(ASSET_PRICES_FILE)
    returns = prices[get_tickers()].pct_change()
    returns["Date"] = pd.to_datetime(prices["Date"])
    returns.set_index("Date", inplace=True)
    return returns


def calculate_weights(cov):
    try:
        rb = RiskBudgeting(cov, POLICY_RISK_BUDGET)
        rb.solve()
        return tuple(rb.x)
    except ValueError:
        return [None, None, None]


def generate_portfolio_weights(look_back_period=60, recalculate=False):
    if not exists(TARGET_WEIGHTS_FILE) or recalculate:
        returns = get_returns()
        cov = returns.rolling(f"{look_back_period}D").cov().dropna()
        weights = pd.DataFrame(columns=get_tickers())
        weights[get_tickers()] = cov.groupby(level=0).apply(
            lambda c: pd.Series([weight for weight in calculate_weights(c)])
        )
        weights.to_csv(TARGET_WEIGHTS_FILE)
    return pd.read_csv(TARGET_WEIGHTS_FILE, index_col="Date", parse_dates=["Date"])


def calculate_portfolio_returns(look_back_period=60, holding_period=20):
    returns = get_returns()
    weights = calculate_portfolio_weights_with_signals(returns, look_back_period)
    portfolio_returns = sum(
        [returns[ticker] * weights.shift(-1)[ticker] 
         for ticker in get_tickers()]).dropna()
    strat_price = 1.0 * (1 + portfolio_returns).cumprod()
    strat_price.index = strat_price.index.astype('datetime64[ns]')
    calc_stats(strat_price).display()
    return portfolio_returns


@jit
def tsmom_risk_weight(row):
    max_value = row.max()
    min_value = row.min()
    new_row = [1/3 + 1/6 if value == max_value else 1/3 - 1/6 if value == min_value else 1/3 for value in row]
    return new_row


def calculate_portfolio_weights_with_signals(returns: pd.DataFrame, look_back_period=60):
    tsmom = returns.shift(21).rolling("231D").apply(lambda x: (1+x).prod()-1, raw=True)
    tsmom_risk_weights = tsmom.apply(lambda row: tsmom_risk_weight(row), axis=1, result_type='expand', raw=True)
    tsmom_risk_weights.columns = get_tickers()
    tsmom_weights = pd.DataFrame().reindex_like(returns)
    for row in returns[look_back_period:].itertuples():
        date = row.Index
        risk_budget = tsmom_risk_weights.loc[date]
        lookback_returns = returns[date - pd.Timedelta(days=look_back_period): date]
        cov = lookback_returns.cov()
        try:
            rb = RiskBudgeting(cov, risk_budget)
            rb.solve()
        except ValueError:
            print(date)
        for i in range(0, len(get_tickers())):
            tsmom_weights.loc[date][get_tickers()[i]] = rb.x[i]
    return tsmom_weights

