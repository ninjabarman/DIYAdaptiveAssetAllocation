from datetime import timedelta
from os.path import exists

import numpy as np
import pandas as pd
from ffn import calc_stats
from pyrb import RiskBudgeting
from pyfolio import create_full_tear_sheet, create_returns_tear_sheet, create_simple_tear_sheet

from yahoo_data import ASSET_PRICES_FILE, DATA_DIR, get_tickers

#  [Risk weight for stocks, bonds, and commodities]
POLICY_RISK_BUDGET = [.6, .2, .2]

TARGET_WEIGHTS_FILE = f"{DATA_DIR}/target_weights.csv"


def get_returns():
    prices = pd.read_csv(ASSET_PRICES_FILE)
    returns = prices[get_tickers()].pct_change()
    returns["Date"] = pd.to_datetime(prices["Date"])#.dt.to_period(freq="D")
    returns.set_index("Date", inplace=True)
    return returns


def calculate_weights(cov):
    try:
        rb = RiskBudgeting(cov, POLICY_RISK_BUDGET)
        rb.solve()
        return tuple(rb.x)
    except ValueError:
        return [None, None, None]


def generate_portfolio_weights(look_back_period=60):
    if not exists(TARGET_WEIGHTS_FILE):
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
    weights = generate_portfolio_weights(look_back_period).shift(1)
    #returns = returns.resample(f"{holding_period}D").sum()
    #returns = np.exp(returns) - 1
    #weights = returns.resample(f"{holding_period}D").mean().shift(1)
    portfolio_returns = []
    for index, row in returns[2:].iterrows():
        s = 0
        for ticker in get_tickers():
            s += row[ticker] * weights.loc[index][ticker]
        portfolio_returns.append({"Date": index, "Portfolio_return": s})
    portfolio_returns = pd.DataFrame(portfolio_returns).set_index(["Date"])
    strat_price = 1.0 * (1 + portfolio_returns["Portfolio_return"]).cumprod()
    strat_price.index = strat_price.index.astype('datetime64[ns]')
    calc_stats(strat_price).display()
    return portfolio_returns
