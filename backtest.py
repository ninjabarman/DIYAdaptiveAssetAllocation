from os.path import exists

import numpy as np
import pandas as pd
from pyrb import RiskBudgeting

from yahoo_data import ASSET_PRICES_FILE, DATA_DIR, get_tickers

DEFAULT_START_DATE = "1993-04-28"

DEFAULT_END_DATE = "2022-11-10"

#  [Risk weight for stocks, bonds, and commodities]
POLICY_RISK_BUDGET = [.8, .1, .1]

TARGET_WEIGHTS_FILE = f"{DATA_DIR}/target_weights.csv"


def get_returns():
    prices = pd.read_csv(ASSET_PRICES_FILE)
    returns = np.log(prices[get_tickers()] / prices.shift(1)[get_tickers()]).dropna()
    returns["Date"] = pd.to_datetime(prices["Date"]).dt.to_period(freq="D")
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
        returns.head()
        cov = returns.rolling(f"{look_back_period}D").cov().dropna()
        weights = pd.DataFrame(columns=get_tickers())
        weights[get_tickers()] = cov.groupby(level=0).apply(
            lambda c: pd.Series([weight for weight in calculate_weights(c)])
        )
        weights.head()
        weights.to_csv(TARGET_WEIGHTS_FILE)
    return pd.read_csv(TARGET_WEIGHTS_FILE, index_col="Date", parse_dates=["Date"])
