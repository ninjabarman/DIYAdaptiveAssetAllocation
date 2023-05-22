import time

import pandas as pd
from ffn import calc_stats
from pyrb import RiskBudgeting

from yahoo_data import ASSET_PRICES_FILE, DATA_DIR, TICKERS

TARGET_WEIGHTS_FILE = f"{DATA_DIR}/target_weights.csv"
ASSET_RETURNS = [f"{ticker}_return" for ticker in TICKERS]
RISK_WEIGHTS = [f"{ticker}_risk_weight" for ticker in TICKERS]
CASH_WEIGHTS = [f"{ticker}_cash_weight" for ticker in TICKERS]
TSMOM = [f"{ticker}_12_month_minus_1_month_return" for ticker in TICKERS]
PORTFOLIO_RETURNS = "Daily Portfolio Returns"
PORTFOLIO_PRICE_RELATIVE = "Portfolio Price Relative"
COVARIANCES = [f"{ticker1}/{ticker2}" for ticker1 in TICKERS for ticker2 in TICKERS]


def load_df():
    return pd.read_csv(ASSET_PRICES_FILE, index_col="Date", parse_dates=["Date"])


def calculate_portfolio_returns(look_back_period=60):
    st = time.time()
    df = load_df()
    df = load_df()
    df[ASSET_RETURNS] = df[TICKERS].pct_change()
    df[TSMOM] = df[ASSET_RETURNS].shift(21).rolling("231D").apply(lambda x: (1 + x).prod() - 1, raw=True)
    df[RISK_WEIGHTS] = df[TSMOM].apply(lambda row: risk_weight(row), axis=1, result_type='expand', raw=True)
    df[COVARIANCES] = df[ASSET_RETURNS].rolling(f"{look_back_period}D").cov().unstack()
    df = df.dropna()
    df[CASH_WEIGHTS] = df[RISK_WEIGHTS + COVARIANCES].apply(lambda row: cash_weights(row), axis=1, result_type='expand')
    df = df.dropna()
    print(df[CASH_WEIGHTS])
    df = df.dropna()
    df[PORTFOLIO_RETURNS] = (df[ASSET_RETURNS].to_numpy() * df[CASH_WEIGHTS].shift(-1).to_numpy()).sum(axis=1)
    df[PORTFOLIO_PRICE_RELATIVE] = 1.0 * (1 + df[PORTFOLIO_RETURNS]).cumprod()
    calc_stats(df[PORTFOLIO_PRICE_RELATIVE]).display()
    et = time.time()
    print(f"Took {et - st} seconds to backtest the portfolio")


def risk_weight(row):
    max_value = row.max()
    min_value = row.min()
    new_row = [1 / 3 + 1 / 6 if value == max_value else 1 / 3 - 1 / 6 if value == min_value else 1 / 3 for value in row]
    return new_row


def cash_weights(row):
    n = len(TICKERS)
    cov = row[n:].array.reshape(3, 3)
    rb = RiskBudgeting(cov, row[0:n])
    rb.solve()
    return list(rb.x)
