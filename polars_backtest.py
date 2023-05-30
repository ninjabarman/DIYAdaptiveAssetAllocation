import time
from datetime import date

import numpy as np
import pandas as pd
import polars as pl
from ffn import calc_stats

from portfolio import return_portfolio
from yahoo_data import ASSET_PRICES_FILE


def calculate_cash_weights(covar_struct, tickers, risk_budget):
    n = len(tickers)
    covars = [covar_struct[f"{ticker_1}/{ticker_2}_covar"]
              for ticker_1 in tickers for ticker_2 in tickers]
    cov = np.array(covars).reshape((n, n))
    risk_contributions = np.dot(cov, risk_budget)
    total_risk_contribution = np.sum(risk_contributions)
    weights = risk_contributions / total_risk_contribution
    weights = np.maximum(weights, 0)
    weights /= np.sum(weights)
    return tuple(weights)


def back_test_portfolio(tickers, risk_budget=None, start_date=date(1994, 1, 1),
                        end_date=date(2023, 5, 1)):
    if risk_budget is None:
        n = len(tickers)
        # Carvers golden commandment: when in doubt, thou shalt use equal risk weights
        risk_budget = [1 / n] * n
    # Download price data using yfinance
    # Create a DataFrame from the price data
    df = pl.read_csv(ASSET_PRICES_FILE).lazy()
    st = time.time()
    df.with_columns(pl.col("Date").str.to_date())
    # Calculate daily returns for each ticker
    df = calculate_asset_returns(df, tickers)
    # Calculate 20-day volatilities
    df = calculate_volatility(df, tickers)
    # Calculate pairwise correlations
    df = calculate_correlations(df, tickers)
    # remove rows with empty values
    df = df.drop_nulls()
    # compute the covariance matrix
    df = calculate_covariances(df, tickers)
    df = assign_asset_weights(df, tickers, risk_budget)
    df = df.collect()
    et = time.time()
    rets_exp = [pl.col(f"{ticker}_Return") for ticker in tickers]
    weights_exp = [pl.col(f"{ticker}_weight") for ticker in tickers]
    dates = df["Date"].to_pandas()

    asset_returns = df.select(rets_exp).to_pandas()
    asset_weights = df.select(weights_exp).to_pandas()
    asset_returns["Date"] = pd.to_datetime(dates)
    asset_weights["Date"] = pd.to_datetime(dates)
    asset_returns.set_index("Date", inplace=True)
    asset_weights.set_index("Date", inplace=True)

    returns = return_portfolio(asset_returns, asset_weights, rebalance_on="D")
    print(f"Took {et - st} seconds to backtest the portfolio")
    return returns


def calculate_asset_returns(df, tickers):
    return df.with_columns([(pl.col(ticker) / pl.col(ticker).shift(1) - 1)
                           .alias(f"{ticker}_Return") for ticker in tickers])


def assign_asset_weights(df, tickers, risk_budget):
    df = df.with_columns([
        pl.struct(
            pl.col(f"{ticker_1}/{ticker_2}_covar")
            for ticker_1 in tickers for ticker_2 in tickers).apply(
            lambda struct: calculate_cash_weights(struct, tickers, risk_budget)).alias("weight_struct")
    ])
    df = df.with_columns([
        pl.col("weight_struct").arr.get(i).alias(f"{ticker}_weight") for i, ticker in enumerate(tickers)
    ])
    return df


def calculate_covariances(df, tickers):
    return df.with_columns([
        pl.col(f"{ticker_1}_Vol").mul(pl.col(f"{ticker_2}_Vol")).mul(pl.col(f"{ticker_1}/{ticker_2}")).alias(
            f"{ticker_1}/{ticker_2}_covar")
        for ticker_1 in tickers for ticker_2 in tickers
    ])


def calculate_correlations(df, tickers, look_back=125):
    return df.with_columns([
        pl.rolling_corr(pl.col(f"{ticker_1}_Return"), pl.col(f"{ticker_2}_Return"), window_size=look_back).alias(
            f"{ticker_1}/{ticker_2}")
        for ticker_1 in tickers for ticker_2 in tickers
    ])


def calculate_volatility(df, tickers, look_back=20):
    return df.with_columns([
        pl.col(f"{ticker}_Return").rolling_std(window_size=look_back).alias(f"{ticker}_Vol") for ticker in tickers
    ])
