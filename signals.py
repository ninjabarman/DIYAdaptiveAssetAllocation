import pandas as pd
import numpy as np

MOMENTUM_LOOKBACK_PERIODS = [63, 126, 189, 252]


def momentum(df: pl.DataFrame, tickers, k=3):
    # compute the average returns for each asset for each lookback window
    rolling_returns = df.select(
        pl.col(f"{ticker}_Return").rolling_mean(window_size).alias(f"{ticker}_{window_size}d_Avg_Return")
        for ticker in tickers for window_size in MOMENTUM_LOOKBACK_PERIODS
    )

    # compute the average of average returns for each look back window for each asset
    rolling_returns = rolling_returns.select(
        pl.concat_list([f"{ticker}_{window_size}d_Avg_Return" for window_size in
                        MOMENTUM_LOOKBACK_PERIODS]).arr.mean().alias(f"{ticker}_avg_return")
        for ticker in tickers
    )

    rank_exp = pl.all().rank("ordinal")

    momentum_rank = rolling_returns.select(
        pl.concat_list(pl.all()).arr.eval(rank_exp, parallel=True)
    )




    momentum_rank = momentum_rank.select([
        pl.all().arr.get(i).alias(f"{ticker}")
        for i, ticker in enumerate(tickers)
    ])

    momentum_rank = momentum_rank.select([
        pl.when(pl.col(ticker).is_in(pl.all().top_k(k))).then(1).when(
            pl.col(ticker).is_in(pl.all().bottom_k(k))).then(-1).otherwise(0).alias(f"{ticker}_rank")
        for ticker in tickers
    ])

    return momentum_rank
