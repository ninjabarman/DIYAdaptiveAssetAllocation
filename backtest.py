from itertools import combinations
import time

from ffn import calc_stats
from portfolio import *
from constants import *
from optimize import *
import time


def load_df() -> pd.DataFrame:
    return pd.read_csv(ASSET_PRICES_FILE, index_col="Date", parse_dates=["Date"])



def test_optimization_methods():
    covariance_matrix = np.array(
        [
            [77.25607201, 4.95549419, -2.1582784],
            [4.95549419, 73.87388998, -3.76609601],
            [-2.1582784, -3.76609601, 259.46734795],
        ]
    )
    cov = np.array([covariance_matrix])
    vectorized_weights = minimum_variance_optimization(cov)
    print(f"vectorized output once:\t{vectorized_weights}")
    cov = np.tile(covariance_matrix, (5, 1, 1))
    vectorized_weights = minimum_variance_optimization(cov)
    print(f"vectorized output repeated:\t{vectorized_weights}")



def calculate_portfolio_returns():
    df = load_df()

    # portfolio of n assets
    n = df.shape[1]

    # calculate daily returns
    returns = df.pct_change().dropna()

    # calculate daily annualized volatilities
    vols = returns.rolling(f"21D").std()
    vols.columns = ASSET_VOLATILITIES

    # calculate linear weighted correlations
    window_sizes = [21, 63, 126, 252]
    look_back_weights = [12, 4, 2, 1]

    corrs = pd.DataFrame(index=returns.index[252:])

    # first calculate the correlations for each window size
    for window_size, weight in zip(window_sizes, look_back_weights):
        window_size_corrs = returns.rolling(f"{window_size}D").corr().unstack() / weight
        window_size_corrs.columns = [
            f"{ticker1}_{ticker2}_{window_size}D_corr"
            for ticker1, ticker2 in window_size_corrs.columns
        ]
        corrs = corrs.join(window_size_corrs)
    corrs = corrs.dropna()

    # then calculate the weighted average of the correlations
    correlations = pd.DataFrame(index=corrs.index)
    for i, (ticker1, ticker2) in enumerate(
        [(ticker1, ticker2) for ticker1 in TICKERS for ticker2 in TICKERS]
    ):
        # if the correlation is between the same asset, set it to 1
        # to ensure that the covariance matrix is positive definite
        if ticker1 == ticker2:
            correlations[f"{ticker1}_{ticker2}_corr"] = np.ones(corrs.shape[0])
        else:
            correlations[f"{ticker1}_{ticker2}_corr"] = (
                corrs.iloc[:, i::9].sum(axis=1) / 19
            )
    del corrs

    # calculate covariance matrix from correlations and volatilities
    cov = pd.DataFrame(index=correlations.index)
    for ticker1 in TICKERS:
        for ticker2 in TICKERS:
            cov[f"{ticker1}_{ticker2}_cov"] = (
                correlations[f"{ticker1}_{ticker2}_corr"]
                * vols[f"{ticker1}_vol"]
                * vols[f"{ticker2}_vol"]
            )

    cov = cov.dropna()
    # calculate optimal weights for portfolio
    start = time.time()
    expected_returns = returns.rolling(f"63D").mean().iloc[252:].values

    weights = risk_budget_optimization(cov.values.reshape((cov.shape[0], n, n)), expected_returns)
    #weights = minimum_variance_optimization(cov.values.reshape((cov.shape[0], n, n)))
    end = time.time()
    print(f"it took {end - start} seconds to perform the rolling optimization")
    df = df.iloc[253:]
    df[CASH_WEIGHTS] = weights
    returns = return_portfolio(returns, df[CASH_WEIGHTS], verbose=True, rebalance_on="D")
    df[PORTFOLIO_RETURNS] = returns["returns"]
    df[PORTFOLIO_TURNOVER] = returns["Two.Way.Turnover"]
    df[PORTFOLIO_RETURNS_RELATIVE] = 1.0 * (1 + df[PORTFOLIO_RETURNS]).cumprod()
    print(calc_stats(df[PORTFOLIO_RETURNS_RELATIVE]).display())
    print(df[CASH_WEIGHTS].describe())
    return df
