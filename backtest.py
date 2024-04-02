from itertools import combinations
import time

from ffn import calc_stats
from portfolio import *
from constants import *
from optimize import *
from signals import *
import time


def trim_df_to_match(df1, df2) -> tuple:
    '''
    Trim two dataframes to match on index
    '''
    common_index = df1.index.intersection(df2.index)
    df1 = df1.loc[common_index]
    df2 = df2.loc[common_index]
    return df1, df2
    

def calculate_covariances(returns: pd.DataFrame) -> pd.DataFrame:
    # calculate daily annualized volatilities
    vols = returns.rolling(f"21D").std()
    vols.columns = ASSET_VOLATILITIES
    
    # calculate linear weighted correlations
    window_sizes = [21, 63, 126, 252]
    look_back_weights = [12, 4, 2, 1]

    window_corrs = pd.DataFrame(index=returns.index[252:])

    # first calculate the correlations for each window size
    for window_size, weight in zip(window_sizes, look_back_weights):
        window_size_corrs = returns.rolling(f"{window_size}D").corr().unstack() / weight
        window_size_corrs.columns = [
            f"{ticker1}_{ticker2}_{window_size}D_corr"
            for ticker1, ticker2 in window_size_corrs.columns
        ]
        window_corrs = window_corrs.join(window_size_corrs)
    window_corrs = window_corrs.dropna()

    # then calculate the weighted average of the correlations
    correlations = pd.DataFrame(index=window_corrs.index)
    for i, (ticker1, ticker2) in enumerate(
        [(ticker1, ticker2) for ticker1 in TICKERS for ticker2 in TICKERS]
    ):
        # if the correlation is between the same asset, set it to 1
        # to ensure that the covariance matrix is positive definite
        if ticker1 == ticker2:
            correlations[f"{ticker1}_{ticker2}_corr"] = np.ones(window_corrs.shape[0])
        else:
            correlations[f"{ticker1}_{ticker2}_corr"] = (
                window_corrs.iloc[:, i::9].sum(axis=1) / 19
            )
    del window_corrs
    
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
    
    return cov


def load_df() -> pd.DataFrame:
    return pd.read_csv(ASSET_PRICES_FILE, index_col="Date", parse_dates=["Date"])[TICKERS]


def calculate_portfolio_returns():
    df = load_df()

    # portfolio of n assets
    n = df.shape[1]

    # calculate daily returns
    returns = df.pct_change().dropna()

    cov = calculate_covariances(returns)
    # calculate optimal weights for portfolio
    start = time.time()

    risk_budget = stock_bond_cross_asset_momentum(returns)
    #risk_budget = trend_risk_weights(returns)
    risk_budget, cov = trim_df_to_match(risk_budget, cov)

    weights = risk_budget_optimization(
        cov.values.reshape((cov.shape[0], n, n)), risk_budget.values
    )
    end = time.time()
    print(f"it took {end - start} seconds to perform the rolling optimization")
    df = df.iloc[253:]
    df[CASH_WEIGHTS] = weights
    df[CASH_WEIGHTS] = df[CASH_WEIGHTS].rolling(f"252D").mean()
    df = df.iloc[252:]
    portfolio_returns = return_portfolio(
        returns, df[CASH_WEIGHTS], verbose=True, rebalance_on="W"
    )
    df[PORTFOLIO_RETURNS] = portfolio_returns["returns"]
    df[PORTFOLIO_TURNOVER] = portfolio_returns["Two.Way.Turnover"]
    df[CASH_WEIGHTS] = portfolio_returns["BOP.Weight"][CASH_WEIGHTS]
    df[PORTFOLIO_RETURNS_RELATIVE] = 1.0 * (1 + df[PORTFOLIO_RETURNS]).cumprod()
    print(calc_stats(df[PORTFOLIO_RETURNS_RELATIVE]).display())
    print(df[CASH_WEIGHTS].describe())
    return df
