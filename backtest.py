import time

from ffn import calc_stats
from pyrb import RiskBudgeting

from portfolio import *
from yahoo_data import ASSET_PRICES_FILE, DATA_DIR, TICKERS

TARGET_WEIGHTS_FILE = f"{DATA_DIR}/target_weights.csv"
ASSET_RETURNS = [f"{ticker}_return" for ticker in TICKERS]
RISK_WEIGHTS = [f"{ticker}_risk_weight" for ticker in TICKERS]
CASH_WEIGHTS = [f"{ticker}_cash_weight" for ticker in TICKERS]
TSMOM = [f"{ticker}_12_month_minus_1_month_return" for ticker in TICKERS]
PORTFOLIO_RETURNS = "Daily Portfolio Returns"
PORTFOLIO_PRICE_RELATIVE = "Portfolio Price Relative"
COVARIANCES = [f"{ticker1}/{ticker2}" for ticker1 in TICKERS for ticker2 in TICKERS]
PORTFOLIO_TURNOVER = "Portfolio Turnover"


def load_df():
    return pd.read_csv(ASSET_PRICES_FILE, index_col="Date", parse_dates=["Date"])


def calculate_portfolio_returns(look_back_period=60):
    st = time.time()
    df = load_df()
    df[ASSET_RETURNS] = df[TICKERS].pct_change()
    df[TSMOM] = df[ASSET_RETURNS].shift(21).rolling("231D").apply(lambda x: (1 + x).prod() - 1, raw=True)
    df[RISK_WEIGHTS] = df[TSMOM].apply(lambda row: risk_weight(row), axis=1, result_type='expand', raw=True)
    df[COVARIANCES] = df[ASSET_RETURNS].rolling(f"{look_back_period}D").cov().unstack()
    df = df.dropna()
    df[CASH_WEIGHTS] = df[RISK_WEIGHTS + COVARIANCES].apply(lambda row: cash_weights(row), axis=1, result_type='expand')
    df = df.dropna()
    returns = return_portfolio(df[ASSET_RETURNS], df[CASH_WEIGHTS], verbose=True)
    df[PORTFOLIO_RETURNS] = returns["returns"]
    df[PORTFOLIO_TURNOVER] = returns["Two.Way.Turnover"]
    df[PORTFOLIO_PRICE_RELATIVE] = 1.0 * (1 + df[PORTFOLIO_RETURNS]).cumprod()
    df.to_csv(f"{DATA_DIR}/portfolio.csv")
    calc_stats(df[PORTFOLIO_PRICE_RELATIVE]).display()
    et = time.time()
    print(f"Took {et - st} seconds to backtest the portfolio")
    return df


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


'''
def vectorized_back_test():
    st = time.time()
    df = load_df()
    n = len(df.columns)
    risk_budget = np.array([1 / n] * n)
    
    df[ASSET_RETURNS] = df[TICKERS].pct_change().dropna()
    df[CASH_WEIGHTS] = df[ASSET_RETURNS].rolling("60D").cov().dropna().groupby(level=0).apply(
        lambda covariance: cash_weights(risk_budget, covariance))
    df[PORTFOLIO_RETURNS] = return_portfolio(df[ASSET_RETURNS], df[CASH_WEIGHTS])["returns"]
    df[PORTFOLIO_PRICE_RELATIVE] = 1.0 * (1 + df[PORTFOLIO_RETURNS]).cumprod()
    
    df[ASSET_RETURNS] = df[TICKERS].pct_change().dropna()
    window_size = "60D"
    cov_matrix = df[ASSET_RETURNS].rolling(window_size).cov()
    cov_matrix = cov_matrix.dropna()
    grouped_cov = cov_matrix.groupby(level=0)

    # Define a function to compute the cash weights
    def cash_weights(risk_budget, covariance_matrix):
        inv_diag = np.diag(1 / np.sqrt(np.diag(covariance_matrix)))
        normalized_cov = inv_diag.dot(covariance_matrix).dot(inv_diag)
        weights = normalized_cov.dot(risk_budget)
        weights = weights / np.sum(weights)
        return weights

    # Apply the cash_weights function to each group of covariance matrices
    cash_weights_array = grouped_cov.apply(lambda x: cash_weights(risk_budget, x.values))

    # Convert the resulting list of arrays into a pandas DataFrame
    df[CASH_WEIGHTS] = pd.DataFrame.from_records(cash_weights_array, index=grouped_cov.groups)
    returns = return_portfolio(df[ASSET_RETURNS], df[CASH_WEIGHTS], verbose=True)
    df[PORTFOLIO_RETURNS] = returns["returns"]
    df[PORTFOLIO_TURNOVER] = returns["Two.Way.Turnover"]
    df[PORTFOLIO_PRICE_RELATIVE] = 1.0 * (1 + df[PORTFOLIO_RETURNS]).cumprod()
    df.to_csv(f"{DATA_DIR}/portfolio.csv")
    calc_stats(df[PORTFOLIO_PRICE_RELATIVE]).display()
    et = time.time()
    print(f"Took {et - st} seconds to backtest the portfolio")

'''
