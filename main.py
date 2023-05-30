import pyfolio

from polars_backtest import back_test_portfolio
from yahoo_data import TICKERS

if __name__ == '__main__':
    df1 = back_test_portfolio(tickers=TICKERS)
    df2 = back_test_portfolio(tickers=TICKERS, risk_budget=[.8, .1, .1])
    df3 = back_test_portfolio(tickers=TICKERS, risk_budget=[.6, .2, .2])
    print("Done")