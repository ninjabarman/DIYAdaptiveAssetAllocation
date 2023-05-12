from backtest import generate_portfolio_weights
from portfolio import run_back_test
from yahoo_data import download_asset_class_data_yahoo
import time

if __name__ == '__main__':
    st = time.time()
    weights = generate_portfolio_weights()
    et = time.time()
    print(f"Took {et - st} seconds to generate portfolio weights")
