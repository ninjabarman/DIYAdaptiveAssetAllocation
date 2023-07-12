from datetime import datetime
from os.path import exists
import yfinance as yf
from constants import ASSET_PRICES_FILE, TICKERS


def download_asset_class_data_yahoo():
    if not exists(ASSET_PRICES_FILE):
        print("Downloading price data from yahoo")
        data = yf.download(
            TICKERS, start="1993-01-29", end=datetime.today().strftime("%Y-%m-%d")
        )["Adj Close"]
        data.to_csv(ASSET_PRICES_FILE)
        print(data.head())
    else:
        print("Price data already downloaded")
