from datetime import datetime
from enum import Enum
from functools import lru_cache
from os.path import exists

import pandas as pd
import yfinance as yf
from dateutil.rrule import rrule, MONTHLY


class SeriesType(Enum):
    TOTAL_RETURN = 1
    PRICE = 2


FRED_DATA_API_KEY = "6dae28976bcf93d722dee9cbf01bc41e"

QUANDLE_API_KEY = "B2CpxbQrnjSYFzWUupPX"

DATA_DIR = "data_dir"

ASSET_RETURNS_FILE = f"{DATA_DIR}/asset_returns.csv"

ASSET_PRICES_FILE = f"{DATA_DIR}/asset_prices.csv"

ASSET_COVARIANCES_FILE = f"{DATA_DIR}/asset_covariances.csv"

asset_data_config = [
    {
        "asset_class": "S&P 500",
        "description": "US Large Cap Equities (Total Return)",
        "ticker": "SPY",
    },
    {
        "asset_class": "Long Term Treasury",
        "description": "Us Long Term Treasury (Total Return)",
        "ticker": "VUSTX",
    },
    {
        "asset_class": "Commodities",
        "description": "Goldman Sachs Commodity Index",
        "ticker": "^SPGSCI"
    }
]


def download_asset_class_data_yahoo():
    if not exists(ASSET_PRICES_FILE):
        print("Downloading price data from yahoo")
        tickers = get_tickers()
        data = yf.download(tickers, start="1993-01-29", end=datetime.today().strftime('%Y-%m-%d'))["Adj Close"]
        data.to_csv(ASSET_PRICES_FILE)
        print(data.head())


@lru_cache(maxsize=None)
def get_tickers():
    return [asset["ticker"] for asset in asset_data_config]
