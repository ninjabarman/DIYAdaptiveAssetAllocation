from datetime import datetime
from os.path import exists

import yfinance as yf

FRED_DATA_API_KEY = "6dae28976bcf93d722dee9cbf01bc41e"

QUANDLE_API_KEY = "B2CpxbQrnjSYFzWUupPX"

NASDAQ_KEY = "sxU6u1b7XNyNPPUFZrYo"

DATA_DIR = "data_dir"

ASSET_PRICES_FILE = f"{DATA_DIR}/asset_prices.csv"

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
    },
    {
        "asset_class": "Real Estate",
        "description": "MSCI US REIT INDEX",
        "ticker": "^RMZ"
    },
    {
        "asset_class": "Gold",
        "description": "U.S. Global Investors Funds Gold and Precious Metals Fund ",
        "ticker": "USERX"
    }
]

TICKERS = [asset["ticker"] for asset in asset_data_config]


def download_asset_class_data_yahoo():
    if not exists(ASSET_PRICES_FILE):
        print("Downloading price data from yahoo")
        data = yf.download(TICKERS, start="1993-01-29", end=datetime.today().strftime('%Y-%m-%d'))["Adj Close"]
        data.to_csv(ASSET_PRICES_FILE)
        print(data.head())
