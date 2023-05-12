import datetime
import pandas as pd

from yahoo_data import DATA_DIR


def get_vix_future_url(date: str):
    return f"https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{date}.csv"


def download_cboe_data():
    date = "2022-06-01"
    url = get_vix_future_url(date)
    df = pd.read_csv(url)
    df.to_csv(f"{DATA_DIR}/vx_future_data_{date}.csv")
