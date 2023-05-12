FRED_DATA_API_KEY = "6dae28976bcf93d722dee9cbf01bc41e"

from os.path import exists

import fredapi as fa
import pandas as pd


def download_asset_class_data_fred():
    fred = fa.Fred(api_key=FRED_DATA_API_KEY)
    for i in [2, 5, 7, 10, 20, 30]:
        maturity = f"DGS{i}"
        file_name = f"{DATA_DIR}/{maturity}.csv"
        if exists(file_name):
            print("Data already downloaded")
            continue
        treasury_yield = pd.DataFrame(fred.get_series(maturity))
        treasury_yield.columns = ["Yield"]
        print(treasury_yield.head())
        print(treasury_yield.tail())
        treasury_yield.to_csv()


def yield_to_returns():
    for i in [2, 5, 7, 10, 20, 30]:
        maturity = f"DGS{i}"
        file_name = f"{DATA_DIR}/{maturity}.csv"
        yields = pd.read_csv(file_name)
        for index, row in yields.head().iterrows():
            constant_maturity_yield = row["Yield"]
            total_return = calculate_daily_return(constant_maturity_yield, i)


# implementation of solution https://quant.stackexchange.com/a/57403
def calculate_daily_return(yt: float, maturity: int):
    dt = 1 / 255
    dy = (yt * dt) - yt
    zt = 1 + (yt / 2)
    duration = calculate_duration(yt, maturity, zt)
    convexity = calculate_convexity(yt, maturity, zt)
    yield_income = ((1 + yt) ** dt) - 1
    print(f"Yield income on {maturity} yr {yield_income}")
    print(f"Duration on {maturity} yr {duration}")
    print(f"convexity on {maturity} yr {convexity}")
    total_return = yield_income - (duration * dy) + ((.5 * convexity) * (dy * 2))
    print(f"Total return on {maturity} yr {total_return}")
    return total_return


def calculate_duration(yt: float, maturity: int, zt: float):
    # return (1 / yt) * (zt ** (2 * maturity))
    return (1 - (1 / (1 + 0.5 * yt) ** (2 * maturity)))


def calculate_convexity(constant_maturity_yield: float, maturity: int, zt: float):
    c_1 = (2 / (constant_maturity_yield ** 2)) * (1 - (zt ** (-2 * maturity)))
    c_2 = ((2 * maturity) / constant_maturity_yield) * (zt ** ((-2 * maturity) - 1))
    return c_1 - c_2
