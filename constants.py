FRED_DATA_API_KEY = "6dae28976bcf93d722dee9cbf01bc41e"
QUANDLE_API_KEY = "B2CpxbQrnjSYFzWUupPX"
NASDAQ_KEY = "sxU6u1b7XNyNPPUFZrYo"
DATA_DIR = "data_dir"
'''
    {
        "asset_class": "Real Estate",
        "description": "MSCI US REIT INDEX",
        "ticker": "^RMZ",
    },
    {
        "asset_class": "Gold",
        "description": "U.S. Global Investors Funds Gold and Precious Metals Fund ",
        "ticker": "USERX",
    },
'''

ASSET_DATA_CONFIG = [
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
        "ticker": "^SPGSCI",
    }
]
TICKERS = [asset["ticker"] for asset in ASSET_DATA_CONFIG]
ASSET_PRICES_FILE = f"{DATA_DIR}/asset_prices.csv"
TARGET_WEIGHTS_FILE = f"{DATA_DIR}/target_weights.csv"
ASSET_RETURNS = [f"{ticker}_return" for ticker in TICKERS]
CASH_WEIGHTS = [f"{ticker}_cash_weight" for ticker in TICKERS]
PORTFOLIO_RETURNS = "Daily Portfolio Returns"
PORTFOLIO_RETURNS_RELATIVE = "Portfolio Returns Relative"
ASSET_VOLATILITIES = [f"{ticker}_vol" for ticker in TICKERS]
RISK_WEIGHTS = [f"{ticker}_risk_weight" for ticker in TICKERS]
TSMOM = [f"{ticker}_12_month_minus_1_month_return" for ticker in TICKERS]
PORTFOLIO_RETURNS = "Daily Portfolio Returns"
ASSET_CORRELATIONS = [f"{ticker1}_{ticker2}_corr" for ticker1 in TICKERS for ticker2 in TICKERS]
ASSET_COVARIANCES = [f"{ticker1}_{ticker2}_cov" for ticker1 in TICKERS for ticker2 in TICKERS]
ASSET_SKEWNESS = [f"{ticker}_skew" for ticker in TICKERS]
ASSET_KURTOSIS = [f"{ticker}_kurt" for ticker in TICKERS]
PORTFOLIO_TURNOVER = "Portfolio Turnover"
STARTING_CASH = 100000