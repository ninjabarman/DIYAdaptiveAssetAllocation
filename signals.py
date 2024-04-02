import pandas as pd
import numpy as np
from constants import *

ASSET_TREND = [f"{ticker}_trend" for ticker in TICKERS]

MOMENTUM_LOOKBACK_PERIODS = [63, 126, 189, 252]


def stock_bond_cross_asset_momentum(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate the cross asset momentum for stocks and bonds
    assumes the df has two columns for stocks and bonds and a datetime index
    
    args:
        df: dataframe with two columns for stocks and bonds and a datetime index
    returns:
        dataframe containing the risk weights for stocks and bonds
        
    if stocks are up:
        if bonds are up:
            stock risk weight = 1
            bond risk weight = 0
        else:
            stock risk weight = 0.5
            bond risk weight = 0.5
    else:
        if bonds are up:
            stock risk weight = 0
            bond risk weight = 1
        else:
            stock risk weight = 0.5
            bond risk weight = 0.5
            
    '''
    ewma_16 = df.ewm(span=64).mean()
    ewma_64 = df.ewm(span=256).mean()
    ewma_cross = ewma_16 - ewma_64
    risk_weights = pd.DataFrame(index=df.iloc[64:].index) 
    
    # allocate risk weights for stocks
    
    risk_weights["SPY_risk_weight"] = 0
    # if stocks are up and bonds are up, stock risk weight = 1
    risk_weights["SPY_risk_weight"].loc[(ewma_cross.iloc[:, 0] > 0) & (ewma_cross.iloc[:, 1] > 0)] = 1.0
    # if stocks are up and bonds are down, stock risk weight = 0.5
    risk_weights["SPY_risk_weight"].loc[(ewma_cross.iloc[:, 0] > 0) & (ewma_cross.iloc[:, 1] <= 0)] = 0.75
    # if stocks are down and bonds are up, stock risk weight = 0.0
    risk_weights["SPY_risk_weight"].loc[(ewma_cross.iloc[:, 0] <= 0) & (ewma_cross.iloc[:, 1] > 0)] = 0.0
    # if stocks are down and bonds are down, stock risk weight = 0.5
    risk_weights["SPY_risk_weight"].loc[(ewma_cross.iloc[:, 0] <= 0) & (ewma_cross.iloc[:, 1] <= 0)] = 0.5

    
    # allocate risk weights for bonds
    
    risk_weights["VUSTX_risk_weight"] = 0
    # if stocks are up and bonds are up, bond risk weight = 0
    risk_weights["VUSTX_risk_weight"].loc[(ewma_cross.iloc[:, 0] > 0) & (ewma_cross.iloc[:, 1] > 0)] = 0.0
    # if stocks are up and bonds are down, bond risk weight = 0.5
    risk_weights["VUSTX_risk_weight"].loc[(ewma_cross.iloc[:, 0] > 0) & (ewma_cross.iloc[:, 1] <= 0)] = 0.25
    # if stocks are down and bonds are up, bond risk weight = 1.0
    risk_weights["VUSTX_risk_weight"].loc[(ewma_cross.iloc[:, 0] <= 0) & (ewma_cross.iloc[:, 1] > 0)] = 1.0
    # if stocks are down and bonds are down, stock risk weight = 0.5
    risk_weights["VUSTX_risk_weight"].loc[(ewma_cross.iloc[:, 0] <= 0) & (ewma_cross.iloc[:, 1] <= 0)] = 0.5
    
    if (risk_weights["SPY_risk_weight"] + risk_weights["VUSTX_risk_weight"]).eq(1.0).all():
        print("risk weights sum to 1")
    else:
        print("risk weights do not sum to 1")
    
    return risk_weights
    
    
def trend_risk_weights(df: pd.DataFrame) -> pd.DataFrame:
    ewma_16 = df.ewm(span=16).mean()
    ewma_64 = df.ewm(span=64).mean()
    ewma_cross = ewma_16 - ewma_64
    risk_weights = pd.DataFrame(index=ewma_cross.index)
    vol = df.rolling(252).std().reindex(ewma_cross.index)
    risk_weights = ewma_cross / vol
    return risk_weights.div(risk_weights.abs().sum(axis=1), axis=0)
