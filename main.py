import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import skew, kurtosis

from yahoo_data import ASSET_PRICES_FILE, TICKERS, download_asset_class_data_yahoo


def kelly_optimization(asset_returns):
    asset_returns = asset_returns.reshape(255, -1)
    cov_matrix = np.cov(asset_returns, rowvar=False)
    expected_returns = np.mean(asset_returns, axis=0)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    weights = np.dot(inv_cov_matrix, expected_returns) / np.sum(
        np.dot(inv_cov_matrix, expected_returns))
    weights[weights < 0] = 0.0
    weights /= np.sum(weights)
    return weights


def kelly_optimization_with_asset_moments(asset_returns):
    asset_returns = asset_returns.reshape(255, -1)
    returns_mean = np.mean(asset_returns, axis=0)
    asset_skewness = skew(asset_returns)
    asset_kurtosis = kurtosis(asset_returns)
    cov_matrix = np.cov(asset_returns, rowvar=False)

    # Calculate the inverse of the covariance matrix
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Calculate the weights using the closed-form solution
    weights = inv_cov_matrix @ (returns_mean - 0.5 * asset_skewness - (1 / 6) * asset_kurtosis) / np.sum(
        inv_cov_matrix @ (returns_mean - 0.5 * asset_skewness - (1 / 6) * asset_kurtosis))

    weights[weights < 0] = 0.0
    weights /= np.sum(weights)

    return weights


def minimum_variance_optimization_with_asset_moments(asset_returns):
    asset_returns = asset_returns.reshape(255, -1)
    asset_skewness = skew(asset_returns)
    asset_kurtosis = kurtosis(asset_returns)
    cov_matrix = np.cov(asset_returns, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Calculate the weight coefficients based on skewness and kurtosis
    coeffs = np.dot(inv_cov_matrix, asset_skewness - (1 / 6) * asset_kurtosis)

    # Calculate the optimal weights without normalization
    weights = np.dot(inv_cov_matrix, coeffs)

    # Apply long-only constraint by setting negative weights to zero
    weights[weights < 0] = 0.0

    # Normalize the weights to sum up to 1
    weights /= np.sum(weights)

    # Enforce no-leverage constraint by scaling down the weights
    max_weight = np.max(weights)
    if max_weight > 1.0:
        weights /= max_weight

    return weights


def erc_portfolio_with_higher_moments(asset_returns):
    asset_returns = asset_returns.reshape(255, -1)
    asset_skewness = skew(asset_returns)
    asset_kurtosis = kurtosis(asset_returns)
    cov_matrix = np.cov(asset_returns, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Calculate the weight coefficients based on skewness and kurtosis
    coeffs = np.dot(inv_cov_matrix, asset_skewness - (1 / 6) * asset_kurtosis)

    # Calculate the optimal weights without normalization
    weights = np.dot(inv_cov_matrix, coeffs)

    # Calculate the risk contributions of each asset
    risk_contributions = np.multiply(cov_matrix @ weights, weights)

    # Calculate the inverse of the risk contributions
    inverse_risk_contributions = 1 / risk_contributions

    # Set negative weights to zero (long-only constraint)
    weights[weights < 0] = 0.0

    # Normalize the inverse risk contributions to sum up to 1
    inverse_risk_contributions /= np.sum(inverse_risk_contributions)

    # Calculate the final weights using the inverse risk contributions
    weights = inverse_risk_contributions / np.sum(inverse_risk_contributions)

    # Enforce no-leverage constraint by scaling down the weights
    max_weight = np.max(weights)
    if max_weight > 1.0:
        weights /= max_weight

    # Set weights to zero for assets with negative weights
    weights[weights < 0] = 0.0

    # Normalize the weights to sum up to 1
    weights /= np.sum(weights)

    return weights


def calculate_portfolio_weights():
    df = pd.read_csv(ASSET_PRICES_FILE, index_col="Date").dropna()
    returns = df.pct_change().fillna(0.0)
    array = returns.to_numpy()
    look_back_period = 255
    slices = sliding_window_view(array, window_shape=(look_back_period, array.shape[1]))
    reshaped_slices = slices.reshape(slices.shape[0], 1, -1, 1)
    weights = np.apply_along_axis(kelly_optimization_with_asset_moments, axis=2, arr=reshaped_slices)
    weights = weights.squeeze(axis=(1, 3))
    weights = pd.DataFrame(weights)
    weights = weights.rolling(255).mean()
    weights.columns = [f"{ticker}_cash_weight" for ticker in TICKERS]
    n = returns.shape[0] - weights.shape[0]
    weights["Date"] = returns.index[n:]

    weights.set_index('Date', inplace=True)
    return returns, weights


if __name__ == "__main__":
    download_asset_class_data_yahoo()
    _ = calculate_portfolio_weights()
