import numpy as np
from numba import guvectorize, float64
from scipy.optimize import minimize

@guvectorize(
    [(float64[:, :, :], float64[:, :])],
    "(k, n, n) -> (k, n)",
    nopython=True,
    cache=True,
    fastmath=True,
    target="parallel",
)
def minimum_variance_optimization(covariance_matrix: np.ndarray, weights: np.ndarray):
    for k in range(covariance_matrix.shape[0]):
        n = covariance_matrix.shape[1]

        # Compute the inverse of the covariance matrix
        inverse_covariance = np.linalg.inv(covariance_matrix[k,])

        ones_vector = np.ones((n, 1))

        denominator = np.dot(np.dot(ones_vector.T, inverse_covariance), ones_vector)

        weights[k] = (np.dot(inverse_covariance, ones_vector) / denominator)[:, 0]


@guvectorize(
    [(float64[:, :, :], float64[:, :], float64[:, :])],
    "(k, n, n), (k, n) -> (k, n)",
    nopython=True,
    cache=True,
    fastmath=True,
    target="parallel",
)
def kelley_optimization(
    covariance_matrix: np.ndarray, expected_returns: np.ndarray, weights: np.ndarray
):
    for k in range(covariance_matrix.shape[0]):
        inv_cov_matrix = np.linalg.inv(covariance_matrix[k,])
        w = np.dot(inv_cov_matrix, expected_returns[k,]) / np.sum(
            np.dot(inv_cov_matrix, expected_returns[k,])
        )
        w[w < 0] = 0.0
        w /= np.sum(w)
        weights[k] = w


@guvectorize(
    [(float64[:, :, :], float64[:, :], float64[:, :])],
    "(k, n, n), (k, n) -> (k, n)",
    nopython=True,
    cache=True,
    fastmath=True,
    target="parallel",
)
def risk_budget_optimization(
    covariance_matrix: np.ndarray, risk_budget: np.ndarray, weights: np.ndarray
):
    for k in range(covariance_matrix.shape[0]):
        rb = risk_budget[k,] / np.sum(risk_budget[k,])
        inv_cov_matrix = np.linalg.inv(covariance_matrix[k,])
        w = np.dot(inv_cov_matrix, rb)
        w[w < 0] = 0.0
        w /= np.sum(w)
        weights[k] = w


def maximum_informatation_ratio_optimization(
    beta: np.array, active_return: np.array, residual_risk: np.array, bounds=None
) -> np.array:
    
    def objective(weights: np.array) -> np.array:
        return -np.dot(weights, active_return) / np.sqrt(np.dot(weights, residual_risk))
    
    def weight_sum_constraint(weights):
        return np.sum(weights) - 1.0
    
    if bounds is not None:
        bounds = [(lower_bound, upper_bound) for lower_bound, upper_bound in bounds]
    
    initial_weights = np.ones_like(beta) / len(beta)
    
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'eq', 'fun': weight_sum_constraint}
    )

    # Get the optimal weights
    weights = result.x

    return weights