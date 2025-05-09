import numpy as np
from typing import Tuple


def atombalance_consistency(C0: np.ndarray, C_next: np.ndarray, E: np.ndarray, tol: float = 1e-10) -> Tuple[bool, float, float]:
    """
    Check if atom balance is consistent between initial and next concentrations.

    Args:
        C0: Initial concentration matrix.
        C_next: Next concentration matrix.
        E: Atom-molecule matrix.
        tol: Tolerance for consistency check.

    Returns:
        A tuple containing:
        - A boolean indicating if atom balance is consistent.
        - Maximum error in atom balance.
        - Mean error in atom balance.
    """
    A_in = C0 @ E.T
    A_out = C_next @ E.T

    max_err = np.max(np.abs(A_in - A_out))
    mean_err = np.mean(np.abs(A_in - A_out))

    return np.allclose(A_in, A_out, atol=tol), max_err, mean_err


def negative_count(C: np.ndarray) -> int:
    """
    Count the number of rows with negative values in the concentration matrix.

    Args:
        C: Concentrations.

    Returns:
        The count of rows with negative values.
    """
    return np.where(np.any(C < 0, axis=1))[0].shape[0]


def RMSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between predictions and targets.

    Args:
        predictions: Predicted values.
        targets: True values.

    Returns:
        The RMSE value.
    """
    return np.sqrt(np.mean((predictions - targets) ** 2))


def RelativeError(c_out_pred: np.ndarray, c_out_true: np.ndarray) -> float:
    """
    Calculate the relative error between predicted and true concentrations.

    Args:
        c_out_pred: Predicted next concentrations.
        c_out_true: True next concentrations.

    Returns:
        The relative error.
    """
    eps = 1e-30  # Small value to avoid division by zero
    return np.mean(np.abs(c_out_pred - c_out_true) / (c_out_true + eps)) * 100


def calculate_error_metrics(c_in: np.ndarray, c_out_pred: np.ndarray, c_out_true: np.ndarray, E: np.ndarray) \
        -> Tuple[float, float, float, float]:
    """
    Calculate error metrics for the given concentrations and atom-molecule matrix.

    Args:
        c_in: Initial concentrations.
        c_out_pred: Predicted next concentrations.
        c_out_true: True next concentrations.
        E: Atom-molecule matrix.

    Returns:
        A tuple containing:
        - Mean error in atom balance.
        - Percentage of rows with negative values.
        - RMSE between predicted and true concentrations.
        - Relative error between predicted and true concentrations.
    """
    _, _, atombalance_mean_err = atombalance_consistency(c_in, c_out_pred, E)
    neg_count = negative_count(c_out_pred)
    neg_count_perc = neg_count / c_out_pred.shape[0] * 100
    rmse = RMSE(c_out_pred, c_out_true)
    rel_err = RelativeError(c_out_pred, c_out_true)

    return atombalance_mean_err, neg_count_perc, rmse, rel_err

