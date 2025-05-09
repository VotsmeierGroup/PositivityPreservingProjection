import numpy as np
import scipy
from typing import Tuple

def positivity_projection(C0: np.ndarray, C_next: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    Projects the concentration changes to those yielding positive concentrations that obey the atom balance.

    Args:
        C0: Initial concentrations.
        C_next: Next concentrations.
        E: Atom-Molecule Matrix.

    Returns:
        Corrected concentrations in the positivity-preserving subspace.
    """
    # Rank of the element-species matrix
    Rank = np.linalg.matrix_rank(E)
    # Basis for the nullspace of E
    _, _, Vt = scipy.linalg.svd(E)
    B = Vt[Rank:].T

    # Calculate the change in concentration
    dcexp = C_next - C0  
    # Weighting factor is predicted concentration
    scaling = 1 / C_next  
    # Reshape scaling for matrix operations
    W = np.einsum('ij,jk->ijk', scaling, np.eye(scaling.shape[1]))

    # Calculate the projected rates by solving the linear system
    W_b_T = np.matmul(W, B[None, :, :]).transpose(0, 2, 1)
    W_dc = np.matmul(W, dcexp[:, :, None])
    dc_corr_S = np.matmul(
        np.matmul(W, B[None, :, :]),
        np.linalg.solve(np.matmul(W_b_T, np.matmul(W, B[None, :, :])), np.matmul(W_b_T, W_dc))
    ).squeeze(axis=2)

    # Corrected concentrations
    cexp_corrected = C0 + dc_corr_S / scaling  

    return cexp_corrected


def orthogonal_projection(C0: np.ndarray, C_next: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    Projects the concentration changes obeying the atom balance by orthogonal projection

    Args:
        C0: Initial concentrations
        C_next: Next concentrations
        E: Atom-Molecule Matrix.

    Returns:
        Corrected concentrations in the orthogonal subspace.
    """
    # Rank of the element-species matrix
    Rank = np.linalg.matrix_rank(E)
    # Basis for the nullspace of E
    _, _, Vt = scipy.linalg.svd(E)
    B = Vt[Rank:].T

    # Calculate the change in concentration
    dcexp = C_next - C0
    # Calculate the corrected rates by orthogonal projection
    dc_corr = (dcexp @ B) @ B.T

    # Corrected concentrations
    cexp_corr = C0 + dc_corr

    return cexp_corr


def backtracking(C0: np.ndarray, C_next_corr: np.ndarray, intersection_point: float = 1e-10) -> np.ndarray:
    """
    Adjusts concentrations to ensure positivity by backtracking along the projection.

    Args:
        C0: Initial concentrations
        C_next_corr: Corrected concentrations
        intersection_point: Minimum allowed value for concentrations

    Returns:
        Concentrations adjusted to ensure positivity
    """
    # Find the indices of negative concentrations
    neg_idx = np.unique(np.where(C_next_corr < 0)[0])
    C_next_corr_copy = np.copy(C_next_corr)
    start_point = C_next_corr_copy[neg_idx]
    end_point = np.clip(C0[neg_idx], a_min=intersection_point, a_max=None)
    direction = end_point - start_point
    # if direction != 0 else np.inf. This may throw a runtime error due to dividing by 0 but is not a problem,
    # because this direction will not be considered in the next step
    t_values = (intersection_point - start_point) / direction
    # find all t_values below 1
    t_values_filtered = np.where(t_values <= 1, t_values, -np.inf)
    # find the maximum t_value for each sample. This then corresponds to the maximum t value between 0 and 1
    t_max = np.expand_dims(np.max(t_values_filtered, axis=1), -1)

    # find the intersection point
    intersection = start_point + t_max * direction
    C_next_corr_copy[neg_idx] = intersection

    return C_next_corr_copy


def orthogonal_backtracking(C0: np.ndarray, C_next: np.ndarray, E: np.ndarray, intersection_point: float = 1e-10) -> np.ndarray:
    """
    Combines orthogonal projection and backtracking to ensure positivity.

    Args:
        C0: Initial concentrations.
        C_next: Next concentrations.
        E: Atom-Molecule Matrix.
        intersection_point: Minimum allowed value for concentrations.

    Returns:
        Concentrations corrected for orthogonality and positivity.
    """
    cexp_corr = orthogonal_projection(C0, C_next, E)
    cexp_corr_positive = backtracking(C0, cexp_corr, intersection_point)

    return cexp_corr_positive


def positivity_backtracking(C0: np.ndarray, C_next: np.ndarray, E: np.ndarray, intersection_point: float = 1e-10) -> np.ndarray:
    """
    Combines positivity-preserving projection and backtracking to ensure positivity.

    Args:
        C0: Initial concentrations.
        C_next: Next concentrations.
        E: Atom-Molecule Matrix.
        intersection_point: Minimum allowed value for concentrations.

    Returns:
        Concentrations corrected for positivity-preserving projection and positivity.
    """
    cexp_corr = positivity_projection(C0, C_next, E)
    cexp_corr_positive = backtracking(C0, cexp_corr, intersection_point)

    return cexp_corr_positive

