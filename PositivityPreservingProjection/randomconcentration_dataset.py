# Generating random concentration dataset
import numpy as np
import scipy
from typing import Tuple

from PositivityPreservingProjection.errormetrics import atombalance_consistency


def sample_elem_species_matrix(n_molecules: int, n_atoms: int, max_atom_per_mol: int) -> np.ndarray:
    """
    Generate a random element-species matrix.

    Args:
        n_molecules (int): Number of molecules.
        n_atoms (int): Number of atoms.
        max_atom_per_mol (int): Maximum number of atoms per molecule.

    Returns:
        np.ndarray: Element-species matrix of shape (n_atoms, n_molecules).
    """
    elem_species_matrix = np.zeros((n_atoms, n_molecules))

    for mol_idx in range(n_molecules):
        for atom_idx in range(n_atoms):
            elem_species_matrix[atom_idx, mol_idx] = np.random.randint(
                0, max_atom_per_mol + 1
            )

    # Ensure no column is all zeros
    if np.any(np.all(elem_species_matrix == 0, axis=0)):
        zero_cols = np.where(np.all(elem_species_matrix == 0, axis=0))[0]
        for col in zero_cols:
            elem_species_matrix[
                np.random.randint(0, n_atoms), col
            ] = np.random.randint(1, max_atom_per_mol + 1)

    return elem_species_matrix


def generate_c1_data(seed_idx: int, n_conditions: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random c1 data.

    Args:
        seed_idx (int): Random seed index.
        n_conditions (int): Number of conditions.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Element-species matrix and c1_true matrix.
    """
    np.random.seed(seed_idx)

    n_atoms = np.random.randint(2, 6)  # Sample between 2 and 6
    n_molecules = np.random.randint(n_atoms + 1, 16)  # Sample between n_atoms and 15

    sampled_elem_species_matrix = sample_elem_species_matrix(
        n_molecules, n_atoms, max_atom_per_mol=4
    )
    c0_true = 10 ** np.random.uniform(-5, 0, (n_conditions, n_molecules))

    return sampled_elem_species_matrix, c0_true


def generate_c0_data(atom_molecule_idx: int, E: np.ndarray, c1_true: np.ndarray) -> np.ndarray:
    """
    Generate c0 data based on stoichiometry and random nullspace directions.

    Args:
        atom_molecule_idx (int): Random seed index.
        E (np.ndarray): atom-molecule matrix.
        c1_true (np.ndarray): True c1 concentrations.

    Returns:
        np.ndarray: Generated c0_true concentrations.
    """
    np.random.seed(atom_molecule_idx)
    batch_size, _ = c1_true.shape

    # Use c0 as our particular solution (c*)
    c_star = c1_true.copy()

    # Compute a basis for the nullspace of E
    _, _, Vt = scipy.linalg.svd(E)
    rank = np.linalg.matrix_rank(E)
    B = Vt[rank:].T

    if B.size == 0:
        print("Nullspace is empty, returning c_star")
        return c_star  # If the nullspace is empty, the solution is unique

    # Generate random nullspace directions for each sample in the batch
    alpha_batch = np.random.randn(batch_size, B.shape[1])  # Shape: (batch_size, d)
    v_batch = alpha_batch @ B.T

    # Compute candidate step sizes for each batch element and each coordinate
    with np.errstate(divide="ignore", invalid="ignore"):
        t_min_candidates = (
            np.min(c_star, axis=-1, keepdims=True) - c_star
        ) / v_batch
        t_max_candidates = (
            np.max(c_star, axis=-1, keepdims=True) - c_star
        ) / v_batch

    # For each element, decide lower/upper candidate based on the sign of v_batch
    lower_candidates = np.where(v_batch > 0, t_min_candidates, t_max_candidates)
    upper_candidates = np.where(v_batch > 0, t_max_candidates, t_min_candidates)

    # Reduce per sample to get overall lower and upper bounds
    t_lower = np.max(lower_candidates, axis=1)
    t_upper = np.min(upper_candidates, axis=1)

    # Ensure valid step sizes and sample t from the uniform distribution per sample
    valid_mask = t_lower <= t_upper
    t = np.zeros(batch_size)
    t[valid_mask] = np.random.uniform(t_lower[valid_mask], t_upper[valid_mask])

    # Compute new concentration vectors
    c0_true = c_star + t[:, None] * v_batch

    # Double-check that the generated concentrations are consistent
    # Ensure non-negativity
    if np.any(c0_true < 0):
        raise ValueError("Negative concentrations generated.")

    # Check stoichiometry
    stoic_consistency = atombalance_consistency(c0_true, c1_true, E)[0]
    if not stoic_consistency:
        raise ValueError("Stoichiometry not satisfied.")

    return c0_true

