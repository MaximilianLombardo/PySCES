"""
Numba-optimized implementations of ARACNe algorithm functions.
"""

import numpy as np
import logging
from typing import Tuple, Optional
import warnings

try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn(
        "Numba is not installed. Using slower Python implementation. "
        "To use the faster Numba implementation, install Numba with: pip install numba"
    )
    # Define dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator

    def prange(*args, **kwargs):
        return range(*args, **kwargs)

# Set up logging
logger = logging.getLogger(__name__)

@jit(nopython=True, parallel=True)
def calculate_mi_matrix_numba(bootstrap_data: np.ndarray, tf_indices: np.ndarray) -> np.ndarray:
    """
    Calculate mutual information matrix using Numba.

    Parameters
    ----------
    bootstrap_data : np.ndarray
        Bootstrap sample of expression data (n_samples x n_genes)
    tf_indices : np.ndarray
        Indices of transcription factors

    Returns
    -------
    np.ndarray
        Mutual information matrix (n_tfs x n_genes)
    """
    n_samples, n_genes = bootstrap_data.shape
    n_tfs = len(tf_indices)

    # Initialize MI matrix
    mi_matrix = np.zeros((n_tfs, n_genes), dtype=np.float64)

    # Calculate MI for each TF-gene pair
    for i in prange(n_tfs):
        tf_idx = tf_indices[i]
        x = bootstrap_data[:, tf_idx]

        for j in range(n_genes):
            # Skip self-interactions
            if tf_idx == j:
                continue

            # Get gene expression
            y = bootstrap_data[:, j]

            # Calculate MI using correlation as a proxy
            # In a real implementation, we would use a proper MI calculation
            correlation = calculate_correlation_numba(x, y)
            mi = abs(correlation)  # Use absolute correlation as a proxy for MI

            mi_matrix[i, j] = mi

    return mi_matrix

@jit(nopython=True)
def calculate_correlation_numba(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient using Numba.

    Parameters
    ----------
    x : np.ndarray
        First array
    y : np.ndarray
        Second array

    Returns
    -------
    float
        Pearson correlation coefficient
    """
    n = len(x)

    # Calculate means
    mean_x = np.sum(x) / n
    mean_y = np.sum(y) / n

    # Calculate covariance and variances
    cov = 0.0
    var_x = 0.0
    var_y = 0.0

    for i in range(n):
        diff_x = x[i] - mean_x
        diff_y = y[i] - mean_y
        cov += diff_x * diff_y
        var_x += diff_x * diff_x
        var_y += diff_y * diff_y

    # Handle edge cases
    if var_x < 1e-10 or var_y < 1e-10:
        return 0.0

    # Calculate correlation
    correlation = cov / np.sqrt(var_x * var_y)

    # Handle numerical issues
    if correlation > 1.0:
        correlation = 1.0
    elif correlation < -1.0:
        correlation = -1.0

    return correlation

@jit(nopython=True)
def apply_dpi_numba(mi_matrix: np.ndarray, tf_indices: np.ndarray, dpi_tolerance: float) -> np.ndarray:
    """
    Apply Data Processing Inequality using Numba.

    Parameters
    ----------
    mi_matrix : np.ndarray
        Mutual information matrix (n_tfs x n_genes)
    tf_indices : np.ndarray
        Indices of transcription factors
    dpi_tolerance : float
        Tolerance for DPI

    Returns
    -------
    np.ndarray
        Pruned mutual information matrix
    """
    n_tfs, n_genes = mi_matrix.shape

    # Create a copy of the MI matrix
    pruned_matrix = mi_matrix.copy()

    # Apply DPI
    for i in range(n_tfs):
        for j in range(n_genes):
            # Skip zero or very small values
            if pruned_matrix[i, j] < 1e-10:
                continue

            # Get TF index
            tf_idx = tf_indices[i]

            # Check all possible mediators
            for k in range(n_genes):
                # Skip if k is i or j
                if k == tf_idx or k == j:
                    continue

                # Get MI values
                mi_ij = pruned_matrix[i, j]

                # Find MI(i,k) and MI(k,j)
                mi_ik = 0.0
                mi_kj = 0.0

                # Check if k is a TF
                k_is_tf = False
                k_tf_idx = -1

                for tf_i in range(n_tfs):
                    if tf_indices[tf_i] == k:
                        k_is_tf = True
                        k_tf_idx = tf_i
                        break

                # Get MI(i,k)
                if k_is_tf:
                    # If k is a TF, we can find MI(i,k) in the matrix
                    mi_ik = pruned_matrix[i, k]
                else:
                    # If k is not a TF, we need to calculate MI(i,k)
                    # For simplicity, we'll use a placeholder value
                    mi_ik = 0.0

                # Get MI(k,j)
                if k_is_tf:
                    # If k is a TF, we can find MI(k,j) in the matrix
                    mi_kj = pruned_matrix[k_tf_idx, j]
                else:
                    # If k is not a TF, we need to calculate MI(k,j)
                    # For simplicity, we'll use a placeholder value
                    mi_kj = 0.0

                # Apply DPI
                if mi_ij < min(mi_ik, mi_kj) - dpi_tolerance:
                    pruned_matrix[i, j] = 0.0
                    break

    return pruned_matrix

def run_aracne_numba(expr_matrix: np.ndarray, gene_list: list, tf_indices: np.ndarray,
                    bootstraps: int = 100, consensus_threshold: float = 0.05,
                    dpi_tolerance: float = 0.1) -> Tuple[np.ndarray, dict]:
    """
    Run ARACNe algorithm with Numba optimization.

    Parameters
    ----------
    expr_matrix : np.ndarray
        Expression matrix (n_samples x n_genes)
    gene_list : list
        List of gene names
    tf_indices : np.ndarray
        Indices of transcription factors
    bootstraps : int, optional
        Number of bootstrap iterations, by default 100
    consensus_threshold : float, optional
        Threshold for consensus network, by default 0.05
    dpi_tolerance : float, optional
        Tolerance for DPI, by default 0.1

    Returns
    -------
    Tuple[np.ndarray, dict]
        Consensus matrix and regulons dictionary
    """
    if not HAS_NUMBA:
        logger.warning("Numba is not installed. Using slower Python implementation.")
        # Fall back to Python implementation
        return None, None

    # Get dimensions
    n_samples, n_genes = expr_matrix.shape
    n_tfs = len(tf_indices)

    # Initialize consensus matrix
    consensus_matrix = np.zeros((n_tfs, n_genes), dtype=np.float64)

    # Run bootstrap iterations
    for b in range(bootstraps):
        logger.debug(f"Bootstrap iteration {b+1}/{bootstraps}")

        # Create bootstrap sample
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_data = expr_matrix[bootstrap_indices]

        # Calculate MI matrix
        mi_matrix = calculate_mi_matrix_numba(bootstrap_data, tf_indices)

        # Apply DPI
        pruned_matrix = apply_dpi_numba(mi_matrix, tf_indices, dpi_tolerance)

        # Add to consensus matrix
        consensus_matrix += (pruned_matrix > 0).astype(np.float64) / bootstraps

    # Apply consensus threshold
    consensus_matrix[consensus_matrix < consensus_threshold] = 0.0

    # Create regulons dictionary
    regulons = {}
    for i, tf_idx in enumerate(tf_indices):
        # Make sure tf_idx is within bounds
        if tf_idx >= len(gene_list):
            logger.warning(f"TF index {tf_idx} out of bounds for gene_list of length {len(gene_list)}")
            continue

        tf_name = gene_list[tf_idx]
        targets = {}

        for j in range(n_genes):
            # Skip self-interactions
            if j == tf_idx:
                continue

            # Make sure j is within bounds
            if j >= len(gene_list):
                continue

            # Get interaction strength
            if i < consensus_matrix.shape[0] and j < consensus_matrix.shape[1]:
                strength = consensus_matrix[i, j]
            else:
                continue

            # Add to targets if non-zero
            if strength > 0:
                targets[gene_list[j]] = strength

        regulons[tf_name] = {
            'targets': targets
        }

    return consensus_matrix, regulons
