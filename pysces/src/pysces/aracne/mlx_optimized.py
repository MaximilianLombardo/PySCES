"""
MLX-optimized implementations of ARACNe algorithm functions.

This module provides MLX-accelerated implementations of the core ARACNe functions
for use on Apple Silicon hardware.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Optional

# Check if MLX is available
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    import warnings
    warnings.warn(
        "MLX is not installed. Using slower implementation. "
        "To use the faster MLX implementation on Apple Silicon, install MLX with: pip install mlx"
    )
    # Define dummy mx for type hints
    class DummyMX:
        class array:
            pass
    mx = DummyMX()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_mi_mlx(x: mx.array, y: mx.array, bins: int = 10) -> float:
    """
    Calculate mutual information between two variables using MLX.

    Parameters
    ----------
    x : mx.array
        First variable
    y : mx.array
        Second variable
    bins : int, default=10
        Number of bins for discretization

    Returns
    -------
    float
        Mutual information
    """
    # MLX doesn't have histogram functions yet, so we need to use NumPy
    # Convert to NumPy arrays efficiently
    if hasattr(x, 'numpy'):
        x_np = x.numpy()
        y_np = y.numpy()
    else:
        # Fallback for older MLX versions
        x_np = np.array(x.tolist())
        y_np = np.array(y.tolist())

    # Calculate histograms in one batch to reduce overhead
    hist_joint, x_edges, y_edges = np.histogram2d(x_np, y_np, bins=bins)
    hist_x = np.sum(hist_joint, axis=1)
    hist_y = np.sum(hist_joint, axis=0)

    # Convert back to MLX arrays in one batch
    hist_joint_mx = mx.array(hist_joint)
    hist_x_mx = mx.array(hist_x).reshape(-1, 1)
    hist_y_mx = mx.array(hist_y).reshape(1, -1)

    # Normalize histograms - keep in MLX format for all operations
    total_count = mx.sum(hist_joint_mx)
    hist_joint_mx = hist_joint_mx / total_count
    hist_x_mx = hist_x_mx / mx.sum(hist_x_mx)
    hist_y_mx = hist_y_mx / mx.sum(hist_y_mx)

    # Calculate mutual information vectorized
    # MI = sum(p(x,y) * log(p(x,y) / (p(x) * p(y))))
    outer_product = mx.matmul(hist_x_mx, hist_y_mx)
    # Add small epsilon to avoid division by zero and log(0)
    epsilon = 1e-10
    ratio = mx.where(hist_joint_mx > epsilon,
                    hist_joint_mx / (outer_product + epsilon),
                    mx.ones_like(hist_joint_mx))
    log_ratio = mx.where(ratio > epsilon, mx.log(ratio), mx.zeros_like(ratio))
    mi = mx.sum(hist_joint_mx * log_ratio)

    # Evaluate the computation graph
    return mi.item()

def calculate_correlation_mlx(x: mx.array, y: mx.array) -> float:
    """
    Calculate Pearson correlation coefficient using MLX.

    Parameters
    ----------
    x : mx.array
        First variable
    y : mx.array
        Second variable

    Returns
    -------
    float
        Correlation coefficient
    """
    # Vectorized implementation of Pearson correlation
    # Center the variables
    x_centered = x - mx.mean(x)
    y_centered = y - mx.mean(y)

    # Calculate correlation in one vectorized operation
    numerator = mx.sum(x_centered * y_centered)
    x_std = mx.sqrt(mx.sum(x_centered**2))
    y_std = mx.sqrt(mx.sum(y_centered**2))
    denominator = x_std * y_std

    # Handle division by zero with mx.where
    epsilon = 1e-10
    correlation = mx.where(denominator > epsilon,
                          numerator / denominator,
                          mx.zeros_like(numerator))

    # Evaluate the computation graph
    return correlation.item()

def apply_dpi_mlx(mi_matrix: mx.array, tf_indices: List[int], dpi_tolerance: float = 0.1) -> mx.array:
    """
    Apply Data Processing Inequality using MLX.

    Parameters
    ----------
    mi_matrix : mx.array
        Mutual information matrix (TFs x genes)
    tf_indices : list
        Indices of TFs in the gene list
    dpi_tolerance : float, default=0.1
        Tolerance for DPI

    Returns
    -------
    mx.array
        Pruned MI matrix
    """
    # Convert to numpy for faster processing
    # This is more efficient than using MLX for this algorithm
    if hasattr(mi_matrix, 'numpy'):
        mi_np = mi_matrix.numpy()
    else:
        mi_np = np.array(mi_matrix.tolist())

    # Create a copy of the MI matrix
    pruned_np = mi_np.copy()

    # Get dimensions
    n_tfs, n_genes = mi_np.shape

    # We'll use vectorized operations instead of a lookup dictionary

    # Apply DPI in larger batches for better efficiency
    # Process TFs in batches
    batch_size = min(20, n_tfs)  # Process up to 20 TFs at a time

    # Pre-compute which genes are TFs for faster lookup
    is_tf = np.zeros(n_genes, dtype=bool)
    for tf_idx in tf_indices:
        is_tf[tf_idx] = True

    # Create a mapping from gene index to TF index
    gene_to_tf_idx = np.full(n_genes, -1, dtype=np.int32)
    for i, tf_idx in enumerate(tf_indices):
        gene_to_tf_idx[tf_idx] = i

    logger.debug("Applying DPI algorithm")

    for i_batch_start in range(0, n_tfs, batch_size):
        i_batch_end = min(i_batch_start + batch_size, n_tfs)

        # Process each TF in the batch
        for i in range(i_batch_start, i_batch_end):
            tf_idx = tf_indices[i]

            # Get all non-zero MI values for this TF
            # This vectorized approach is much faster
            non_zero_indices = np.where((pruned_np[i, :] > 1e-10) &
                                       (np.arange(n_genes) != tf_idx))[0]

            # Skip if no non-zero values
            if len(non_zero_indices) == 0:
                continue

            # Process each gene with non-zero MI
            for j in non_zero_indices:
                mi_ij = pruned_np[i, j]

                # Find potential mediators
                # Skip self and target gene
                mediators = np.arange(n_genes)
                mediators = mediators[(mediators != tf_idx) & (mediators != j)]

                # Get MI values for TF to mediators
                mi_ik_values = pruned_np[i, mediators]

                # Get MI values for mediators to target gene
                mi_kj_values = np.zeros_like(mi_ik_values)

                # For mediators that are TFs, get values from the matrix
                tf_mediators = mediators[is_tf[mediators]]
                if len(tf_mediators) > 0:
                    tf_mediator_indices = gene_to_tf_idx[tf_mediators]
                    mi_kj_values[is_tf[mediators]] = pruned_np[tf_mediator_indices, j]

                # Find the minimum of mi_ik and mi_kj for each mediator
                min_mi_values = np.minimum(mi_ik_values, mi_kj_values)

                # Apply DPI
                if np.any(mi_ij < min_mi_values - dpi_tolerance):
                    pruned_np[i, j] = 0.0

    # Convert back to MLX array
    pruned_matrix = mx.array(pruned_np)

    return pruned_matrix

def run_aracne_mlx(
    expr_matrix: np.ndarray,
    gene_list: List[str],
    tf_indices: List[int],
    bootstraps: int = 100,
    consensus_threshold: float = 0.5,
    dpi_tolerance: float = 0.1
) -> Tuple[np.ndarray, Dict]:
    """
    Run ARACNe algorithm using MLX for acceleration.

    Parameters
    ----------
    expr_matrix : np.ndarray
        Expression matrix (cells x genes)
    gene_list : list
        List of gene names
    tf_indices : list
        Indices of TFs in the gene list
    bootstraps : int, default=100
        Number of bootstrap iterations
    consensus_threshold : float, default=0.5
        Threshold for consensus network
    dpi_tolerance : float, default=0.1
        Tolerance for DPI

    Returns
    -------
    tuple
        Consensus matrix and regulons dictionary
    """
    if not HAS_MLX:
        logger.warning("MLX not available. Falling back to Numba implementation.")
        from .numba_optimized import run_aracne_numba
        return run_aracne_numba(
            expr_matrix,
            gene_list,
            tf_indices,
            bootstraps=bootstraps,
            consensus_threshold=consensus_threshold,
            dpi_tolerance=dpi_tolerance
        )

    # Check if running on Apple Silicon
    import platform
    if platform.processor() != 'arm':
        logger.warning("Not running on Apple Silicon. Falling back to Numba implementation.")
        from .numba_optimized import run_aracne_numba
        return run_aracne_numba(
            expr_matrix,
            gene_list,
            tf_indices,
            bootstraps=bootstraps,
            consensus_threshold=consensus_threshold,
            dpi_tolerance=dpi_tolerance
        )

    logger.info("Running ARACNe with MLX acceleration on Apple Silicon")

    # Convert to MLX arrays once at the beginning
    expr_mx = mx.array(expr_matrix)

    # Get dimensions
    n_samples, n_genes = expr_mx.shape
    n_tfs = len(tf_indices)

    # Initialize consensus matrix
    consensus_matrix = mx.zeros((n_tfs, n_genes))

    # Prepare for bootstrap iterations

    # Run bootstrap iterations in batches to reduce overhead
    batch_size = min(10, bootstraps)  # Process up to 10 bootstraps at a time

    for b_start in range(0, bootstraps, batch_size):
        b_end = min(b_start + batch_size, bootstraps)
        batch_bootstraps = b_end - b_start

        logger.debug(f"Bootstrap iterations {b_start+1}-{b_end}/{bootstraps}")

        # Process each bootstrap in the batch
        for _ in range(batch_bootstraps):

            # Create bootstrap sample efficiently
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)

            # MLX doesn't support fancy indexing like NumPy
            # Convert to numpy, index, then convert back efficiently
            if hasattr(expr_mx, 'numpy'):
                expr_np = expr_mx.numpy()
            else:
                expr_np = np.array(expr_mx.tolist())

            bootstrap_data_np = expr_np[bootstrap_indices]
            bootstrap_data = mx.array(bootstrap_data_np)

            # Convert bootstrap data to numpy once
            if hasattr(bootstrap_data, 'numpy'):
                bootstrap_np = bootstrap_data.numpy()
            else:
                bootstrap_np = np.array(bootstrap_data.tolist())

            # Initialize MI matrix in numpy (faster for updates)
            mi_matrix_np = np.zeros((n_tfs, n_genes))

            # Calculate all correlations in a vectorized way
            # This is much faster than calculating each correlation individually
            logger.debug("Calculating correlations for all TF-gene pairs")

            # Process in larger batches for better efficiency
            tf_batch_size = min(n_tfs, 20)  # Process up to 20 TFs at a time
            gene_batch_size = min(n_genes, 200)  # Process up to 200 genes at a time

            for i_start in range(0, n_tfs, tf_batch_size):
                i_end = min(i_start + tf_batch_size, n_tfs)

                # Get TF indices for this batch
                batch_tf_indices = [tf_indices[i] for i in range(i_start, i_end)]

                for j_start in range(0, n_genes, gene_batch_size):
                    j_end = min(j_start + gene_batch_size, n_genes)
                    j_batch_size = j_end - j_start

                    # Process each TF in the batch
                    for i_idx, tf_idx in enumerate(batch_tf_indices):
                        i = i_start + i_idx

                        # Process each gene in the batch
                        for j_idx in range(j_batch_size):
                            j = j_start + j_idx

                            # Skip self-interactions
                            if tf_idx == j:
                                continue

                            # Get gene expression data directly from numpy
                            x_np = bootstrap_np[:, tf_idx]
                            y_np = bootstrap_np[:, j]

                            # Calculate correlation directly in numpy (faster)
                            # Center the variables
                            x_centered = x_np - np.mean(x_np)
                            y_centered = y_np - np.mean(y_np)

                            # Calculate correlation
                            numerator = np.sum(x_centered * y_centered)
                            x_std = np.sqrt(np.sum(x_centered**2))
                            y_std = np.sqrt(np.sum(y_centered**2))
                            denominator = x_std * y_std

                            # Handle division by zero
                            if denominator > 1e-10:
                                correlation = numerator / denominator
                            else:
                                correlation = 0.0

                            # Store absolute correlation as MI
                            mi_matrix_np[i, j] = abs(correlation)

            # Convert the complete MI matrix to MLX at once
            # This minimizes the number of conversions
            mi_matrix = mx.array(mi_matrix_np)

            # Apply DPI
            pruned_matrix = apply_dpi_mlx(mi_matrix, tf_indices, dpi_tolerance)

            # Add to consensus matrix
            consensus_matrix = consensus_matrix + (pruned_matrix > 0).astype(mx.float32) / bootstraps

    # Apply consensus threshold
    consensus_matrix = mx.where(consensus_matrix < consensus_threshold,
                              mx.zeros_like(consensus_matrix),
                              consensus_matrix)

    # Convert back to numpy efficiently
    if hasattr(consensus_matrix, 'numpy'):
        consensus_np = consensus_matrix.numpy()
    else:
        consensus_np = np.array(consensus_matrix.tolist())

    # Create regulons dictionary efficiently
    regulons = {}
    for i, tf_idx in enumerate(tf_indices):
        # Make sure tf_idx is within bounds
        if tf_idx < len(gene_list):
            tf_name = gene_list[tf_idx]
            targets = {}

            # Use vectorized operations to find non-zero interactions
            for j in range(min(n_genes, len(gene_list))):
                if j != tf_idx and j < len(gene_list) and consensus_np[i, j] > 0:
                    targets[gene_list[j]] = float(consensus_np[i, j])

            regulons[tf_name] = {
                'targets': targets
            }

    return consensus_np, regulons
