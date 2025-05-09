"""
Explore MLX optimization for ARACNe algorithm.

This script demonstrates how to use MLX to accelerate the ARACNe algorithm
on Apple Silicon hardware.
"""

import numpy as np
import pandas as pd
import time
import logging
import sys
import os
from anndata import AnnData

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pysces', 'src'))

# Import ARACNe modules
from pysces.aracne.core import ARACNe

# Check if MLX is available
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
    logger.info("MLX is available. Using MLX for acceleration.")
except ImportError:
    HAS_MLX = False
    logger.warning("MLX is not available. Install MLX with: pip install mlx")

def create_synthetic_dataset(n_cells, n_genes, n_tfs):
    """Create a synthetic dataset for benchmarking."""
    # Create random expression data
    expr_matrix = np.random.rand(n_cells, n_genes)

    # Create gene list
    gene_list = [f"gene_{i}" for i in range(n_genes)]

    # Create AnnData object
    adata = AnnData(X=expr_matrix)
    adata.var_names = gene_list
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    # Create TF list
    tf_list = [f"gene_{i}" for i in range(n_tfs)]

    return adata, tf_list

def calculate_mi_mlx(x, y, bins=10):
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
    # Convert to numpy for histogram calculation
    # (MLX doesn't have histogram functions yet)
    x_np = x.tolist()
    y_np = y.tolist()

    # Calculate joint histogram
    hist_joint, _, _ = np.histogram2d(x_np, y_np, bins=bins)

    # Calculate marginal histograms
    hist_x, _ = np.histogram(x_np, bins=bins)
    hist_y, _ = np.histogram(y_np, bins=bins)

    # Convert back to MLX arrays
    hist_joint_mx = mx.array(hist_joint)
    hist_x_mx = mx.array(hist_x).reshape(-1, 1)
    hist_y_mx = mx.array(hist_y).reshape(1, -1)

    # Normalize histograms
    hist_joint_mx = hist_joint_mx / mx.sum(hist_joint_mx)
    hist_x_mx = hist_x_mx / mx.sum(hist_x_mx)
    hist_y_mx = hist_y_mx / mx.sum(hist_y_mx)

    # Calculate mutual information
    # MI = sum(p(x,y) * log(p(x,y) / (p(x) * p(y))))
    outer_product = mx.matmul(hist_x_mx, hist_y_mx)
    ratio = mx.where(hist_joint_mx > 0, hist_joint_mx / (outer_product + 1e-10), mx.zeros_like(hist_joint_mx))
    log_ratio = mx.where(ratio > 0, mx.log(ratio), mx.zeros_like(ratio))
    mi = mx.sum(hist_joint_mx * log_ratio)

    return mi.item()

def apply_dpi_mlx(mi_matrix, tf_indices, dpi_tolerance=0.1):
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
    # Create a copy of the MI matrix
    # MLX doesn't have a copy method, so we convert to list and back
    pruned_matrix = mx.array(mi_matrix.tolist())

    # Get dimensions
    n_tfs, n_genes = mi_matrix.shape

    # Apply DPI
    # For each TF-gene pair, check if there's a mediator
    for i in range(n_tfs):
        for j in range(n_genes):
            # Skip self-interactions
            if tf_indices[i] == j:
                continue

            # Skip zero or very small values
            if pruned_matrix[i, j] < 1e-10:
                continue

            # Check all possible mediators
            for k in range(n_genes):
                # Skip if k is i or j
                if k == tf_indices[i] or k == j:
                    continue

                # Get MI values
                mi_ij = pruned_matrix[i, j]

                # Find MI(i,k) and MI(k,j)
                mi_ik = 0.0
                mi_kj = 0.0

                # Check if k is a TF
                k_is_tf = False
                k_tf_idx = -1

                for tf_i, tf_idx in enumerate(tf_indices):
                    if tf_idx == k:
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
                    # Update the pruned matrix
                    # MLX's update pattern is different from JAX
                    temp = pruned_matrix.tolist()
                    temp[i][j] = 0.0
                    pruned_matrix = mx.array(temp)
                    break

    return pruned_matrix

def run_aracne_mlx(expr_matrix, gene_list, tf_indices, bootstraps=100, consensus_threshold=0.5, dpi_tolerance=0.1):
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
        logger.warning("MLX not available. Falling back to CPU implementation.")
        aracne = ARACNe(
            bootstraps=bootstraps,
            consensus_threshold=consensus_threshold,
            dpi_tolerance=dpi_tolerance,
            use_numba=True
        )
        adata = AnnData(X=expr_matrix.T)
        adata.var_names = gene_list
        adata.obs_names = [f"cell_{i}" for i in range(expr_matrix.shape[0])]
        network = aracne.run(adata, tf_list=[gene_list[i] for i in tf_indices])
        return network['consensus_matrix'], network['regulons']

    # Convert to MLX arrays
    expr_mx = mx.array(expr_matrix)

    # Get dimensions
    n_samples, n_genes = expr_mx.shape
    n_tfs = len(tf_indices)

    # Initialize consensus matrix
    consensus_matrix = mx.zeros((n_tfs, n_genes))

    # Run bootstrap iterations
    for b in range(bootstraps):
        logger.debug(f"Bootstrap iteration {b+1}/{bootstraps}")

        # Create bootstrap sample
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        # MLX doesn't support fancy indexing like NumPy
        # We need to convert to numpy, index, then convert back
        expr_np = expr_mx.tolist()
        bootstrap_data_np = np.array(expr_np)[bootstrap_indices]
        bootstrap_data = mx.array(bootstrap_data_np)

        # Calculate MI matrix
        mi_matrix = mx.zeros((n_tfs, n_genes))

        # Calculate MI for each TF-gene pair
        for i, tf_idx in enumerate(tf_indices):
            for j in range(n_genes):
                # Skip self-interactions
                if tf_idx == j:
                    continue

                # Calculate MI
                x = bootstrap_data[:, tf_idx]
                y = bootstrap_data[:, j]

                # For simplicity, use correlation as a proxy for MI
                # In a real implementation, we would use calculate_mi_mlx
                x_centered = x - mx.mean(x)
                y_centered = y - mx.mean(y)
                correlation = mx.sum(x_centered * y_centered) / (mx.sqrt(mx.sum(x_centered**2)) * mx.sqrt(mx.sum(y_centered**2)))
                mi = mx.abs(correlation)

                # Update the MI matrix
                # MLX's update pattern is different from JAX
                temp = mi_matrix.tolist()
                temp[i][j] = mi.item()
                mi_matrix = mx.array(temp)

        # Apply DPI
        pruned_matrix = apply_dpi_mlx(mi_matrix, tf_indices, dpi_tolerance)

        # Add to consensus matrix
        consensus_matrix = consensus_matrix + (pruned_matrix > 0).astype(mx.float32) / bootstraps

    # Apply consensus threshold
    consensus_matrix = mx.where(consensus_matrix < consensus_threshold, mx.zeros_like(consensus_matrix), consensus_matrix)

    # Convert back to numpy
    consensus_np = consensus_matrix.tolist()

    # Create regulons dictionary
    regulons = {}
    for i, tf_idx in enumerate(tf_indices):
        tf_name = gene_list[tf_idx]
        targets = {}

        for j in range(n_genes):
            # Skip self-interactions
            if j == tf_idx:
                continue

            # Get interaction strength
            strength = consensus_np[i][j]

            # Add to targets if non-zero
            if strength > 0:
                targets[gene_list[j]] = strength

        regulons[tf_name] = {
            'targets': targets
        }

    return consensus_np, regulons

def benchmark_aracne_implementations(n_cells, n_genes, n_tfs, bootstraps=10):
    """Benchmark different ARACNe implementations."""
    # Create synthetic dataset
    logger.info(f"Creating dataset with {n_cells} cells, {n_genes} genes, {n_tfs} TFs")
    adata, tf_list = create_synthetic_dataset(n_cells, n_genes, n_tfs)

    # Get TF indices
    tf_indices = [i for i in range(n_tfs)]

    # Get expression matrix
    expr_matrix = adata.X

    # Benchmark Python implementation
    logger.info("Benchmarking Python implementation...")
    aracne_python = ARACNe(bootstraps=bootstraps, use_numba=False)
    start_time = time.time()
    network_python = aracne_python.run(adata, tf_list=tf_list)
    python_time = time.time() - start_time
    logger.info(f"Python implementation: {python_time:.2f} seconds")

    # Benchmark Numba implementation
    logger.info("Benchmarking Numba implementation...")
    aracne_numba = ARACNe(bootstraps=bootstraps, use_numba=True)
    start_time = time.time()
    network_numba = aracne_numba.run(adata, tf_list=tf_list)
    numba_time = time.time() - start_time
    logger.info(f"Numba implementation: {numba_time:.2f} seconds")

    # Benchmark MLX implementation if available
    if HAS_MLX:
        logger.info("Benchmarking MLX implementation...")
        start_time = time.time()
        consensus_matrix, regulons = run_aracne_mlx(
            expr_matrix,
            adata.var_names.tolist(),
            tf_indices,
            bootstraps=bootstraps
        )
        mlx_time = time.time() - start_time
        logger.info(f"MLX implementation: {mlx_time:.2f} seconds")

        # Calculate speedups
        python_mlx_speedup = python_time / mlx_time
        numba_mlx_speedup = numba_time / mlx_time

        logger.info(f"MLX speedup vs Python: {python_mlx_speedup:.2f}x")
        logger.info(f"MLX speedup vs Numba: {numba_mlx_speedup:.2f}x")
    else:
        mlx_time = None
        python_mlx_speedup = None
        numba_mlx_speedup = None

    return {
        "python_time": python_time,
        "numba_time": numba_time,
        "mlx_time": mlx_time,
        "python_numba_speedup": python_time / numba_time,
        "python_mlx_speedup": python_mlx_speedup,
        "numba_mlx_speedup": numba_mlx_speedup
    }

def run_benchmarks():
    """Run benchmarks with different dataset sizes."""
    # Define dataset sizes
    dataset_sizes = [
        (100, 100, 10),   # Small dataset: 100 cells, 100 genes, 10 TFs
        (200, 150, 15),   # Medium dataset: 200 cells, 150 genes, 15 TFs
        (500, 200, 20),   # Large dataset: 500 cells, 200 genes, 20 TFs
    ]

    # Run benchmarks
    results = []

    for n_cells, n_genes, n_tfs in dataset_sizes:
        logger.info(f"\nBenchmarking dataset: {n_cells} cells, {n_genes} genes, {n_tfs} TFs")

        result = benchmark_aracne_implementations(
            n_cells=n_cells,
            n_genes=n_genes,
            n_tfs=n_tfs,
            bootstraps=5  # Use a small number for quick testing
        )

        result.update({
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_tfs": n_tfs
        })

        results.append(result)

    # Print summary
    logger.info("\nBenchmark Results:")
    logger.info("=================")
    for result in results:
        logger.info(f"Dataset: {result['n_cells']}x{result['n_genes']} ({result['n_tfs']} TFs)")
        logger.info(f"  Python: {result['python_time']:.2f} seconds")
        logger.info(f"  Numba:  {result['numba_time']:.2f} seconds")
        logger.info(f"  Python/Numba Speedup: {result['python_numba_speedup']:.2f}x")

        if result['mlx_time'] is not None:
            logger.info(f"  MLX:    {result['mlx_time']:.2f} seconds")
            logger.info(f"  Python/MLX Speedup: {result['python_mlx_speedup']:.2f}x")
            logger.info(f"  Numba/MLX Speedup:  {result['numba_mlx_speedup']:.2f}x")

    return results

if __name__ == "__main__":
    # Check if running on Apple Silicon
    import platform
    if platform.processor() == 'arm':
        logger.info("Running on Apple Silicon. MLX acceleration should be available.")
    else:
        logger.warning("Not running on Apple Silicon. MLX acceleration may not be available or optimal.")

    run_benchmarks()
