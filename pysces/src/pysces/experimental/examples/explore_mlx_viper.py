"""
Explore MLX optimization for VIPER algorithm.

This script demonstrates how to use MLX to accelerate the VIPER algorithm
on Apple Silicon hardware.
"""

import numpy as np
import pandas as pd
import time
import logging
import sys
import os
import scipy.stats
import scipy.sparse
from anndata import AnnData

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pysces', 'src'))

# Import VIPER modules
from pysces.viper.regulons import Regulon, GeneSet
from pysces.viper.core import viper_scores

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

def create_synthetic_regulons(gene_list, n_tfs, targets_per_tf=20):
    """Create synthetic regulons for benchmarking."""
    regulons = []

    for i in range(n_tfs):
        tf_name = f"gene_{i}"
        regulon = Regulon(tf_name)

        # Add random targets
        n_genes = len(gene_list)
        target_indices = np.random.choice(n_genes, targets_per_tf, replace=False)

        for idx in target_indices:
            target_gene = gene_list[idx]
            mode = 1 if np.random.rand() > 0.5 else -1
            likelihood = np.random.rand()
            regulon.add_target(target_gene, mode, likelihood)

        regulons.append(regulon)

    return regulons

def calculate_signature_mlx(expr_mx, method='rank'):
    """
    Calculate gene expression signatures using MLX.

    Parameters
    ----------
    expr_mx : mx.array
        Expression matrix (genes x cells)
    method : str, default='rank'
        Method for calculating signatures

    Returns
    -------
    mx.array
        Gene expression signatures (genes x cells)
    """
    if method == 'scale':
        # Center and scale the data
        mean = mx.mean(expr_mx, axis=1, keepdims=True)
        std = mx.std(expr_mx, axis=1, keepdims=True)
        signatures = (expr_mx - mean) / std
        return signatures

    elif method == 'rank':
        # For rank method, we need to convert to numpy, rank, and convert back
        # This is because MLX doesn't have a rank function yet
        expr_np = expr_mx.tolist()
        n_genes, n_cells = expr_np.shape

        signatures_np = np.zeros_like(expr_np)
        for i in range(n_cells):
            # Rank transform the column
            ranks = np.argsort(np.argsort(expr_np[:, i])) + 1
            # Scale to [0, 1]
            ranks = ranks / (n_genes + 1)
            # Convert to z-scores
            signatures_np[:, i] = scipy.stats.norm.ppf(ranks)

        # Convert back to MLX
        signatures = mx.array(signatures_np)
        return signatures

    else:
        raise ValueError(f"Unsupported method: {method}")

def calculate_enrichment_score_mlx(signature_mx, indices, weights, method='gsea', abs_score=False):
    """
    Calculate enrichment score for a gene set in a signature using MLX.

    Parameters
    ----------
    signature_mx : mx.array
        Gene expression signature (genes,)
    indices : mx.array
        Indices of genes in the gene set
    weights : mx.array
        Weights of genes in the gene set
    method : str, default='gsea'
        Method for calculating enrichment
    abs_score : bool, default=False
        Whether to use absolute values of the signature

    Returns
    -------
    float
        Enrichment score
    """
    if len(indices) == 0:
        return 0.0

    # Get signature values for genes in the gene set
    values = mx.take(signature_mx, indices)

    # Use absolute values if requested
    if abs_score:
        values = mx.abs(values)

    # Calculate enrichment score based on method
    if method == 'mean':
        # Weighted mean
        weight_sum = mx.sum(weights)
        if weight_sum == 0:
            return 0.0
        return mx.sum(values * weights) / weight_sum

    elif method == 'sum':
        # Weighted sum
        return mx.sum(values * weights)

    elif method == 'gsea':
        # For GSEA, we need to convert to numpy for now
        # This is because MLX doesn't have all the operations we need
        values_np = values.tolist()
        weights_np = weights.tolist()

        # Sort genes by signature value
        sorted_indices = np.argsort(values_np)[::-1]
        sorted_values = np.array(values_np)[sorted_indices]
        sorted_weights = np.array(weights_np)[sorted_indices]

        # Calculate running sum
        weight_sum = np.sum(sorted_weights)
        if weight_sum == 0:
            return 0.0

        running_sum = np.cumsum(sorted_weights) / weight_sum

        # Calculate null distribution (uniform)
        null_dist = np.arange(1, len(running_sum) + 1) / len(running_sum)

        # Find maximum deviation from null distribution
        deviations = running_sum - null_dist
        max_pos = np.max(deviations)
        max_neg = np.min(deviations)

        # Return enrichment score with sign
        if max_pos > -max_neg:
            return float(max_pos)
        else:
            return float(max_neg)

    else:
        raise ValueError(f"Unsupported method: {method}")

def viper_scores_mlx(adata, regulons, layer=None, signature_method='rank', enrichment_method='gsea', abs_score=False, normalize=True):
    """
    Calculate VIPER scores using MLX for GPU acceleration.

    Parameters
    ----------
    adata : AnnData
        Gene expression data (cells x genes)
    regulons : list of Regulon
        List of regulons to calculate enrichment for
    layer : str, optional
        Layer in AnnData to use. If None, uses .X
    signature_method : str, default='rank'
        Method for calculating signatures
    enrichment_method : str, default='gsea'
        Method for calculating enrichment
    abs_score : bool, default=False
        Whether to use absolute values of the signatures
    normalize : bool, default=True
        Whether to normalize enrichment scores

    Returns
    -------
    pd.DataFrame
        VIPER scores (regulons x cells)
    """
    if not HAS_MLX:
        logger.warning("MLX not available. Falling back to CPU implementation.")
        return viper_scores(
            adata,
            regulons,
            layer=layer,
            signature_method=signature_method,
            enrichment_method=enrichment_method,
            abs_score=abs_score,
            normalize=normalize
        )

    # Get expression data
    if layer is None:
        expr = adata.X
    else:
        expr = adata.layers[layer]

    # If sparse, convert to dense
    if hasattr(expr, 'toarray'):
        expr = expr.toarray()

    # Transpose if needed (ensure genes x cells)
    if expr.shape[0] == adata.n_obs and expr.shape[1] == adata.n_vars:
        # Data is in cells x genes format, transpose to genes x cells
        expr = expr.T

    # Convert to MLX array
    expr_mx = mx.array(expr)

    # Calculate signatures
    if signature_method == 'scale':
        # Center and scale the data
        mean = mx.mean(expr_mx, axis=1, keepdims=True)
        std = mx.std(expr_mx, axis=1, keepdims=True)
        signatures = (expr_mx - mean) / std
    else:
        # For other methods, fall back to numpy and convert to MLX
        # Get expression data
        if layer is None:
            expr_np = adata.X
        else:
            expr_np = adata.layers[layer]

        # If sparse, convert to dense
        if hasattr(expr_np, 'toarray'):
            expr_np = expr_np.toarray()

        # Transpose if needed (ensure genes x cells)
        if expr_np.shape[0] == adata.n_obs and expr_np.shape[1] == adata.n_vars:
            # Data is in cells x genes format, transpose to genes x cells
            expr_np = expr_np.T

        # Apply simple scaling
        mean = np.mean(expr_np, axis=1, keepdims=True)
        std = np.std(expr_np, axis=1, keepdims=True)
        signatures_np = (expr_np - mean) / (std + 1e-10)
        signatures = mx.array(signatures_np)

    # Create gene name to index mapping
    gene_names = adata.var_names.tolist()
    gene_indices = {gene: i for i, gene in enumerate(gene_names)}

    # Initialize NES matrix
    n_regulons = len(regulons)
    n_cells = expr_mx.shape[1]
    nes_matrix = mx.zeros((n_regulons, n_cells))

    # Calculate enrichment for each regulon
    for i, regulon in enumerate(regulons):
        # Get genes in the regulon that are also in the data
        common_genes = [gene for gene in regulon.genes if gene in gene_indices]

        if not common_genes:
            continue

        # Get indices and weights of common genes
        indices = mx.array([gene_indices[gene] for gene in common_genes])
        weights = mx.array([regulon.weights.get(gene, 1.0) for gene in common_genes])

        # Calculate enrichment for each cell
        for j in range(n_cells):
            score = calculate_enrichment_score_mlx(
                signatures[:, j],
                indices,
                weights,
                method=enrichment_method,
                abs_score=abs_score
            )
            # Create a new array with the updated value
            # MLX's update pattern is different from JAX
            temp = nes_matrix.tolist()
            temp[i][j] = score
            nes_matrix = mx.array(temp)

    # Normalize if requested
    if normalize:
        # Normalize by standard deviation across cells
        std = mx.std(nes_matrix, axis=1, keepdims=True)
        nes_matrix = nes_matrix / std

    # Convert back to numpy and create DataFrame
    nes_np = nes_matrix.tolist()
    viper_df = pd.DataFrame(
        nes_np,
        index=[r.tf_name for r in regulons],
        columns=adata.obs_names
    )

    return viper_df

def benchmark_viper_implementations(n_cells, n_genes, n_tfs, targets_per_tf=20):
    """Benchmark different VIPER implementations."""
    # Create synthetic dataset
    logger.info(f"Creating dataset with {n_cells} cells, {n_genes} genes, {n_tfs} TFs")
    adata, tf_list = create_synthetic_dataset(n_cells, n_genes, n_tfs)

    # Create synthetic regulons
    logger.info(f"Creating {n_tfs} regulons with {targets_per_tf} targets each")
    regulons = create_synthetic_regulons(adata.var_names.tolist(), n_tfs, targets_per_tf)

    # Benchmark Python implementation
    logger.info("Benchmarking Python implementation...")
    start_time = time.time()
    python_result = viper_scores(
        adata,
        regulons,
        signature_method='rank',
        enrichment_method='gsea',
        use_numba=False
    )
    python_time = time.time() - start_time
    logger.info(f"Python implementation: {python_time:.2f} seconds")

    # Benchmark Numba implementation
    logger.info("Benchmarking Numba implementation...")
    start_time = time.time()
    numba_result = viper_scores(
        adata,
        regulons,
        signature_method='rank',
        enrichment_method='gsea',
        use_numba=True
    )
    numba_time = time.time() - start_time
    logger.info(f"Numba implementation: {numba_time:.2f} seconds")

    # Benchmark MLX implementation if available
    if HAS_MLX:
        logger.info("Benchmarking MLX implementation...")
        start_time = time.time()
        mlx_result = viper_scores_mlx(
            adata,
            regulons,
            signature_method='rank',
            enrichment_method='gsea'
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
        (100, 100, 10, 20),   # Small dataset: 100 cells, 100 genes, 10 TFs, 20 targets per TF
        (500, 200, 20, 30),   # Medium dataset: 500 cells, 200 genes, 20 TFs, 30 targets per TF
        (1000, 500, 50, 50),  # Large dataset: 1000 cells, 500 genes, 50 TFs, 50 targets per TF
    ]

    # Run benchmarks
    results = []

    for n_cells, n_genes, n_tfs, targets_per_tf in dataset_sizes:
        logger.info(f"\nBenchmarking dataset: {n_cells} cells, {n_genes} genes, {n_tfs} TFs, {targets_per_tf} targets per TF")

        result = benchmark_viper_implementations(
            n_cells=n_cells,
            n_genes=n_genes,
            n_tfs=n_tfs,
            targets_per_tf=targets_per_tf
        )

        result.update({
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_tfs": n_tfs,
            "targets_per_tf": targets_per_tf
        })

        results.append(result)

    # Print summary
    logger.info("\nBenchmark Results:")
    logger.info("=================")
    for result in results:
        logger.info(f"Dataset: {result['n_cells']}x{result['n_genes']} ({result['n_tfs']} TFs, {result['targets_per_tf']} targets per TF)")
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
