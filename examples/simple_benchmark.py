"""
Simple benchmark script for ARACNe implementation with and without Numba acceleration.
"""

import numpy as np
import time
import logging
import sys
import os
import importlib.util

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core module
core_path = os.path.join(os.path.dirname(__file__), '..', 'pysces', 'src', 'pysces', 'aracne', 'core.py')
spec = importlib.util.spec_from_file_location("core", core_path)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

# Import numba_optimized module
numba_path = os.path.join(os.path.dirname(__file__), '..', 'pysces', 'src', 'pysces', 'aracne', 'numba_optimized.py')
spec = importlib.util.spec_from_file_location("numba_optimized", numba_path)
numba_optimized = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numba_optimized)

# Set up ARACNe class
ARACNe = core.ARACNe

# Make sure the core module can find the numba_optimized module
core.numba_optimized = numba_optimized
core.run_aracne_numba = numba_optimized.run_aracne_numba
core.HAS_NUMBA = numba_optimized.HAS_NUMBA

def create_synthetic_dataset(n_cells, n_genes, n_tfs):
    """Create a synthetic dataset for benchmarking."""
    # Create random expression data
    expr_matrix = np.random.rand(n_cells, n_genes)

    # Create gene list
    gene_list = [f"gene_{i}" for i in range(n_genes)]

    # Create TF indices
    tf_indices = np.arange(n_tfs)

    return expr_matrix, gene_list, tf_indices

def benchmark_aracne_python(expr_matrix, gene_list, tf_indices, bootstraps=3):
    """Benchmark ARACNe Python implementation."""
    # Initialize ARACNe
    aracne = ARACNe(
        bootstraps=bootstraps,
        p_value=0.05,
        dpi_tolerance=0.1,
        consensus_threshold=0.5,
        use_numba=False
    )

    # Run ARACNe and measure time
    start_time = time.time()
    result = aracne._run_aracne_python(expr_matrix, gene_list, tf_indices)
    elapsed_time = time.time() - start_time

    return elapsed_time, result

def benchmark_aracne_numba(expr_matrix, gene_list, tf_indices, bootstraps=3):
    """Benchmark ARACNe Numba implementation."""
    # Perform a warmup run for Numba to compile the functions
    logger.info("Performing warmup run for Numba compilation...")
    # Create a smaller dataset for warmup
    warmup_expr, warmup_genes, warmup_tfs = create_synthetic_dataset(50, 50, 5)
    # Run with smaller bootstraps
    numba_optimized.run_aracne_numba(warmup_expr, warmup_genes, warmup_tfs, bootstraps=1)
    logger.info("Warmup complete")

    # Run ARACNe and measure time
    start_time = time.time()
    consensus_matrix, regulons = numba_optimized.run_aracne_numba(
        expr_matrix, gene_list, tf_indices, bootstraps=bootstraps
    )
    elapsed_time = time.time() - start_time

    return elapsed_time, (consensus_matrix, regulons)

def run_benchmarks():
    """Run benchmarks with different dataset sizes."""
    # Define dataset sizes
    dataset_sizes = [
        (100, 100, 10),   # Small dataset
        (200, 150, 15),   # Medium dataset
        (500, 200, 20),   # Large dataset
        (1000, 500, 50),  # Very large dataset
    ]

    # Run benchmarks
    results = []

    for n_cells, n_genes, n_tfs in dataset_sizes:
        logger.info(f"Creating dataset with {n_cells} cells, {n_genes} genes, {n_tfs} TFs")
        expr_matrix, gene_list, tf_indices = create_synthetic_dataset(n_cells, n_genes, n_tfs)

        # Benchmark Python implementation
        logger.info("Benchmarking Python implementation...")
        python_time, _ = benchmark_aracne_python(expr_matrix, gene_list, tf_indices)
        logger.info(f"Python implementation: {python_time:.2f} seconds")

        # Benchmark Numba implementation
        logger.info("Benchmarking Numba implementation...")
        numba_time, _ = benchmark_aracne_numba(expr_matrix, gene_list, tf_indices)
        logger.info(f"Numba implementation: {numba_time:.2f} seconds")

        # Calculate speedup
        speedup = python_time / numba_time
        logger.info(f"Speedup: {speedup:.2f}x")

        results.append({
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_tfs": n_tfs,
            "python_time": python_time,
            "numba_time": numba_time,
            "speedup": speedup
        })

    # Print summary
    logger.info("\nBenchmark Results:")
    logger.info("=================")
    for result in results:
        logger.info(f"Dataset: {result['n_cells']}x{result['n_genes']} ({result['n_tfs']} TFs)")
        logger.info(f"  Python: {result['python_time']:.2f} seconds")
        logger.info(f"  Numba:  {result['numba_time']:.2f} seconds")
        logger.info(f"  Speedup: {result['speedup']:.2f}x")

    return results

if __name__ == "__main__":
    run_benchmarks()
