"""
Robust stratified benchmark for ARACNe implementations using the Tabula Sapiens dataset.

This script benchmarks ARACNe with stratification by cell type on the Tabula Sapiens dataset,
focusing only on cell types with at least 200 cells.
"""

import os
import sys
import numpy as np
import logging
import time
import anndata as ad
import scipy.sparse
import platform
import pandas as pd
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print debug information
print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("AnnData version:", ad.__version__)
print("Platform:", platform.platform())
print("Processor:", platform.processor())

# Monkey patch NumPy for compatibility with older code
if not hasattr(np, 'float_'):
    print("Monkey patching np.float_ for NumPy 2.0 compatibility")
    np.float_ = np.float64

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pysces/src')))

# Import ARACNe directly from the core module
from pysces.aracne.core import ARACNe

def load_and_subset_dataset():
    """Load the Tabula Sapiens dataset and subset it for faster testing."""
    try:
        # Load the dataset
        logger.info("Loading Tabula Sapiens testis dataset...")
        adata = ad.read_h5ad("./tabula_sapiens_testis.h5ad")
        logger.info(f"Loaded dataset with {adata.n_obs} cells and {adata.n_vars} genes")

        # Print cell type distribution
        logger.info("Cell type distribution:")
        cell_counts = adata.obs['cell_type'].value_counts()
        for cell_type, count in cell_counts.items():
            logger.info(f"  {cell_type}: {count} cells")

        # Filter to keep only cell types with at least 200 cells
        logger.info("Filtering to keep only cell types with at least 200 cells...")
        large_cell_types = cell_counts[cell_counts >= 200].index.tolist()
        logger.info(f"Cell types with at least 200 cells: {large_cell_types}")

        # Subset the dataset
        adata = adata[adata.obs['cell_type'].isin(large_cell_types)].copy()
        logger.info(f"Dataset after cell type filtering: {adata.n_obs} cells and {adata.n_vars} genes")

        # Subset to top 1000 highly variable genes
        logger.info("Subsetting to top 1000 highly variable genes...")
        if scipy.sparse.issparse(adata.X):
            gene_means = np.array(adata.X.mean(axis=0)).flatten()
            gene_vars = np.array(adata.X.power(2).mean(axis=0)).flatten() - gene_means**2
        else:
            gene_means = np.mean(adata.X, axis=0)
            gene_vars = np.var(adata.X, axis=0)

        # Calculate coefficient of variation
        cv = gene_vars / (gene_means + 1e-8)

        # Get top 1000 genes by CV
        top_genes = np.argsort(cv)[-1000:]
        adata = adata[:, top_genes].copy()
        logger.info(f"Final dataset: {adata.n_obs} cells and {adata.n_vars} genes")

        return adata

    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def run_warmup(adata, tf_list, bootstraps=2):
    """Run a warmup to compile Numba functions."""
    logger.info("Running warmup to compile Numba functions...")

    # Run with Numba backend
    aracne = ARACNe(bootstraps=bootstraps, backend='numba')
    try:
        _ = aracne.run(adata, tf_list=tf_list)
        logger.info("Warmup completed successfully")
    except Exception as e:
        logger.error(f"Warmup failed: {str(e)}")

def benchmark_backend(adata, tf_list, backend, stratify=False, bootstraps=2):
    """Benchmark a specific backend with optional stratification."""
    strat_desc = "with stratification" if stratify else "without stratification"
    logger.info(f"Benchmarking {backend} backend {strat_desc}...")

    # Create ARACNe object
    if stratify:
        aracne = ARACNe(bootstraps=bootstraps, backend=backend, stratify_by_cell_type=True)
    else:
        aracne = ARACNe(bootstraps=bootstraps, backend=backend)

    # Run benchmark
    start_time = time.time()
    try:
        network = aracne.run(adata, tf_list=tf_list)
        run_time = time.time() - start_time

        # Count edges per cell type if stratified
        if stratify and 'stratified_networks' in network:
            edge_counts = {}
            for cell_type, strat_network in network['stratified_networks'].items():
                if 'edges' in strat_network:
                    edge_counts[cell_type] = len(strat_network['edges'])
                else:
                    edge_counts[cell_type] = 0

            # Log edge counts per cell type
            logger.info(f"  {backend} {strat_desc}: {run_time:.2f} seconds")
            logger.info(f"  Total edges: {len(network['edges'])}")
            logger.info("  Edges per cell type:")
            for cell_type, count in edge_counts.items():
                logger.info(f"    {cell_type}: {count} edges")
        else:
            logger.info(f"  {backend} {strat_desc}: {run_time:.2f} seconds, {len(network['edges'])} edges")

        return run_time, len(network['edges']), True, network.get('stratified_networks', {})

    except Exception as e:
        logger.error(f"  {backend} {strat_desc} failed: {str(e)}")
        return None, None, False, {}

def run_benchmark():
    """Run a comprehensive benchmark of all backends with stratification."""
    # Load and subset dataset
    adata = load_and_subset_dataset()
    if adata is None:
        logger.error("Failed to load dataset. Exiting.")
        return

    # Create a list of TFs (for simplicity, use the first 10 genes)
    tf_list = adata.var_names[:10].tolist()
    logger.info(f"Using {len(tf_list)} TFs for benchmark: {tf_list}")

    # Run warmup to compile Numba functions
    run_warmup(adata, tf_list)

    # Benchmark parameters
    bootstraps = 2  # Number of bootstrap iterations (reduced for faster testing)

    # Results storage
    results = defaultdict(dict)

    # Benchmark each backend with and without stratification
    backends = ['numba', 'pytorch']
    if platform.processor() == 'arm':
        backends.append('mlx')

    for backend in backends:
        # Without stratification
        time_no_strat, edges_no_strat, success_no_strat, _ = benchmark_backend(
            adata, tf_list, backend, stratify=False, bootstraps=bootstraps
        )
        results[backend]['no_strat'] = (time_no_strat, edges_no_strat, success_no_strat)

        # With stratification
        time_strat, edges_strat, success_strat, strat_networks = benchmark_backend(
            adata, tf_list, backend, stratify=True, bootstraps=bootstraps
        )
        results[backend]['strat'] = (time_strat, edges_strat, success_strat)
        results[backend]['strat_networks'] = strat_networks

    # Print summary
    logger.info("\nBenchmark Summary:")
    logger.info("=================")
    logger.info(f"Dataset: {adata.n_obs} cells, {adata.n_vars} genes, {len(tf_list)} TFs")
    logger.info(f"Cell types: {adata.obs['cell_type'].nunique()}")
    logger.info(f"Bootstraps: {bootstraps}")
    logger.info("")

    # Print results for each backend
    for backend in backends:
        logger.info(f"{backend.capitalize()} Backend:")

        # Without stratification
        time_no_strat, edges_no_strat, success_no_strat = results[backend]['no_strat']
        if success_no_strat:
            logger.info(f"  Without stratification: {time_no_strat:.2f} seconds, {edges_no_strat} edges")
        else:
            logger.info("  Without stratification: Failed")

        # With stratification
        time_strat, edges_strat, success_strat = results[backend]['strat']
        if success_strat:
            logger.info(f"  With stratification: {time_strat:.2f} seconds, {edges_strat} edges")

            # Print edge counts per cell type
            strat_networks = results[backend]['strat_networks']
            if strat_networks:
                logger.info("  Edges per cell type:")
                for cell_type, network in strat_networks.items():
                    if 'edges' in network:
                        logger.info(f"    {cell_type}: {len(network['edges'])} edges")
        else:
            logger.info("  With stratification: Failed")

    # Calculate speedups
    logger.info("\nSpeedup Comparison (without stratification):")
    logger.info("=========================================")

    # Get Numba time as baseline
    numba_time = results['numba']['no_strat'][0]
    if numba_time:
        for backend in backends:
            if backend != 'numba':
                backend_time = results[backend]['no_strat'][0]
                if backend_time:
                    speedup = numba_time / backend_time
                    logger.info(f"{backend.capitalize()} vs Numba: {speedup:.2f}x")

    logger.info("\nSpeedup Comparison (with stratification):")
    logger.info("======================================")

    # Get Numba time as baseline
    numba_time = results['numba']['strat'][0]
    if numba_time:
        for backend in backends:
            if backend != 'numba':
                backend_time = results[backend]['strat'][0]
                if backend_time:
                    speedup = numba_time / backend_time
                    logger.info(f"{backend.capitalize()} vs Numba: {speedup:.2f}x")

if __name__ == "__main__":
    run_benchmark()
