"""
Stratified benchmark for ARACNe implementations using real AnnData dataset.

This script benchmarks ARACNe with stratification by cell type on a real dataset.
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

# Monkey patch NumPy for compatibility with older code
if not hasattr(np, 'float_'):
    print("Monkey patching np.float_ for NumPy 2.0 compatibility")
    np.float_ = np.float64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print debug information
print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("AnnData version:", ad.__version__)
print("Platform:", platform.platform())
print("Processor:", platform.processor())

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pysces/src')))

# Import ARACNe directly from the core module
from pysces.aracne.core import ARACNe

def load_dataset():
    """Load a real AnnData dataset and subset it for faster testing."""
    try:
        # Load the Tabula Sapiens testis dataset
        logger.info("Loading Tabula Sapiens testis dataset...")
        adata = ad.read_h5ad("./tabula_sapiens_testis.h5ad")
        logger.info(f"Loaded dataset with {adata.n_obs} cells and {adata.n_vars} genes")

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
        adata = adata[:, top_genes]
        logger.info(f"Subset dataset has {adata.n_obs} cells and {adata.n_vars} genes")

        # Subset to a smaller number of cells for faster testing
        logger.info("Subsetting to 1000 random cells...")
        cell_indices = np.random.choice(adata.n_obs, size=1000, replace=False)
        adata = adata[cell_indices, :]
        logger.info(f"Final dataset has {adata.n_obs} cells and {adata.n_vars} genes")

        # Check if cell_type column exists
        if 'cell_type' not in adata.obs.columns:
            logger.warning("No 'cell_type' column found in dataset. Creating synthetic cell types.")
            # Create synthetic cell types
            n_cells = adata.n_obs
            adata.obs['cell_type'] = np.random.choice(['type_A', 'type_B', 'type_C'], size=n_cells)

        # Print cell type distribution
        logger.info("Cell type distribution:")
        for cell_type, count in adata.obs['cell_type'].value_counts().items():
            logger.info(f"  {cell_type}: {count} cells")

        return adata

    except Exception as e:
        logger.error(f"Error loading real dataset: {str(e)}")
        logger.info("Creating synthetic dataset instead...")

        # Create synthetic dataset
        n_cells = 500
        n_genes = 200

        # Create random expression matrix
        X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))

        # Create gene names
        gene_names = [f"gene_{i}" for i in range(n_genes)]

        # Create cell names
        cell_names = [f"cell_{i}" for i in range(n_cells)]

        # Create cell types
        cell_types = np.random.choice(['type_A', 'type_B', 'type_C'], size=n_cells)

        # Create AnnData object
        adata = ad.AnnData(X=X)
        adata.var_names = gene_names
        adata.obs_names = cell_names
        adata.obs['cell_type'] = cell_types

        # Print cell type distribution
        logger.info("Cell type distribution (synthetic):")
        for cell_type, count in pd.Series(cell_types).value_counts().items():
            logger.info(f"  {cell_type}: {count} cells")

        return adata

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

def benchmark_backend(adata, tf_list, backend, stratify=False, bootstraps=3):
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

        return run_time, len(network['edges']), True

    except Exception as e:
        logger.error(f"  {backend} {strat_desc} failed: {str(e)}")
        return None, None, False

def run_benchmark():
    """Run a comprehensive benchmark of all backends with stratification."""
    # Load real dataset
    adata = load_dataset()

    # Create a list of TFs (for simplicity, use the first 10 genes)
    tf_list = adata.var_names[:10].tolist()

    # Run warmup to compile Numba functions
    run_warmup(adata, tf_list)

    # Benchmark parameters
    bootstraps = 2  # Number of bootstrap iterations (reduced for faster testing)

    # Results storage
    results = defaultdict(dict)

    # Benchmark each backend with and without stratification
    backends = ['python', 'numba', 'pytorch']
    if platform.processor() == 'arm':
        backends.append('mlx')

    for backend in backends:
        # Without stratification
        time_no_strat, edges_no_strat, success_no_strat = benchmark_backend(
            adata, tf_list, backend, stratify=False, bootstraps=bootstraps
        )
        results[backend]['no_strat'] = (time_no_strat, edges_no_strat, success_no_strat)

        # With stratification
        time_strat, edges_strat, success_strat = benchmark_backend(
            adata, tf_list, backend, stratify=True, bootstraps=bootstraps
        )
        results[backend]['strat'] = (time_strat, edges_strat, success_strat)

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
        else:
            logger.info("  With stratification: Failed")

    # Calculate speedups
    logger.info("\nSpeedup Comparison (without stratification):")
    logger.info("=========================================")

    # Get Python time as baseline
    python_time = results['python']['no_strat'][0]
    if python_time:
        for backend in backends:
            if backend != 'python':
                backend_time = results[backend]['no_strat'][0]
                if backend_time:
                    speedup = python_time / backend_time
                    logger.info(f"{backend.capitalize()} vs Python: {speedup:.2f}x")

    logger.info("\nSpeedup Comparison (with stratification):")
    logger.info("======================================")

    # Get Python time as baseline
    python_time = results['python']['strat'][0]
    if python_time:
        for backend in backends:
            if backend != 'python':
                backend_time = results[backend]['strat'][0]
                if backend_time:
                    speedup = python_time / backend_time
                    logger.info(f"{backend.capitalize()} vs Python: {speedup:.2f}x")

if __name__ == "__main__":
    run_benchmark()
