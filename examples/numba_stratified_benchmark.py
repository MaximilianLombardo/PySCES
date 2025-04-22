"""
Benchmark for Numba-accelerated ARACNe with stratification.

This script benchmarks the Numba-accelerated ARACNe implementation with cell type stratification
on the Tabula Sapiens dataset, focusing only on cell types with at least 200 cells.
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

def benchmark_numba_stratified(adata, tf_list, bootstraps=2):
    """Benchmark Numba-accelerated ARACNe with cell type stratification."""
    logger.info("Benchmarking Numba-accelerated ARACNe with cell type stratification...")

    # Create ARACNe object with stratification
    aracne = ARACNe(bootstraps=bootstraps, backend='numba', stratify_by_cell_type=True)

    # Run benchmark
    start_time = time.time()
    try:
        network = aracne.run(adata, tf_list=tf_list)
        run_time = time.time() - start_time

        # Get cell types from the dataset
        cell_types = adata.obs['cell_type'].unique()

        # Count edges per cell type (if available)
        edge_counts = {}
        if 'stratified_networks' in network:
            for cell_type, strat_network in network['stratified_networks'].items():
                if 'edges' in strat_network:
                    edge_counts[cell_type] = len(strat_network['edges'])
                else:
                    edge_counts[cell_type] = 0
        else:
            # If stratified networks are not available, just report the total edges
            logger.info("Stratified networks not available in the result")
            for cell_type in cell_types:
                edge_counts[cell_type] = 'N/A'

        # Log results
        logger.info(f"Total runtime: {run_time:.2f} seconds")
        logger.info(f"Total edges: {len(network['edges'])}")
        logger.info("Edges per cell type:")
        for cell_type, count in edge_counts.items():
            logger.info(f"  {cell_type}: {count} edges" if count != 'N/A' else f"  {cell_type}: {count}")

        return run_time, network, edge_counts

    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        return None, None, None

def run_benchmark():
    """Run the Numba stratified benchmark."""
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

    # Run benchmark
    run_time, network, edge_counts = benchmark_numba_stratified(adata, tf_list, bootstraps)

    if run_time is not None:
        # Print summary
        logger.info("\nBenchmark Summary:")
        logger.info("=================")
        logger.info(f"Dataset: {adata.n_obs} cells and {adata.n_vars} genes")
        logger.info(f"TFs: {len(tf_list)}")
        logger.info(f"Bootstraps: {bootstraps}")
        logger.info(f"Total runtime: {run_time:.2f} seconds")
        logger.info(f"Total edges: {len(network['edges'])}")

        # Print edges per cell type
        logger.info("\nEdges per cell type:")
        if 'stratified_networks' in network:
            for cell_type, strat_network in network['stratified_networks'].items():
                if 'edges' in strat_network:
                    logger.info(f"  {cell_type}: {len(strat_network['edges'])} edges")
        else:
            for cell_type, count in edge_counts.items():
                logger.info(f"  {cell_type}: {count}")

if __name__ == "__main__":
    run_benchmark()
