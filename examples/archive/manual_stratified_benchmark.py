"""
Manual stratified benchmark for ARACNe implementations.

This script manually stratifies the data by cell type and runs ARACNe separately on each stratum,
allowing for more direct comparison of performance across different backends.
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

def stratify_dataset(adata):
    """Stratify the dataset by cell type."""
    logger.info("Stratifying dataset by cell type...")
    
    # Get unique cell types
    cell_types = adata.obs['cell_type'].unique()
    
    # Create a dictionary to store stratified datasets
    strata = {}
    
    # Stratify by cell type
    for cell_type in cell_types:
        # Get indices for this cell type
        indices = adata.obs[adata.obs['cell_type'] == cell_type].index
        
        # Create a subset for this cell type
        stratum = adata[indices].copy()
        
        # Add to strata dictionary
        strata[cell_type] = stratum
        
        logger.info(f"  {cell_type}: {stratum.n_obs} cells")
    
    return strata

def benchmark_backend_on_stratum(stratum, tf_list, backend, bootstraps=2):
    """Benchmark a specific backend on a single stratum."""
    # Create ARACNe object
    aracne = ARACNe(bootstraps=bootstraps, backend=backend)
    
    # Run benchmark
    start_time = time.time()
    try:
        network = aracne.run(stratum, tf_list=tf_list)
        run_time = time.time() - start_time
        
        # Count edges
        edge_count = len(network['edges'])
        
        return run_time, edge_count, True
    
    except Exception as e:
        logger.error(f"  Benchmark failed: {str(e)}")
        return None, None, False

def run_benchmark():
    """Run a comprehensive benchmark of backends on stratified data."""
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
    
    # Stratify dataset
    strata = stratify_dataset(adata)
    
    # Benchmark parameters
    bootstraps = 2  # Number of bootstrap iterations (reduced for faster testing)
    
    # Backends to benchmark
    backends = ['numba', 'pytorch']
    
    # Results storage
    results = defaultdict(lambda: defaultdict(dict))
    
    # Benchmark each backend on each stratum
    for backend in backends:
        logger.info(f"\nBenchmarking {backend} backend on each cell type...")
        
        for cell_type, stratum in strata.items():
            logger.info(f"  Running on {cell_type} ({stratum.n_obs} cells)...")
            
            run_time, edge_count, success = benchmark_backend_on_stratum(
                stratum, tf_list, backend, bootstraps
            )
            
            if success:
                logger.info(f"    {run_time:.2f} seconds, {edge_count} edges")
                results[backend][cell_type] = {
                    'time': run_time,
                    'edges': edge_count,
                    'success': True
                }
            else:
                logger.info(f"    Failed")
                results[backend][cell_type] = {
                    'time': None,
                    'edges': None,
                    'success': False
                }
    
    # Print summary
    logger.info("\nBenchmark Summary:")
    logger.info("=================")
    logger.info(f"Dataset: {adata.n_obs} cells and {adata.n_vars} genes")
    logger.info(f"TFs: {len(tf_list)}")
    logger.info(f"Bootstraps: {bootstraps}")
    logger.info("")
    
    # Print results for each backend and cell type
    for backend in backends:
        logger.info(f"{backend.capitalize()} Backend:")
        
        for cell_type in strata.keys():
            result = results[backend][cell_type]
            
            if result['success']:
                logger.info(f"  {cell_type}: {result['time']:.2f} seconds, {result['edges']} edges")
            else:
                logger.info(f"  {cell_type}: Failed")
    
    # Calculate speedups between backends
    logger.info("\nSpeedup Comparison:")
    logger.info("=================")
    
    if len(backends) > 1:
        for cell_type in strata.keys():
            logger.info(f"Cell type: {cell_type}")
            
            # Compare each pair of backends
            for i, backend1 in enumerate(backends):
                for backend2 in backends[i+1:]:
                    result1 = results[backend1][cell_type]
                    result2 = results[backend2][cell_type]
                    
                    if result1['success'] and result2['success']:
                        speedup = result1['time'] / result2['time']
                        logger.info(f"  {backend2} vs {backend1}: {speedup:.2f}x")

if __name__ == "__main__":
    run_benchmark()
