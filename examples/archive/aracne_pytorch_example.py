"""
Example script demonstrating the PyTorch-accelerated ARACNe implementation.

This script loads a sample dataset and runs ARACNe with PyTorch acceleration.
"""

import os
import sys
import numpy as np
import logging
import time
import anndata as ad

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pysces/src')))

# Import PySCES
from pysces.aracne import ARACNe

def load_sample_data():
    """Load a sample dataset for testing."""
    try:
        # Try to load the Tabula Sapiens testis dataset
        adata = ad.read_h5ad("data/tabula_sapiens_testis.h5ad")
        logger.info(f"Loaded dataset with {adata.n_obs} cells and {adata.n_vars} genes")
        return adata
    except Exception as e:
        logger.error(f"Error loading sample data: {str(e)}")

        # Create a synthetic dataset
        logger.info("Creating synthetic dataset")
        n_cells = 200  # Reduced from 1000 for faster testing
        n_genes = 100  # Reduced from 500 for faster testing

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

        logger.info(f"Created synthetic dataset with {adata.n_obs} cells and {adata.n_vars} genes")
        return adata

def run_aracne_benchmark(adata):
    """Run ARACNe with different acceleration methods and compare performance."""
    # Create a list of TFs (for simplicity, use the first 10 genes)
    tf_list = adata.var_names[:10].tolist()  # Reduced from 50 for faster testing

    # Run ARACNe with Python implementation
    logger.info("Running ARACNe with Python implementation...")
    aracne_python = ARACNe(bootstraps=3, backend='python')  # Reduced from 10 for faster testing
    start_time = time.time()
    network_python = aracne_python.run(adata, tf_list=tf_list)
    python_time = time.time() - start_time
    logger.info(f"Python implementation: {python_time:.2f} seconds")

    # Run ARACNe with Numba acceleration
    logger.info("Running ARACNe with Numba acceleration...")
    aracne_numba = ARACNe(bootstraps=3, backend='numba')  # Reduced from 10 for faster testing
    start_time = time.time()
    network_numba = aracne_numba.run(adata, tf_list=tf_list)
    numba_time = time.time() - start_time
    logger.info(f"Numba implementation: {numba_time:.2f} seconds")

    # Run ARACNe with PyTorch acceleration
    logger.info("Running ARACNe with PyTorch acceleration...")
    aracne_pytorch = ARACNe(bootstraps=3, backend='pytorch')  # Reduced from 10 for faster testing
    start_time = time.time()
    network_pytorch = aracne_pytorch.run(adata, tf_list=tf_list)
    pytorch_time = time.time() - start_time
    logger.info(f"PyTorch implementation: {pytorch_time:.2f} seconds")

    # Run ARACNe with PyTorch acceleration and cell type stratification
    logger.info("Running ARACNe with PyTorch acceleration and cell type stratification...")
    aracne_pytorch_cell_type = ARACNe(bootstraps=3, backend='pytorch', stratify_by_cell_type=True)  # Reduced from 10 for faster testing
    start_time = time.time()
    network_pytorch_cell_type = aracne_pytorch_cell_type.run(adata, tf_list=tf_list)
    pytorch_cell_type_time = time.time() - start_time
    logger.info(f"PyTorch implementation with cell type stratification: {pytorch_cell_type_time:.2f} seconds")

    # Run ARACNe with PyTorch acceleration and tissue stratification
    if 'tissue' in adata.obs.columns:
        logger.info("Running ARACNe with PyTorch acceleration and tissue stratification...")
        aracne_pytorch_tissue = ARACNe(bootstraps=3, backend='pytorch', stratify_by_tissue=True)  # Reduced from 10 for faster testing
        start_time = time.time()
        network_pytorch_tissue = aracne_pytorch_tissue.run(adata, tf_list=tf_list)
        pytorch_tissue_time = time.time() - start_time
        logger.info(f"PyTorch implementation with tissue stratification: {pytorch_tissue_time:.2f} seconds")
    else:
        logger.info("Skipping tissue stratification (no 'tissue' column in adata.obs)")
        network_pytorch_tissue = None
        pytorch_tissue_time = None

    # Run ARACNe with PyTorch acceleration and both stratifications
    if 'tissue' in adata.obs.columns:
        logger.info("Running ARACNe with PyTorch acceleration and both stratifications...")
        aracne_pytorch_both = ARACNe(bootstraps=3, backend='pytorch',
                                    stratify_by_tissue=True, stratify_by_cell_type=True)  # Reduced from 10 for faster testing
        start_time = time.time()
        network_pytorch_both = aracne_pytorch_both.run(adata, tf_list=tf_list)
        pytorch_both_time = time.time() - start_time
        logger.info(f"PyTorch implementation with both stratifications: {pytorch_both_time:.2f} seconds")
    else:
        logger.info("Skipping combined stratification (no 'tissue' column in adata.obs)")
        network_pytorch_both = None
        pytorch_both_time = None

    # Calculate speedups
    numba_speedup = python_time / numba_time
    pytorch_speedup = python_time / pytorch_time
    pytorch_cell_type_speedup = python_time / pytorch_cell_type_time

    logger.info("\nPerformance Summary:")
    logger.info(f"Python implementation: {python_time:.2f} seconds")
    logger.info(f"Numba implementation: {numba_time:.2f} seconds (speedup: {numba_speedup:.2f}x)")
    logger.info(f"PyTorch implementation: {pytorch_time:.2f} seconds (speedup: {pytorch_speedup:.2f}x)")
    logger.info(f"PyTorch with cell type stratification: {pytorch_cell_type_time:.2f} seconds (speedup: {pytorch_cell_type_speedup:.2f}x)")

    if pytorch_tissue_time is not None:
        pytorch_tissue_speedup = python_time / pytorch_tissue_time
        logger.info(f"PyTorch with tissue stratification: {pytorch_tissue_time:.2f} seconds (speedup: {pytorch_tissue_speedup:.2f}x)")

    if pytorch_both_time is not None:
        pytorch_both_speedup = python_time / pytorch_both_time
        logger.info(f"PyTorch with both stratifications: {pytorch_both_time:.2f} seconds (speedup: {pytorch_both_speedup:.2f}x)")

    # Compare network sizes
    logger.info("\nNetwork Comparison:")
    logger.info(f"Python network: {len(network_python['edges'])} edges")
    logger.info(f"Numba network: {len(network_numba['edges'])} edges")
    logger.info(f"PyTorch network: {len(network_pytorch['edges'])} edges")
    logger.info(f"PyTorch with cell type stratification: {len(network_pytorch_cell_type['edges'])} edges")

    if network_pytorch_tissue is not None:
        logger.info(f"PyTorch with tissue stratification: {len(network_pytorch_tissue['edges'])} edges")

    if network_pytorch_both is not None:
        logger.info(f"PyTorch with both stratifications: {len(network_pytorch_both['edges'])} edges")

    # Prepare results dictionary
    results = {
        'python_time': python_time,
        'numba_time': numba_time,
        'pytorch_time': pytorch_time,
        'pytorch_cell_type_time': pytorch_cell_type_time,
        'python_network': network_python,
        'numba_network': network_numba,
        'pytorch_network': network_pytorch,
        'pytorch_cell_type_network': network_pytorch_cell_type
    }

    # Add optional results
    if pytorch_tissue_time is not None:
        results['pytorch_tissue_time'] = pytorch_tissue_time
        results['pytorch_tissue_network'] = network_pytorch_tissue

    if pytorch_both_time is not None:
        results['pytorch_both_time'] = pytorch_both_time
        results['pytorch_both_network'] = network_pytorch_both

    return results

if __name__ == "__main__":
    # Load sample data
    adata = load_sample_data()

    # Run ARACNe benchmark
    results = run_aracne_benchmark(adata)
