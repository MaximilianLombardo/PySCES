"""
Simple test script for ARACNe implementation.

This script tests the ARACNe implementation with a small synthetic dataset.
"""

import os
import sys
import numpy as np
import logging
import time
import anndata as ad

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print debug information
print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("AnnData version:", ad.__version__)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pysces/src')))

# Import ARACNe directly from the core module
from pysces.aracne.core import ARACNe

def create_synthetic_dataset():
    """Create a small synthetic dataset for testing."""
    logger.info("Creating synthetic dataset")
    n_cells = 100  # Small dataset for quick testing
    n_genes = 50   # Small dataset for quick testing

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

def run_aracne_test():
    """Run a simple test of ARACNe with different backends."""
    # Create synthetic dataset
    adata = create_synthetic_dataset()

    # Create a list of TFs (for simplicity, use the first 5 genes)
    tf_list = adata.var_names[:5].tolist()

    # Test parameters
    bootstraps = 2  # Minimal for testing

    # Test Python backend
    logger.info("Testing Python backend...")
    aracne_python = ARACNe(bootstraps=bootstraps, backend='python')
    start_time = time.time()
    try:
        network_python = aracne_python.run(adata, tf_list=tf_list)
        python_time = time.time() - start_time
        logger.info(f"Python backend: {python_time:.2f} seconds, {len(network_python['edges'])} edges")
        python_success = True
    except Exception as e:
        logger.error(f"Python backend failed: {str(e)}")
        python_success = False

    # Test Numba backend
    logger.info("Testing Numba backend...")
    aracne_numba = ARACNe(bootstraps=bootstraps, backend='numba')
    start_time = time.time()
    try:
        network_numba = aracne_numba.run(adata, tf_list=tf_list)
        numba_time = time.time() - start_time
        logger.info(f"Numba backend: {numba_time:.2f} seconds, {len(network_numba['edges'])} edges")
        numba_success = True
    except Exception as e:
        logger.error(f"Numba backend failed: {str(e)}")
        numba_success = False

    # Test PyTorch backend
    logger.info("Testing PyTorch backend...")
    aracne_pytorch = ARACNe(bootstraps=bootstraps, backend='pytorch')
    start_time = time.time()
    try:
        network_pytorch = aracne_pytorch.run(adata, tf_list=tf_list)
        pytorch_time = time.time() - start_time
        logger.info(f"PyTorch backend: {pytorch_time:.2f} seconds, {len(network_pytorch['edges'])} edges")
        pytorch_success = True
    except Exception as e:
        logger.error(f"PyTorch backend failed: {str(e)}")
        pytorch_success = False

    # Test MLX backend (if on Apple Silicon)
    import platform
    if platform.processor() == 'arm':
        logger.info("Testing MLX backend...")
        aracne_mlx = ARACNe(bootstraps=bootstraps, backend='mlx')
        start_time = time.time()
        try:
            network_mlx = aracne_mlx.run(adata, tf_list=tf_list)
            mlx_time = time.time() - start_time
            logger.info(f"MLX backend: {mlx_time:.2f} seconds, {len(network_mlx['edges'])} edges")
            mlx_success = True
        except Exception as e:
            logger.error(f"MLX backend failed: {str(e)}")
            mlx_success = False
    else:
        logger.info("Skipping MLX backend (not on Apple Silicon)")
        mlx_success = False

    # Print summary
    logger.info("\nSummary:")
    logger.info(f"Python backend: {'Success' if python_success else 'Failed'}")
    logger.info(f"Numba backend: {'Success' if numba_success else 'Failed'}")
    logger.info(f"PyTorch backend: {'Success' if pytorch_success else 'Failed'}")
    if platform.processor() == 'arm':
        logger.info(f"MLX backend: {'Success' if mlx_success else 'Failed'}")

    # Calculate speedups if successful
    if python_success and numba_success:
        logger.info(f"Numba speedup vs Python: {python_time / numba_time:.2f}x")

    if python_success and pytorch_success:
        logger.info(f"PyTorch speedup vs Python: {python_time / pytorch_time:.2f}x")

    if numba_success and pytorch_success:
        logger.info(f"PyTorch speedup vs Numba: {numba_time / pytorch_time:.2f}x")

    if python_success and mlx_success:
        logger.info(f"MLX speedup vs Python: {python_time / mlx_time:.2f}x")

    if numba_success and mlx_success:
        logger.info(f"MLX speedup vs Numba: {numba_time / mlx_time:.2f}x")

    if pytorch_success and mlx_success:
        logger.info(f"MLX speedup vs PyTorch: {pytorch_time / mlx_time:.2f}x")

if __name__ == "__main__":
    run_aracne_test()
