"""
Robust benchmark for ARACNe implementations.

This script provides a more robust benchmark for comparing different ARACNe implementations.
"""

import os
import sys
import numpy as np
import logging
import time
import anndata as ad
import platform

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

def create_synthetic_dataset(n_cells=500, n_genes=200):
    """Create a synthetic dataset for benchmarking."""
    logger.info(f"Creating synthetic dataset with {n_cells} cells and {n_genes} genes")
    
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

def benchmark_backend(adata, tf_list, backend, bootstraps=5, iterations=3):
    """Benchmark a specific backend."""
    logger.info(f"Benchmarking {backend} backend...")
    
    # Create ARACNe object
    aracne = ARACNe(bootstraps=bootstraps, backend=backend)
    
    # Run multiple iterations
    times = []
    edges = []
    success = False
    
    for i in range(iterations):
        logger.info(f"  Iteration {i+1}/{iterations}")
        start_time = time.time()
        try:
            network = aracne.run(adata, tf_list=tf_list)
            iteration_time = time.time() - start_time
            times.append(iteration_time)
            edges.append(len(network['edges']))
            logger.info(f"  Iteration {i+1}: {iteration_time:.2f} seconds, {len(network['edges'])} edges")
            success = True
        except Exception as e:
            logger.error(f"  Iteration {i+1} failed: {str(e)}")
    
    # Calculate average time and edges
    if success:
        avg_time = sum(times) / len(times)
        avg_edges = sum(edges) / len(edges)
        logger.info(f"{backend} backend: {avg_time:.2f} seconds (avg), {avg_edges:.0f} edges (avg)")
        return avg_time, avg_edges, success
    else:
        logger.error(f"{backend} backend failed all iterations")
        return None, None, False

def run_benchmark():
    """Run a comprehensive benchmark of all backends."""
    # Create synthetic dataset
    adata = create_synthetic_dataset(n_cells=500, n_genes=200)
    
    # Create a list of TFs (for simplicity, use the first 20 genes)
    tf_list = adata.var_names[:20].tolist()
    
    # Run warmup to compile Numba functions
    run_warmup(adata, tf_list)
    
    # Benchmark parameters
    bootstraps = 5  # Number of bootstrap iterations
    iterations = 3  # Number of benchmark iterations
    
    # Benchmark Python backend
    python_time, python_edges, python_success = benchmark_backend(
        adata, tf_list, 'python', bootstraps, iterations
    )
    
    # Benchmark Numba backend
    numba_time, numba_edges, numba_success = benchmark_backend(
        adata, tf_list, 'numba', bootstraps, iterations
    )
    
    # Benchmark PyTorch backend
    pytorch_time, pytorch_edges, pytorch_success = benchmark_backend(
        adata, tf_list, 'pytorch', bootstraps, iterations
    )
    
    # Benchmark MLX backend (if on Apple Silicon)
    if platform.processor() == 'arm':
        mlx_time, mlx_edges, mlx_success = benchmark_backend(
            adata, tf_list, 'mlx', bootstraps, iterations
        )
    else:
        logger.info("Skipping MLX backend (not on Apple Silicon)")
        mlx_time, mlx_edges, mlx_success = None, None, False
    
    # Print summary
    logger.info("\nBenchmark Summary:")
    logger.info("=================")
    logger.info(f"Dataset: {adata.n_obs} cells, {adata.n_vars} genes, {len(tf_list)} TFs")
    logger.info(f"Bootstraps: {bootstraps}, Iterations: {iterations}")
    logger.info("")
    
    if python_success:
        logger.info(f"Python backend: {python_time:.2f} seconds, {python_edges:.0f} edges")
    else:
        logger.info("Python backend: Failed")
    
    if numba_success:
        logger.info(f"Numba backend: {numba_time:.2f} seconds, {numba_edges:.0f} edges")
    else:
        logger.info("Numba backend: Failed")
    
    if pytorch_success:
        logger.info(f"PyTorch backend: {pytorch_time:.2f} seconds, {pytorch_edges:.0f} edges")
    else:
        logger.info("PyTorch backend: Failed")
    
    if platform.processor() == 'arm':
        if mlx_success:
            logger.info(f"MLX backend: {mlx_time:.2f} seconds, {mlx_edges:.0f} edges")
        else:
            logger.info("MLX backend: Failed")
    
    # Calculate speedups
    logger.info("\nSpeedup Comparison:")
    logger.info("=================")
    
    if python_success and numba_success:
        logger.info(f"Numba vs Python: {python_time / numba_time:.2f}x")
    
    if python_success and pytorch_success:
        logger.info(f"PyTorch vs Python: {python_time / pytorch_time:.2f}x")
    
    if numba_success and pytorch_success:
        logger.info(f"PyTorch vs Numba: {numba_time / pytorch_time:.2f}x")
    
    if python_success and mlx_success:
        logger.info(f"MLX vs Python: {python_time / mlx_time:.2f}x")
    
    if numba_success and mlx_success:
        logger.info(f"MLX vs Numba: {numba_time / mlx_time:.2f}x")
    
    if pytorch_success and mlx_success:
        logger.info(f"MLX vs PyTorch: {pytorch_time / mlx_time:.2f}x")

if __name__ == "__main__":
    run_benchmark()
