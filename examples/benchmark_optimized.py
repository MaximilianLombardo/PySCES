"""
Optimized benchmark for comparing Python, Numba, and MLX implementations.

This script benchmarks optimized implementations of correlation matrix calculation
using Python, Numba, and MLX.
"""

import numpy as np
import time
import logging
import platform
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if Numba is available
try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
    logger.info("Numba is available. Using Numba for acceleration.")
except ImportError:
    HAS_NUMBA = False
    logger.warning("Numba is not available. Install Numba with: pip install numba")
    # Define dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    
    def prange(*args, **kwargs):
        return range(*args, **kwargs)

# Check if MLX is available
try:
    import mlx.core as mx
    HAS_MLX = True
    logger.info("MLX is available. Using MLX for acceleration.")
except ImportError:
    HAS_MLX = False
    logger.warning("MLX is not available. Install MLX with: pip install mlx")

# Check if running on Apple Silicon
IS_APPLE_SILICON = platform.processor() == 'arm'
if IS_APPLE_SILICON:
    logger.info("Running on Apple Silicon. MLX acceleration should be available.")
else:
    logger.warning("Not running on Apple Silicon. MLX acceleration may not be available or optimal.")

def create_synthetic_data(n_samples, n_features):
    """Create synthetic data for benchmarking."""
    return np.random.rand(n_samples, n_features)

# Python implementation (baseline)
def calculate_mi_matrix_python(data, indices1, indices2, bins=10):
    """Calculate mutual information matrix using Python."""
    n1 = len(indices1)
    n2 = len(indices2)
    mi_matrix = np.zeros((n1, n2))
    
    for i in range(n1):
        idx1 = indices1[i]
        for j in range(n2):
            idx2 = indices2[j]
            if idx1 == idx2:
                continue
            
            # Get data
            x = data[:, idx1]
            y = data[:, idx2]
            
            # Calculate mutual information
            mi = calculate_mi_python(x, y, bins)
            mi_matrix[i, j] = mi
    
    return mi_matrix

def calculate_mi_python(x, y, bins=10):
    """Calculate mutual information between two variables using Python."""
    # Calculate joint histogram
    hist_joint, _, _ = np.histogram2d(x, y, bins=bins)
    
    # Calculate marginal histograms
    hist_x = np.sum(hist_joint, axis=1)
    hist_y = np.sum(hist_joint, axis=0)
    
    # Normalize histograms
    hist_joint = hist_joint / np.sum(hist_joint)
    hist_x = hist_x / np.sum(hist_x)
    hist_y = hist_y / np.sum(hist_y)
    
    # Calculate mutual information
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if hist_joint[i, j] > 0:
                p_xy = hist_joint[i, j]
                p_x = hist_x[i]
                p_y = hist_y[j]
                mi += p_xy * np.log(p_xy / (p_x * p_y))
    
    return mi

# Numba implementation (optimized)
@jit(nopython=True)
def calculate_mi_numba(x, y, bins=10):
    """Calculate mutual information between two variables using Numba."""
    # Calculate joint histogram
    hist_joint = np.zeros((bins, bins))
    
    # Find min and max for binning
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    # Handle edge case where min == max
    if x_min == x_max:
        x_min = x_min - 0.1
        x_max = x_max + 0.1
    if y_min == y_max:
        y_min = y_min - 0.1
        y_max = y_max + 0.1
    
    # Calculate bin widths
    x_bin_width = (x_max - x_min) / bins
    y_bin_width = (y_max - y_min) / bins
    
    # Fill joint histogram
    for k in range(len(x)):
        i = min(bins - 1, int((x[k] - x_min) / x_bin_width))
        j = min(bins - 1, int((y[k] - y_min) / y_bin_width))
        hist_joint[i, j] += 1
    
    # Calculate marginal histograms
    hist_x = np.zeros(bins)
    hist_y = np.zeros(bins)
    
    for i in range(bins):
        for j in range(bins):
            hist_x[i] += hist_joint[i, j]
            hist_y[j] += hist_joint[i, j]
    
    # Normalize histograms
    total = np.sum(hist_joint)
    hist_joint = hist_joint / total
    hist_x = hist_x / total
    hist_y = hist_y / total
    
    # Calculate mutual information
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if hist_joint[i, j] > 0:
                p_xy = hist_joint[i, j]
                p_x = hist_x[i]
                p_y = hist_y[j]
                if p_x > 0 and p_y > 0:
                    mi += p_xy * np.log(p_xy / (p_x * p_y))
    
    return mi

@jit(nopython=True, parallel=True)
def calculate_mi_matrix_numba(data, indices1, indices2, bins=10):
    """Calculate mutual information matrix using Numba with parallelization."""
    n1 = len(indices1)
    n2 = len(indices2)
    mi_matrix = np.zeros((n1, n2))
    
    for i in prange(n1):
        idx1 = indices1[i]
        for j in range(n2):
            idx2 = indices2[j]
            if idx1 == idx2:
                continue
            
            # Get data
            x = data[:, idx1]
            y = data[:, idx2]
            
            # Calculate mutual information
            mi = calculate_mi_numba(x, y, bins)
            mi_matrix[i, j] = mi
    
    return mi_matrix

# MLX implementation
def calculate_mi_mlx(x, y, bins=10):
    """Calculate mutual information between two variables using MLX."""
    if not HAS_MLX:
        return calculate_mi_python(x, y, bins)
    
    # Convert to MLX arrays
    x_mx = mx.array(x)
    y_mx = mx.array(y)
    
    # MLX doesn't have histogram functions yet, so we need to use NumPy
    # Calculate joint histogram
    hist_joint, _, _ = np.histogram2d(x, y, bins=bins)
    
    # Calculate marginal histograms
    hist_x = np.sum(hist_joint, axis=1)
    hist_y = np.sum(hist_joint, axis=0)
    
    # Convert to MLX arrays
    hist_joint_mx = mx.array(hist_joint)
    hist_x_mx = mx.array(hist_x).reshape(-1, 1)
    hist_y_mx = mx.array(hist_y).reshape(1, -1)
    
    # Normalize histograms
    total = mx.sum(hist_joint_mx)
    hist_joint_mx = hist_joint_mx / total
    hist_x_mx = hist_x_mx / mx.sum(hist_x_mx)
    hist_y_mx = hist_y_mx / mx.sum(hist_y_mx)
    
    # Calculate mutual information
    # MI = sum(p(x,y) * log(p(x,y) / (p(x) * p(y))))
    outer_product = mx.matmul(hist_x_mx, hist_y_mx)
    epsilon = 1e-10
    ratio = mx.where(hist_joint_mx > epsilon, 
                    hist_joint_mx / (outer_product + epsilon), 
                    mx.ones_like(hist_joint_mx))
    log_ratio = mx.where(ratio > epsilon, mx.log(ratio), mx.zeros_like(ratio))
    mi = mx.sum(hist_joint_mx * log_ratio)
    
    return mi.item()

def calculate_mi_matrix_mlx(data, indices1, indices2, bins=10):
    """Calculate mutual information matrix using MLX."""
    if not HAS_MLX:
        return calculate_mi_matrix_python(data, indices1, indices2, bins)
    
    # Convert data to numpy for easier indexing
    data_np = data
    
    # Initialize MI matrix
    n1 = len(indices1)
    n2 = len(indices2)
    mi_matrix = np.zeros((n1, n2))
    
    # Calculate MI for each pair
    for i in range(n1):
        idx1 = indices1[i]
        for j in range(n2):
            idx2 = indices2[j]
            if idx1 == idx2:
                continue
            
            # Get data
            x = data_np[:, idx1]
            y = data_np[:, idx2]
            
            # Calculate MI
            mi = calculate_mi_mlx(x, y, bins)
            mi_matrix[i, j] = mi
    
    return mi_matrix

def benchmark_mi_matrix(n_samples, n_features, n_indices, bins=10, n_iterations=3):
    """Benchmark mutual information matrix calculation."""
    # Create synthetic data
    data = create_synthetic_data(n_samples, n_features)
    
    # Create indices
    indices1 = np.random.choice(n_features, n_indices, replace=False)
    indices2 = np.random.choice(n_features, n_indices, replace=False)
    
    # Warm-up runs to account for compilation overhead
    logger.info("Running warm-up iterations...")
    _ = calculate_mi_matrix_python(data, indices1, indices2, bins)
    if HAS_NUMBA:
        _ = calculate_mi_matrix_numba(data, indices1, indices2, bins)
    if HAS_MLX and IS_APPLE_SILICON:
        _ = calculate_mi_matrix_mlx(data, indices1, indices2, bins)
    
    # Benchmark Python implementation
    logger.info("Benchmarking Python implementation...")
    python_times = []
    for _ in range(n_iterations):
        start_time = time.time()
        _ = calculate_mi_matrix_python(data, indices1, indices2, bins)
        python_times.append(time.time() - start_time)
    python_time = sum(python_times) / len(python_times)
    logger.info(f"Python implementation: {python_time:.4f} seconds (avg of {n_iterations} runs)")
    
    # Benchmark Numba implementation
    numba_time = None
    if HAS_NUMBA:
        logger.info("Benchmarking Numba implementation...")
        numba_times = []
        for _ in range(n_iterations):
            start_time = time.time()
            _ = calculate_mi_matrix_numba(data, indices1, indices2, bins)
            numba_times.append(time.time() - start_time)
        numba_time = sum(numba_times) / len(numba_times)
        logger.info(f"Numba implementation: {numba_time:.4f} seconds (avg of {n_iterations} runs)")
        
        # Calculate speedup
        python_numba_speedup = python_time / numba_time
        logger.info(f"Numba speedup vs Python: {python_numba_speedup:.2f}x")
    else:
        python_numba_speedup = None
    
    # Benchmark MLX implementation
    mlx_time = None
    if HAS_MLX and IS_APPLE_SILICON:
        logger.info("Benchmarking MLX implementation...")
        mlx_times = []
        for _ in range(n_iterations):
            start_time = time.time()
            _ = calculate_mi_matrix_mlx(data, indices1, indices2, bins)
            mlx_times.append(time.time() - start_time)
        mlx_time = sum(mlx_times) / len(mlx_times)
        logger.info(f"MLX implementation: {mlx_time:.4f} seconds (avg of {n_iterations} runs)")
        
        # Calculate speedups
        python_mlx_speedup = python_time / mlx_time
        logger.info(f"MLX speedup vs Python: {python_mlx_speedup:.2f}x")
        
        if numba_time is not None:
            numba_mlx_speedup = numba_time / mlx_time
            logger.info(f"MLX speedup vs Numba: {numba_mlx_speedup:.2f}x")
        else:
            numba_mlx_speedup = None
    else:
        python_mlx_speedup = None
        numba_mlx_speedup = None
    
    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_indices": n_indices,
        "bins": bins,
        "python_time": python_time,
        "numba_time": numba_time,
        "mlx_time": mlx_time,
        "python_numba_speedup": python_numba_speedup,
        "python_mlx_speedup": python_mlx_speedup,
        "numba_mlx_speedup": numba_mlx_speedup
    }

def run_benchmarks():
    """Run benchmarks with different dataset sizes."""
    # Define dataset sizes
    dataset_sizes = [
        (100, 100, 10),    # Small dataset: 100 samples, 100 features, 10 indices
        (500, 200, 20),    # Medium dataset: 500 samples, 200 features, 20 indices
        (1000, 500, 30),   # Large dataset: 1000 samples, 500 features, 30 indices
    ]
    
    # Run benchmarks
    results = []
    
    for n_samples, n_features, n_indices in dataset_sizes:
        logger.info(f"\nBenchmarking dataset: {n_samples} samples, {n_features} features, {n_indices} indices")
        
        result = benchmark_mi_matrix(
            n_samples=n_samples,
            n_features=n_features,
            n_indices=n_indices,
            bins=10,
            n_iterations=3
        )
        
        results.append(result)
    
    # Print summary
    logger.info("\nBenchmark Results:")
    logger.info("=================")
    for result in results:
        logger.info(f"Dataset: {result['n_samples']}×{result['n_features']}×{result['n_indices']}")
        logger.info(f"  Python: {result['python_time']:.4f} seconds")
        
        if result['numba_time'] is not None:
            logger.info(f"  Numba:  {result['numba_time']:.4f} seconds")
            logger.info(f"  Python/Numba Speedup: {result['python_numba_speedup']:.2f}x")
        
        if result['mlx_time'] is not None:
            logger.info(f"  MLX:    {result['mlx_time']:.4f} seconds")
            logger.info(f"  Python/MLX Speedup: {result['python_mlx_speedup']:.2f}x")
            
            if result['numba_time'] is not None:
                logger.info(f"  Numba/MLX Speedup:  {result['numba_mlx_speedup']:.2f}x")
    
    return results

if __name__ == "__main__":
    run_benchmarks()
