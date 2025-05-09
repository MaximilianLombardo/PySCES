"""
Benchmark MLX implementation of ARACNe algorithm.

This script benchmarks the MLX-optimized implementation of ARACNe against
the Python and Numba implementations.
"""

import numpy as np
import time
import logging
import sys
import os
import platform

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def calculate_correlation_python(x, y):
    """Calculate Pearson correlation coefficient using Python."""
    # Center the variables
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    # Calculate correlation
    numerator = np.sum(x_centered * y_centered)
    x_std = np.sqrt(np.sum(x_centered**2))
    y_std = np.sqrt(np.sum(y_centered**2))
    denominator = x_std * y_std

    # Handle division by zero
    if denominator < 1e-10:
        return 0.0

    return numerator / denominator

def calculate_correlation_matrix_python(data, indices1, indices2):
    """Calculate correlation matrix using Python."""
    n1 = len(indices1)
    n2 = len(indices2)
    corr_matrix = np.zeros((n1, n2))

    for i, idx1 in enumerate(indices1):
        for j, idx2 in enumerate(indices2):
            if idx1 == idx2:
                continue

            corr = calculate_correlation_python(data[:, idx1], data[:, idx2])
            corr_matrix[i, j] = abs(corr)

    return corr_matrix

def calculate_correlation_numba(x, y):
    """Calculate Pearson correlation coefficient using Numba."""
    try:
        import numba

        @numba.jit(nopython=True)
        def _correlation(x, y):
            # Center the variables
            x_mean = x.mean()
            y_mean = y.mean()
            x_centered = x - x_mean
            y_centered = y - y_mean

            # Calculate correlation
            numerator = (x_centered * y_centered).sum()
            x_std = np.sqrt((x_centered ** 2).sum())
            y_std = np.sqrt((y_centered ** 2).sum())
            denominator = x_std * y_std

            # Handle division by zero
            if denominator < 1e-10:
                return 0.0

            return numerator / denominator

        return _correlation(x, y)
    except ImportError:
        logger.warning("Numba not available. Using Python implementation.")
        return calculate_correlation_python(x, y)

def calculate_correlation_matrix_numba(data, indices1, indices2):
    """Calculate correlation matrix using Numba."""
    try:
        import numba

        @numba.jit(nopython=True)
        def _correlation_matrix(data, indices1, indices2):
            n1 = len(indices1)
            n2 = len(indices2)
            corr_matrix = np.zeros((n1, n2))

            for i in range(n1):
                idx1 = indices1[i]
                for j in range(n2):
                    idx2 = indices2[j]
                    if idx1 == idx2:
                        continue

                    # Get data
                    x = data[:, idx1]
                    y = data[:, idx2]

                    # Center the variables
                    x_mean = x.mean()
                    y_mean = y.mean()
                    x_centered = x - x_mean
                    y_centered = y - y_mean

                    # Calculate correlation
                    numerator = (x_centered * y_centered).sum()
                    x_std = np.sqrt((x_centered ** 2).sum())
                    y_std = np.sqrt((y_centered ** 2).sum())
                    denominator = x_std * y_std

                    # Handle division by zero
                    if denominator < 1e-10:
                        corr = 0.0
                    else:
                        corr = numerator / denominator

                    corr_matrix[i, j] = abs(corr)

            return corr_matrix

        return _correlation_matrix(data, indices1, indices2)
    except ImportError:
        logger.warning("Numba not available. Using Python implementation.")
        return calculate_correlation_matrix_python(data, indices1, indices2)

def calculate_correlation_mlx(x, y):
    """Calculate Pearson correlation coefficient using MLX."""
    if not HAS_MLX:
        return calculate_correlation_python(x, y)

    # Convert to MLX arrays
    x_mx = mx.array(x)
    y_mx = mx.array(y)

    # Center the variables
    x_centered = x_mx - mx.mean(x_mx)
    y_centered = y_mx - mx.mean(y_mx)

    # Calculate correlation
    numerator = mx.sum(x_centered * y_centered)
    x_std = mx.sqrt(mx.sum(x_centered**2))
    y_std = mx.sqrt(mx.sum(y_centered**2))
    denominator = x_std * y_std

    # Handle division by zero
    epsilon = 1e-10
    correlation = mx.where(denominator > epsilon,
                          numerator / denominator,
                          mx.zeros_like(numerator))

    return correlation.item()

def calculate_correlation_matrix_mlx(data, indices1, indices2):
    """Calculate correlation matrix using MLX."""
    if not HAS_MLX:
        return calculate_correlation_matrix_python(data, indices1, indices2)

    # Convert data to numpy for easier indexing
    data_np = data

    # Initialize correlation matrix
    n1 = len(indices1)
    n2 = len(indices2)
    corr_matrix = np.zeros((n1, n2))

    # Calculate correlations
    for i, idx1 in enumerate(indices1):
        for j, idx2 in enumerate(indices2):
            if idx1 == idx2:
                continue

            # Get data
            x = data_np[:, idx1]
            y = data_np[:, idx2]

            # Calculate correlation directly in numpy (faster)
            # Center the variables
            x_centered = x - np.mean(x)
            y_centered = y - np.mean(y)

            # Calculate correlation
            numerator = np.sum(x_centered * y_centered)
            x_std = np.sqrt(np.sum(x_centered**2))
            y_std = np.sqrt(np.sum(y_centered**2))
            denominator = x_std * y_std

            # Handle division by zero
            if denominator > 1e-10:
                correlation = numerator / denominator
            else:
                correlation = 0.0

            # Store absolute correlation
            corr_matrix[i, j] = abs(correlation)

    return corr_matrix

def benchmark_correlation_matrix(n_samples, n_features, n_indices, n_iterations=5):
    """Benchmark correlation matrix calculation."""
    # Create synthetic data
    data = create_synthetic_data(n_samples, n_features)

    # Create indices
    indices1 = np.random.choice(n_features, n_indices, replace=False)
    indices2 = np.random.choice(n_features, n_indices, replace=False)

    # Warm-up runs to account for compilation overhead
    logger.info("Running warm-up iterations...")
    _ = calculate_correlation_matrix_python(data, indices1, indices2)
    _ = calculate_correlation_matrix_numba(data, indices1, indices2)
    if HAS_MLX and IS_APPLE_SILICON:
        _ = calculate_correlation_matrix_mlx(data, indices1, indices2)

    # Benchmark Python implementation
    logger.info("Benchmarking Python implementation...")
    python_times = []
    for _ in range(n_iterations):
        start_time = time.time()
        _ = calculate_correlation_matrix_python(data, indices1, indices2)
        python_times.append(time.time() - start_time)
    python_time = sum(python_times) / len(python_times)
    logger.info(f"Python implementation: {python_time:.4f} seconds (avg of {n_iterations} runs)")

    # Benchmark Numba implementation
    logger.info("Benchmarking Numba implementation...")
    numba_times = []
    for _ in range(n_iterations):
        start_time = time.time()
        _ = calculate_correlation_matrix_numba(data, indices1, indices2)
        numba_times.append(time.time() - start_time)
    numba_time = sum(numba_times) / len(numba_times)
    logger.info(f"Numba implementation: {numba_time:.4f} seconds (avg of {n_iterations} runs)")

    # Benchmark MLX implementation
    if HAS_MLX and IS_APPLE_SILICON:
        logger.info("Benchmarking MLX implementation...")
        mlx_times = []
        for _ in range(n_iterations):
            start_time = time.time()
            _ = calculate_correlation_matrix_mlx(data, indices1, indices2)
            mlx_times.append(time.time() - start_time)
        mlx_time = sum(mlx_times) / len(mlx_times)
        logger.info(f"MLX implementation: {mlx_time:.4f} seconds (avg of {n_iterations} runs)")

        # Calculate speedups
        python_mlx_speedup = python_time / mlx_time
        numba_mlx_speedup = numba_time / mlx_time

        logger.info(f"MLX speedup vs Python: {python_mlx_speedup:.2f}x")
        logger.info(f"MLX speedup vs Numba: {numba_mlx_speedup:.2f}x")
    else:
        mlx_time = None
        python_mlx_speedup = None
        numba_mlx_speedup = None

    # Calculate Numba speedup
    python_numba_speedup = python_time / numba_time
    logger.info(f"Numba speedup vs Python: {python_numba_speedup:.2f}x")

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_indices": n_indices,
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
        (1000, 500, 50),   # Large dataset: 1000 samples, 500 features, 50 indices
    ]

    # Run benchmarks
    results = []

    for n_samples, n_features, n_indices in dataset_sizes:
        logger.info(f"\nBenchmarking dataset: {n_samples} samples, {n_features} features, {n_indices} indices")

        result = benchmark_correlation_matrix(
            n_samples=n_samples,
            n_features=n_features,
            n_indices=n_indices
        )

        results.append(result)

    # Print summary
    logger.info("\nBenchmark Results:")
    logger.info("=================")
    for result in results:
        logger.info(f"Dataset: {result['n_samples']}×{result['n_features']}×{result['n_indices']}")
        logger.info(f"  Python: {result['python_time']:.4f} seconds")
        logger.info(f"  Numba:  {result['numba_time']:.4f} seconds")
        logger.info(f"  Python/Numba Speedup: {result['python_numba_speedup']:.2f}x")

        if result['mlx_time'] is not None:
            logger.info(f"  MLX:    {result['mlx_time']:.4f} seconds")
            logger.info(f"  Python/MLX Speedup: {result['python_mlx_speedup']:.2f}x")
            logger.info(f"  Numba/MLX Speedup:  {result['numba_mlx_speedup']:.2f}x")

    return results

if __name__ == "__main__":
    run_benchmarks()
