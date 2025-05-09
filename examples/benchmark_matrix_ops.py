"""
Benchmark for matrix operations using Python, Numba, and MLX.

This script benchmarks matrix operations that are more suited to MLX's strengths.
"""

import numpy as np
import time
import logging
import platform

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

def create_random_matrices(n, m, k):
    """Create random matrices for benchmarking."""
    A = np.random.rand(n, m)
    B = np.random.rand(m, k)
    return A, B

# Python implementation (using NumPy)
def matrix_ops_python(A, B):
    """Perform matrix operations using NumPy."""
    # Matrix multiplication
    C = np.matmul(A, B)
    
    # Element-wise operations
    D = np.sin(C) + np.cos(C)
    
    # Reduction operations
    E = np.mean(D, axis=1, keepdims=True)
    
    # More element-wise operations
    F = D / (E + 1e-10)
    
    # Final reduction
    result = np.sum(F)
    
    return result

# Numba implementation
@jit(nopython=True)
def matrix_multiply_numba(A, B):
    """Matrix multiplication using Numba."""
    n, m = A.shape
    m, k = B.shape
    C = np.zeros((n, k))
    
    for i in range(n):
        for j in range(k):
            for l in range(m):
                C[i, j] += A[i, l] * B[l, j]
    
    return C

@jit(nopython=True, parallel=True)
def matrix_ops_numba(A, B):
    """Perform matrix operations using Numba."""
    # Matrix multiplication
    C = matrix_multiply_numba(A, B)
    
    # Element-wise operations
    D = np.zeros_like(C)
    for i in prange(C.shape[0]):
        for j in range(C.shape[1]):
            D[i, j] = np.sin(C[i, j]) + np.cos(C[i, j])
    
    # Reduction operations
    E = np.zeros((C.shape[0], 1))
    for i in range(C.shape[0]):
        row_sum = 0.0
        for j in range(C.shape[1]):
            row_sum += D[i, j]
        E[i, 0] = row_sum / C.shape[1]
    
    # More element-wise operations
    F = np.zeros_like(D)
    for i in prange(D.shape[0]):
        for j in range(D.shape[1]):
            F[i, j] = D[i, j] / (E[i, 0] + 1e-10)
    
    # Final reduction
    result = 0.0
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            result += F[i, j]
    
    return result

# MLX implementation
def matrix_ops_mlx(A, B):
    """Perform matrix operations using MLX."""
    if not HAS_MLX:
        return matrix_ops_python(A, B)
    
    # Convert to MLX arrays
    A_mx = mx.array(A)
    B_mx = mx.array(B)
    
    # Matrix multiplication
    C = mx.matmul(A_mx, B_mx)
    
    # Element-wise operations
    D = mx.sin(C) + mx.cos(C)
    
    # Reduction operations
    E = mx.mean(D, axis=1, keepdims=True)
    
    # More element-wise operations
    F = D / (E + 1e-10)
    
    # Final reduction
    result = mx.sum(F)
    
    # Evaluate the computation graph
    return result.item()

def benchmark_matrix_ops(n, m, k, n_iterations=5):
    """Benchmark matrix operations."""
    # Create random matrices
    A, B = create_random_matrices(n, m, k)
    
    # Warm-up runs to account for compilation overhead
    logger.info("Running warm-up iterations...")
    _ = matrix_ops_python(A, B)
    if HAS_NUMBA:
        _ = matrix_ops_numba(A, B)
    if HAS_MLX and IS_APPLE_SILICON:
        _ = matrix_ops_mlx(A, B)
    
    # Benchmark Python implementation
    logger.info("Benchmarking Python (NumPy) implementation...")
    python_times = []
    for _ in range(n_iterations):
        start_time = time.time()
        _ = matrix_ops_python(A, B)
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
            _ = matrix_ops_numba(A, B)
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
            _ = matrix_ops_mlx(A, B)
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
        "n": n,
        "m": m,
        "k": k,
        "python_time": python_time,
        "numba_time": numba_time,
        "mlx_time": mlx_time,
        "python_numba_speedup": python_numba_speedup,
        "python_mlx_speedup": python_mlx_speedup,
        "numba_mlx_speedup": numba_mlx_speedup
    }

def run_benchmarks():
    """Run benchmarks with different matrix sizes."""
    # Define matrix sizes
    matrix_sizes = [
        (100, 100, 100),    # Small matrices: 100x100 * 100x100
        (500, 500, 500),    # Medium matrices: 500x500 * 500x500
        (1000, 1000, 1000), # Large matrices: 1000x1000 * 1000x1000
    ]
    
    # Run benchmarks
    results = []
    
    for n, m, k in matrix_sizes:
        logger.info(f"\nBenchmarking matrices: {n}x{m} * {m}x{k}")
        
        result = benchmark_matrix_ops(
            n=n,
            m=m,
            k=k,
            n_iterations=3
        )
        
        results.append(result)
    
    # Print summary
    logger.info("\nBenchmark Results:")
    logger.info("=================")
    for result in results:
        logger.info(f"Matrices: {result['n']}x{result['m']} * {result['m']}x{result['k']}")
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
