# Performance Optimizations in PySCES

This document outlines the performance optimizations implemented in PySCES to accelerate the ARACNe and VIPER algorithms for single-cell data analysis.

## Overview

PySCES has been optimized to handle large-scale single-cell datasets efficiently. The key optimizations include:

1. **Numba Acceleration**: Just-in-time compilation of performance-critical functions for all platforms
2. **MLX Integration**: Comprehensive support for Apple Silicon GPU acceleration leveraging unified memory
3. **Algorithmic Improvements**: Enhanced implementations of core algorithms with vectorization and parallelization
4. **Automatic Acceleration Selection**: Intelligent selection of the best available acceleration method

These optimizations significantly reduce computation time, especially for large datasets, making it practical to analyze datasets with thousands of cells and genes. The implementation is designed to provide the best performance across different hardware platforms, with special optimizations for Apple Silicon devices.

## Benchmark Results

### ARACNe Optimization

The ARACNe algorithm has been accelerated using Numba, resulting in dramatic performance improvements:

| Dataset Size (cells × genes × TFs) | Python Implementation | Numba Implementation | Speedup |
|-----------------------------------|----------------------|---------------------|---------|
| 100 × 100 × 10                    | 0.14 seconds         | 0.00 seconds        | 62.92x  |
| 200 × 150 × 15                    | 0.54 seconds         | 0.01 seconds        | 89.25x  |
| 500 × 200 × 20                    | 1.59 seconds         | 0.02 seconds        | 78.12x  |
| 1000 × 500 × 50                   | 38.34 seconds        | 0.63 seconds        | 61.26x  |

The Numba-accelerated implementation provides a 60-90x speedup across different dataset sizes, with the most significant improvements seen in medium to large datasets.

![ARACNe Benchmark](images/aracne_benchmark.png)

![ARACNe Speedup](images/aracne_speedup.png)

### VIPER Optimization

The VIPER algorithm has also been accelerated using Numba:

| Dataset Size (cells × genes × TFs) | Python Implementation | Numba Implementation | Speedup |
|-----------------------------------|----------------------|---------------------|---------|
| 100 × 100 × 10                    | 0.04 seconds         | 0.00 seconds        | 10.51x  |
| 500 × 200 × 20                    | 0.45 seconds         | 0.04 seconds        | 12.57x  |
| 1000 × 500 × 50                   | 3.06 seconds         | 0.22 seconds        | 14.15x  |

The Numba-accelerated implementation provides a 10-14x speedup, with the speedup increasing as the dataset size grows.

![VIPER Benchmark](images/viper_benchmark.png)

![VIPER Speedup](images/viper_speedup.png)

### Speedup Comparison

The following chart compares the speedup factors for both ARACNe and VIPER across different dataset sizes:

![Speedup Comparison](images/speedup_comparison.png)

As shown in the chart, ARACNe achieves significantly higher speedups (60-90x) compared to VIPER (10-14x). This is because ARACNe's core algorithms involve more intensive numerical computations that benefit more from Numba's optimizations.

## How Numba Acceleration Works

[Numba](https://numba.pydata.org/) is a just-in-time (JIT) compiler that translates Python functions to optimized machine code at runtime. Here's how it accelerates our code:

### 1. Just-in-Time Compilation

Numba compiles Python functions to optimized machine code the first time they're called. This compilation has a one-time overhead but results in much faster execution for subsequent calls.

```python
@jit(nopython=True)
def calculate_enrichment_score_numba(signature, indices, weights, method='gsea', abs_score=False):
    # Function implementation
    # ...
```

### 2. Vectorization

Numba automatically vectorizes operations where possible, leveraging CPU SIMD (Single Instruction, Multiple Data) instructions to process multiple data elements in parallel.

### 3. Loop Optimization

Numba optimizes loops by:
- Eliminating Python's interpretation overhead
- Unrolling loops to reduce branch prediction misses
- Optimizing memory access patterns

### 4. Parallelization

Numba can parallelize operations across multiple CPU cores:

```python
@jit(nopython=True, parallel=True)
def calculate_signature_numba(expr, method='rank'):
    # Function implementation with parallel execution
    # ...
```

## Why Speedup Increases with Dataset Size

An interesting observation from our benchmarks is that the speedup often increases with the size of the dataset. This is due to several factors:

1. **Compilation Overhead Amortization**:
   - Numba's compilation overhead is a fixed cost
   - For larger datasets, this overhead becomes a smaller percentage of the total execution time

2. **Vectorization Efficiency**:
   - Vectorized operations become more efficient with larger datasets
   - CPU pipelines can be kept fuller, reducing the impact of branch mispredictions

3. **Memory Access Patterns**:
   - Larger datasets enable more predictable memory access patterns
   - This allows CPU prefetching to work more effectively

4. **Parallelization Benefits**:
   - The benefits of parallelization outweigh the overhead costs for larger datasets

5. **Loop Optimization Effects**:
   - Loop unrolling and other optimizations have a greater impact on larger loops

This makes our optimized implementation particularly valuable for large-scale single-cell analyses, where the datasets can contain thousands of cells and genes.

## MLX Acceleration Strategy for Apple Silicon

For users with Apple Silicon hardware (M1/M2/M3 chips), we've implemented support for MLX, Apple's machine learning framework that leverages the unified memory architecture and GPU acceleration capabilities of Apple Silicon.

### Key MLX Features Leveraged

#### 1. Lazy Evaluation

MLX uses a lazy computation model where operations are recorded in a compute graph but not executed until explicitly evaluated. This allows for:

- **Operation Batching**: Multiple operations can be batched together before evaluation
- **Graph Optimization**: The computation graph can be optimized before execution
- **Reduced Overhead**: Minimizing the number of evaluations reduces overhead

#### 2. Unified Memory Architecture

Apple Silicon has a unified memory architecture where the CPU and GPU share the same memory pool:

- **Zero-Copy Operations**: No need to transfer data between CPU and GPU memory
- **Device Flexibility**: Operations can run on either CPU or GPU without data movement
- **Parallel Execution**: CPU and GPU can work simultaneously on different parts of the computation

### Implementation Strategy

#### Platform Detection and Initialization

The MLX implementation automatically detects Apple Silicon hardware and the availability of MLX:

```python
def is_apple_silicon():
    """Check if running on Apple Silicon."""
    import platform
    return platform.processor() == 'arm'

def initialize_mlx():
    """Initialize MLX if available."""
    if is_apple_silicon():
        try:
            import mlx.core as mx
            return True, mx
        except ImportError:
            return False, None
    return False, None
```

#### Optimized Core Functions

We've implemented MLX-optimized versions of the core computational functions:

1. **For ARACNe**:
   - Mutual Information calculation using MLX's vectorized operations
   - Data Processing Inequality (DPI) algorithm using MLX's array operations
   - Bootstrap sampling and consensus network construction

2. **For VIPER**:
   - Signature calculation using MLX's statistical functions
   - Enrichment score calculation using MLX's sorting and cumulative operations
   - NES matrix calculation with efficient broadcasting

#### Vectorization and Batch Processing

The MLX implementation emphasizes:

- **Vectorized Operations**: Replacing loops with MLX's array operations
- **Batch Processing**: Processing data in batches to reduce overhead
- **Strategic Evaluation**: Evaluating the compute graph at optimal points

#### Memory Efficiency

To maximize performance, we:

- **Minimize Conversions**: Keep data in MLX format throughout the computation
- **Reuse Arrays**: Avoid creating unnecessary temporary arrays
- **Leverage Unified Memory**: Let operations run on the most appropriate device

### Automatic Fallback Mechanism

The implementation includes a robust fallback mechanism:

```python
def run_algorithm(adata, **kwargs):
    """Run algorithm with automatic acceleration selection."""
    has_mlx, mx = initialize_mlx()

    if has_mlx and is_apple_silicon():
        # Use MLX implementation
        return run_algorithm_mlx(adata, **kwargs)
    elif HAS_NUMBA:
        # Fall back to Numba implementation
        return run_algorithm_numba(adata, **kwargs)
    else:
        # Fall back to Python implementation
        return run_algorithm_python(adata, **kwargs)
```

### Usage

The MLX acceleration can be enabled explicitly:

```python
# ARACNe with MLX acceleration on Apple Silicon
aracne = ARACNe(use_mlx=True)
network = aracne.run(adata, tf_list=tf_names)

# VIPER with MLX acceleration on Apple Silicon
activity = viper_scores(adata, regulons, use_mlx=True)
```

By default, the implementation will automatically use MLX if running on Apple Silicon and MLX is available.

### Performance Expectations

The MLX implementation is expected to provide significant speedups for:

- **Large Matrices**: Operations on large matrices benefit most from GPU acceleration
- **Complex Mathematical Operations**: Functions with high arithmetic intensity
- **Batch Processing**: Processing multiple samples or bootstraps in parallel

The actual speedup will vary depending on the specific hardware, dataset size, and operation type.

## Practical Impact for Scientists

These optimizations enable scientists to:

1. **Analyze Larger Datasets**: Process datasets with thousands of cells and genes that would be impractical with the original implementation

2. **Perform More Iterations**: Run more bootstrap iterations or permutations for statistical robustness

3. **Iterate Faster**: Reduce the time from data to insight, enabling more rapid hypothesis testing and refinement

4. **Scale to Full Datasets**: Move beyond downsampling to analyze complete datasets

## Using the Optimized Implementations

The optimized implementations are used by default, but can be explicitly controlled:

### Numba Acceleration (All Platforms)

```python
# ARACNe with Numba acceleration (default)
aracne = ARACNe(use_numba=True)
network = aracne.run(adata, tf_list=tf_names)

# VIPER with Numba acceleration (default)
activity = viper_scores(adata, regulons, use_numba=True)
```

### MLX Acceleration (Apple Silicon Only)

```python
# ARACNe with MLX acceleration on Apple Silicon
aracne = ARACNe(use_mlx=True)
network = aracne.run(adata, tf_list=tf_names)

# VIPER with MLX acceleration on Apple Silicon
activity = viper_scores(adata, regulons, use_mlx=True)
```

### Combined Acceleration Options

```python
# Try MLX first, fall back to Numba if MLX is not available
aracne = ARACNe(use_mlx=True, use_numba=True)
network = aracne.run(adata, tf_list=tf_names)

# Same for VIPER
activity = viper_scores(adata, regulons, use_mlx=True, use_numba=True)
```

### Python Implementation (for Debugging)

To use the Python implementation (e.g., for debugging or comparison):

```python
# ARACNe without any acceleration
aracne = ARACNe(use_numba=False, use_mlx=False)
network = aracne.run(adata, tf_list=tf_names)

# VIPER without any acceleration
activity = viper_scores(adata, regulons, use_numba=False, use_mlx=False)
```

## Future Optimization Directions

We're continuing to explore additional optimization strategies:

1. **Unified Acceleration Interface**: Create a common interface for all acceleration backends (Numba, MLX, CUDA)

2. **NVIDIA GPU Support**: Extend the acceleration framework to support NVIDIA GPUs via CUDA

3. **Advanced MLX Optimizations**: Further optimize the MLX implementation with:
   - Custom Metal kernels for specialized operations
   - MLX's compilation capabilities for frequently used function patterns
   - Advanced memory management techniques

4. **Hybrid CPU/GPU Approach**: Intelligently distribute workloads between CPU and GPU based on operation characteristics

5. **Memory Optimization**: Reduce memory usage for very large datasets through:
   - Streaming processing of large matrices
   - Sparse matrix representations where appropriate
   - Incremental computation strategies

6. **Distributed Computing**: Enable processing across multiple machines for extremely large datasets

## Conclusion

The performance optimizations in PySCES significantly accelerate the ARACNe and VIPER algorithms, enabling the analysis of larger datasets in less time. These improvements make it practical to apply these powerful algorithms to the scale of modern single-cell datasets, facilitating new biological insights.
