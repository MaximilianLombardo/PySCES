# Numba Optimization Guide for PySCES

This guide provides detailed information about the Numba optimizations used in PySCES and how to get the best performance from the Numba-accelerated implementation.

## What is Numba?

Numba is a Just-In-Time (JIT) compiler for Python that translates Python functions to optimized machine code at runtime. It's particularly effective for numerical computations and can provide significant speedups for algorithms like ARACNe and VIPER.

Key benefits of Numba include:
- **Performance**: Numba can provide speedups of 10-100x for numerical code
- **Ease of Use**: Numba requires minimal changes to existing Python code
- **Compatibility**: Numba works on all major platforms (Linux, macOS, Windows)
- **No Specialized Hardware**: Numba doesn't require GPUs or other specialized hardware

## How Numba is Used in PySCES

PySCES uses Numba to accelerate performance-critical functions in both the ARACNe and VIPER algorithms:

### ARACNe Optimizations

1. **Mutual Information Calculation**: The most computationally intensive part of ARACNe is calculating mutual information between gene pairs. Numba accelerates this by:
   - Optimizing nested loops
   - Efficient memory access patterns
   - Vectorizing operations where possible

2. **Data Processing Inequality (DPI)**: The DPI algorithm, which prunes indirect interactions from the network, is optimized with Numba for faster execution.

3. **Bootstrapping**: The bootstrapping process, which improves network robustness, is optimized to efficiently handle multiple iterations.

### VIPER Optimizations

1. **Signature Calculation**: The calculation of gene expression signatures is optimized for faster processing.

2. **Enrichment Score Calculation**: The calculation of enrichment scores for gene sets is optimized with Numba.

3. **Bootstrap Analysis**: The bootstrapping process for statistical robustness is optimized.

## Numba Optimization Techniques Used

### 1. JIT Compilation with `@numba.jit`

The `@numba.jit` decorator is used to compile functions at runtime:

```python
import numba

@numba.jit(nopython=True, parallel=True)
def calculate_mi_matrix(data, tf_indices):
    # Function implementation
    ...
```

Key parameters:
- `nopython=True`: Forces Numba to compile the function without using the Python interpreter, resulting in better performance
- `parallel=True`: Enables automatic parallelization of loops
- `cache=True`: Caches the compiled function to avoid recompilation

### 2. Vectorization with `@numba.vectorize`

The `@numba.vectorize` decorator is used to create NumPy universal functions (ufuncs) that operate element-wise on arrays:

```python
@numba.vectorize(['float64(float64, float64)'], nopython=True)
def fast_correlation(x, y):
    # Function implementation
    ...
```

### 3. Parallel Processing with `prange`

Numba's `prange` (parallel range) is used to parallelize loops:

```python
from numba import prange

@numba.jit(nopython=True, parallel=True)
def parallel_function(data):
    result = np.zeros(data.shape[0])
    for i in prange(data.shape[0]):
        # This loop will be parallelized
        result[i] = process(data[i])
    return result
```

### 4. Memory Layout Optimization

Numba performs best with contiguous arrays in memory. PySCES ensures optimal memory layout:

```python
# Ensure C-contiguous memory layout
data = np.ascontiguousarray(data)
```

### 5. Type Specialization

Numba performs best when types are consistent. PySCES uses explicit type conversions:

```python
# Ensure consistent types
data = np.asarray(data, dtype=np.float64)
indices = np.asarray(indices, dtype=np.int32)
```

## Performance Tuning

### Controlling Number of Threads

You can control the number of threads used by Numba:

```python
# In your code
aracne = ARACNe(n_threads=4)  # Use 4 threads

# Or via environment variable
import os
os.environ['NUMBA_NUM_THREADS'] = '4'
```

Setting `n_threads=0` will use all available CPU cores.

### Memory Usage Considerations

For large datasets, memory usage can be a concern. Consider:

1. **Stratification**: Process data by tissue or cell type to reduce memory requirements
2. **Chunking**: Process data in chunks if possible
3. **Sparse Matrices**: Use sparse matrices for large, sparse datasets

### Compilation Overhead

Numba has a compilation overhead the first time a function is called. For short-running tasks, this overhead can be significant. Options to mitigate this:

1. **Use `cache=True`**: Cache compiled functions to disk
2. **Pre-compile**: Call functions with small inputs first to trigger compilation
3. **Batch Processing**: Process data in larger batches to amortize compilation overhead

## Benchmarking

PySCES includes benchmarking scripts to measure performance:

```python
# Run ARACNe benchmark
python examples/benchmark_aracne.py

# Run VIPER benchmark
python examples/benchmark_viper.py
```

Typical performance improvements with Numba:

| Algorithm | Dataset Size | Python Time | Numba Time | Speedup |
|-----------|--------------|-------------|------------|---------|
| ARACNe    | Small        | ~10s        | ~3s        | 3x      |
| ARACNe    | Medium       | ~10min      | ~1min      | 10x     |
| ARACNe    | Large        | ~2h         | ~10min     | 12x     |
| VIPER     | Small        | ~5s         | ~2s        | 2.5x    |
| VIPER     | Medium       | ~1min       | ~15s       | 4x      |
| VIPER     | Large        | ~10min      | ~1min      | 10x     |

## Troubleshooting

### Common Issues

1. **"Function compilation failed"**: Usually indicates a type error or unsupported operation
   - Solution: Check that all arrays have consistent types and operations are Numba-compatible

2. **"Object mode fallback"**: Indicates Numba couldn't compile in nopython mode
   - Solution: Simplify the function or check for Python objects that Numba can't handle

3. **"No implementation of function..."**: Indicates a function Numba doesn't support
   - Solution: Rewrite using Numba-compatible functions or move that part outside the JIT-compiled function

4. **Poor Performance**: If Numba isn't providing expected speedups
   - Check that `nopython=True` is being used
   - Ensure arrays are contiguous and have consistent types
   - Use `parallel=True` and `prange` for parallelizable loops
   - Profile to identify bottlenecks

### Debugging Numba Code

Numba provides debugging options:

```python
import numba

# Enable debug output
numba.config.NUMBA_DEBUG = 1

# Disable JIT for debugging
numba.config.DISABLE_JIT = 1
```

## Conclusion

The Numba-accelerated implementation in PySCES provides excellent performance across all platforms without requiring specialized hardware. By understanding how Numba is used and following the performance tuning guidelines, you can get the best performance from PySCES for your specific use case.

For most users, the default settings will provide good performance, but for very large datasets or performance-critical applications, the tuning options described in this guide can help optimize performance further.
