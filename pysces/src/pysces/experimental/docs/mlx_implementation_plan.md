# MLX Implementation Plan for PySCES (Experimental)

> **Note**: This is an experimental implementation that is not part of the main PySCES codebase. The primary implementation of PySCES uses Numba for acceleration, which provides excellent performance across all platforms. This MLX implementation is provided for research purposes and for users with Apple Silicon hardware who want to experiment with alternative acceleration approaches.

This document outlines the detailed implementation plan for optimizing PySCES algorithms using MLX on Apple Silicon hardware.

## Background

MLX is Apple's machine learning framework designed to leverage the unified memory architecture and GPU capabilities of Apple Silicon chips. It provides a NumPy-like API with lazy evaluation and automatic differentiation, making it well-suited for accelerating scientific computing workloads.

## Implementation Goals

1. Achieve significant speedup over Numba implementation on Apple Silicon
2. Maintain numerical accuracy and compatibility with existing code
3. Provide seamless fallback to Numba when MLX is not available
4. Lay groundwork for a unified acceleration interface

## Key MLX Features to Leverage

### 1. Lazy Evaluation

MLX uses a lazy computation model where operations are recorded in a compute graph but not executed until explicitly evaluated. This allows for:

- **Operation Batching**: Multiple operations can be batched together before evaluation
- **Graph Optimization**: The computation graph can be optimized before execution
- **Reduced Overhead**: Minimizing the number of evaluations reduces overhead

Example:
```python
import mlx.core as mx

# These operations are recorded but not executed
x = mx.ones((1000, 1000))
y = mx.ones((1000, 1000))
z = mx.matmul(x, y) + x

# The computation is executed only when we evaluate z
result = z.item()  # or mx.eval(z)
```

### 2. Unified Memory Architecture

Apple Silicon has a unified memory architecture where the CPU and GPU share the same memory pool:

- **Zero-Copy Operations**: No need to transfer data between CPU and GPU memory
- **Device Flexibility**: Operations can run on either CPU or GPU without data movement
- **Parallel Execution**: CPU and GPU can work simultaneously on different parts of the computation

## Implementation Phases

### Phase 1: Core Mathematical Operations

Implement MLX-optimized versions of the core mathematical operations:

#### For ARACNe:
1. **Mutual Information Calculation**:
   ```python
   def calculate_mi_mlx(x, y, bins=10):
       """Calculate mutual information using MLX."""
       # Implementation using MLX's array operations
       # Keep data in MLX format throughout
       # Use vectorized operations
   ```

2. **DPI Algorithm**:
   ```python
   def apply_dpi_mlx(mi_matrix, tf_indices, dpi_tolerance=0.1):
       """Apply DPI algorithm using MLX."""
       # Implementation using MLX's array operations
       # Leverage broadcasting
       # Avoid in-place updates that force eager execution
   ```

#### For VIPER:
1. **Signature Calculation**:
   ```python
   def calculate_signature_mlx(expr_mx, method='rank'):
       """Calculate gene expression signatures using MLX."""
       # Implementation using MLX's statistical functions
       # Vectorize operations where possible
   ```

2. **Enrichment Score Calculation**:
   ```python
   def calculate_enrichment_score_mlx(signature, indices, weights, method='gsea'):
       """Calculate enrichment score using MLX."""
       # Implementation using MLX's sorting and cumulative operations
       # Vectorize operations where possible
   ```

### Phase 2: Algorithm-Level Integration

Implement MLX-optimized versions of the main algorithm functions:

1. **ARACNe Algorithm**:
   ```python
   def run_aracne_mlx(expr_matrix, gene_list, tf_indices, **kwargs):
       """Run ARACNe algorithm using MLX."""
       # Implementation using MLX's lazy evaluation model
       # Batch operations together
       # Evaluate only when necessary
   ```

2. **VIPER Algorithm**:
   ```python
   def viper_scores_mlx(adata, regulons, **kwargs):
       """Calculate VIPER scores using MLX."""
       # Implementation using MLX's vectorized operations
       # Leverage unified memory architecture
       # Evaluate strategically
   ```

### Phase 3: Integration with Main Pipeline

Integrate MLX-optimized implementations into the main pipeline:

1. **Platform Detection**:
   ```python
   def is_apple_silicon():
       """Check if running on Apple Silicon."""
       import platform
       return platform.processor() == 'arm'
   ```

2. **Automatic Acceleration Selection**:
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

## Optimization Techniques

### Vectorization

Replace loops with vectorized operations:

```python
# Instead of this:
for i in range(n):
    for j in range(m):
        result[i, j] = x[i] * y[j]

# Do this:
result = mx.outer(x, y)
```

### Batch Processing

Process data in batches to reduce overhead:

```python
# Instead of processing each item individually:
for i in range(n):
    result[i] = process(data[i])

# Process in batches:
batch_size = 100
for i in range(0, n, batch_size):
    batch = data[i:i+batch_size]
    result[i:i+batch_size] = process_batch(batch)
```

### Memory Efficiency

Minimize conversions between MLX and NumPy arrays:

```python
# Instead of this:
x_np = x.numpy()
result_np = some_numpy_operation(x_np)
result = mx.array(result_np)

# Do this if possible:
result = some_mlx_operation(x)
```

### Strategic Evaluation

Evaluate at the end of major computation steps:

```python
# Instead of evaluating after each operation:
for i in range(n):
    x = mx.add(x, y)
    x = mx.eval(x)  # Unnecessary evaluation

# Evaluate only when necessary:
for i in range(n):
    x = mx.add(x, y)
x = mx.eval(x)  # Evaluate once at the end
```

## Benchmarking and Validation

### Benchmarking

Implement comprehensive benchmarks to compare performance:

```python
def benchmark_implementations(dataset_sizes, **kwargs):
    """Benchmark different implementations."""
    results = []

    for n_cells, n_genes, n_tfs in dataset_sizes:
        # Create synthetic dataset
        adata = create_synthetic_dataset(n_cells, n_genes, n_tfs)

        # Benchmark Python implementation
        start_time = time.time()
        result_python = run_algorithm_python(adata, **kwargs)
        python_time = time.time() - start_time

        # Benchmark Numba implementation
        start_time = time.time()
        result_numba = run_algorithm_numba(adata, **kwargs)
        numba_time = time.time() - start_time

        # Benchmark MLX implementation
        start_time = time.time()
        result_mlx = run_algorithm_mlx(adata, **kwargs)
        mlx_time = time.time() - start_time

        # Calculate speedups
        python_numba_speedup = python_time / numba_time
        python_mlx_speedup = python_time / mlx_time
        numba_mlx_speedup = numba_time / mlx_time

        results.append({
            "dataset_size": (n_cells, n_genes, n_tfs),
            "python_time": python_time,
            "numba_time": numba_time,
            "mlx_time": mlx_time,
            "python_numba_speedup": python_numba_speedup,
            "python_mlx_speedup": python_mlx_speedup,
            "numba_mlx_speedup": numba_mlx_speedup
        })

    return results
```

### Validation

Ensure numerical accuracy across implementations:

```python
def validate_implementations(dataset_sizes, **kwargs):
    """Validate different implementations."""
    for n_cells, n_genes, n_tfs in dataset_sizes:
        # Create synthetic dataset
        adata = create_synthetic_dataset(n_cells, n_genes, n_tfs)

        # Run different implementations
        result_python = run_algorithm_python(adata, **kwargs)
        result_numba = run_algorithm_numba(adata, **kwargs)
        result_mlx = run_algorithm_mlx(adata, **kwargs)

        # Compare results
        np.testing.assert_allclose(result_python, result_numba, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(result_python, result_mlx, rtol=1e-5, atol=1e-8)
```

## Expected Challenges and Solutions

### Challenge 1: MLX API Differences

MLX's API differs from NumPy in some areas, particularly for in-place operations and advanced indexing.

**Solution**: Create wrapper functions to handle API differences and provide consistent behavior.

### Challenge 2: Lazy Evaluation Complexity

Lazy evaluation can make debugging more difficult as errors may not be raised until evaluation.

**Solution**: Add strategic evaluation points during development and comprehensive error handling.

### Challenge 3: Limited MLX Functions

MLX may not have all the functions available in NumPy or SciPy.

**Solution**: Implement custom functions where needed or fall back to NumPy for specific operations.

## Timeline and Milestones

1. **Week 1**: Implement core mathematical operations for ARACNe and VIPER
2. **Week 2**: Implement algorithm-level integration and testing
3. **Week 3**: Integrate with main pipeline and implement automatic acceleration selection
4. **Week 4**: Comprehensive benchmarking, validation, and documentation

## Conclusion

This implementation plan provides a structured approach to optimizing PySCES algorithms using MLX on Apple Silicon hardware. By following this plan, we aim to achieve significant performance improvements while maintaining numerical accuracy and code readability.
