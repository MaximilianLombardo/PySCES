# Experimental Implementations in PySCES

This document provides information about the experimental implementations available in PySCES and when you might want to use them.

## Overview

PySCES primarily uses Numba for acceleration, which provides excellent performance across all platforms without requiring specialized hardware. However, during development, we explored several alternative acceleration approaches that might be beneficial in specific scenarios.

These alternative implementations have been moved to the `pysces.experimental` package to keep the main codebase clean and focused on the Numba implementation. They are not recommended for production use but are available for research purposes.

## Available Experimental Implementations

### PyTorch-Based Implementation

The PyTorch-based implementation uses GPU acceleration for performance-critical operations in ARACNe and VIPER.

**Location**: `pysces.experimental.aracne.pytorch_optimized`

**Potential Benefits**:
- May provide better performance on systems with powerful NVIDIA GPUs
- Can leverage CUDA for parallel processing
- Integrates with other PyTorch-based workflows

**Limitations**:
- Requires PyTorch with CUDA support
- Performance varies significantly based on GPU hardware
- More complex to use and debug
- Not as well-tested as the Numba implementation

### MLX-Based Implementation

The MLX-based implementation is optimized for Apple Silicon hardware (M1/M2/M3 chips) and leverages the unified memory architecture and GPU capabilities of these chips.

**Location**: `pysces.experimental.aracne.mlx_optimized` and `pysces.experimental.viper.mlx_optimized`

**Potential Benefits**:
- May provide better performance on Apple Silicon hardware
- Leverages the unified memory architecture for efficient computation
- Uses lazy evaluation for optimized computation graphs

**Limitations**:
- Only works on Apple Silicon hardware
- Requires MLX to be installed
- Less mature than the Numba implementation
- More complex to use and debug

## When to Use Experimental Implementations

You might consider using the experimental implementations in the following scenarios:

1. **Research on Acceleration Approaches**: If you're researching different acceleration approaches for bioinformatics algorithms, these implementations provide a starting point for comparison.

2. **Specialized Hardware**: If you have specialized hardware (powerful NVIDIA GPUs or Apple Silicon) and want to explore potential performance improvements.

3. **Integration with Existing Workflows**: If you're already using PyTorch or MLX in your workflow and want to integrate PySCES more seamlessly.

## How to Use Experimental Implementations

### PyTorch Implementation

```python
from pysces.experimental.aracne.pytorch_optimized import run_aracne_pytorch

# Run ARACNe with PyTorch acceleration
network = run_aracne_pytorch(
    expr_matrix,
    gene_list,
    tf_indices,
    bootstraps=100,
    consensus_threshold=0.05,
    dpi_tolerance=0.0
)
```

### MLX Implementation

```python
from pysces.experimental.aracne.mlx_optimized import run_aracne_mlx

# Run ARACNe with MLX acceleration
network = run_aracne_mlx(
    expr_matrix,
    gene_list,
    tf_indices,
    bootstraps=100,
    consensus_threshold=0.05,
    dpi_tolerance=0.0
)
```

## Installation Requirements

To use the experimental implementations, you'll need to install additional dependencies:

### For PyTorch Implementation

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

### For MLX Implementation

```bash
pip install mlx
```

Note that MLX only works on Apple Silicon hardware (M1/M2/M3 chips).

## Benchmarking and Comparison

We've provided benchmarking scripts in the `pysces.experimental.examples` package to compare the performance of different implementations:

```python
from pysces.experimental.examples.benchmark_mlx import benchmark_implementations

# Run benchmarks
results = benchmark_implementations(
    dataset_sizes=[(100, 100, 10), (500, 200, 20), (1000, 500, 50)],
    bootstraps=10
)

# Print results
for result in results:
    print(f"Dataset size: {result['dataset_size']}")
    print(f"Python time: {result['python_time']:.2f}s")
    print(f"Numba time: {result['numba_time']:.2f}s")
    print(f"MLX time: {result['mlx_time']:.2f}s")
    print(f"Numba speedup: {result['python_numba_speedup']:.2f}x")
    print(f"MLX speedup: {result['python_mlx_speedup']:.2f}x")
    print(f"MLX vs Numba: {result['numba_mlx_speedup']:.2f}x")
    print()
```

## Contributing to Experimental Implementations

If you make improvements to these experimental implementations, please consider contributing them back to the project. While they are not part of the main codebase, they may still be valuable to other researchers.

## Conclusion

The experimental implementations in PySCES provide alternative acceleration approaches for specific hardware or research purposes. While the Numba implementation is recommended for most users due to its excellent balance of performance, compatibility, and ease of use, these experimental implementations may be valuable in specific scenarios.

For most users, we recommend sticking with the Numba implementation, which provides excellent performance across all platforms without requiring specialized hardware.
