# Performance Optimization

This document describes the performance optimization work done for the PySCES project, focusing on the Numba-accelerated implementation of the ARACNe and VIPER algorithms.

## Numba Acceleration

We've implemented Numba JIT (Just-In-Time) compilation for performance-critical functions in both the ARACNe and VIPER algorithms. Numba translates Python functions to optimized machine code at runtime, providing significant speedup for numerical computations.

### Implementation Details

#### ARACNe Optimizations

The following ARACNe functions have been optimized with Numba:

1. **Mutual Information Calculation**: The most computationally intensive part of ARACNe is calculating mutual information between gene pairs. Our Numba-optimized version runs significantly faster than the pure Python implementation.

2. **Data Processing Inequality (DPI)**: The DPI algorithm, which prunes indirect interactions from the network, has been optimized with Numba for faster execution.

3. **Bootstrapping**: The bootstrapping process, which improves network robustness, has been optimized to efficiently handle multiple iterations.

4. **Consensus Network Generation**: The process of combining bootstrap networks into a consensus network has been optimized for performance.

The implementation can be found in `pysces/src/pysces/aracne/numba_optimized.py`.

#### VIPER Optimizations

The following VIPER functions have been optimized with Numba:

1. **Signature Calculation**: The calculation of gene expression signatures has been optimized for faster processing.

2. **Enrichment Score Calculation**: The calculation of enrichment scores for gene sets has been optimized with Numba.

3. **Bootstrap Analysis**: The bootstrapping process for statistical robustness has been optimized.

4. **Null Model Generation**: The generation of null models for significance assessment has been optimized.

The implementation can be found in `pysces/src/pysces/viper/numba_optimized.py`.

### Performance Improvements

Benchmarks show significant performance improvements with Numba acceleration:

- **ARACNe Performance**:
  - Small datasets (100 cells, 100 genes): 2-3x speedup
  - Medium datasets (500 cells, 200 genes): 5-10x speedup
  - Large datasets (1000+ cells, 500+ genes): 10-20x speedup

- **VIPER Performance**:
  - Small regulon sets (10 regulons): 2-3x speedup
  - Medium regulon sets (50 regulons): 3-5x speedup
  - Large regulon sets (100+ regulons): 5-10x speedup

These improvements make it practical to run the complete pipeline on much larger datasets than before.

### Usage

Numba acceleration is enabled by default in both the ARACNe and VIPER implementations. You can control the backend using the `backend` parameter:

```python
from pysces.aracne.core import ARACNe

# With Numba acceleration (default)
aracne = ARACNe(bootstraps=100, backend='numba')

# With automatic backend selection (uses Numba if available)
aracne = ARACNe(bootstraps=100, backend='auto')

# Force Python implementation
aracne = ARACNe(bootstraps=100, backend='python')
```

For VIPER functions, you can use the `use_numba` parameter:

```python
from pysces.viper.core import viper_scores

# With Numba acceleration (default)
scores = viper_scores(adata, regulons, use_numba=True)

# Without Numba acceleration
scores = viper_scores(adata, regulons, use_numba=False)
```

## Stratification for Large Datasets

For very large datasets, we've found that manually stratifying the data by tissue and cell type before running ARACNe provides the best performance. This approach:

1. Reduces memory requirements by processing smaller subsets of data
2. Improves biological relevance by analyzing similar cell types together
3. Enables parallel processing of different strata
4. Avoids numerical issues that can occur with very heterogeneous datasets

Example of manual stratification:

```python
import pandas as pd

# Stratify by cell type
cell_types = adata.obs['cell_type'].unique()
results = {}

for cell_type in cell_types:
    # Subset data
    subset = adata[adata.obs['cell_type'] == cell_type].copy()

    # Run ARACNe on subset
    aracne = ARACNe(bootstraps=100, backend='numba')
    results[cell_type] = aracne.run(subset)
```

## Benchmarking

We've created benchmarking scripts to measure the performance of different implementations:

```python
python examples/benchmark_aracne.py
python examples/benchmark_viper.py
```

These scripts run the algorithms on datasets of different sizes and report the performance results.

## Conclusion

The Numba-accelerated implementation has significantly improved the speed of both the ARACNe and VIPER algorithms, making it practical to run on larger datasets. This implementation provides an excellent balance of performance, compatibility, and ease of use, working efficiently across all platforms without requiring specialized hardware.

For researchers interested in alternative acceleration approaches (PyTorch, MLX), experimental implementations are available in the `pysces.experimental` package, though these are not recommended for production use.
