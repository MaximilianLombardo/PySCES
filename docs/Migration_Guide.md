# PySCES Migration Guide

This guide helps users migrate from previous versions of PySCES to the current Numba-focused architecture.

## Overview of Changes

PySCES has undergone a significant architectural change to focus on Numba as the primary acceleration backend. Key changes include:

1. **Simplified Backend Selection**: The `backend` parameter now focuses on Numba with options 'auto', 'numba', and 'python'
2. **Removed C++ Extensions**: The C++ extensions have been removed in favor of the more reliable and cross-platform Numba implementation
3. **Moved Alternative Implementations**: PyTorch and MLX implementations have been moved to the `pysces.experimental` package
4. **Improved Stratification**: Added separate parameters for tissue and cell type stratification
5. **Updated API**: Some parameter names and defaults have changed to reflect the Numba focus

## API Changes

### ARACNe Class

#### Before (v0.2.x):

```python
from pysces.aracne.core import ARACNe

aracne = ARACNe(
    bootstraps=100,
    p_value=0.05,
    dpi_tolerance=0.1,
    consensus_threshold=0.5,
    use_cpp=True,  # Use C++ extensions
    use_gpu=False,  # Use GPU acceleration
    n_threads=0
)
```

#### After (v0.3.x):

```python
from pysces.aracne.core import ARACNe

aracne = ARACNe(
    bootstraps=100,
    p_value=0.05,
    dpi_tolerance=0.1,
    consensus_threshold=0.5,
    backend='numba',  # Use Numba acceleration
    n_threads=0,
    stratify_by_tissue=False,
    stratify_by_cell_type=False,
    tissue_col='tissue',
    cell_type_col='cell_type'
)
```

### VIPER Functions

#### Before (v0.2.x):

```python
from pysces.viper.core import viper_scores

scores = viper_scores(
    adata,
    regulons,
    layer=None,
    signature_method='rank',
    enrichment_method='gsea',
    abs_score=False,
    normalize=True,
    use_gpu=False  # Use GPU acceleration
)
```

#### After (v0.3.x):

```python
from pysces.viper.core import viper_scores

scores = viper_scores(
    adata,
    regulons,
    layer=None,
    signature_method='rank',
    enrichment_method='gsea',
    abs_score=False,
    normalize=True,
    use_numba=True  # Use Numba acceleration
)
```

## Migration Steps

### 1. Update Installation

If you were using C++ extensions or GPU acceleration, you'll need to update your installation:

```bash
# Remove old version
pip uninstall pysces

# Install new version
pip install pysces
```

Make sure you have Numba installed:

```bash
pip install numba>=0.56
```

### 2. Update ARACNe Code

Replace `use_cpp` and `use_gpu` parameters with `backend`:

```python
# Old code
aracne = ARACNe(use_cpp=True, use_gpu=False)

# New code
aracne = ARACNe(backend='numba')
```

### 3. Update VIPER Code

Replace `use_gpu` parameter with `use_numba`:

```python
# Old code
scores = viper_scores(adata, regulons, use_gpu=False)

# New code
scores = viper_scores(adata, regulons, use_numba=True)
```

### 4. Update Stratification Code

If you were manually stratifying data, you can now use the built-in stratification parameters:

```python
# Old code
results = {}
for tissue in adata.obs['tissue'].unique():
    subset = adata[adata.obs['tissue'] == tissue].copy()
    aracne = ARACNe()
    results[tissue] = aracne.run(subset)

# New code
aracne = ARACNe(stratify_by_tissue=True, tissue_col='tissue')
results = aracne.run(adata)
```

### 5. Update Experimental Code

If you were using PyTorch or MLX implementations, you'll need to import them from the experimental package:

```python
# Old code
from pysces.aracne.pytorch_optimized import run_aracne_pytorch

# New code
from pysces.experimental.aracne.pytorch_optimized import run_aracne_pytorch
```

## Handling Deprecated Features

### C++ Extensions

The C++ extensions have been removed entirely. If your code relied on them, you'll need to switch to the Numba implementation:

```python
# Old code
try:
    from pysces.aracne._cpp import aracne_ext
    has_cpp = True
except ImportError:
    has_cpp = False

# New code - just use Numba
from pysces.aracne.core import ARACNe
aracne = ARACNe(backend='numba')
```

### GPU Acceleration

GPU acceleration through PyTorch is now in the experimental package:

```python
# Old code
aracne = ARACNe(use_gpu=True)

# New code
from pysces.experimental.aracne.pytorch_optimized import run_aracne_pytorch
# Use PyTorch implementation directly
```

## Performance Considerations

The Numba implementation provides excellent performance across all platforms without requiring specialized hardware. However, there are some differences to be aware of:

1. **First-Run Compilation**: Numba compiles functions the first time they're called, which can cause a delay on the first run
2. **Memory Usage**: The Numba implementation may use more memory than the C++ implementation in some cases
3. **Parallelization**: The Numba implementation uses a different parallelization strategy, so optimal `n_threads` values may differ

## Troubleshooting

### Common Issues

1. **"No module named 'pysces.aracne._cpp'"**: This error occurs because the C++ extensions have been removed
   - Solution: Update your code to use the Numba implementation

2. **"'ARACNe' object has no attribute 'use_gpu'"**: This error occurs because the `use_gpu` parameter has been replaced with `backend`
   - Solution: Update your code to use `backend='numba'` instead

3. **Performance Regression**: If you experience performance regression after migration
   - Try increasing the number of threads: `aracne = ARACNe(n_threads=8)`
   - Consider using stratification for large datasets: `aracne = ARACNe(stratify_by_cell_type=True)`
   - Check that Numba is being used: `print(aracne.use_numba)`

## Getting Help

If you encounter issues during migration, please:

1. Check the [documentation](https://github.com/your-org/pysces/docs)
2. Open an issue on the [GitHub repository](https://github.com/your-org/pysces/issues)
3. Contact the maintainers for assistance

## Conclusion

The migration to a Numba-focused architecture simplifies the codebase and provides excellent performance across all platforms without requiring specialized hardware. By following this guide, you should be able to update your code to work with the new architecture with minimal changes.
