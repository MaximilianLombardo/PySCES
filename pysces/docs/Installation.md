# PySCES Installation Guide

This guide provides instructions for installing PySCES with Numba acceleration for optimal performance.

## Prerequisites

PySCES requires the following:

- Python 3.10 or later
- Numba 0.56 or later (for performance optimization)

### Platform-Specific Requirements

PySCES with Numba acceleration works on all major platforms (Linux, macOS, Windows) without any platform-specific requirements. This is one of the key advantages of the Numba-based implementation.

## Installation Methods

### Using pip

The simplest way to install PySCES is using pip:

```bash
pip install pysces
```

This will install PySCES with all required dependencies, including Numba.

### Using Conda

You can also install PySCES using Conda:

```bash
conda install -c conda-forge pysces
```

### From Source

To install from source:

```bash
git clone https://github.com/your-organization/pysces.git
cd pysces
pip install -e .
```

For development, you may want to install with all optional dependencies:

```bash
pip install -e ".[dev,census]"
```

## Verifying Installation

To verify that PySCES is installed correctly with Numba acceleration:

```python
import pysces
import numba

# Check versions
print(f"PySCES version: {pysces.__version__}")
print(f"Numba version: {numba.__version__}")

# Run a simple test
from pysces.aracne.core import ARACNe
aracne = ARACNe(backend='numba')
print(f"Using Numba: {aracne.use_numba}")
```

If Numba is available, you should see `Using Numba: True` in the output.

## Troubleshooting

### Numba Not Available

If you see a warning about Numba not being available:

1. Make sure you have Numba installed: `pip install numba>=0.56`
2. Check that your Python version is compatible with Numba
3. Try installing with verbose output to see any errors:

```bash
pip install -v numba
```

### Performance Issues

If you're experiencing performance issues:

1. Make sure Numba is being used (check with `aracne.use_numba`)
2. Try running with different numbers of threads (`n_threads` parameter)
3. For large datasets, consider using stratification by tissue or cell type

## Using PySCES with Alternative Backends

PySCES primarily uses Numba for acceleration, but also provides a pure Python fallback:

```python
from pysces.aracne.core import ARACNe

# Use Numba (default)
aracne = ARACNe(backend='numba')

# Force Python implementation
aracne = ARACNe(backend='python')

# Auto-detect (uses Numba if available)
aracne = ARACNe(backend='auto')
```

## Experimental Implementations

PySCES includes experimental implementations of alternative backends (PyTorch, MLX) in the `pysces.experimental` package. These are not recommended for production use but are available for research purposes. See the [Experimental README](../src/pysces/experimental/README.md) for more information.
