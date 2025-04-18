# PySCES Installation Guide

This guide provides instructions for installing PySCES with C++ extensions for optimal performance.

## Prerequisites

PySCES requires the following:

- Python 3.10 or later
- C++ compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- OpenMP (optional, for parallel processing)

### Platform-Specific Requirements

#### Linux

On Linux, you need GCC and OpenMP:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential libomp-dev

# CentOS/RHEL
sudo yum install gcc-c++ libgomp
```

#### macOS

On macOS, you need Clang and OpenMP (via Homebrew or Conda):

```bash
# Using Homebrew
brew install libomp

# Using Conda
conda install -c conda-forge compilers libomp
```

#### Windows

On Windows, you need Visual Studio 2019 or later with C++ support.

## Installation Methods

### Using pip

The simplest way to install PySCES is using pip:

```bash
pip install pysces
```

This will automatically compile the C++ extensions if possible.

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

## Verifying Installation

To verify that PySCES is installed correctly with C++ extensions:

```python
import pysces
from pysces.aracne._cpp import aracne_ext

# Check version
print(f"PySCES version: {pysces.__version__}")
print(f"ARACNe C++ extensions version: {getattr(aracne_ext, '__version__', 'not available')}")
```

If the C++ extensions are not available, you will see a warning message.

## Troubleshooting

### C++ Extensions Not Available

If you see a warning about C++ extensions not being available:

1. Make sure you have a compatible C++ compiler installed.
2. For macOS, make sure OpenMP is installed via Homebrew or Conda.
3. Try installing with verbose output:

```bash
pip install -v pysces
```

### OpenMP Not Found

If you see a warning about OpenMP not being found:

1. For macOS, install OpenMP via Homebrew or Conda.
2. For Linux, install the OpenMP development package.
3. For Windows, make sure Visual Studio includes C++ and OpenMP support.

### Compilation Errors

If you encounter compilation errors:

1. Make sure your compiler is up to date.
2. Check that you have the necessary development packages installed.
3. Try installing with verbose output to see the exact error:

```bash
pip install -v pysces
```

## Using PySCES without C++ Extensions

If you cannot install the C++ extensions, PySCES will automatically fall back to a pure Python implementation. While this is slower, it provides the same functionality.

To explicitly disable C++ extensions:

```python
from pysces.aracne.core import ARACNe

# Create ARACNe instance with C++ extensions disabled
aracne = ARACNe(use_gpu=True)  # This will force Python implementation
```

## GPU Support (Experimental)

PySCES includes experimental GPU support for some operations. To use GPU acceleration:

1. Install PyTorch with CUDA support:

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

2. Enable GPU support in your code:

```python
from pysces.aracne.core import ARACNe

# Create ARACNe instance with GPU support
aracne = ARACNe(use_gpu=True)
```

Note that GPU support is still experimental and may not be available for all operations.
