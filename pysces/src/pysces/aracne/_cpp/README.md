# ARACNe C++ Extensions

This directory contains C++ extensions for the ARACNe algorithm, which provide significant performance improvements over the Python implementation.

## Overview

The C++ extensions implement the following components of the ARACNe algorithm:

- Mutual Information calculation using adaptive partitioning
- MI matrix calculation for all gene pairs
- Data Processing Inequality (DPI) algorithm
- Bootstrap sampling
- Full ARACNe algorithm with bootstrapping

## Implementation Details

### Mutual Information Calculation

The mutual information calculation uses adaptive partitioning to estimate the mutual information between two vectors. The implementation includes:

- Handling of edge cases (constant arrays, NaN values, etc.)
- Optimization for perfect correlation/anti-correlation
- Efficient partitioning algorithm

### MI Matrix Calculation

The MI matrix calculation computes mutual information for all TF-gene pairs. The implementation includes:

- Parallelization using OpenMP
- Efficient memory management
- Handling of self-interactions

### Data Processing Inequality

The DPI algorithm removes indirect interactions from the network. The implementation includes:

- Parallelization using OpenMP
- Tolerance parameter for controlling pruning
- Efficient matrix operations

### Bootstrap Sampling

The bootstrap sampling creates resampled datasets for robust network inference. The implementation includes:

- Efficient sampling with replacement
- Random number generation
- Preservation of data structure

## Building the Extensions

The extensions are built automatically when installing PySCES. The build process requires:

- C++11 compatible compiler
- pybind11
- OpenMP (optional, for parallelization)

## Platform Compatibility

The extensions are compatible with:

- Linux (GCC 9+)
- macOS (Clang 10+ with libomp)
- Windows (MSVC 2019+)

## Performance

The C++ extensions provide significant performance improvements over the Python implementation:

- MI calculation: 10-100x faster
- Full ARACNe algorithm: 5-50x faster

## Version History

- 1.0.0: Initial release with fixed implementation
- 0.1.0: Original implementation (with known issues)

## Known Issues

The original implementation had several issues that have been fixed in the current version:

- Handling of perfect correlation/anti-correlation
- Handling of constant arrays
- MI matrix shape issues
- Segmentation faults in matrix operations

## Future Improvements

Planned improvements include:

- GPU acceleration for large datasets
- Further optimization of memory usage
- Support for sparse matrices
- Additional parallelization strategies
