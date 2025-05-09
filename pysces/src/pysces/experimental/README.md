# Experimental Implementations

This directory contains alternative implementations of ARACNe and VIPER algorithms that were explored during development but are not part of the main codebase. These implementations are kept for reference and research purposes but are not actively maintained or supported.

## Contents

- `aracne/`: Alternative ARACNe implementations (PyTorch, MLX)
- `viper/`: Alternative VIPER implementations (MLX)
- `examples/`: Example scripts demonstrating the use of alternative implementations
- `docs/`: Documentation related to alternative implementations

## Background

During the development of PySCES, multiple implementation strategies were explored:

1. __Numba Implementation__: CPU-optimized code using JIT compilation
2. __PyTorch Implementation__: GPU-accelerated code with CUDA support
3. __MLX Implementation__: Optimized for Apple Silicon (M1/M2/M3 chips)
4. __Pure Python Implementation__: Fallback with no optimization

After extensive testing, the __Numba implementation__ emerged as the clear winner for several reasons:

- __Performance__: Provides excellent speed without specialized hardware
- __Compatibility__: Works across all platforms (Windows, Mac, Linux) without GPU requirements
- __Simplicity__: Minimal dependencies, making installation and maintenance easier
- __Reliability__: Robust performance across different datasets and cell types

The implementations in this directory are provided for researchers who may want to experiment with alternative acceleration strategies or who have specialized hardware that could benefit from these approaches.

## Usage

These implementations are not intended for production use and are not actively maintained. If you wish to use them, you will need to:

1. Install the required dependencies (PyTorch, MLX, etc.)
2. Import the modules directly from this experimental directory
3. Handle any compatibility issues that may arise

## Contributing

If you make improvements to these experimental implementations, please consider contributing them back to the project. While they are not part of the main codebase, they may still be valuable to other researchers.
