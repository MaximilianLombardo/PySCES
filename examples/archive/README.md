# Archived Benchmark Scripts

This directory contains archived benchmark scripts that were used during the development and testing of the PySCES ARACNe implementation. These scripts are kept for reference and may be useful for future development.

## Script Descriptions

### aracne_pytorch_example.py
Example script demonstrating the PyTorch-accelerated ARACNe implementation. Includes benchmarks comparing Python, Numba, and PyTorch implementations.

### robust_aracne_benchmark.py
A more robust benchmark for comparing different ARACNe implementations. Includes multiple iterations and warmup runs to account for JIT compilation overhead.

### stratified_aracne_benchmark.py
Benchmark script for testing ARACNe with stratification by cell type on the Tabula Sapiens dataset.

### robust_stratified_benchmark.py
An improved version of the stratified benchmark that focuses on cell types with at least 200 cells and includes more detailed performance metrics.

### aracne_subset_test.py
Test script for running ARACNe on a subset of the Tabula Sapiens dataset. Includes NumPy 2.0 compatibility fixes.

### test_dataset_load.py
Simple script to test loading the Tabula Sapiens dataset and checking its properties.

### simple_aracne_test.py
A simplified test script for verifying that the ARACNe implementation works correctly with different backends.

### manual_stratified_benchmark.py
Benchmark script that manually stratifies the data by cell type and runs ARACNe separately on each stratum, allowing for more direct comparison of performance across different backends.

## Usage Notes

These scripts are provided for reference only and may require modifications to work with the current version of PySCES. They include various approaches to benchmarking and testing the ARACNe implementation, which may be useful for future development.

To use any of these scripts, copy them from the archive directory to the examples directory and modify as needed.
