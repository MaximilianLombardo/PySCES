# PySCES Examples

This directory contains example scripts for using the PySCES package.

## Benchmark Scripts

### numba_cell_type_benchmark.py
Benchmark script for testing the Numba-accelerated ARACNe implementation on different cell types from the Tabula Sapiens testis dataset. This script:

1. Loads the Tabula Sapiens testis dataset
2. Filters to keep only cell types with at least 200 cells
3. Subsets to the top 1000 highly variable genes
4. Stratifies the data by cell type
5. Runs ARACNe separately on each cell type
6. Reports performance metrics for each cell type

#### Usage

```bash
python examples/numba_cell_type_benchmark.py
```

#### Output

The script outputs detailed performance metrics for each cell type, including:
- Runtime in seconds
- Total number of edges in the network
- Number of edges per TF
- Edges per second
- Seconds per 1000 cells

This benchmark is useful for understanding how ARACNe performs on different cell types and for optimizing the implementation for specific use cases.

## Archive

The `archive` directory contains older benchmark scripts that were used during development. These scripts are kept for reference and may be useful for future development.
