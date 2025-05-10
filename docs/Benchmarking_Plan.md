# PySCES Benchmarking Plan

This document outlines the comprehensive plan for benchmarking the Numba-accelerated implementation of PySCES on real datasets to validate its performance.

## 1. Benchmark Objectives

Our benchmarking aims to:

1. **Validate Performance**: Confirm that the Numba implementation delivers the expected performance improvements
2. **Establish Baselines**: Create performance baselines for different dataset sizes and types
3. **Identify Bottlenecks**: Find any remaining performance bottlenecks in the pipeline
4. **Document Scalability**: Understand how performance scales with dataset size
5. **Compare Stratification Approaches**: Evaluate the impact of tissue and cell type stratification

## 2. Datasets Selection

We will select a range of real datasets with varying characteristics:

### Small Datasets (< 10,000 cells)
- **Tabula Sapiens Testis**: Already available in the repository
- **PBMC 3K**: Common benchmark dataset, easily available

### Medium Datasets (10,000 - 100,000 cells)
- **Tabula Sapiens Lung**: Larger tissue from Tabula Sapiens
- **PBMC 10K**: Larger PBMC dataset

### Large Datasets (> 100,000 cells)
- **CELLxGENE Census Subset**: A subset of cells from multiple tissues
- **Human Cell Atlas Subset**: Representative subset from HCA

## 3. Benchmark Metrics

We will measure:

1. **Execution Time**: Total runtime for each component of the pipeline
   - ARACNe (MI calculation, DPI, bootstrapping, consensus)
   - VIPER (signature calculation, enrichment scoring)
   
2. **Memory Usage**: Peak memory consumption during execution

3. **Scalability Metrics**:
   - Time vs. number of cells
   - Time vs. number of genes
   - Time vs. number of TFs

4. **Stratification Impact**:
   - Performance with vs. without tissue stratification
   - Performance with vs. without cell type stratification
   - Combined stratification approaches

## 4. Benchmark Implementation

We will create two main scripts:

1. **benchmark_pipeline.py**: Core benchmarking functions
   - Functions to benchmark ARACNe, VIPER, and the full pipeline
   - Functions to measure execution time and memory usage
   - Functions to benchmark different stratification approaches
   - Functions to analyze and visualize results

2. **run_benchmarks.py**: Script to run all benchmarks
   - Load datasets of different sizes
   - Run benchmarks on each dataset
   - Save results to CSV files
   - Generate visualization plots

## 5. Analysis and Documentation

After running the benchmarks, we will:

1. **Analyze Results**:
   - Compare execution times across dataset sizes
   - Evaluate the impact of stratification
   - Identify any performance bottlenecks
   - Determine optimal parameters for different dataset sizes

2. **Create Visualization**:
   - Plot execution time vs. dataset size
   - Plot memory usage vs. dataset size
   - Compare stratification approaches
   - Visualize scaling behavior

3. **Document Findings**:
   - Create a benchmark report (markdown document)
   - Update performance expectations in documentation
   - Provide recommendations for different dataset sizes

## 6. Implementation Timeline

1. **Week 1: Setup and Small Datasets**
   - Implement benchmark scripts
   - Acquire and prepare small datasets
   - Run initial benchmarks on small datasets

2. **Week 2: Medium and Large Datasets**
   - Acquire and prepare medium and large datasets
   - Run benchmarks on medium datasets
   - Run benchmarks on large datasets with stratification

3. **Week 3: Analysis and Documentation**
   - Analyze benchmark results
   - Create visualizations
   - Document findings and recommendations
   - Update project documentation

## 7. Expected Outcomes

1. **Performance Validation**: Confirmation that the Numba implementation delivers the expected performance improvements
2. **Optimization Opportunities**: Identification of any remaining bottlenecks
3. **User Guidance**: Clear recommendations for users on optimal parameters for different dataset sizes
4. **Documentation Updates**: Updated performance expectations in the project documentation
5. **Benchmark Suite**: Reusable benchmark suite for future performance testing

## 8. Potential Challenges and Mitigations

1. **Challenge**: Very large datasets may exceed available memory
   - **Mitigation**: Implement chunking or streaming approaches, use stratification

2. **Challenge**: Benchmarking may take a long time for large datasets
   - **Mitigation**: Use smaller bootstrap values for initial testing, run longer benchmarks overnight

3. **Challenge**: Datasets may have different characteristics affecting performance
   - **Mitigation**: Document dataset characteristics alongside results, normalize where appropriate

4. **Challenge**: Some datasets may not have tissue or cell type annotations
   - **Mitigation**: Add synthetic annotations for testing or focus on datasets with proper annotations

## 9. Resources Needed

1. **Computational Resources**:
   - High-memory machine for large datasets (32GB+ RAM)
   - Multi-core CPU for parallel processing
   - Sufficient disk space for datasets and results

2. **Datasets**:
   - Access to CELLxGENE Census API
   - Storage for downloaded datasets
   - Pre-processing scripts for data preparation

3. **Software**:
   - Memory profiling tools (memory_profiler)
   - Visualization libraries (matplotlib, seaborn)
   - Data manipulation libraries (pandas, numpy)

## Appendix: Benchmark Script Templates

### benchmark_pipeline.py

```python
# Core benchmarking functions will be implemented here
# Including:
# - Functions to benchmark ARACNe
# - Functions to benchmark VIPER
# - Functions to benchmark the full pipeline
# - Functions to measure execution time and memory usage
# - Functions to benchmark different stratification approaches
# - Functions to analyze and visualize results
```

### run_benchmarks.py

```python
# Script to run all benchmarks will be implemented here
# Including:
# - Loading datasets of different sizes
# - Running benchmarks on each dataset
# - Saving results to CSV files
# - Generating visualization plots
```

This benchmarking plan will be implemented in the coming weeks to provide valuable insights into the performance characteristics of the Numba-accelerated implementation of PySCES and help guide future optimization efforts.
