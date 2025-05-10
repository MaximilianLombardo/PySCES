# PySCES Current Status and Next Steps

## Current Status

PySCES has successfully implemented a complete pipeline for single-cell regulatory network analysis, featuring:

### ARACNe Implementation

- **Numba-Accelerated Core**: High-performance implementation using Numba JIT compilation
- **Streamlined Interface**: Single `backend` parameter with options: 'auto', 'numba', 'python'
- **Automatic Backend Selection**: Detects Numba availability with fallback to Python implementation
- **Stratification**: Separate parameters for tissue and cell type stratification with hierarchical processing
- **Cross-Platform Performance**: Numba implementation provides excellent performance across all platforms without specialized hardware

### VIPER Implementation

- **Complete Implementation**: Fully functional with Numba acceleration
- **Core Functions**: Implemented calculate_sig, viper_scores, viper_bootstrap, viper_null_model
- **ARACNe Integration**: Successfully integrated with ARACNe output
- **AnnData Compatibility**: Works with AnnData objects for single-cell analysis

### Performance Benchmarks

Our benchmarks on the Tabula Sapiens testis dataset demonstrate that the Numba-accelerated implementation can process thousands of cells in seconds, with performance varying by cell type:

- **Spermatid cells** (4,009 cells): 1.26 seconds, 7,860 edges/second
- **Spermatocyte cells** (3,070 cells): 0.51 seconds, 14,906 edges/second

### Manual Stratification Approach

We've found that manually stratifying the data (processing each cell type separately) is the most effective approach for large datasets:

- Avoids memory issues with large datasets
- Allows for cell type-specific optimization
- Provides more control over the analysis process
- Enables parallel processing of different cell types

## Next Steps

### 1. Run Complete Pipeline on Real Datasets

Our immediate priority is to test the entire Numba-optimized pipeline on real datasets of varying sizes:

- **Small Dataset**: Tabula Sapiens testis (~7K cells)
- **Medium Dataset**: 50-100K cells
- **Large Dataset**: Approaching production scale (500K+ cells)

For each dataset, we will:
- Measure performance metrics (runtime, memory usage)
- Validate results against expected biological patterns
- Identify bottlenecks and optimization opportunities

### 2. Implement Appropriate Pre-processing

We need to modify the pipeline to include appropriate pre-processing steps:

- **Data Validation**:
  - Verify raw counts (integers) vs. normalized expression (floats)
  - Check for missing values and outliers
  - Validate gene and cell annotations

- **Quality Control**:
  - Filter low-quality cells (high mitochondrial content, low gene count)
  - Filter low-expressed genes
  - Handle batch effects if present

- **Normalization**:
  - Implement appropriate normalization methods for ARACNe input
  - Document normalization impact on network inference

- **Feature Selection**:
  - Select highly variable genes
  - Include known transcription factors
  - Optimize feature selection for performance

### 3. Refine Manual Stratification Strategy

Based on our findings, manual stratification (processing each cell type separately) is the most effective approach for large datasets:

- **Implementation**:
  - Create utility functions to easily stratify AnnData objects
  - Implement parallel processing of strata
  - Develop methods to combine results from different strata

- **Optimization**:
  - Determine optimal stratum size for performance
  - Implement memory-efficient processing
  - Add progress tracking and error handling

### 4. Documentation and Code Cleanup

Several areas need documentation updates and code consolidation:

- **Documentation**:
  - Update function and class docstrings
  - Create tutorial notebooks for the complete pipeline
  - Document performance characteristics and recommendations
  - Add examples for different stratification approaches

- **Code Cleanup**:
  - Replace NumPy 2.0 monkey patch with proper type handling
  - Consolidate duplicate code in different implementations
  - Standardize error handling and logging
  - Remove deprecated or unused functions

## Items to Clean Up/Consolidate

1. **NumPy 2.0 Compatibility**:
   - Replace `np.float_` monkey patch with proper `np.float64` usage throughout the codebase
   - Update type annotations to reflect NumPy 2.0 types

2. **Documentation Updates**:
   - Update all documentation to reflect the Numba-focused architecture
   - Create comprehensive API reference for Numba-accelerated functions
   - Add examples of using the Numba backend effectively

3. **Stratification Implementation**:
   - Consolidate stratification code between different implementations
   - Create helper functions for common stratification patterns
   - Improve documentation of stratification parameters

4. **Benchmark Scripts**:
   - Further consolidate benchmark scripts
   - Create standardized benchmark suite for Numba implementation
   - Add automated performance regression testing

5. **Error Handling**:
   - Standardize error messages
   - Improve error recovery
   - Add more informative debugging information

6. **Experimental Directory**:
   - Ensure all alternative implementations are properly documented
   - Add clear warnings about experimental status
   - Provide examples of when to use experimental implementations

## Implementation Plan

1. **Week 1**: Run complete pipeline on real datasets
   - Set up test datasets of different sizes
   - Create benchmark scripts for the complete pipeline
   - Document performance characteristics

2. **Week 2**: Implement pre-processing steps
   - Add data validation functions
   - Implement quality control steps
   - Create normalization utilities
   - Document pre-processing impact

3. **Week 3**: Refine stratification strategy
   - Create utility functions for stratification
   - Implement parallel processing
   - Optimize memory usage
   - Test on large datasets

4. **Week 4**: Documentation and cleanup
   - Update function and class docstrings
   - Create tutorial notebooks
   - Clean up code and consolidate implementations
   - Update repository organization

## Conclusion

PySCES has successfully implemented a complete, Numba-optimized pipeline for single-cell regulatory network analysis. By focusing on the Numba implementation, we've simplified the codebase while maintaining excellent performance across all platforms. The next phase focuses on testing with real datasets of varying sizes, implementing appropriate pre-processing, refining the stratification strategy, and improving documentation and code organization.

The manual stratification approach has proven effective and will be further developed for production use with larger datasets. By processing each cell type separately, we can avoid memory issues and optimize performance for specific biological contexts.
