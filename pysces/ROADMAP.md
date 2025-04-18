# PySCES Roadmap

This document outlines the development roadmap for PySCES (Python Single-Cell Expression System), a Python port of the PISCES single-cell regulatory-network pipeline with integrated support for CELLxGENE Census data.

## Short-term Goals (0-3 months)

### 1. Enhance ARACNe Implementation
- [x] Implement full mutual information calculation in C++
- [x] Add data processing inequality (DPI) algorithm
- [x] Implement bootstrapping for network robustness
- [x] Optimize memory usage for large datasets
- [x] Add OpenMP parallelization for performance
- [ ] Add comprehensive tests against reference outputs
- [ ] Optimize algorithm parameters for single-cell data

### 2. Expand Test Suite
- [x] Add basic unit tests for ARACNe
- [x] Implement CI/CD pipeline for automated testing
- [ ] Add unit tests for all core components
- [ ] Create integration tests for end-to-end workflows
- [ ] Add performance benchmarks
- [ ] Create test datasets with known regulatory relationships

### 3. Improve Documentation
- [x] Add example script for ARACNe with Census data
- [ ] Complete API documentation with examples
- [ ] Add tutorials for common use cases
- [ ] Create user guide with installation and usage instructions
- [ ] Add developer guide for contributors
- [ ] Document algorithm parameters and their effects

### 4. Package Distribution
- [x] Update package structure and dependencies
- [ ] Set up PyPI distribution
- [ ] Create conda package
- [ ] Add versioning and release process
- [ ] Create binary wheels for common platforms

## Medium-term Goals (3-6 months)

### 1. GPU Acceleration
- [ ] Implement GPU-accelerated mutual information calculation
- [ ] Add GPU support for network pruning
- [ ] Optimize GPU memory usage
- [ ] Create benchmarks comparing CPU and GPU performance
- [ ] Add fallback mechanisms for systems without GPU
- [ ] Support mixed precision for performance

### 2. Advanced Analysis Features
- [ ] Implement additional clustering methods
- [ ] Add differential network analysis
- [ ] Implement network comparison tools
- [ ] Add support for multi-omics integration
- [ ] Create advanced visualization tools
- [ ] Add network validation metrics

### 3. Census Integration Enhancements
- [ ] Update batch processing approach to use TileDB-SOMA-ML API
- [ ] Optimize Census data loading for large datasets
- [ ] Add support for incremental processing
- [ ] Implement caching mechanisms
- [ ] Add support for additional Census data types
- [ ] Create specialized preprocessing for Census data

### 4. Performance Optimization
- [x] Implement parallel processing for CPU
- [ ] Profile and optimize critical code paths
- [ ] Further optimize memory usage for large datasets
- [ ] Add support for out-of-core processing
- [ ] Implement chunking for very large datasets

## Long-term Vision (6+ months)

### 1. Ecosystem Integration
- [ ] Integrate with popular single-cell analysis tools (Scanpy, Seurat, etc.)
- [ ] Add support for additional data sources
- [ ] Create plugins for common visualization platforms
- [ ] Develop interoperability with other network inference tools
- [ ] Create web interface for interactive analysis

### 2. Advanced GPU Features
- [ ] Implement multi-GPU support
- [ ] Add distributed computing capabilities
- [ ] Optimize for different GPU architectures
- [ ] Create cloud deployment options
- [ ] Support for TPU and other accelerators

### 3. Community Building
- [ ] Create user forum or discussion platform
- [ ] Develop contributor guidelines
- [ ] Host workshops and tutorials
- [ ] Establish governance model for long-term maintenance
- [ ] Create comprehensive documentation website

### 4. Research Applications
- [ ] Develop specialized modules for specific research domains
- [ ] Create case studies and example applications
- [ ] Collaborate with research groups for validation
- [ ] Publish methods papers and benchmarks
- [ ] Create benchmark datasets for network inference

## Completed Milestones

### Project Setup
- [x] Create repository structure
- [x] Set up basic package configuration
- [x] Create conda environment
- [x] Add license and contribution guidelines

### Core Implementation
- [x] Implement data loading and preprocessing
- [x] Create basic ARACNe framework
- [x] Implement VIPER algorithm
- [x] Add basic analysis tools
- [x] Create visualization functions
- [x] Make Census dependencies optional with graceful fallbacks
- [x] Create test installation script

### ARACNe Implementation
- [x] Implement adaptive partitioning algorithm for mutual information
- [x] Add data processing inequality (DPI) algorithm
- [x] Implement bootstrapping for network robustness
- [x] Add consensus network generation
- [x] Implement OpenMP parallelization for performance
- [x] Add platform-specific optimizations
- [x] Create example script for ARACNe with Census data
