# PySCES Roadmap

This document outlines the development roadmap for PySCES (Python Single-Cell Expression System), a Python port of the PISCES single-cell regulatory-network pipeline with integrated support for CELLxGENE Census data.

## Current Priorities (0-1 month)

### 1. VIPER Algorithm Implementation
- [ ] Create `Regulon` and `GeneSet` classes for network representation
- [ ] Implement core VIPER functions (enrichment calculation, activity scoring)
- [ ] Add analytical methods (single-sample, multi-sample analysis)
- [ ] Implement AnnData compatibility for single-cell workflows
- [ ] Create integration with ARACNe output
- [ ] Add comprehensive tests for VIPER functionality

### 2. ARACNe Optimization
- [ ] Optimize ARACNe for large datasets with sampling strategies
- [ ] Improve memory usage for large datasets
- [ ] Add progress reporting for long-running operations
- [ ] Implement chunking for very large datasets
- [ ] Add comprehensive tests with real data

### 3. Improve Documentation
- [ ] Complete API documentation with examples
- [ ] Add tutorials for common use cases
- [ ] Create user guide with installation and usage instructions
- [ ] Add developer guide for contributors
- [ ] Document algorithm parameters and their effects

## Medium-term Goals (1-3 months)

### 1. MLX/GPU Acceleration
- [ ] Evaluate MLX for GPU-accelerated mutual information calculation
- [ ] Implement MLX-based matrix operations
- [ ] Add GPU support for bootstrapping
- [ ] Optimize GPU memory usage
- [ ] Create benchmarks comparing CPU and GPU performance

### 2. Census Integration Improvements
- [ ] Update batch processing approach to use TileDB-SOMA-ML API
- [ ] Implement chunked processing for very large datasets
- [ ] Add more detailed error messages and suggestions
- [ ] Create examples with Census data

### 3. Advanced Analysis Tools
- [ ] Implement differential network analysis
- [ ] Add network-based clustering
- [ ] Implement cell type identification based on master regulators
- [ ] Add regulatory module detection
- [ ] Create visualization tools for network analysis

### 4. Package Distribution
- [ ] Set up PyPI distribution
- [ ] Create conda package
- [ ] Add versioning and release process
- [ ] Create binary wheels for common platforms
- [ ] Add CI/CD pipeline for automated releases

## Long-term Vision (3+ months)

### 1. Ecosystem Integration
- [ ] Integrate with popular single-cell analysis tools (Scanpy, Seurat, etc.)
- [ ] Add support for additional data sources
- [ ] Create plugins for common analysis platforms
- [ ] Develop interactive visualization tools
- [ ] Add support for spatial transcriptomics data

### 2. Performance and Scalability
- [ ] Implement distributed computing support
- [ ] Add support for cloud-based computation
- [ ] Optimize for very large datasets (millions of cells)
- [ ] Implement out-of-core processing for memory-constrained environments
- [ ] Add support for streaming data processing

### 3. Advanced Algorithms
- [ ] Implement causal network inference
- [ ] Add time-series analysis for dynamic networks
- [ ] Implement multi-omics integration
- [ ] Add support for perturbation data
- [ ] Implement network-based feature selection

## VIPER Implementation Plan

### 1. Core Algorithm Implementation (2-3 weeks)

#### 1.1 Data Structures
- Create a `Regulon` class to represent regulatory networks
  - Attributes: TF name, target genes, interaction modes (activation/repression)
  - Methods: adding targets, filtering, statistical operations
- Implement the `GeneSet` class for msigDB integration
  - Support standard gene set formats (GMT, etc.)
  - Methods for enrichment analysis

#### 1.2 VIPER Core Functions
- Port core functions from the PISCES R implementation:
  - `calculate_sig` - Calculate enrichment signatures
  - `viper_scores` - Calculate VIPER activity scores
  - `viper_bootstrap` - Implement bootstrapping for robust estimates
  - `viper_null_model` - Create null models for significance testing

#### 1.3 Analytical Methods
- Implement single-sample and multi-sample analysis modes
- Add differential activity analysis between conditions
- Create shadow analysis for handling redundancy among regulons
- Implement leading-edge analysis for key target identification

### 2. Integration with ARACNe (1-2 weeks)

#### 2.1 Network Format Conversion
- Create utility functions to convert ARACNe output to VIPER-compatible regulons
- Implement methods to filter and threshold networks based on statistical measures
- Add handling for string-based gene identifiers (as seen in our testing)

#### 2.2 Pipeline Integration
- Develop a streamlined workflow from expression data → ARACNe → VIPER
- Create helper functions for common analysis patterns
- Add AnnData integration for single-cell workflows
- Ensure compatibility with the current ARACNe output format

### 3. Performance Optimization (2-3 weeks)

#### 3.1 Algorithmic Optimizations
- Implement sparse matrix operations for large networks
- Add parallel processing support for matrix operations
- Optimize memory usage for large datasets

#### 3.2 MLX GPU Acceleration (Future Phase)
- Evaluate MLX for GPU-accelerated matrix operations
- Create MLX-based implementations of key operations:
  - Matrix multiplication with MLX tensors
  - Enrichment score calculation
  - Bootstrapping operations
- Implement automatic fallback to CPU if MLX/GPU not available
- Add benchmarking for GPU vs CPU performance comparison

### 4. AnnData Integration (1 week)

#### 4.1 AnnData Compatibility
- Focus on making the implementation compatible with AnnData objects
- Create utility functions for extracting relevant data from AnnData objects
- Implement methods for storing VIPER results back in AnnData objects

#### 4.2 CELLxGENE Integration
- Create dedicated loader functions for CELLxGENE datasets
- Implement caching for frequently accessed data
- Add support for metadata-based filtering

### 5. Documentation and Examples (1-2 weeks)

#### 5.1 API Documentation
- Document all VIPER functions with detailed parameter descriptions
- Create usage examples with real data
- Document expected input/output formats

#### 5.2 Tutorial Notebooks
- Create example notebooks demonstrating:
  - Basic VIPER analysis workflow
  - Integration with ARACNe output
  - Differential regulatory activity analysis
  - Cell type identification with master regulators
  - Visualizing regulatory networks and activities

## Implementation Details

### Key VIPER Functions

```python
def calculate_regulon_enrichment(
    expression_data: np.ndarray | anndata.AnnData,
    regulons: List[Regulon],
    pleiotropy: bool = True,
    method: str = "weighted",
    min_targets: int = 10
) -> np.ndarray:
    """
    Calculate enrichment of regulons in expression data using the aREA algorithm.
    
    Parameters
    ----------
    expression_data : np.ndarray or anndata.AnnData
        Differential expression values, with genes as rows and samples as columns
        or an AnnData object with expression data
    regulons : List[Regulon]
        List of regulon objects containing TF and target information
    pleiotropy : bool, default=True
        Whether to correct for regulon pleiotropy
    method : str, default="weighted"
        Enrichment method: "weighted", "unweighted", or "integrated"
    min_targets : int, default=10
        Minimum number of targets required for regulon analysis
        
    Returns
    -------
    np.ndarray
        Matrix of enrichment scores, with regulons as rows and samples as columns
    """
    # Implementation here
    pass

def viper_analysis(
    expression_data: np.ndarray | anndata.AnnData,
    regulons: List[Regulon],
    sample_labels: Optional[List[str]] = None,
    bootstraps: int = 100,
    min_targets: int = 10,
    cores: int = 1,
    use_gpu: bool = False
) -> Dict:
    """
    Run VIPER analysis on expression data using the provided regulons.
    
    Parameters
    ----------
    expression_data : np.ndarray or anndata.AnnData
        Expression data, with genes as rows and samples as columns
        or an AnnData object with expression data
    regulons : List[Regulon]
        List of regulon objects containing TF and target information
    sample_labels : List[str], optional
        Sample labels for the columns of the expression data
    bootstraps : int, default=100
        Number of bootstrap iterations for null model
    min_targets : int, default=10
        Minimum number of targets required for regulon analysis
    cores : int, default=1
        Number of CPU cores to use
    use_gpu : bool, default=False
        Whether to use GPU acceleration via MLX
        
    Returns
    -------
    Dict
        Dictionary containing VIPER results including:
        - 'activity': Normalized enrichment scores for each regulator
        - 'pvalue': P-values for each regulator
        - 'regulons': Filtered regulons used in the analysis
    """
    # Implementation here
    pass
```

## Testing Strategy

- **Unit Tests**: Create comprehensive tests for each VIPER component
- **Integration Tests**: Test the full ARACNe → VIPER pipeline
- **Small Sample Tests**: Create tests with small, synthetic datasets for quick validation
- **Sampling Tests**: Implement tests that sample larger datasets for faster testing
- **Performance Profiling**: Add tests that profile the performance of key functions
- **Memory Usage Tests**: Monitor memory usage during testing to identify potential issues
- **Real Data Tests**: Validate with established benchmark datasets and CELLxGENE data

## Timeline

1. **Core Algorithm Implementation**: 2-3 weeks
2. **Integration with ARACNe**: 1-2 weeks
3. **AnnData Integration**: 1 week
4. **Basic Performance Optimization**: 1-2 weeks
5. **Documentation and Examples**: 1-2 weeks
6. **MLX GPU Acceleration**: 2-3 weeks (second phase)
7. **Census API Integration**: 2-3 weeks (second phase)
