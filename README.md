# PySCES Project

This repository contains the PySCES (Python Single-Cell Expression System) project, a Python port of the PISCES single-cell regulatory-network pipeline with integrated support for CELLxGENE Census data.

## Project Structure

The repository is organized as follows:

### Main Package

- **[pysces/](pysces/)**: The main PySCES package, containing the Python implementation of the PISCES pipeline.
  - See the [PySCES README](pysces/README.md) for detailed information about the package, including installation instructions, usage examples, and development status.

### Documentation

- **[docs/](docs/)**: Comprehensive documentation for the PySCES project.
  - [Current Status and Next Steps](docs/Current_Status_and_Next_Steps.md): Latest status and detailed next steps.
  - [Project Status](docs/PySCES_Project_Status.md): Comprehensive project status and implementation details.
  - [Implementation Plan](docs/pysces_implementation_plan.md): Original implementation plan and system architecture.
  - See the [Documentation README](docs/README.md) for a complete list of available documentation.

### Reference Implementations

- **[references/](references/)**: Reference implementations and resources used as inspiration and guidance for the PySCES project.
  - PISCES: Original R implementation of the PISCES pipeline.
  - GPU-ARACNE: CUDA implementation of the ARACNe algorithm.
  - single-cell-pipeline: Pipeline for single-cell analysis.
  - TileDB-SOMA: Storage format for single-cell data.
  - cellxgene-census: Documentation for CELLxGENE Census data access.
  - See the [References README](references/README.md) for detailed information about each reference implementation.

### Archive

- **[archive/](archive/)**: Archived files and documents that are no longer actively used but are kept for reference.
  - [debug_scripts/](archive/debug_scripts/): Archived debug and test scripts related to the C++ extensions.

## Current Status

PySCES is currently in active development. The following components are implemented and functional:

- ✅ **ARACNe Algorithm**: Core algorithm for inferring gene regulatory networks
  - Mutual information calculation using Python implementation
  - Data processing inequality (DPI) algorithm
  - Bootstrapping for network robustness
  - Consensus network generation
  - Network to regulons conversion
  - Verified with real biological data from CELLxGENE

- ✅ **VIPER Algorithm**: Core algorithm for inferring protein activity
  - GeneSet and Regulon data structures
  - Core VIPER functions (calculate_sig, viper_scores, viper_bootstrap, viper_null_model)
  - ARACNe integration (aracne_to_regulons, aracne_to_viper)
  - AnnData compatibility for single-cell analysis
  - Verified with real biological data from Tabula Sapiens

- ✅ **Data Handling**: Tools for loading and preprocessing data
  - Basic data loading functionality
  - Census integration with direct loading approach
  - Data preprocessing functions
  - Rank transformation for ARACNe
  - Verified compatibility with AnnData objects

- ✅ **Analysis Tools**: Basic analysis functionality
  - Clustering functionality
  - Master regulator identification
  - Similarity calculation functions
  - Protein activity analysis

- ✅ **Visualization**: Basic visualization tools
  - UMAP visualization
  - Heatmap generation
  - Network visualization utilities
  - Protein activity visualization

## Current Development Focus

We are currently focusing on the following areas:

1. **Testing on Real Datasets**: Running the complete Numba-optimized pipeline on real datasets of varying sizes to measure performance and validate results.

2. **Pre-processing Implementation**: Modifying the pipeline to include appropriate pre-processing steps for data validation, quality control, normalization, and feature selection.

3. **Manual Stratification Strategy**: Refining our approach to manually stratify data by cell type, which has proven to be the most effective approach for large datasets.

4. **Documentation and Code Cleanup**: Updating documentation to reflect the current status and plan, and cleaning up code to improve maintainability and performance.

5. **NumPy 2.0 Compatibility**: Replacing temporary monkey patches with proper type handling throughout the codebase.

For a detailed breakdown of the project status and next steps, see the [Current Status and Next Steps](docs/Current_Status_and_Next_Steps.md) document.

## Getting Started

To get started with the PySCES project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/pysces-project.git
   cd pysces-project
   ```

2. **Install the PySCES package**:
   ```bash
   cd pysces
   conda env create -f environment.yml
   conda activate pysces
   pip install -e .
   ```

3. **Run the example scripts**:
   ```bash
   python examples/test_installation.py
   python examples/pipeline_validation_test_direct.py
   python examples/aracne_census_example.py
   ```

## Contributing

Contributions to the PySCES project are welcome! Please see the [Contributing Guidelines](pysces/CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
