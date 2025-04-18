# PySCES Project

This repository contains the PySCES (Python Single-Cell Expression System) project, a Python port of the PISCES single-cell regulatory-network pipeline with integrated support for CELLxGENE Census data.

## Project Structure

The repository is organized as follows:

### Main Package

- **[pysces/](pysces/)**: The main PySCES package, containing the Python implementation of the PISCES pipeline.
  - See the [PySCES README](pysces/README.md) for detailed information about the package, including installation instructions, usage examples, and development status.

### Documentation

- **[docs/](docs/)**: Comprehensive documentation for the PySCES project.
  - [Project Status](docs/PySCES_Project_Status.md): Current status, implementation details, and future plans.
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

## Current Status

PySCES is currently in active development. The following components are implemented and functional:

- ✅ **ARACNe Algorithm**: Core algorithm for inferring gene regulatory networks
  - Mutual information calculation with C++ extensions
  - Data processing inequality (DPI) algorithm
  - Bootstrapping for network robustness
  - Consensus network generation
  - Python fallback implementation
  - Verified with real biological data from CELLxGENE

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

- ✅ **Visualization**: Basic visualization tools
  - UMAP visualization
  - Heatmap generation
  - Network visualization utilities

## Current Development Focus

We are currently focusing on the following areas:

1. **VIPER Algorithm Implementation**: Implementing the VIPER algorithm for inferring transcription factor activity from gene expression data and regulatory networks.

2. **Performance Optimization**: Optimizing the ARACNe algorithm for large datasets, including sampling strategies and memory optimization.

3. **MLX/GPU Acceleration**: Exploring MLX and GPU acceleration as alternatives to C++ extensions for performance-critical operations.

4. **Documentation and Examples**: Improving documentation and adding examples with real data.

For a detailed breakdown of the project status and roadmap, see the [Project Status](docs/PySCES_Project_Status.md) document and the [Roadmap](pysces/ROADMAP.md).

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
   python examples/aracne_census_example.py
   ```

## Contributing

Contributions to the PySCES project are welcome! Please see the [Contributing Guidelines](pysces/CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
