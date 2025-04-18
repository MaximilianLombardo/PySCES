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
   python examples/basic_workflow.ipynb
   ```

4. **Read the documentation**:
   - [Project Status](docs/PySCES_Project_Status.md) for an overview of the current state of the project.
   - [PySCES README](pysces/README.md) for detailed information about the package.

## Contributing

Contributions to the PySCES project are welcome! Please see the [Contributing Guidelines](pysces/CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](pysces/LICENSE) file for details.

## Acknowledgments

PySCES is a port of the PISCES pipeline developed by the Califano Lab at Columbia University. We acknowledge the original authors of the PISCES pipeline and related projects for their valuable contributions to the field of single-cell analysis and regulatory network inference.
