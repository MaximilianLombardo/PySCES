# PySCES: Python Single-Cell Expression System

PySCES is a Python port of the PISCES single-cell regulatory-network pipeline, with integrated support for CELLxGENE Census data.

> **Note**: This project is under active development. See the [Project Status](../docs/PySCES_Project_Status.md) document for detailed information about the current state, implementation details, and future plans.

## Features

- **ARACNe Network Inference**: Infer gene regulatory networks from single-cell expression data using mutual information and data processing inequality
- **VIPER/metaVIPER**: Infer protein activity from gene expression and regulatory networks
- **CELLxGENE Census Integration**: Directly load and process data from the CELLxGENE Census
- **Analysis Tools**: Clustering, master regulator identification, and visualization
- **High Performance**: C++ extensions with OpenMP parallelization for performance-critical operations
- **GPU Acceleration**: Optional GPU support for performance-critical operations (coming soon)

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-org/pysces.git
cd pysces

# Create and activate conda environment
conda env create -f environment.yml
conda activate pysces

# Install the package
pip install -e .
```

### With Census Integration

To use the CELLxGENE Census integration, you need to install additional dependencies:

```bash
# Install with Census support
pip install -e ".[census]"
```

Alternatively, you can edit the `environment.yml` file to uncomment the Census dependencies before creating the conda environment.

### With GPU Support (Coming Soon)

```bash
# Install with GPU support
pip install -e ".[gpu]"
```

## Quick Start

```python
import pysces
import anndata as ad

# Load data
adata = pysces.read_census_direct(
    "homo_sapiens", 
    obs_value_filter="tissue_general == 'blood'",
    census_version="latest"
)

# Preprocess data
adata = pysces.preprocess_data(adata)

# Run ARACNe
aracne = pysces.ARACNe(
    p_value=1e-8,              # P-value threshold for MI significance
    bootstraps=100,            # Number of bootstrap iterations
    dpi_tolerance=0.0,         # Tolerance for Data Processing Inequality
    consensus_threshold=0.05,  # Threshold for consensus network
    n_threads=0                # Auto-detect number of threads
)
network = aracne.run(adata)

# Run VIPER
regulons = pysces.aracne_to_regulons(network)
activity = pysces.viper(adata, regulons)

# Cluster cells by regulatory activity
adata = pysces.cluster_activity(adata, activity)

# Find master regulators
mrs = pysces.identify_mrs(activity, adata.obs["clusters"])
```

## ARACNe Implementation

The ARACNe algorithm in PySCES is being implemented using C++ extensions with OpenMP parallelization. The implementation plan includes:

1. **Mutual Information Calculation**: Using adaptive partitioning algorithm for accurate MI estimation
2. **Data Processing Inequality (DPI)**: Removing indirect interactions from the network
3. **Bootstrapping**: Improving network robustness through multiple bootstrap iterations
4. **Consensus Network**: Aggregating results from bootstrap iterations

> **Note**: The C++ implementation is currently in progress. The current version provides a framework with placeholder implementations that will be replaced with optimized C++ code.

### ARACNe Parameters

- `p_value`: P-value threshold for mutual information significance
- `bootstraps`: Number of bootstrap iterations
- `dpi_tolerance`: Tolerance for Data Processing Inequality
- `consensus_threshold`: Threshold for consensus network (fraction of bootstrap networks)
- `n_threads`: Number of threads to use (0 = auto)
- `use_gpu`: Whether to use GPU acceleration (if available)

## Documentation

For detailed documentation, see the [examples](examples/) directory. Key examples include:

- [Basic Workflow](examples/basic_workflow.ipynb): End-to-end workflow from data loading to master regulator analysis
- [ARACNe Census Example](examples/aracne_census_example.py): Example of using ARACNe with Census data

## Project Structure

```
pysces/
├── pysces/                      # Main package
│   ├── data/                    # Data handling
│   │   ├── loaders.py           # Data loaders
│   │   ├── census.py            # Census integration
│   │   └── preprocessing.py     # Data preprocessing
│   ├── aracne/                  # ARACNe implementation
│   │   ├── core.py              # Python interface
│   │   └── _cpp/                # C++ extensions
│   ├── viper/                   # VIPER implementation
│   │   ├── regulons.py          # Regulon handling
│   │   └── activity.py          # Activity inference
│   ├── analysis/                # Analysis tools
│   │   ├── clustering.py        # Clustering methods
│   │   └── master_regulators.py # MR identification
│   └── plotting/                # Visualization
│       └── plots.py             # Standard plots
├── tests/                       # Test suite
├── examples/                    # Example notebooks
├── docs/                        # Documentation
├── environment.yml              # Conda environment
├── setup.py                     # Package metadata
└── README.md                    # This file
```

## Development Status

PySCES is currently in active development. The following components are implemented:

- [x] Data loading and preprocessing
- [x] Census integration with direct loading approach
- [x] Basic ARACNe framework with C++ extension structure
- [x] VIPER implementation
- [x] Analysis tools (clustering, master regulators)
- [x] Visualization
- [x] Basic test suite

The following components are in progress or pending:

- [ ] Complete ARACNe C++ implementation
- [ ] Fix batch processing approach for Census data
- [ ] GPU acceleration
- [ ] Comprehensive documentation
- [ ] Comprehensive test suite

For a detailed breakdown of the project status, see the [Project Status](../docs/PySCES_Project_Status.md) document.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

PySCES is a port of the PISCES pipeline developed by the Califano Lab at Columbia University.
