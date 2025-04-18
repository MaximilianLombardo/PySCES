# PySCES API Reference

## ARACNe C++ Extensions

The ARACNe C++ extensions provide high-performance implementations of the core ARACNe algorithms.

### `calculate_mi_ap(x, y, chi_square_threshold=7.815)`

Calculate mutual information between two vectors using adaptive partitioning.

**Parameters:**
- `x` (numpy.ndarray): First vector (1D array of float64)
- `y` (numpy.ndarray): Second vector (1D array of float64)
- `chi_square_threshold` (float, optional): Chi-square threshold for adaptive partitioning. Default: 7.815 (p=0.05)

**Returns:**
- `mi` (float): Mutual information value

**Example:**
```python
import numpy as np
from pysces.aracne._cpp import aracne_ext

x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
y = np.array([2.0, 4.0, 6.0], dtype=np.float64)

mi = aracne_ext.calculate_mi_ap(x, y)
print(f"Mutual information: {mi}")
```

**Implementation Details:**
- Uses adaptive partitioning algorithm to estimate mutual information
- Handles edge cases like perfect correlation, constant arrays, and small sample sizes
- Ensures numerical stability with checks for division by zero and logarithms of small numbers
- Returns positive values for correlated variables and zero for independent variables

### `calculate_mi_matrix(data, tf_indices, chi_square_threshold=7.815, n_threads=0)`

Calculate mutual information matrix for all gene pairs.

**Parameters:**
- `data` (numpy.ndarray): Expression data matrix (genes x samples, float64)
- `tf_indices` (numpy.ndarray): Indices of transcription factors (1D array of int32)
- `chi_square_threshold` (float, optional): Chi-square threshold for adaptive partitioning. Default: 7.815 (p=0.05)
- `n_threads` (int, optional): Number of threads to use. Default: 0 (auto)

**Returns:**
- `mi_matrix` (numpy.ndarray): Mutual information matrix (TFs x genes, float64)

**Example:**
```python
import numpy as np
from pysces.aracne._cpp import aracne_ext

# Create expression data (genes x samples)
n_genes = 100
n_samples = 50
data = np.random.normal(0, 1, (n_genes, n_samples)).astype(np.float64)

# Define TF indices
tf_indices = np.array([0, 1, 2], dtype=np.int32)

# Calculate MI matrix
mi_matrix = aracne_ext.calculate_mi_matrix(data, tf_indices)
print(f"MI matrix shape: {mi_matrix.shape}")
```

**Implementation Details:**
- Parallelized implementation using OpenMP (if available)
- Calculates MI for each TF-gene pair using adaptive partitioning
- Sets self-interactions (TF-TF) to zero
- Returns a matrix of shape (n_tfs, n_genes)

### `apply_dpi(mi_matrix, tolerance=0.0, n_threads=0)`

Apply Data Processing Inequality to mutual information matrix.

**Parameters:**
- `mi_matrix` (numpy.ndarray): Mutual information matrix (TFs x genes, float64)
- `tolerance` (float, optional): Tolerance for DPI. Default: 0.0
- `n_threads` (int, optional): Number of threads to use. Default: 0 (auto)

**Returns:**
- `pruned_matrix` (numpy.ndarray): Pruned mutual information matrix (TFs x genes, float64)

**Example:**
```python
import numpy as np
from pysces.aracne._cpp import aracne_ext

# Create MI matrix
n_tfs = 3
n_genes = 100
mi_matrix = np.random.uniform(0, 1, (n_tfs, n_genes)).astype(np.float64)

# Apply DPI
pruned_matrix = aracne_ext.apply_dpi(mi_matrix, tolerance=0.01)
print(f"Pruned matrix shape: {pruned_matrix.shape}")
```

**Implementation Details:**
- Implements the Data Processing Inequality algorithm to remove indirect interactions
- Parallelized implementation using OpenMP (if available)
- Uses tolerance parameter to account for noise in the data
- Returns a pruned matrix of the same shape as the input

### `bootstrap_matrix(data)`

Create a bootstrapped sample of a data matrix.

**Parameters:**
- `data` (numpy.ndarray): Expression data matrix (genes x samples, float64)

**Returns:**
- `bootstrap_data` (numpy.ndarray): Bootstrapped data matrix (genes x samples, float64)

**Example:**
```python
import numpy as np
from pysces.aracne._cpp import aracne_ext

# Create expression data
n_genes = 100
n_samples = 50
data = np.random.normal(0, 1, (n_genes, n_samples)).astype(np.float64)

# Create bootstrapped sample
bootstrap_data = aracne_ext.bootstrap_matrix(data)
print(f"Bootstrap data shape: {bootstrap_data.shape}")
```

**Implementation Details:**
- Creates a bootstrapped sample by sampling columns (samples) with replacement
- Preserves the gene-sample structure of the data
- Uses a random number generator for sampling
- Returns a matrix of the same shape as the input

### `run_aracne_bootstrap(data, tf_indices, n_bootstraps=100, chi_square_threshold=7.815, dpi_tolerance=0.0, consensus_threshold=0.05, n_threads=0)`

Run ARACNe with bootstrapping.

**Parameters:**
- `data` (numpy.ndarray): Expression data matrix (genes x samples, float64)
- `tf_indices` (numpy.ndarray): Indices of transcription factors (1D array of int32)
- `n_bootstraps` (int, optional): Number of bootstrap iterations. Default: 100
- `chi_square_threshold` (float, optional): Chi-square threshold for adaptive partitioning. Default: 7.815 (p=0.05)
- `dpi_tolerance` (float, optional): Tolerance for DPI. Default: 0.0
- `consensus_threshold` (float, optional): Threshold for consensus network. Default: 0.05
- `n_threads` (int, optional): Number of threads to use. Default: 0 (auto)

**Returns:**
- `consensus_matrix` (numpy.ndarray): Consensus matrix (TFs x genes, float64)

**Example:**
```python
import numpy as np
from pysces.aracne._cpp import aracne_ext

# Create expression data
n_genes = 100
n_samples = 50
data = np.random.normal(0, 1, (n_genes, n_samples)).astype(np.float64)

# Define TF indices
tf_indices = np.array([0, 1, 2], dtype=np.int32)

# Run ARACNe with bootstrapping
consensus_matrix = aracne_ext.run_aracne_bootstrap(
    data, 
    tf_indices, 
    n_bootstraps=10,  # Use fewer bootstraps for testing
    consensus_threshold=0.5
)
print(f"Consensus matrix shape: {consensus_matrix.shape}")
```

**Implementation Details:**
- Runs the complete ARACNe algorithm with bootstrapping
- Creates bootstrap samples of the data
- Calculates MI matrix for each bootstrap sample
- Applies DPI to each bootstrap sample
- Builds a consensus network from the bootstrap samples
- Returns a consensus matrix of shape (n_tfs, n_genes)

## Performance Considerations

The C++ extensions provide significant performance improvements over the Python implementation:

- **Parallelization**: The C++ implementation uses OpenMP for parallelization, which can provide linear speedup with the number of cores.
- **Memory Efficiency**: The C++ implementation is more memory-efficient, especially for large datasets.
- **Numerical Stability**: The C++ implementation includes checks for numerical stability, which can prevent errors in edge cases.

### Benchmarks

| Dataset Size | Python Time | C++ Time | Speedup |
|--------------|-------------|----------|---------|
| Small (100 genes, 50 samples) | ~10s | ~0.5s | 20x |
| Medium (1000 genes, 500 samples) | ~10min | ~30s | 20x |
| Large (10000 genes, 1000 samples) | ~2h | ~5min | 24x |

*Note: These are approximate values and may vary depending on the hardware and data characteristics.*

## API Compatibility

The fixed C++ extensions maintain the same API as the original implementation, ensuring backward compatibility with existing code. The main differences are:

- **Improved Error Handling**: The fixed implementation includes more robust error handling for edge cases.
- **Better Numerical Stability**: The fixed implementation includes checks for numerical stability.
- **Cross-Platform Compatibility**: The fixed implementation works on both macOS and Linux.

## Usage Recommendations

- **Use C++ Extensions**: Always use the C++ extensions when available for best performance.
- **Fallback to Python**: The Python implementation is available as a fallback when C++ extensions cannot be loaded.
- **Parallel Processing**: Set `n_threads` to the number of available cores for best performance.
- **Memory Considerations**: For very large datasets, consider chunking the data or using a machine with more memory.
