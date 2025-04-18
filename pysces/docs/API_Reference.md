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

**Notes:**
- The function handles edge cases such as constant arrays (returns 0) and arrays with NaN values (returns NaN).
- For perfect correlation or anti-correlation, the function returns a value based on the correlation strength.
- For other cases, it uses the adaptive partitioning algorithm.

### `calculate_mi_matrix(data, tf_indices, chi_square_threshold=7.815, n_threads=0)`

Calculate mutual information matrix for all gene pairs.

**Parameters:**
- `data` (numpy.ndarray): Expression matrix (samples x genes)
- `tf_indices` (numpy.ndarray): Indices of transcription factors
- `chi_square_threshold` (float, optional): Chi-square threshold for adaptive partitioning. Default: 7.815 (p=0.05)
- `n_threads` (int, optional): Number of threads to use. Default: 0 (auto)

**Returns:**
- `mi_matrix` (numpy.ndarray): Mutual information matrix (n_tfs x n_genes)

**Example:**
```python
import numpy as np
from pysces.aracne._cpp import aracne_ext

# Generate random expression data
n_samples = 100
n_genes = 20
data = np.random.normal(0, 1, (n_samples, n_genes)).astype(np.float64)

# Define TF indices
tf_indices = np.array([0, 1, 2], dtype=np.int32)

# Calculate MI matrix
mi_matrix = aracne_ext.calculate_mi_matrix(data, tf_indices)
print(f"MI matrix shape: {mi_matrix.shape}")
```

**Notes:**
- The function automatically sets diagonal elements (self-interactions) to zero.
- The function uses OpenMP for parallelization if available.
- If `n_threads` is 0, it uses all available cores.

### `apply_dpi(mi_matrix, tolerance=0.0, n_threads=0)`

Apply Data Processing Inequality to a mutual information matrix.

**Parameters:**
- `mi_matrix` (numpy.ndarray): Mutual information matrix (n_tfs x n_genes)
- `tolerance` (float, optional): Tolerance for DPI. Default: 0.0
- `n_threads` (int, optional): Number of threads to use. Default: 0 (auto)

**Returns:**
- `pruned_matrix` (numpy.ndarray): Pruned mutual information matrix (n_tfs x n_genes)

**Example:**
```python
import numpy as np
from pysces.aracne._cpp import aracne_ext

# Create a simple MI matrix
mi_matrix = np.array([
    [0.0, 0.5, 0.3],  # TF1 -> Gene1, Gene2, Gene3
    [0.4, 0.0, 0.6]   # TF2 -> Gene1, Gene2, Gene3
], dtype=np.float64)

# Apply DPI
pruned_matrix = aracne_ext.apply_dpi(mi_matrix)
print(pruned_matrix)
```

**Notes:**
- The function removes indirect interactions based on the DPI algorithm.
- The function uses OpenMP for parallelization if available.
- If `n_threads` is 0, it uses all available cores.

### `bootstrap_matrix(data)`

Create a bootstrapped sample of a data matrix.

**Parameters:**
- `data` (numpy.ndarray): Expression matrix (samples x genes)

**Returns:**
- `bootstrap_data` (numpy.ndarray): Bootstrapped expression matrix (samples x genes)

**Example:**
```python
import numpy as np
from pysces.aracne._cpp import aracne_ext

# Generate random expression data
n_samples = 100
n_genes = 20
data = np.random.normal(0, 1, (n_samples, n_genes)).astype(np.float64)

# Create bootstrap sample
bootstrap_data = aracne_ext.bootstrap_matrix(data)
print(f"Bootstrap data shape: {bootstrap_data.shape}")
```

**Notes:**
- The function samples with replacement from the original data.
- The function preserves the shape of the original data.

### `run_aracne_bootstrap(data, tf_indices, n_bootstraps=100, chi_square_threshold=7.815, dpi_tolerance=0.0, consensus_threshold=0.05, n_threads=0)`

Run ARACNe with bootstrapping.

**Parameters:**
- `data` (numpy.ndarray): Expression matrix (samples x genes)
- `tf_indices` (numpy.ndarray): Indices of transcription factors
- `n_bootstraps` (int, optional): Number of bootstrap iterations. Default: 100
- `chi_square_threshold` (float, optional): Chi-square threshold for adaptive partitioning. Default: 7.815 (p=0.05)
- `dpi_tolerance` (float, optional): Tolerance for DPI. Default: 0.0
- `consensus_threshold` (float, optional): Threshold for consensus network. Default: 0.05
- `n_threads` (int, optional): Number of threads to use. Default: 0 (auto)

**Returns:**
- `consensus_matrix` (numpy.ndarray): Consensus network matrix (n_tfs x n_genes)

**Example:**
```python
import numpy as np
from pysces.aracne._cpp import aracne_ext

# Generate random expression data
n_samples = 100
n_genes = 20
data = np.random.normal(0, 1, (n_samples, n_genes)).astype(np.float64)

# Define TF indices
tf_indices = np.array([0, 1, 2], dtype=np.int32)

# Run ARACNe with bootstrapping
consensus_matrix = aracne_ext.run_aracne_bootstrap(
    data, tf_indices, n_bootstraps=10)
print(f"Consensus matrix shape: {consensus_matrix.shape}")
```

**Notes:**
- The function runs the full ARACNe algorithm with bootstrapping.
- The function uses OpenMP for parallelization if available.
- If `n_threads` is 0, it uses all available cores.
- The consensus matrix contains the fraction of bootstrap networks in which each interaction appears.
- Interactions that appear in less than `consensus_threshold` of bootstrap networks are set to zero.

## Performance Considerations

The C++ extensions provide significant performance improvements over the Python implementation:

- **Mutual Information Calculation**: The C++ implementation is typically 10-100x faster than the Python implementation.
- **Matrix Operations**: The C++ implementation uses efficient data structures and algorithms for matrix operations.
- **Parallelization**: The C++ implementation uses OpenMP for parallelization, which can provide additional speedup on multi-core systems.

For large datasets, the performance difference can be substantial. For example, running ARACNe on a dataset with 10,000 genes and 1,000 samples can take hours with the Python implementation but only minutes with the C++ implementation.

## Platform Compatibility

The C++ extensions are compatible with the following platforms:

- **Linux**: Full support with OpenMP parallelization.
- **macOS**: Support with OpenMP if libomp is installed (via Homebrew or Conda).
- **Windows**: Support with OpenMP.

If the C++ extensions cannot be compiled on your platform, PySCES will automatically fall back to the Python implementation with a warning message.
