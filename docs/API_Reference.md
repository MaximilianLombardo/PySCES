# PySCES API Reference

## ARACNe Numba-Accelerated Implementation

The ARACNe Numba-accelerated implementation provides high-performance functions for the core ARACNe algorithms.

### ARACNe Class

```python
from pysces.aracne.core import ARACNe

aracne = ARACNe(
    bootstraps=100,
    p_value=0.05,
    dpi_tolerance=0.1,
    consensus_threshold=0.5,
    backend='numba',
    n_threads=0,
    stratify_by_tissue=False,
    stratify_by_cell_type=False,
    tissue_col='tissue',
    cell_type_col='cell_type'
)
```

**Parameters:**
- `bootstraps` (int, default=100): Number of bootstrap iterations
- `p_value` (float, default=0.05): P-value threshold for mutual information significance
- `dpi_tolerance` (float, default=0.1): Tolerance for Data Processing Inequality
- `consensus_threshold` (float, default=0.5): Threshold for consensus network (fraction of bootstrap networks)
- `backend` (str, default='auto'): Computation backend to use. Options: 'auto', 'numba', 'python'
- `n_threads` (int, default=0): Number of threads to use (0 = auto)
- `stratify_by_tissue` (bool, default=False): Whether to stratify analysis by tissue
- `stratify_by_cell_type` (bool, default=False): Whether to stratify analysis by cell type
- `tissue_col` (str, default='tissue'): Column name in AnnData.obs containing tissue information
- `cell_type_col` (str, default='cell_type'): Column name in AnnData.obs containing cell type information

### ARACNe.run()

Run ARACNe network inference on expression data.

```python
network = aracne.run(
    adata,
    tf_list=None,
    layer=None,
    validate=True
)
```

**Parameters:**
- `adata` (AnnData): Expression data (cells x genes)
- `tf_list` (list of str, optional): List of transcription factor names. If None, all genes are considered TFs.
- `layer` (str, optional): Which layer of the AnnData to use. If None, uses .X
- `validate` (bool, default=True): Whether to validate the input data before running ARACNe

**Returns:**
- `network` (dict): Dictionary containing network data and metadata:
  - `regulons`: Dictionary of regulons, where each regulon is a dictionary with targets and weights
  - `tf_names`: List of transcription factor names
  - `consensus_matrix`: Consensus matrix (TFs x genes)
  - `edges`: List of edges, where each edge is a dictionary with source, target, and weight
  - `metadata`: Dictionary of metadata (p_value, bootstraps, dpi_tolerance, consensus_threshold)

**Example:**
```python
import pysces
import anndata as ad
import numpy as np

# Create test data
n_genes = 100
n_cells = 50
X = np.random.normal(0, 1, (n_cells, n_genes))
adata = ad.AnnData(X=X)
adata.var_names = [f"gene_{i}" for i in range(n_genes)]
adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

# Define TF list (first 10 genes)
tf_list = [f"gene_{i}" for i in range(10)]

# Run ARACNe
aracne = pysces.ARACNe(bootstraps=10, backend='numba')
network = aracne.run(adata, tf_list=tf_list)

# Print network statistics
print(f"Number of TFs: {len(network['tf_names'])}")
print(f"Number of edges: {len(network['edges'])}")
```

### aracne_to_regulons()

Convert ARACNe network to a list of Regulon objects.

```python
from pysces.aracne.core import aracne_to_regulons

regulons = aracne_to_regulons(network)
```

**Parameters:**
- `network` (dict): ARACNe network output from ARACNe.run()

**Returns:**
- `regulons` (list): List of Regulon objects

**Example:**
```python
import pysces

# Run ARACNe
aracne = pysces.ARACNe(bootstraps=10)
network = aracne.run(adata)

# Convert to regulons
regulons = pysces.aracne_to_regulons(network)

# Print regulon information
for regulon in regulons:
    print(f"TF: {regulon.tf_name}, Targets: {len(regulon.targets)}")
```

## VIPER Numba-Accelerated Implementation

The VIPER Numba-accelerated implementation provides high-performance functions for the core VIPER algorithms.

### viper_scores()

Calculate VIPER scores for regulons.

```python
from pysces.viper.core import viper_scores

scores = viper_scores(
    adata,
    regulons,
    layer=None,
    signature_method='rank',
    enrichment_method='gsea',
    abs_score=False,
    normalize=True,
    use_numba=True
)
```

**Parameters:**
- `adata` (AnnData): Gene expression data (cells x genes)
- `regulons` (list of Regulon): List of regulons to calculate enrichment for
- `layer` (str, optional): Layer in AnnData to use. If None, uses .X
- `signature_method` (str, default='rank'): Method for calculating signatures
- `enrichment_method` (str, default='gsea'): Method for calculating enrichment
- `abs_score` (bool, default=False): Whether to use absolute values of the signatures
- `normalize` (bool, default=True): Whether to normalize enrichment scores
- `use_numba` (bool, default=True): Whether to use Numba acceleration if available

**Returns:**
- `scores` (pd.DataFrame): VIPER scores (regulons x cells)

**Example:**
```python
import pysces

# Run ARACNe
aracne = pysces.ARACNe(bootstraps=10)
network = aracne.run(adata)

# Convert to regulons
regulons = pysces.aracne_to_regulons(network)

# Calculate VIPER scores
scores = pysces.viper_scores(adata, regulons)

# Print scores shape
print(f"Scores shape: {scores.shape}")
```

### viper_bootstrap()

Calculate VIPER scores with bootstrapping.

```python
from pysces.viper.core import viper_bootstrap

mean_scores, std_scores = viper_bootstrap(
    adata,
    regulons,
    n_bootstraps=100,
    sample_fraction=0.8,
    layer=None,
    signature_method='rank',
    enrichment_method='gsea',
    abs_score=False,
    normalize=True,
    seed=None,
    use_numba=True
)
```

**Parameters:**
- `adata` (AnnData): Gene expression data (cells x genes)
- `regulons` (list of Regulon): List of regulons to calculate enrichment for
- `n_bootstraps` (int, default=100): Number of bootstrap iterations
- `sample_fraction` (float, default=0.8): Fraction of cells to sample in each bootstrap
- `layer` (str, optional): Layer in AnnData to use. If None, uses .X
- `signature_method` (str, default='rank'): Method for calculating signatures
- `enrichment_method` (str, default='gsea'): Method for calculating enrichment
- `abs_score` (bool, default=False): Whether to use absolute values of the signatures
- `normalize` (bool, default=True): Whether to normalize enrichment scores
- `seed` (int, optional): Random seed for reproducibility
- `use_numba` (bool, default=True): Whether to use Numba acceleration if available

**Returns:**
- `mean_scores` (pd.DataFrame): Mean VIPER scores (regulons x cells)
- `std_scores` (pd.DataFrame): Standard deviations of VIPER scores (regulons x cells)

### viper_null_model()

Calculate VIPER scores with a null model for statistical significance.

```python
from pysces.viper.core import viper_null_model

scores, p_values = viper_null_model(
    adata,
    regulons,
    n_permutations=1000,
    layer=None,
    signature_method='rank',
    enrichment_method='gsea',
    abs_score=False,
    normalize=True,
    seed=None,
    use_numba=True
)
```

**Parameters:**
- `adata` (AnnData): Gene expression data (cells x genes)
- `regulons` (list of Regulon): List of regulons to calculate enrichment for
- `n_permutations` (int, default=1000): Number of permutations for the null model
- `layer` (str, optional): Layer in AnnData to use. If None, uses .X
- `signature_method` (str, default='rank'): Method for calculating signatures
- `enrichment_method` (str, default='gsea'): Method for calculating enrichment
- `abs_score` (bool, default=False): Whether to use absolute values of the signatures
- `normalize` (bool, default=True): Whether to normalize enrichment scores
- `seed` (int, optional): Random seed for reproducibility
- `use_numba` (bool, default=True): Whether to use Numba acceleration if available

**Returns:**
- `scores` (pd.DataFrame): VIPER scores (regulons x cells)
- `p_values` (pd.DataFrame): P-values for VIPER scores (regulons x cells)

## Performance Considerations

The Numba-accelerated implementation provides significant performance improvements over the pure Python implementation:

- **JIT Compilation**: Numba translates Python functions to optimized machine code at runtime, providing significant speedup for numerical computations.
- **Parallelization**: Numba can automatically parallelize certain operations, providing additional speedup on multi-core systems.
- **Memory Efficiency**: The Numba implementation is designed to be memory-efficient, especially for large datasets.
- **Cross-Platform Compatibility**: The Numba implementation works on all major platforms (Linux, macOS, Windows) without any platform-specific requirements.

### Benchmarks

| Dataset Size | Python Time | Numba Time | Speedup |
|--------------|-------------|------------|---------|
| Small (100 genes, 50 samples) | ~10s | ~3s | 3x |
| Medium (1000 genes, 500 samples) | ~10min | ~1min | 10x |
| Large (10000 genes, 1000 samples) | ~2h | ~10min | 12x |

*Note: These are approximate values and may vary depending on the hardware and data characteristics.*

## Usage Recommendations

- **Use Numba Acceleration**: Always use Numba acceleration when available for best performance.
- **Stratify Large Datasets**: For very large datasets, consider stratifying by tissue or cell type to reduce memory requirements and improve performance.
- **Parallel Processing**: Set `n_threads` to the number of available cores for best performance.
- **Memory Considerations**: For very large datasets, consider processing in batches or using a machine with more memory.
