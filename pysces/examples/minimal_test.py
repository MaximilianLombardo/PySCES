"""
Minimal test script for the validation module.
"""

import pysces
import numpy as np
import pandas as pd
import anndata as ad
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create a synthetic dataset
n_cells = 50
n_genes = 100
X = np.random.randint(0, 10, size=(n_cells, n_genes))
gene_names = [f"gene_{i}" for i in range(n_genes)]
cell_names = [f"cell_{i}" for i in range(n_cells)]
adata = ad.AnnData(
    X=X, 
    obs=pd.DataFrame(index=cell_names),
    var=pd.DataFrame(index=gene_names)
)

# Validate the dataset
print("\n=== Validating dataset ===")
is_valid, issues = pysces.validate_anndata_structure(adata)
if not is_valid:
    print("❌ AnnData structure has issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✅ AnnData structure is valid")

# Create an invalid dataset with duplicate gene names
gene_names_dup = gene_names.copy()
gene_names_dup[0] = gene_names_dup[1]  # Create a duplicate
adata_invalid = ad.AnnData(
    X=X, 
    obs=pd.DataFrame(index=cell_names),
    var=pd.DataFrame(index=gene_names_dup)
)

# Validate the invalid dataset
print("\n=== Validating invalid dataset ===")
is_valid, issues = pysces.validate_gene_names(adata_invalid)
if not is_valid:
    print("✅ Correctly identified invalid gene names:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("❌ Failed to identify invalid gene names")

print("\n=== Test complete ===")
