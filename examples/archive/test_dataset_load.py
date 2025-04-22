"""
Simple script to test loading the Tabula Sapiens dataset.
"""

import os
import sys
import numpy as np
import anndata as ad
import scipy.sparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print NumPy version
print(f"NumPy version: {np.__version__}")
print(f"AnnData version: {ad.__version__}")

try:
    # Try to load the dataset
    logger.info("Loading Tabula Sapiens testis dataset...")
    adata = ad.read_h5ad("./tabula_sapiens_testis.h5ad")
    logger.info(f"Successfully loaded dataset with {adata.n_obs} cells and {adata.n_vars} genes")

    # Check if cell_type column exists
    if 'cell_type' in adata.obs.columns:
        logger.info(f"Found 'cell_type' column with {adata.obs['cell_type'].nunique()} unique cell types")
        for cell_type, count in adata.obs['cell_type'].value_counts().items():
            logger.info(f"  {cell_type}: {count} cells")
    else:
        logger.warning("No 'cell_type' column found in dataset")

    # Check data type of expression matrix
    logger.info(f"Expression matrix data type: {adata.X.dtype}")
    logger.info(f"Expression matrix is sparse: {scipy.sparse.issparse(adata.X)}")

    # Check for NaN or infinite values
    if scipy.sparse.issparse(adata.X):
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X

    logger.info(f"NaN values: {np.isnan(X_dense).sum()}")
    logger.info(f"Infinite values: {np.isinf(X_dense).sum()}")

except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
