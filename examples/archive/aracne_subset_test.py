"""
Test ARACNe on a subset of the Tabula Sapiens dataset.
"""

import os
import sys
import numpy as np
import anndata as ad
import scipy.sparse
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print versions
print(f"NumPy version: {np.__version__}")
print(f"AnnData version: {ad.__version__}")

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pysces/src')))

# Monkey patch NumPy for compatibility with older code
if not hasattr(np, 'float_'):
    logger.info("Monkey patching np.float_ for NumPy 2.0 compatibility")
    np.float_ = np.float64

# Import ARACNe directly from the core module
try:
    from pysces.aracne.core import ARACNe
    logger.info("Successfully imported ARACNe")
except Exception as e:
    logger.error(f"Error importing ARACNe: {str(e)}")
    sys.exit(1)

def load_and_subset_data():
    """Load and subset the Tabula Sapiens dataset."""
    try:
        # Load the dataset
        logger.info("Loading Tabula Sapiens testis dataset...")
        adata = ad.read_h5ad("./tabula_sapiens_testis.h5ad")
        logger.info(f"Loaded dataset with {adata.n_obs} cells and {adata.n_vars} genes")
        
        # Subset to top 1000 highly variable genes
        logger.info("Subsetting to top 1000 highly variable genes...")
        if scipy.sparse.issparse(adata.X):
            gene_means = np.array(adata.X.mean(axis=0)).flatten()
            gene_vars = np.array(adata.X.power(2).mean(axis=0)).flatten() - gene_means**2
        else:
            gene_means = np.mean(adata.X, axis=0)
            gene_vars = np.var(adata.X, axis=0)
        
        # Calculate coefficient of variation
        cv = gene_vars / (gene_means + 1e-8)
        
        # Get top 1000 genes by CV
        top_genes = np.argsort(cv)[-1000:]
        adata = adata[:, top_genes]
        logger.info(f"Subset dataset has {adata.n_obs} cells and {adata.n_vars} genes")
        
        # Subset to a smaller number of cells for faster testing
        logger.info("Subsetting to 1000 random cells...")
        cell_indices = np.random.choice(adata.n_obs, size=1000, replace=False)
        adata = adata[cell_indices, :]
        logger.info(f"Final dataset has {adata.n_obs} cells and {adata.n_vars} genes")
        
        return adata
    
    except Exception as e:
        logger.error(f"Error loading and subsetting data: {str(e)}")
        return None

def run_aracne_test():
    """Run ARACNe on a subset of the data."""
    # Load and subset data
    adata = load_and_subset_data()
    if adata is None:
        return
    
    # Create a list of TFs (for simplicity, use the first 10 genes)
    tf_list = adata.var_names[:10].tolist()
    
    # Test parameters
    bootstraps = 2  # Minimal for testing
    
    # Test backends
    backends = ['python', 'numba']
    
    for backend in backends:
        logger.info(f"Testing {backend} backend...")
        aracne = ARACNe(bootstraps=bootstraps, backend=backend)
        
        start_time = time.time()
        try:
            network = aracne.run(adata, tf_list=tf_list)
            run_time = time.time() - start_time
            logger.info(f"{backend} backend: {run_time:.2f} seconds, {len(network['edges'])} edges")
        except Exception as e:
            logger.error(f"{backend} backend failed: {str(e)}")

if __name__ == "__main__":
    run_aracne_test()
