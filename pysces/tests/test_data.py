"""
Tests for the data module.
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from pysces.data import preprocessing

def test_preprocess_data():
    """Test the preprocess_data function."""
    # Create a test AnnData object
    n_cells = 100
    n_genes = 200
    
    # Create random data
    X = np.random.randint(0, 10, size=(n_cells, n_genes))
    
    # Create gene names with some mitochondrial genes
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    gene_names[0:10] = [f"MT-{i}" for i in range(10)]  # Add some mitochondrial genes
    
    # Create AnnData object
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=gene_names))
    
    # Run preprocessing
    processed = preprocessing.preprocess_data(
        adata,
        min_genes=10,
        min_cells=5,
        max_pct_mito=20,
        normalize=True
    )
    
    # Check that the output is an AnnData object
    assert isinstance(processed, ad.AnnData)
    
    # Check that mitochondrial percentage was calculated
    assert 'pct_counts_mt' in processed.obs.columns
    
    # Check that normalization was performed
    assert 'counts' in processed.layers
    
    # Check that filtering was performed
    assert processed.n_obs <= n_cells
    assert processed.n_vars <= n_genes

def test_rank_transform():
    """Test the rank_transform function."""
    # Create a test AnnData object
    n_cells = 50
    n_genes = 100
    
    # Create random data
    X = np.random.randint(0, 10, size=(n_cells, n_genes))
    
    # Create AnnData object
    adata = ad.AnnData(X=X)
    
    # Run rank transform
    ranked = preprocessing.rank_transform(adata)
    
    # Check that the output is an AnnData object
    assert isinstance(ranked, ad.AnnData)
    
    # Check that the rank layer was added
    assert 'rank' in ranked.layers
    
    # Check that the rank values are as expected
    for i in range(n_genes):
        gene_expr = X[:, i]
        expected_ranks = pd.Series(gene_expr).rank(method='average').values
        np.testing.assert_allclose(ranked.layers['rank'][:, i], expected_ranks)
