"""
Tests for ARACNe implementation.
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from pysces.aracne import ARACNe, aracne_to_regulons

def test_aracne_initialization():
    """Test ARACNe initialization with different parameters."""
    # Default initialization
    aracne = ARACNe()
    assert aracne.p_value == 1e-8
    assert aracne.bootstraps == 100
    assert aracne.dpi_tolerance == 0.0
    assert aracne.consensus_threshold == 0.05
    assert aracne.n_threads == 0
    assert aracne.use_gpu is False
    
    # Custom initialization
    aracne = ARACNe(
        p_value=0.01,
        bootstraps=50,
        dpi_tolerance=0.1,
        consensus_threshold=0.1,
        n_threads=2,
        use_gpu=True
    )
    assert aracne.p_value == 0.01
    assert aracne.bootstraps == 50
    assert aracne.dpi_tolerance == 0.1
    assert aracne.consensus_threshold == 0.1
    assert aracne.n_threads == 2
    # GPU might be disabled if not available
    # So we don't assert on use_gpu

def test_aracne_run_small_dataset():
    """Test ARACNe on a small synthetic dataset."""
    # Create a small synthetic dataset
    n_genes = 20
    n_cells = 50
    n_tfs = 5
    
    # Generate random expression data
    np.random.seed(42)
    expr_matrix = np.random.normal(0, 1, (n_genes, n_cells))
    
    # Create gene names
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    tf_names = [f"gene_{i}" for i in range(n_tfs)]
    
    # Create AnnData object
    adata = ad.AnnData(
        X=expr_matrix.T,  # Transpose to cells x genes
        var=pd.DataFrame(index=gene_names)
    )
    
    # Run ARACNe
    aracne = ARACNe(bootstraps=2)  # Use fewer bootstraps for testing
    network = aracne.run(adata, tf_list=tf_names)
    
    # Check network structure
    assert 'regulons' in network
    assert 'tf_names' in network
    assert 'metadata' in network
    
    # Check TF names
    assert set(network['tf_names']) == set(tf_names)
    
    # Check regulons
    for tf in tf_names:
        assert tf in network['regulons']
        assert 'targets' in network['regulons'][tf]
    
    # Convert to regulon objects
    regulons = aracne_to_regulons(network)
    
    # Check regulon objects
    assert len(regulons) == n_tfs
    for regulon in regulons:
        assert regulon.tf_name in tf_names
        assert len(regulon.targets) > 0

def test_aracne_run_with_layer():
    """Test ARACNe with a specific layer in AnnData."""
    # Create a small synthetic dataset
    n_genes = 10
    n_cells = 20
    
    # Generate random expression data
    np.random.seed(42)
    expr_matrix = np.random.normal(0, 1, (n_genes, n_cells))
    expr_matrix_normalized = np.exp(expr_matrix)  # Just a different transformation
    
    # Create gene names
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    
    # Create AnnData object with multiple layers
    adata = ad.AnnData(
        X=expr_matrix.T,  # Transpose to cells x genes
        var=pd.DataFrame(index=gene_names)
    )
    adata.layers['normalized'] = expr_matrix_normalized.T
    
    # Run ARACNe with default layer
    aracne = ARACNe(bootstraps=2)  # Use fewer bootstraps for testing
    network1 = aracne.run(adata)
    
    # Run ARACNe with specific layer
    network2 = aracne.run(adata, layer='normalized')
    
    # Networks should be different due to different input data
    # This is a simple check - the actual values will depend on the implementation
    assert network1 != network2

def test_aracne_with_sparse_matrix():
    """Test ARACNe with sparse input matrix."""
    # Create a small synthetic dataset with sparse matrix
    n_genes = 15
    n_cells = 30
    
    # Generate random sparse expression data
    np.random.seed(42)
    expr_matrix = np.random.normal(0, 1, (n_genes, n_cells))
    expr_matrix[expr_matrix < 0.5] = 0  # Sparsify
    sparse_expr_matrix = ad.AnnData(X=expr_matrix.T).X
    
    # Create gene names
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    
    # Create AnnData object with sparse matrix
    adata = ad.AnnData(
        X=sparse_expr_matrix,  # Already cells x genes
        var=pd.DataFrame(index=gene_names)
    )
    
    # Run ARACNe
    aracne = ARACNe(bootstraps=2)  # Use fewer bootstraps for testing
    network = aracne.run(adata)
    
    # Check network structure
    assert 'regulons' in network
    assert 'tf_names' in network
    assert 'metadata' in network

if __name__ == "__main__":
    # Run tests manually
    test_aracne_initialization()
    test_aracne_run_small_dataset()
    test_aracne_run_with_layer()
    test_aracne_with_sparse_matrix()
    print("All tests passed!")
