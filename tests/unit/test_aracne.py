"""
Tests for the ARACNe implementation with a focus on the Numba backend.
"""
import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
from pysces.src.pysces.aracne.core import ARACNe, aracne_to_regulons

def test_aracne_initialization():
    """Test that ARACNe initializes correctly with different parameters."""
    # Default initialization
    aracne = ARACNe()
    assert aracne.bootstraps == 100
    assert aracne.p_value == 0.05
    assert aracne.dpi_tolerance == 0.1

    # Custom parameters
    aracne = ARACNe(bootstraps=50, p_value=0.01, dpi_tolerance=0.2)
    assert aracne.bootstraps == 50
    assert aracne.p_value == 0.01
    assert aracne.dpi_tolerance == 0.2

    # Test backend selection
    aracne = ARACNe(backend='auto')
    # The result depends on whether Numba is installed
    # Either way, use_numba should be a boolean
    assert isinstance(aracne.use_numba, bool)

    aracne = ARACNe(backend='numba')
    # Should prefer Numba, but might fall back to Python if Numba not available
    assert isinstance(aracne.use_numba, bool)

    aracne = ARACNe(backend='python')
    assert not aracne.use_numba

def test_aracne_run():
    """Test running ARACNe on a small random dataset."""
    # Create a small random dataset
    n_genes = 20
    n_cells = 50
    X = np.random.rand(n_cells, n_genes)
    obs = pd.DataFrame(index=[f'cell_{i}' for i in range(n_cells)])
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    adata = AnnData(X=X, obs=obs, var=var)

    # Run ARACNe with default parameters
    aracne = ARACNe(bootstraps=2)  # Use just 2 bootstraps for quick testing
    network = aracne.run(adata)

    # Check network structure
    assert 'regulons' in network
    assert 'edges' in network
    assert 'tf_names' in network
    assert 'consensus_matrix' in network
    assert 'metadata' in network

    # Test with specific TF list
    tf_list = [f'gene_{i}' for i in range(5)]  # First 5 genes as TFs
    network = aracne.run(adata, tf_list=tf_list)
    assert len(network['tf_names']) == 5

def test_aracne_to_regulons():
    """Test converting ARACNe network to regulons."""
    # Create a small random dataset
    n_genes = 20
    n_cells = 50
    X = np.random.rand(n_cells, n_genes)
    obs = pd.DataFrame(index=[f'cell_{i}' for i in range(n_cells)])
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    adata = AnnData(X=X, obs=obs, var=var)

    # Run ARACNe with default parameters
    aracne = ARACNe(bootstraps=2)  # Use just 2 bootstraps for quick testing
    network = aracne.run(adata)

    # Convert to regulons
    regulons = aracne_to_regulons(network)

    # Check regulons
    assert isinstance(regulons, list)
    assert len(regulons) == len(network['tf_names'])

    # Check first regulon
    if regulons:
        regulon = regulons[0]
        assert hasattr(regulon, 'tf_name')
        assert hasattr(regulon, 'targets')

if __name__ == "__main__":
    # Run tests manually
    test_aracne_initialization()
    test_aracne_run()
    test_aracne_to_regulons()
    print("All tests passed!")
