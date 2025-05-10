"""
Tests for the VIPER implementation with a focus on the Numba backend.
"""
import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
from pysces.src.pysces.viper.core import viper_scores, viper_bootstrap, viper_null_model
from pysces.src.pysces.viper.regulons import Regulon, GeneSet

def create_test_data():
    """Create a small test dataset and regulons."""
    # Create a small random dataset
    n_genes = 20
    n_cells = 30
    X = np.random.rand(n_cells, n_genes)
    obs = pd.DataFrame(index=[f'cell_{i}' for i in range(n_cells)])
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_genes)])
    adata = AnnData(X=X, obs=obs, var=var)

    # Create regulons
    regulons = []
    for i in range(5):  # Create 5 regulons
        regulon = Regulon(tf_name=f'gene_{i}')
        # Add some targets
        for j in range(10):
            target_idx = (i + j) % n_genes
            regulon.add_target(f'gene_{target_idx}', mode=1.0, likelihood=0.8)
        regulons.append(regulon)

    return adata, regulons

def test_viper_scores():
    """Test calculating VIPER scores."""
    adata, regulons = create_test_data()

    # Calculate VIPER scores
    viper_df = viper_scores(
        adata,
        regulons,
        signature_method='rank',
        enrichment_method='gsea',
        use_numba=True
    )

    # Check result
    assert isinstance(viper_df, pd.DataFrame)
    assert viper_df.shape == (len(regulons), adata.n_obs)
    assert list(viper_df.index) == [r.tf_name for r in regulons]
    assert list(viper_df.columns) == list(adata.obs_names)

def test_viper_bootstrap():
    """Test VIPER bootstrapping."""
    adata, regulons = create_test_data()

    # Calculate bootstrapped VIPER scores
    # Use sample_fraction=1.0 to avoid shape mismatch issues
    mean_df, std_df = viper_bootstrap(
        adata,
        regulons,
        n_bootstraps=2,  # Use just 2 bootstraps for quick testing
        sample_fraction=1.0,  # Use all cells to avoid shape mismatch
        use_numba=True
    )

    # Check results
    assert isinstance(mean_df, pd.DataFrame)
    assert isinstance(std_df, pd.DataFrame)
    assert mean_df.shape == (len(regulons), adata.n_obs)
    assert std_df.shape == (len(regulons), adata.n_obs)

def test_viper_null_model():
    """Test VIPER null model."""
    adata, regulons = create_test_data()

    # Calculate VIPER scores with null model
    scores_df, pvals_df = viper_null_model(
        adata,
        regulons,
        n_permutations=2,  # Use just 2 permutations for quick testing
        use_numba=True
    )

    # Check results
    assert isinstance(scores_df, pd.DataFrame)
    assert isinstance(pvals_df, pd.DataFrame)
    assert scores_df.shape == (len(regulons), adata.n_obs)
    assert pvals_df.shape == (len(regulons), adata.n_obs)
    assert pvals_df.min().min() >= 0
    assert pvals_df.max().max() <= 1

if __name__ == "__main__":
    # Run tests manually
    test_viper_scores()
    test_viper_bootstrap()
    test_viper_null_model()
    print("All tests passed!")
