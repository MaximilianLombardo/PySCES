"""Tests for metacell generation."""

import numpy as np
import pandas as pd
import anndata as ad
import pysces


def create_test_adata(n_cells=20, n_genes=15, seed=0):
    np.random.seed(seed)
    counts = np.random.poisson(1.0, (n_cells, n_genes))
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    return ad.AnnData(X=counts, obs=obs, var=var)


def test_generate_metacells_shape():
    adata = create_test_adata()
    mdata = pysces.generate_metacells(adata, k=5, n_pcs=5)
    assert isinstance(mdata, ad.AnnData)
    assert mdata.shape == adata.shape
    assert "metacell_members" in mdata.uns
    assert len(mdata.uns["metacell_members"]) == adata.n_obs


def test_generate_metacells_members():
    adata = create_test_adata(n_cells=10, n_genes=5)
    mdata = pysces.generate_metacells(adata, k=3, n_pcs=3)
    for key, cells in mdata.uns["metacell_members"].items():
        assert len(cells) >= 1
        for c in cells:
            assert c in adata.obs_names
