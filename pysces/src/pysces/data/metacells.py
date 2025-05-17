"""Utilities for generating metacells by pooling k nearest neighbours."""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy import sparse


def generate_metacells(adata: ad.AnnData, k: int = 50, n_pcs: int = 30) -> ad.AnnData:
    """Generate metacells by averaging each cell with its k nearest neighbours.

    Parameters
    ----------
    adata : AnnData
        Input data containing raw counts in ``adata.X``.
    k : int, default=50
        Number of neighbours to pool for each metacell.
    n_pcs : int, default=30
        Number of principal components to use when building the kNN graph.

    Returns
    -------
    AnnData
        New :class:`~anndata.AnnData` object containing pooled counts. A mapping
        of metacell identifiers to the contributing cell names is stored in
        ``uns['metacell_members']``.
    """
    # Work on a copy for PCA computation
    temp = adata.copy()
    sc.pp.normalize_total(temp, target_sum=1e6)
    sc.pp.log1p(temp)
    sc.pp.pca(temp, n_comps=n_pcs)

    # Build kNN graph in PCA space
    pcs = temp.obsm["X_pca"]
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(pcs)
    neighbours = nn.kneighbors(return_distance=False)

    # Use original counts for pooling
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    pooled_counts = np.zeros_like(X, dtype=float)
    members = {}
    for i in range(adata.n_obs):
        group = np.append(neighbours[i], i)
        pooled_counts[i] = X[group].mean(axis=0)
        members[f"metacell_{i}"] = adata.obs_names[group].tolist()

    pooled = ad.AnnData(
        X=pooled_counts,
        obs=pd.DataFrame(index=[f"metacell_{i}" for i in range(adata.n_obs)]),
        var=adata.var.copy(),
    )
    pooled.uns["metacell_members"] = members
    pooled.obs["n_members"] = [len(members[f"metacell_{i}"]) for i in range(adata.n_obs)]

    return pooled
