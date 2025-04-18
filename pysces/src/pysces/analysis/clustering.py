"""
Clustering functionality for protein activity profiles.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Dict, List, Tuple, Union, Optional

def viper_similarity(activity_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate similarity matrix based on VIPER activity scores.
    
    Parameters
    ----------
    activity_matrix : DataFrame
        Protein activity matrix (proteins x cells)
        
    Returns
    -------
    DataFrame with cell-cell similarity
    
    Examples
    --------
    >>> import pysces
    >>> adata = pysces.read_census("homo_sapiens", max_cells=1000)
    >>> adata = pysces.preprocess_data(adata)
    >>> aracne = pysces.ARACNe()
    >>> network = aracne.run(adata)
    >>> regulons = pysces.aracne_to_regulons(network)
    >>> activity = pysces.viper(adata, regulons)
    >>> similarity = pysces.viper_similarity(activity)
    """
    # Normalize activity
    normalized = activity_matrix.T.copy()
    normalized = (normalized - normalized.mean()) / normalized.std()
    
    # Calculate correlation
    similarity = normalized.corr()
    
    return similarity

def cluster_activity(
    adata: ad.AnnData,
    activity_matrix: pd.DataFrame,
    method: str = 'louvain',
    key_added: str = 'activity_clusters',
    resolution: float = 1.0,
    n_clusters: Optional[int] = None,
    random_state: int = 42,
    **kwargs
) -> ad.AnnData:
    """
    Cluster cells based on protein activity profiles.
    
    Parameters
    ----------
    adata : AnnData
        Original data
    activity_matrix : DataFrame
        Protein activity matrix (proteins x cells)
    method : str, default='louvain'
        Clustering method to use. Options:
        - 'louvain': Louvain community detection
        - 'leiden': Leiden community detection
        - 'kmeans': K-means clustering
    key_added : str, default='activity_clusters'
        Key to add to adata.obs for the clustering results
    resolution : float, default=1.0
        Resolution parameter for community detection methods
    n_clusters : int, optional
        Number of clusters for k-means
    random_state : int, default=42
        Random seed for reproducibility
    **kwargs : 
        Additional parameters for the clustering method
        
    Returns
    -------
    Updated AnnData object
    
    Examples
    --------
    >>> import pysces
    >>> adata = pysces.read_census("homo_sapiens", max_cells=1000)
    >>> adata = pysces.preprocess_data(adata)
    >>> aracne = pysces.ARACNe()
    >>> network = aracne.run(adata)
    >>> regulons = pysces.aracne_to_regulons(network)
    >>> activity = pysces.viper(adata, regulons)
    >>> adata = pysces.cluster_activity(adata, activity)
    """
    # Make a copy to avoid modifying the original
    adata = adata.copy()
    
    # Create temporary AnnData with activities
    activity_adata = ad.AnnData(
        X=activity_matrix.T.values,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=activity_matrix.index)
    )
    
    # Compute neighbors
    sc.pp.neighbors(activity_adata, random_state=random_state, **kwargs)
    
    # Apply clustering
    if method == 'louvain':
        sc.tl.louvain(activity_adata, resolution=resolution, key_added=key_added, random_state=random_state)
    elif method == 'leiden':
        sc.tl.leiden(activity_adata, resolution=resolution, key_added=key_added, random_state=random_state)
    elif method == 'kmeans':
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for k-means clustering")
        sc.tl.kmeans(activity_adata, n_clusters=n_clusters, key_added=key_added, random_state=random_state)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    # Transfer clustering to original object
    adata.obs[key_added] = activity_adata.obs[key_added]
    
    # Add UMAP coordinates if not present
    if 'X_umap' not in activity_adata.obsm:
        sc.tl.umap(activity_adata, random_state=random_state)
    
    if 'X_umap' not in adata.obsm:
        adata.obsm['X_umap'] = activity_adata.obsm['X_umap']
    
    # Add activity matrix to AnnData object
    adata.obsm['X_activity'] = activity_matrix.T.values
    
    return adata
