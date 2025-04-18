"""
Data preprocessing functionality for single-cell expression data.
"""

import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
from typing import Optional, Union, Dict, List, Tuple

def preprocess_data(
    adata: ad.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    max_counts: Optional[int] = None,
    min_counts: Optional[int] = None,
    max_pct_mito: Optional[float] = None,
    normalize: bool = True,
    log_transform: bool = False
) -> ad.AnnData:
    """
    Run standard preprocessing pipeline on AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to preprocess
    min_genes : int, default=200
        Minimum number of genes expressed for a cell to pass filtering
    min_cells : int, default=3
        Minimum number of cells expressing a gene for it to pass filtering
    max_counts : int, optional
        Maximum number of counts for a cell to pass filtering
    min_counts : int, optional
        Minimum number of counts for a cell to pass filtering
    max_pct_mito : float, optional
        Maximum percentage of mitochondrial genes for a cell to pass filtering
    normalize : bool, default=True
        Whether to normalize the data to CPM (counts per million)
    log_transform : bool, default=False
        Whether to log-transform the data after normalization
        
    Returns
    -------
    AnnData object with preprocessed data
    
    Examples
    --------
    >>> import pysces
    >>> adata = pysces.read_census("homo_sapiens", max_cells=1000)
    >>> adata = pysces.preprocess_data(adata, max_pct_mito=10)
    """
    # Make a copy to avoid modifying the original
    adata = adata.copy()
    
    # Filter cells
    if min_genes:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    
    # Filter genes
    if min_cells:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith(('MT-', 'mt-'))
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True
    )
    
    # Filter by counts
    if max_counts:
        adata = adata[adata.obs.n_counts < max_counts, :]
    if min_counts:
        adata = adata[adata.obs.n_counts > min_counts, :]
    
    # Filter by mitochondrial percentage
    if max_pct_mito:
        adata = adata[adata.obs.pct_counts_mt < max_pct_mito, :]
    
    # Normalize
    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e6)  # CPM normalization
        # Store raw counts
        adata.layers['counts'] = adata.X.copy()
    
    # Log transform
    if log_transform:
        sc.pp.log1p(adata)
        adata.layers['log1p'] = adata.X.copy()
    
    return adata

def filter_cells(
    adata: ad.AnnData,
    min_genes: int = 200,
    max_genes: Optional[int] = None,
    min_counts: Optional[int] = None,
    max_counts: Optional[int] = None,
    max_pct_mito: Optional[float] = None
) -> ad.AnnData:
    """
    Filter cells based on quality control metrics.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to filter
    min_genes : int, default=200
        Minimum number of genes expressed for a cell to pass filtering
    max_genes : int, optional
        Maximum number of genes expressed for a cell to pass filtering
    min_counts : int, optional
        Minimum number of counts for a cell to pass filtering
    max_counts : int, optional
        Maximum number of counts for a cell to pass filtering
    max_pct_mito : float, optional
        Maximum percentage of mitochondrial genes for a cell to pass filtering
        
    Returns
    -------
    AnnData object with filtered cells
    """
    # Make a copy to avoid modifying the original
    adata = adata.copy()
    
    # Calculate QC metrics if not already present
    if 'n_genes' not in adata.obs:
        adata.var['mt'] = adata.var_names.str.startswith(('MT-', 'mt-'))
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True
        )
    
    # Apply filters
    if min_genes:
        adata = adata[adata.obs.n_genes > min_genes, :]
    if max_genes:
        adata = adata[adata.obs.n_genes < max_genes, :]
    if min_counts:
        adata = adata[adata.obs.n_counts > min_counts, :]
    if max_counts:
        adata = adata[adata.obs.n_counts < max_counts, :]
    if max_pct_mito:
        adata = adata[adata.obs.pct_counts_mt < max_pct_mito, :]
    
    return adata

def filter_genes(
    adata: ad.AnnData,
    min_cells: int = 3,
    min_counts: Optional[int] = None
) -> ad.AnnData:
    """
    Filter genes based on expression across cells.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to filter
    min_cells : int, default=3
        Minimum number of cells expressing a gene for it to pass filtering
    min_counts : int, optional
        Minimum number of counts across all cells for a gene to pass filtering
        
    Returns
    -------
    AnnData object with filtered genes
    """
    # Make a copy to avoid modifying the original
    adata = adata.copy()
    
    # Apply filters
    if min_cells:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    
    if min_counts:
        gene_counts = np.array(adata.X.sum(axis=0)).flatten()
        adata = adata[:, gene_counts >= min_counts]
    
    return adata

def normalize_data(
    adata: ad.AnnData,
    target_sum: float = 1e6,
    layer: Optional[str] = None,
    target_layer: Optional[str] = None
) -> ad.AnnData:
    """
    Normalize expression data to counts per million (CPM).
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to normalize
    target_sum : float, default=1e6
        Target sum for normalization (1e6 = CPM)
    layer : str, optional
        Layer to normalize. If None, uses .X
    target_layer : str, optional
        Layer to store normalized data. If None, overwrites the input layer or .X
        
    Returns
    -------
    AnnData object with normalized data
    """
    # Make a copy to avoid modifying the original
    adata = adata.copy()
    
    # Normalize
    if layer is None and target_layer is None:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    else:
        # Get data to normalize
        if layer is None:
            data = adata.X
        else:
            data = adata.layers[layer]
        
        # Normalize
        from scipy import sparse
        if sparse.issparse(data):
            data = data.copy()
            sc.pp._normalize_data(data, target_sum)
        else:
            data = sc.pp._normalize_data_array(data, target_sum)
        
        # Store normalized data
        if target_layer is None:
            if layer is None:
                adata.X = data
            else:
                adata.layers[layer] = data
        else:
            adata.layers[target_layer] = data
    
    return adata

def rank_transform(
    adata: ad.AnnData,
    layer: Optional[str] = None,
    target_layer: str = 'rank'
) -> ad.AnnData:
    """
    Apply rank transformation to expression data.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object to transform
    layer : str, optional
        Layer to transform. If None, uses .X
    target_layer : str, default='rank'
        Layer to store transformed data
        
    Returns
    -------
    AnnData object with rank-transformed data
    """
    # Make a copy to avoid modifying the original
    adata = adata.copy()
    
    # Get data matrix to transform
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]
    
    # Convert to dense if sparse
    from scipy import sparse
    if sparse.issparse(X):
        X = X.toarray()
    
    # Apply ranking (gene-wise)
    ranked_X = np.zeros_like(X, dtype=np.float64)  # Change to float64 to preserve decimals
    for i in range(X.shape[1]):
        gene_expr = X[:, i]
        # Use pandas rank with method='average' to handle ties correctly
        ranked_X[:, i] = pd.Series(gene_expr).rank(method='average').values
    
    # Store in target layer
    adata.layers[target_layer] = ranked_X
    
    return adata
