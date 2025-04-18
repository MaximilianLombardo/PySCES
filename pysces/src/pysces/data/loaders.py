"""
Basic data loading functionality for various file formats.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from typing import Optional, Union, Dict, List

def read_anndata(path_or_adata: Union[str, ad.AnnData]) -> ad.AnnData:
    """
    Read data from AnnData object or file.
    
    Parameters
    ----------
    path_or_adata : str or AnnData
        Path to h5ad file or AnnData object
        
    Returns
    -------
    AnnData object
    
    Examples
    --------
    >>> import pysces
    >>> # Load from file
    >>> adata = pysces.read_anndata("path/to/data.h5ad")
    >>> # Or pass existing AnnData object
    >>> adata = pysces.read_anndata(existing_adata)
    """
    if isinstance(path_or_adata, str):
        return ad.read_h5ad(path_or_adata)
    elif isinstance(path_or_adata, ad.AnnData):
        return path_or_adata
    else:
        raise TypeError("Expected AnnData object or path to h5ad file")

def read_csv(
    path: str,
    gene_column: int = 0,
    transpose: bool = False,
    **kwargs
) -> ad.AnnData:
    """
    Read expression data from CSV file.
    
    Parameters
    ----------
    path : str
        Path to CSV file
    gene_column : int, default=0
        Column index to use as gene names
    transpose : bool, default=False
        Whether to transpose the data (if True, rows are cells and columns are genes)
    **kwargs
        Additional arguments to pass to pandas.read_csv
        
    Returns
    -------
    AnnData object
    
    Examples
    --------
    >>> import pysces
    >>> # Load from CSV where rows are genes and columns are cells
    >>> adata = pysces.read_csv("path/to/data.csv")
    >>> # Load from CSV where rows are cells and columns are genes
    >>> adata = pysces.read_csv("path/to/data.csv", gene_column=None, transpose=True)
    """
    df = pd.read_csv(path, index_col=gene_column, **kwargs)
    
    if transpose:
        # Rows are cells, columns are genes
        return ad.AnnData(X=df.values, obs=pd.DataFrame(index=df.index), var=pd.DataFrame(index=df.columns))
    else:
        # Rows are genes, columns are cells
        return ad.AnnData(X=df.values.T, var=pd.DataFrame(index=df.index), obs=pd.DataFrame(index=df.columns))

def read_10x(path: str) -> ad.AnnData:
    """
    Read data from 10X directory.
    
    Parameters
    ----------
    path : str
        Path to 10X directory containing matrix.mtx, genes.tsv, and barcodes.tsv
        
    Returns
    -------
    AnnData object
    
    Examples
    --------
    >>> import pysces
    >>> adata = pysces.read_10x("path/to/10x_data/")
    """
    return sc.read_10x_mtx(path)
