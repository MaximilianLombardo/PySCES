"""
Protein activity inference functionality using VIPER.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse
from typing import Dict, List, Tuple, Union, Optional
from .regulons import Regulon

def viper(
    adata: ad.AnnData,
    regulons: List[Regulon],
    layer: Optional[str] = None,
    method: str = 'none'
) -> pd.DataFrame:
    """
    Run VIPER algorithm to infer protein activity.
    
    Parameters
    ----------
    adata : AnnData
        Gene expression data (cells x genes)
    regulons : list of Regulon
        List of regulon objects
    layer : str, optional
        Layer in AnnData to use. If None, uses .X
    method : str, default='none'
        Method for handling pleiotropy. Options:
        - 'none': No pleiotropy correction
        - 'scale': Scale by number of targets
        - 'adaptive': Adaptive pleiotropy correction
        
    Returns
    -------
    DataFrame with protein activity scores (proteins x cells)
    
    Examples
    --------
    >>> import pysces
    >>> adata = pysces.read_census("homo_sapiens", max_cells=1000)
    >>> adata = pysces.preprocess_data(adata)
    >>> aracne = pysces.ARACNe()
    >>> network = aracne.run(adata)
    >>> regulons = pysces.aracne_to_regulons(network)
    >>> activity = pysces.viper(adata, regulons)
    """
    # Get expression data
    if layer is None:
        expr = adata.X
    else:
        expr = adata.layers[layer]
    
    # If sparse, convert to dense
    if scipy.sparse.issparse(expr):
        expr = expr.toarray()
    
    # Transpose if needed (ensure genes x cells)
    if expr.shape[0] == adata.n_obs and expr.shape[1] == adata.n_vars:
        # Data is in cells x genes format, transpose to genes x cells
        expr = expr.T
    elif expr.shape[0] == adata.n_vars and expr.shape[1] == adata.n_obs:
        # Data is already in genes x cells format
        pass
    else:
        raise ValueError(f"Expression matrix shape {expr.shape} does not match AnnData dimensions ({adata.n_vars} genes, {adata.n_obs} cells)")
    
    # Get gene names and create lookup
    gene_names = adata.var_names
    gene_lookup = {gene: i for i, gene in enumerate(gene_names)}
    
    # Initialize activity matrix
    n_tfs = len(regulons)
    n_cells = expr.shape[1]
    activity = np.zeros((n_tfs, n_cells))
    
    # Calculate activity for each TF
    for i, regulon in enumerate(regulons):
        # Get indices of targets present in the data
        target_indices = []
        target_modes = []
        
        for target, mode in regulon.targets.items():
            if target in gene_lookup:
                target_indices.append(gene_lookup[target])
                target_modes.append(mode)
        
        if not target_indices:
            continue  # Skip if no targets found
        
        # Extract target expression
        target_expr = expr[target_indices, :]
        target_modes = np.array(target_modes).reshape(-1, 1)
        
        # Calculate NES (simplified version)
        # For full implementation, need to handle pleiotropy and signature significance
        if method == 'none':
            # Simple weighted average
            activity[i, :] = np.mean(target_expr * target_modes, axis=0)
        elif method == 'scale':
            # Scale by number of targets
            activity[i, :] = np.sum(target_expr * target_modes, axis=0) / np.sqrt(len(target_indices))
        elif method == 'adaptive':
            # Adaptive pleiotropy correction (simplified)
            # For full implementation, need to account for target correlation
            weights = 1.0 / np.sqrt(np.sum(np.abs(target_modes)))
            activity[i, :] = weights * np.sum(target_expr * target_modes, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # Create DataFrame
    activity_df = pd.DataFrame(
        activity,
        index=[r.tf_name for r in regulons],
        columns=adata.obs_names
    )
    
    return activity_df

def metaviper(
    adata: ad.AnnData,
    regulon_sets: Dict[str, List[Regulon]],
    layer: Optional[str] = None,
    method: str = 'none',
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Run metaVIPER with multiple regulon sets.
    
    Parameters
    ----------
    adata : AnnData
        Gene expression data
    regulon_sets : dict
        Dictionary mapping set names to lists of regulons
    layer : str, optional
        Layer in AnnData to use
    method : str, default='none'
        Method for handling pleiotropy in individual VIPER runs
    weights : dict, optional
        Dictionary mapping set names to weights. If None, all sets are weighted equally.
        
    Returns
    -------
    DataFrame with integrated protein activity
    
    Examples
    --------
    >>> import pysces
    >>> adata = pysces.read_census("homo_sapiens", max_cells=1000)
    >>> adata = pysces.preprocess_data(adata)
    >>> # Run with multiple regulon sets
    >>> activity = pysces.metaviper(adata, {'set1': regulons1, 'set2': regulons2})
    """
    # Run VIPER for each regulon set
    viper_results = {}
    for name, regulons in regulon_sets.items():
        viper_results[name] = viper(adata, regulons, layer, method)
    
    # Get weights
    if weights is None:
        weights = {name: 1.0 for name in regulon_sets.keys()}
    else:
        # Ensure all sets have weights
        for name in regulon_sets.keys():
            if name not in weights:
                weights[name] = 1.0
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {name: w / total_weight for name, w in weights.items()}
    
    # Get all TFs
    all_tfs = set()
    for result in viper_results.values():
        all_tfs.update(result.index)
    
    # Initialize integrated result
    integrated = pd.DataFrame(
        index=list(all_tfs),
        columns=adata.obs_names
    )
    
    # Fill with weighted values
    for tf in all_tfs:
        weighted_sum = 0
        total_weight_used = 0
        
        for name, result in viper_results.items():
            if tf in result.index:
                w = weights[name]
                weighted_sum += w * result.loc[tf]
                total_weight_used += w
        
        if total_weight_used > 0:
            # Normalize by weights used
            integrated.loc[tf] = weighted_sum / total_weight_used
    
    return integrated
