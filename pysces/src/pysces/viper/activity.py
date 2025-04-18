"""
Protein activity inference functionality using VIPER.

This module provides high-level functions for running the VIPER (Virtual Inference of
Protein-activity by Enriched Regulon analysis) algorithm on gene expression data.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from .regulons import Regulon, GeneSet
from .core import (
    calculate_signature,
    calculate_enrichment_score,
    calculate_nes_matrix,
    viper_scores as _viper_scores,
    viper_bootstrap as _viper_bootstrap,
    viper_null_model as _viper_null_model,
    viper_similarity as _viper_similarity,
    viper_cluster as _viper_cluster,
    viper_mlx as _viper_mlx
)

# Configure logging
logger = logging.getLogger(__name__)

def viper(
    adata: ad.AnnData,
    regulons: List[Regulon],
    layer: Optional[str] = None,
    method: str = 'gsea',
    signature_method: str = 'rank',
    abs_score: bool = False,
    normalize: bool = True,
    use_gpu: bool = False
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
    method : str, default='gsea'
        Method for calculating enrichment. Options:
        - 'gsea': Gene Set Enrichment Analysis
        - 'mean': Mean of gene expression values
        - 'sum': Sum of gene expression values
        - 'median': Median of gene expression values
    signature_method : str, default='rank'
        Method for calculating signatures. Options:
        - 'rank': Rank transform the data
        - 'scale': Center and scale the data
        - 'mad': Median absolute deviation normalization
        - 'ttest': T-test against all other cells
        - 'diff': Simple difference from reference
    abs_score : bool, default=False
        Whether to use absolute values of the signatures
    normalize : bool, default=True
        Whether to normalize enrichment scores
    use_gpu : bool, default=False
        Whether to use GPU acceleration with MLX (if available)

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
    # Check if regulons have targets
    valid_regulons = [r for r in regulons if len(r.targets) > 0]
    if len(valid_regulons) < len(regulons):
        logger.warning(f"Removed {len(regulons) - len(valid_regulons)} regulons with no targets.")
        regulons = valid_regulons

    if len(regulons) == 0:
        raise ValueError("No valid regulons provided.")

    # Use GPU if requested
    if use_gpu:
        try:
            return _viper_mlx(
                adata,
                regulons,
                layer=layer,
                signature_method=signature_method,
                enrichment_method=method,
                abs_score=abs_score,
                normalize=normalize
            )
        except ImportError:
            logger.warning("MLX not available. Falling back to CPU implementation.")
            use_gpu = False

    # Use CPU implementation
    return _viper_scores(
        adata,
        regulons,
        layer=layer,
        signature_method=signature_method,
        enrichment_method=method,
        abs_score=abs_score,
        normalize=normalize
    )

def metaviper(
    adata: ad.AnnData,
    regulon_sets: Dict[str, List[Regulon]],
    layer: Optional[str] = None,
    method: str = 'gsea',
    signature_method: str = 'rank',
    abs_score: bool = False,
    normalize: bool = True,
    use_gpu: bool = False,
    weights: Optional[Dict[str, float]] = None,
    weight_method: str = 'equal'
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
    method : str, default='gsea'
        Method for calculating enrichment
    signature_method : str, default='rank'
        Method for calculating signatures
    abs_score : bool, default=False
        Whether to use absolute values of the signatures
    normalize : bool, default=True
        Whether to normalize enrichment scores
    use_gpu : bool, default=False
        Whether to use GPU acceleration with MLX (if available)
    weights : dict, optional
        Dictionary mapping set names to weights. If None, weights are determined by weight_method.
    weight_method : str, default='equal'
        Method for determining weights if not provided. Options:
        - 'equal': Equal weights for all regulon sets
        - 'size': Weight by number of regulons in each set
        - 'adaptive': Adaptive weights based on enrichment strength

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
    >>> # Run with custom weights
    >>> activity = pysces.metaviper(adata, {'set1': regulons1, 'set2': regulons2}, weights={'set1': 0.7, 'set2': 0.3})
    """
    # Run VIPER for each regulon set
    viper_results = {}
    for name, regulons in regulon_sets.items():
        viper_results[name] = viper(
            adata,
            regulons,
            layer=layer,
            method=method,
            signature_method=signature_method,
            abs_score=abs_score,
            normalize=normalize,
            use_gpu=use_gpu
        )

    # Determine weights
    if weights is None:
        if weight_method == 'equal':
            # Equal weights for all regulon sets
            weights = {name: 1.0 for name in regulon_sets.keys()}
        elif weight_method == 'size':
            # Weight by number of regulons in each set
            weights = {name: len(regulons) for name, regulons in regulon_sets.items()}
        elif weight_method == 'adaptive':
            # Adaptive weights based on enrichment strength
            weights = {}
            for name, result in viper_results.items():
                # Use mean absolute NES as weight
                weights[name] = np.mean(np.abs(result.values))
        else:
            raise ValueError(f"Unknown weight_method: {weight_method}")
    else:
        # Ensure all sets have weights
        for name in regulon_sets.keys():
            if name not in weights:
                weights[name] = 1.0

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {name: w / total_weight for name, w in weights.items()}
    else:
        # If all weights are zero, use equal weights
        weights = {name: 1.0 / len(regulon_sets) for name in regulon_sets.keys()}

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


def viper_bootstrap_wrapper(
    adata: ad.AnnData,
    regulons: List[Regulon],
    n_bootstraps: int = 100,
    sample_fraction: float = 0.8,
    layer: Optional[str] = None,
    method: str = 'gsea',
    signature_method: str = 'rank',
    abs_score: bool = False,
    normalize: bool = True,
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate VIPER scores with bootstrapping.

    Parameters
    ----------
    adata : AnnData
        Gene expression data (cells x genes)
    regulons : list of Regulon
        List of regulons to calculate enrichment for
    n_bootstraps : int, default=100
        Number of bootstrap iterations
    sample_fraction : float, default=0.8
        Fraction of cells to sample in each bootstrap
    layer : str, optional
        Layer in AnnData to use. If None, uses .X
    method : str, default='gsea'
        Method for calculating enrichment
    signature_method : str, default='rank'
        Method for calculating signatures
    abs_score : bool, default=False
        Whether to use absolute values of the signatures
    normalize : bool, default=True
        Whether to normalize enrichment scores
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple of pd.DataFrame
        Mean VIPER scores and standard deviations (regulons x cells)

    Examples
    --------
    >>> import pysces
    >>> adata = pysces.read_census("homo_sapiens", max_cells=1000)
    >>> adata = pysces.preprocess_data(adata)
    >>> aracne = pysces.ARACNe()
    >>> network = aracne.run(adata)
    >>> regulons = pysces.aracne_to_regulons(network)
    >>> mean_activity, std_activity = pysces.viper_bootstrap_wrapper(adata, regulons, n_bootstraps=50)
    """
    return _viper_bootstrap(
        adata,
        regulons,
        n_bootstraps=n_bootstraps,
        sample_fraction=sample_fraction,
        layer=layer,
        signature_method=signature_method,
        enrichment_method=method,
        abs_score=abs_score,
        normalize=normalize,
        seed=seed
    )


def viper_null_model_wrapper(
    adata: ad.AnnData,
    regulons: List[Regulon],
    n_permutations: int = 1000,
    layer: Optional[str] = None,
    method: str = 'gsea',
    signature_method: str = 'rank',
    abs_score: bool = False,
    normalize: bool = True,
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate VIPER scores with a null model for statistical significance.

    Parameters
    ----------
    adata : AnnData
        Gene expression data (cells x genes)
    regulons : list of Regulon
        List of regulons to calculate enrichment for
    n_permutations : int, default=1000
        Number of permutations for the null model
    layer : str, optional
        Layer in AnnData to use. If None, uses .X
    method : str, default='gsea'
        Method for calculating enrichment
    signature_method : str, default='rank'
        Method for calculating signatures
    abs_score : bool, default=False
        Whether to use absolute values of the signatures
    normalize : bool, default=True
        Whether to normalize enrichment scores
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple of pd.DataFrame
        VIPER scores and p-values (regulons x cells)

    Examples
    --------
    >>> import pysces
    >>> adata = pysces.read_census("homo_sapiens", max_cells=1000)
    >>> adata = pysces.preprocess_data(adata)
    >>> aracne = pysces.ARACNe()
    >>> network = aracne.run(adata)
    >>> regulons = pysces.aracne_to_regulons(network)
    >>> activity, p_values = pysces.viper_null_model_wrapper(adata, regulons, n_permutations=100)
    """
    return _viper_null_model(
        adata,
        regulons,
        n_permutations=n_permutations,
        layer=layer,
        signature_method=signature_method,
        enrichment_method=method,
        abs_score=abs_score,
        normalize=normalize,
        seed=seed
    )


def viper_similarity_wrapper(activity_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate similarity between cells based on VIPER activity profiles.

    Parameters
    ----------
    activity_matrix : pd.DataFrame
        VIPER activity matrix (regulons x cells)

    Returns
    -------
    pd.DataFrame
        Similarity matrix (cells x cells)

    Examples
    --------
    >>> import pysces
    >>> adata = pysces.read_census("homo_sapiens", max_cells=1000)
    >>> adata = pysces.preprocess_data(adata)
    >>> aracne = pysces.ARACNe()
    >>> network = aracne.run(adata)
    >>> regulons = pysces.aracne_to_regulons(network)
    >>> activity = pysces.viper(adata, regulons)
    >>> similarity = pysces.viper_similarity_wrapper(activity)
    """
    return _viper_similarity(activity_matrix)


def viper_cluster_wrapper(
    activity_matrix: pd.DataFrame,
    n_clusters: int = 10,
    method: str = 'kmeans',
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Cluster cells based on VIPER activity profiles.

    Parameters
    ----------
    activity_matrix : pd.DataFrame
        VIPER activity matrix (regulons x cells)
    n_clusters : int, default=10
        Number of clusters
    method : str, default='kmeans'
        Clustering method. Options:
        - 'kmeans': K-means clustering
        - 'hierarchical': Hierarchical clustering
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Cluster labels for each cell

    Examples
    --------
    >>> import pysces
    >>> adata = pysces.read_census("homo_sapiens", max_cells=1000)
    >>> adata = pysces.preprocess_data(adata)
    >>> aracne = pysces.ARACNe()
    >>> network = aracne.run(adata)
    >>> regulons = pysces.aracne_to_regulons(network)
    >>> activity = pysces.viper(adata, regulons)
    >>> clusters = pysces.viper_cluster_wrapper(activity, n_clusters=5)
    """
    return _viper_cluster(
        activity_matrix,
        n_clusters=n_clusters,
        method=method,
        random_state=random_state
    )
