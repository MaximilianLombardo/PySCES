"""
Core VIPER algorithm implementation.

This module provides the core functions for the VIPER (Virtual Inference of Protein-activity
by Enriched Regulon analysis) algorithm, including signature calculation, enrichment analysis,
and statistical significance assessment.
"""

import numpy as np
import pandas as pd
import scipy.stats
import scipy.sparse
import logging
from typing import Dict, List, Tuple, Union, Optional, Set, Iterable, Any
from anndata import AnnData
from .regulons import Regulon, GeneSet

# Check if Numba is available
try:
    from .numba_optimized import (
        viper_scores_numba,
        viper_bootstrap_numba,
        viper_null_model_numba,
        HAS_NUMBA
    )
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False

# Configure logging
logger = logging.getLogger(__name__)

def calculate_signature(
    adata: AnnData,
    layer: Optional[str] = None,
    method: str = 'rank',
    reference: Optional[np.ndarray] = None,
    reference_groups: Optional[np.ndarray] = None,
    target_groups: Optional[np.ndarray] = None,
    scale: bool = True
) -> np.ndarray:
    """
    Calculate gene expression signatures from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Gene expression data (cells x genes)
    layer : str, optional
        Layer in AnnData to use. If None, uses .X
    method : str, default='rank'
        Method for calculating signatures. Options:
        - 'rank': Rank transform the data
        - 'scale': Center and scale the data
        - 'mad': Median absolute deviation normalization
        - 'ttest': T-test against reference or all other cells
        - 'diff': Simple difference from reference
    reference : np.ndarray, optional
        Reference expression data for comparison methods
    reference_groups : np.ndarray, optional
        Group labels for reference samples
    target_groups : np.ndarray, optional
        Group labels for target samples
    scale : bool, default=True
        Whether to scale the signatures to have mean 0 and std 1

    Returns
    -------
    np.ndarray
        Gene expression signatures (genes x cells)
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

    # Calculate signatures based on method
    if method == 'rank':
        # Rank transform the data
        signatures = np.zeros_like(expr)
        for i in range(expr.shape[1]):
            signatures[:, i] = scipy.stats.rankdata(expr[:, i])

        # Scale ranks to [0, 1]
        signatures = signatures / (expr.shape[0] + 1)

        # Convert to z-scores
        signatures = scipy.stats.norm.ppf(signatures)

    elif method == 'scale':
        # Center and scale the data
        signatures = (expr - np.mean(expr, axis=1, keepdims=True)) / np.std(expr, axis=1, keepdims=True)

    elif method == 'mad':
        # Median absolute deviation normalization
        median = np.median(expr, axis=1, keepdims=True)
        mad = np.median(np.abs(expr - median), axis=1, keepdims=True) * 1.4826  # Factor for normal distribution
        signatures = (expr - median) / mad

    elif method == 'ttest':
        # T-test against reference or all other cells
        signatures = np.zeros_like(expr)

        if reference is not None:
            # T-test against reference
            for i in range(expr.shape[1]):
                t_stat, _ = scipy.stats.ttest_ind(expr[:, i], reference, equal_var=False)
                signatures[:, i] = t_stat
        else:
            # T-test against all other cells
            for i in range(expr.shape[1]):
                other_cells = np.delete(expr, i, axis=1)
                t_stat, _ = scipy.stats.ttest_ind(
                    expr[:, i].reshape(-1, 1),
                    other_cells,
                    axis=1,
                    equal_var=False
                )
                signatures[:, i] = t_stat

    elif method == 'diff':
        # Simple difference from reference
        if reference is None:
            reference = np.mean(expr, axis=1, keepdims=True)
        signatures = expr - reference.reshape(-1, 1)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Scale if requested
    if scale:
        signatures = (signatures - np.mean(signatures, axis=1, keepdims=True)) / np.std(signatures, axis=1, keepdims=True)

    return signatures

def calculate_enrichment_score(
    signature: np.ndarray,
    gene_set: Union[GeneSet, Regulon],
    gene_names: List[str],
    method: str = 'gsea',
    abs_score: bool = False
) -> float:
    """
    Calculate enrichment score for a gene set in a signature.

    Parameters
    ----------
    signature : np.ndarray
        Gene expression signature (genes,)
    gene_set : GeneSet or Regulon
        Gene set to calculate enrichment for
    gene_names : list of str
        Names of genes in the signature
    method : str, default='gsea'
        Method for calculating enrichment. Options:
        - 'gsea': Gene Set Enrichment Analysis
        - 'mean': Mean of gene expression values
        - 'sum': Sum of gene expression values
        - 'median': Median of gene expression values
    abs_score : bool, default=False
        Whether to use absolute values of the signature

    Returns
    -------
    float
        Enrichment score
    """
    # Create gene name to index mapping
    gene_indices = {gene: i for i, gene in enumerate(gene_names)}

    # Get genes in the gene set that are also in the signature
    common_genes = [gene for gene in gene_set.genes if gene in gene_indices]

    if not common_genes:
        return 0.0

    # Get indices and weights of common genes
    indices = [gene_indices[gene] for gene in common_genes]
    weights = [gene_set.weights.get(gene, 1.0) for gene in common_genes]

    # Get signature values for common genes
    if abs_score:
        values = np.abs(signature[indices])
    else:
        values = signature[indices]

    # Calculate enrichment score based on method
    if method == 'gsea':
        # Sort genes by signature value
        sorted_indices = np.argsort(values)[::-1]
        sorted_values = values[sorted_indices]
        sorted_weights = np.array([weights[i] for i in sorted_indices])

        # Calculate running sum
        running_sum = np.cumsum(sorted_weights) / np.sum(sorted_weights)

        # Find maximum deviation from zero
        max_deviation = np.max(np.abs(running_sum - np.arange(len(running_sum)) / len(running_sum)))

        # Return enrichment score with sign from correlation
        return max_deviation * np.sign(np.corrcoef(values, weights)[0, 1])

    elif method == 'mean':
        # Weighted mean
        return np.average(values, weights=weights)

    elif method == 'sum':
        # Weighted sum
        return np.sum(values * weights)

    elif method == 'median':
        # Median (weights not used)
        return np.median(values)

    else:
        raise ValueError(f"Unknown method: {method}")

def calculate_nes_matrix(
    signatures: np.ndarray,
    regulons: List[Regulon],
    gene_names: List[str],
    method: str = 'gsea',
    abs_score: bool = False,
    normalize: bool = True
) -> np.ndarray:
    """
    Calculate Normalized Enrichment Score (NES) matrix for regulons.

    Parameters
    ----------
    signatures : np.ndarray
        Gene expression signatures (genes x cells)
    regulons : list of Regulon
        List of regulons to calculate enrichment for
    gene_names : list of str
        Names of genes in the signatures
    method : str, default='gsea'
        Method for calculating enrichment
    abs_score : bool, default=False
        Whether to use absolute values of the signatures
    normalize : bool, default=True
        Whether to normalize enrichment scores

    Returns
    -------
    np.ndarray
        NES matrix (regulons x cells)
    """
    n_regulons = len(regulons)
    n_cells = signatures.shape[1]

    # Initialize NES matrix
    nes_matrix = np.zeros((n_regulons, n_cells))

    # Calculate enrichment for each regulon and cell
    for i, regulon in enumerate(regulons):
        for j in range(n_cells):
            nes_matrix[i, j] = calculate_enrichment_score(
                signatures[:, j],
                regulon,
                gene_names,
                method=method,
                abs_score=abs_score
            )

    # Normalize if requested
    if normalize:
        # Normalize by standard deviation across cells
        nes_matrix = nes_matrix / np.std(nes_matrix, axis=1, keepdims=True)

    return nes_matrix

def viper_scores(
    adata: AnnData,
    regulons: List[Regulon],
    layer: Optional[str] = None,
    signature_method: str = 'rank',
    enrichment_method: str = 'gsea',
    abs_score: bool = False,
    normalize: bool = True,
    use_numba: bool = True
) -> pd.DataFrame:
    """
    Calculate VIPER scores for regulons.

    Parameters
    ----------
    adata : AnnData
        Gene expression data (cells x genes)
    regulons : list of Regulon
        List of regulons to calculate enrichment for
    layer : str, optional
        Layer in AnnData to use. If None, uses .X
    signature_method : str, default='rank'
        Method for calculating signatures
    enrichment_method : str, default='gsea'
        Method for calculating enrichment
    abs_score : bool, default=False
        Whether to use absolute values of the signatures
    normalize : bool, default=True
        Whether to normalize enrichment scores
    use_numba : bool, default=True
        Whether to use Numba acceleration if available

    Returns
    -------
    pd.DataFrame
        VIPER scores (regulons x cells)
    """
    # Use Numba if available and requested
    if USE_NUMBA and use_numba:
        logger.info("Using Numba acceleration for VIPER")
        return viper_scores_numba(
            adata,
            regulons,
            layer=layer,
            signature_method=signature_method,
            enrichment_method=enrichment_method,
            abs_score=abs_score,
            normalize=normalize
        )

    # Fall back to Python implementation
    if use_numba:
        logger.warning("Numba not available. Using Python implementation.")

    # Calculate signatures
    signatures = calculate_signature(
        adata,
        layer=layer,
        method=signature_method
    )

    # Calculate NES matrix
    nes_matrix = calculate_nes_matrix(
        signatures,
        regulons,
        adata.var_names.tolist(),
        method=enrichment_method,
        abs_score=abs_score,
        normalize=normalize
    )

    # Create DataFrame
    viper_df = pd.DataFrame(
        nes_matrix,
        index=[r.tf_name for r in regulons],
        columns=adata.obs_names
    )

    return viper_df

def viper_bootstrap(
    adata: AnnData,
    regulons: List[Regulon],
    n_bootstraps: int = 100,
    sample_fraction: float = 0.8,
    layer: Optional[str] = None,
    signature_method: str = 'rank',
    enrichment_method: str = 'gsea',
    abs_score: bool = False,
    normalize: bool = True,
    seed: Optional[int] = None,
    use_numba: bool = True
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
    signature_method : str, default='rank'
        Method for calculating signatures
    enrichment_method : str, default='gsea'
        Method for calculating enrichment
    abs_score : bool, default=False
        Whether to use absolute values of the signatures
    normalize : bool, default=True
        Whether to normalize enrichment scores
    seed : int, optional
        Random seed for reproducibility
    use_numba : bool, default=True
        Whether to use Numba acceleration if available

    Returns
    -------
    tuple of pd.DataFrame
        Mean VIPER scores and standard deviations (regulons x cells)
    """
    # Use Numba if available and requested
    if USE_NUMBA and use_numba:
        logger.info("Using Numba acceleration for VIPER bootstrapping")
        return viper_bootstrap_numba(
            adata,
            regulons,
            n_bootstraps=n_bootstraps,
            sample_fraction=sample_fraction,
            layer=layer,
            signature_method=signature_method,
            enrichment_method=enrichment_method,
            abs_score=abs_score,
            normalize=normalize,
            seed=seed
        )

    # Fall back to Python implementation
    if use_numba:
        logger.warning("Numba not available. Using Python implementation.")

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

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

    # Initialize arrays for bootstrap results
    n_regulons = len(regulons)
    n_cells = expr.shape[1]
    bootstrap_results = np.zeros((n_bootstraps, n_regulons, n_cells))

    # Run bootstrap iterations
    for i in range(n_bootstraps):
        # Sample cells with replacement
        n_samples = int(n_cells * sample_fraction)
        sample_indices = np.random.choice(n_cells, n_samples, replace=True)

        # Create bootstrap sample
        bootstrap_expr = expr[:, sample_indices]

        # Calculate signatures
        bootstrap_signatures = calculate_signature(
            AnnData(X=bootstrap_expr.T, var=adata.var),
            method=signature_method
        )

        # Calculate NES matrix
        bootstrap_results[i] = calculate_nes_matrix(
            bootstrap_signatures,
            regulons,
            adata.var_names.tolist(),
            method=enrichment_method,
            abs_score=abs_score,
            normalize=normalize
        )

    # Calculate mean and standard deviation
    mean_scores = np.mean(bootstrap_results, axis=0)
    std_scores = np.std(bootstrap_results, axis=0)

    # Create DataFrames
    mean_df = pd.DataFrame(
        mean_scores,
        index=[r.tf_name for r in regulons],
        columns=adata.obs_names
    )

    std_df = pd.DataFrame(
        std_scores,
        index=[r.tf_name for r in regulons],
        columns=adata.obs_names
    )

    return mean_df, std_df

def viper_null_model(
    adata: AnnData,
    regulons: List[Regulon],
    n_permutations: int = 1000,
    layer: Optional[str] = None,
    signature_method: str = 'rank',
    enrichment_method: str = 'gsea',
    abs_score: bool = False,
    normalize: bool = True,
    seed: Optional[int] = None,
    use_numba: bool = True
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
    signature_method : str, default='rank'
        Method for calculating signatures
    enrichment_method : str, default='gsea'
        Method for calculating enrichment
    abs_score : bool, default=False
        Whether to use absolute values of the signatures
    normalize : bool, default=True
        Whether to normalize enrichment scores
    seed : int, optional
        Random seed for reproducibility
    use_numba : bool, default=True
        Whether to use Numba acceleration if available

    Returns
    -------
    tuple of pd.DataFrame
        VIPER scores and p-values (regulons x cells)
    """
    # Use Numba if available and requested
    if USE_NUMBA and use_numba:
        logger.info("Using Numba acceleration for VIPER null model")
        return viper_null_model_numba(
            adata,
            regulons,
            n_permutations=n_permutations,
            layer=layer,
            signature_method=signature_method,
            enrichment_method=enrichment_method,
            abs_score=abs_score,
            normalize=normalize,
            seed=seed
        )

    # Fall back to Python implementation
    if use_numba:
        logger.warning("Numba not available. Using Python implementation.")

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Calculate actual VIPER scores
    viper_df = viper_scores(
        adata,
        regulons,
        layer=layer,
        signature_method=signature_method,
        enrichment_method=enrichment_method,
        abs_score=abs_score,
        normalize=normalize,
        use_numba=use_numba
    )

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

    # Calculate signatures
    signatures = calculate_signature(
        adata,
        layer=layer,
        method=signature_method
    )

    # Initialize arrays for null model results
    n_regulons = len(regulons)
    n_cells = expr.shape[1]
    null_results = np.zeros((n_permutations, n_regulons, n_cells))

    # Run permutation iterations
    for i in range(n_permutations):
        # Create permuted regulons
        permuted_regulons = []
        for regulon in regulons:
            # Randomly sample genes to create a permuted regulon
            n_targets = len(regulon.targets)
            random_genes = np.random.choice(adata.var_names, n_targets, replace=False)

            # Create new regulon with random genes
            permuted_regulon = Regulon(regulon.tf_name)
            for j, gene in enumerate(random_genes):
                # Use original modes and likelihoods
                target = list(regulon.targets.keys())[j % len(regulon.targets)]
                mode = regulon.targets[target]
                likelihood = regulon.likelihood.get(target, 1.0)
                permuted_regulon.add_target(gene, mode, likelihood)

            permuted_regulons.append(permuted_regulon)

        # Calculate NES matrix for permuted regulons
        null_results[i] = calculate_nes_matrix(
            signatures,
            permuted_regulons,
            adata.var_names.tolist(),
            method=enrichment_method,
            abs_score=abs_score,
            normalize=normalize
        )

    # Calculate p-values
    p_values = np.zeros((n_regulons, n_cells))
    for i in range(n_regulons):
        for j in range(n_cells):
            actual_score = viper_df.iloc[i, j]
            null_scores = null_results[:, i, j]

            if actual_score >= 0:
                # One-sided p-value for positive scores
                p_values[i, j] = np.mean(null_scores >= actual_score)
            else:
                # One-sided p-value for negative scores
                p_values[i, j] = np.mean(null_scores <= actual_score)

    # Create p-value DataFrame
    p_value_df = pd.DataFrame(
        p_values,
        index=viper_df.index,
        columns=viper_df.columns
    )

    return viper_df, p_value_df

def viper_similarity(activity_matrix: pd.DataFrame) -> pd.DataFrame:
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
    """
    # Normalize activity
    normalized = activity_matrix.T.copy()
    normalized = (normalized - normalized.mean()) / normalized.std()

    # Calculate correlation
    similarity = normalized.corr()

    return similarity

def viper_cluster(
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
    """
    # Normalize activity
    normalized = activity_matrix.T.copy()
    normalized = (normalized - normalized.mean()) / normalized.std()

    # Perform clustering
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(normalized)

    elif method == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hierarchical.fit_predict(normalized)

    else:
        raise ValueError(f"Unknown method: {method}")

    return labels

def viper_mlx(
    adata: AnnData,
    regulons: List[Regulon],
    layer: Optional[str] = None,
    signature_method: str = 'rank',
    enrichment_method: str = 'gsea',
    abs_score: bool = False,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Calculate VIPER scores using MLX for GPU acceleration.

    Parameters
    ----------
    adata : AnnData
        Gene expression data (cells x genes)
    regulons : list of Regulon
        List of regulons to calculate enrichment for
    layer : str, optional
        Layer in AnnData to use. If None, uses .X
    signature_method : str, default='rank'
        Method for calculating signatures
    enrichment_method : str, default='gsea'
        Method for calculating enrichment
    abs_score : bool, default=False
        Whether to use absolute values of the signatures
    normalize : bool, default=True
        Whether to normalize enrichment scores

    Returns
    -------
    pd.DataFrame
        VIPER scores (regulons x cells)
    """
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError:
        logger.warning("MLX not available. Falling back to CPU implementation.")
        return viper_scores(
            adata,
            regulons,
            layer=layer,
            signature_method=signature_method,
            enrichment_method=enrichment_method,
            abs_score=abs_score,
            normalize=normalize
        )

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

    # Convert to MLX array
    expr_mx = mx.array(expr)

    # Calculate signatures
    if signature_method == 'rank':
        # Rank transform the data
        signatures = mx.zeros_like(expr_mx)
        for i in range(expr_mx.shape[1]):
            # Convert to numpy for ranking, then back to MLX
            col_np = expr_mx[:, i].tolist()
            ranks_np = scipy.stats.rankdata(col_np)
            signatures = signatures.at[:, i].set(mx.array(ranks_np))

        # Scale ranks to [0, 1]
        signatures = signatures / (expr_mx.shape[0] + 1)

        # Convert to z-scores
        # MLX doesn't have ppf, so convert to numpy, apply ppf, then back to MLX
        signatures_np = signatures.tolist()
        signatures_z = scipy.stats.norm.ppf(signatures_np)
        signatures = mx.array(signatures_z)

    elif signature_method == 'scale':
        # Center and scale the data
        mean = mx.mean(expr_mx, axis=1, keepdims=True)
        std = mx.std(expr_mx, axis=1, keepdims=True)
        signatures = (expr_mx - mean) / std

    else:
        # For other methods, fall back to CPU implementation
        signatures_np = calculate_signature(
            adata,
            layer=layer,
            method=signature_method
        )
        signatures = mx.array(signatures_np)

    # Create gene name to index mapping
    gene_names = adata.var_names.tolist()
    gene_indices = {gene: i for i, gene in enumerate(gene_names)}

    # Initialize NES matrix
    n_regulons = len(regulons)
    n_cells = expr_mx.shape[1]
    nes_matrix = mx.zeros((n_regulons, n_cells))

    # Calculate enrichment for each regulon
    for i, regulon in enumerate(regulons):
        # Get genes in the regulon that are also in the data
        common_genes = [gene for gene in regulon.genes if gene in gene_indices]

        if not common_genes:
            continue

        # Get indices and weights of common genes
        indices = mx.array([gene_indices[gene] for gene in common_genes])
        weights = mx.array([regulon.weights.get(gene, 1.0) for gene in common_genes])

        # Get signature values for common genes
        regulon_signatures = mx.take(signatures, indices, axis=0)

        if abs_score:
            regulon_signatures = mx.abs(regulon_signatures)

        # Calculate enrichment based on method
        if enrichment_method == 'mean':
            # Weighted mean
            weighted_sum = mx.sum(regulon_signatures * weights.reshape(-1, 1), axis=0)
            total_weight = mx.sum(weights)
            nes_matrix = nes_matrix.at[i].set(weighted_sum / total_weight)

        elif enrichment_method == 'sum':
            # Weighted sum
            weighted_sum = mx.sum(regulon_signatures * weights.reshape(-1, 1), axis=0)
            nes_matrix = nes_matrix.at[i].set(weighted_sum)

        else:
            # For GSEA and other methods, fall back to CPU implementation
            for j in range(n_cells):
                sig_np = regulon_signatures[:, j].tolist()
                score = calculate_enrichment_score(
                    np.array(sig_np),
                    regulon,
                    [gene_names[idx] for idx in indices.tolist()],
                    method=enrichment_method,
                    abs_score=False  # Already applied above if needed
                )
                nes_matrix = nes_matrix.at[i, j].set(score)

    # Normalize if requested
    if normalize:
        # Normalize by standard deviation across cells
        std = mx.std(nes_matrix, axis=1, keepdims=True)
        nes_matrix = nes_matrix / std

    # Convert back to numpy and create DataFrame
    nes_np = nes_matrix.tolist()
    viper_df = pd.DataFrame(
        nes_np,
        index=[r.tf_name for r in regulons],
        columns=adata.obs_names
    )

    return viper_df
