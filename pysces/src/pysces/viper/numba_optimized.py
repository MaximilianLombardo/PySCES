"""
Numba-optimized implementations of VIPER algorithm functions.
"""

import numpy as np
import pandas as pd
import scipy.stats
import scipy.sparse
import logging
from typing import Dict, List, Tuple, Union, Optional, Set, Iterable, Any
from anndata import AnnData
from .regulons import Regulon, GeneSet

try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    import warnings
    warnings.warn(
        "Numba is not installed. Using slower Python implementation. "
        "To use the faster Numba implementation, install Numba with: pip install numba"
    )
    # Define dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator

    def prange(*args, **kwargs):
        return range(*args, **kwargs)

# Configure logging
logger = logging.getLogger(__name__)

@jit(nopython=True)
def _rank_transform_numba(data: np.ndarray) -> np.ndarray:
    """
    Rank transform a 1D array using Numba.

    Parameters
    ----------
    data : np.ndarray
        1D array to rank transform

    Returns
    -------
    np.ndarray
        Rank transformed array
    """
    n = len(data)
    ranks = np.zeros(n, dtype=np.float64)

    # Create sorted indices
    sorted_indices = np.argsort(data)

    # Assign ranks
    for i in range(n):
        ranks[sorted_indices[i]] = i + 1

    # Scale ranks to [0, 1]
    ranks = ranks / (n + 1)

    return ranks

@jit(nopython=True)
def _ppf_approx_numba(p: np.ndarray) -> np.ndarray:
    """
    Approximate normal PPF (percent point function) using Numba.
    This is a fast approximation of scipy.stats.norm.ppf.

    Parameters
    ----------
    p : np.ndarray
        Probabilities

    Returns
    -------
    np.ndarray
        Approximate z-scores
    """
    # Ensure p is in (0, 1)
    p = np.clip(p, 0.0001, 0.9999)

    # Constants for the approximation
    a = 2.30753
    b = 0.27061
    c = 0.99229
    d = 0.04481

    # Calculate the approximation
    t = np.sqrt(-2.0 * np.log(p))
    z = -(t - (c + d * t) / (1.0 + b * t)) / a

    # For p > 0.5, use symmetry
    mask = p > 0.5
    z[mask] = -z[mask]
    p_adj = 1.0 - p[mask]
    t = np.sqrt(-2.0 * np.log(p_adj))
    z[mask] = (t - (c + d * t) / (1.0 + b * t)) / a

    return z

@jit(nopython=True, parallel=True)
def calculate_signature_numba(expr: np.ndarray, method: str = 'rank') -> np.ndarray:
    """
    Calculate gene expression signatures using Numba.

    Parameters
    ----------
    expr : np.ndarray
        Expression matrix (genes x cells)
    method : str, default='rank'
        Method for calculating signatures. Options:
        - 'rank': Rank transform the data
        - 'scale': Center and scale the data
        - 'mad': Median absolute deviation normalization

    Returns
    -------
    np.ndarray
        Gene expression signatures (genes x cells)
    """
    n_genes, n_cells = expr.shape
    signatures = np.zeros((n_genes, n_cells), dtype=np.float64)

    if method == 'rank':
        # Rank transform the data
        for i in prange(n_cells):
            ranks = _rank_transform_numba(expr[:, i])
            signatures[:, i] = _ppf_approx_numba(ranks)

    elif method == 'scale':
        # Center and scale the data
        for i in prange(n_genes):
            mean = np.mean(expr[i, :])
            std = np.std(expr[i, :])
            if std > 0:
                signatures[i, :] = (expr[i, :] - mean) / std

    elif method == 'mad':
        # Median absolute deviation normalization
        for i in prange(n_genes):
            median = np.median(expr[i, :])
            mad = np.median(np.abs(expr[i, :] - median)) * 1.4826  # Factor for normal distribution
            if mad > 0:
                signatures[i, :] = (expr[i, :] - median) / mad

    return signatures

@jit(nopython=True)
def calculate_enrichment_score_numba(
    signature: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    method: str = 'gsea',
    abs_score: bool = False
) -> float:
    """
    Calculate enrichment score for a gene set in a signature using Numba.

    Parameters
    ----------
    signature : np.ndarray
        Gene expression signature (genes,)
    indices : np.ndarray
        Indices of genes in the gene set
    weights : np.ndarray
        Weights of genes in the gene set
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
    if len(indices) == 0:
        return 0.0

    # Get signature values for genes in the gene set
    values = signature[indices]

    # Use absolute values if requested
    if abs_score:
        values = np.abs(values)

    # Calculate enrichment score based on method
    if method == 'gsea':
        # Sort genes by signature value
        sorted_indices = np.argsort(values)[::-1]
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]

        # Calculate running sum
        weight_sum = np.sum(sorted_weights)
        if weight_sum == 0:
            return 0.0

        running_sum = np.cumsum(sorted_weights) / weight_sum

        # Calculate null distribution (uniform)
        null_dist = np.arange(1, len(running_sum) + 1) / len(running_sum)

        # Find maximum deviation from null distribution
        deviations = running_sum - null_dist
        max_pos = np.max(deviations)
        max_neg = np.min(deviations)

        # Return enrichment score with sign
        if max_pos > -max_neg:
            return max_pos
        else:
            return max_neg

    elif method == 'mean':
        # Weighted mean
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            return 0.0
        return np.sum(values * weights) / weight_sum

    elif method == 'sum':
        # Weighted sum
        return np.sum(values * weights)

    elif method == 'median':
        # Median (weights not used)
        return np.median(values)

    # Default to mean if method not recognized
    return np.mean(values)

def calculate_nes_matrix_numba(
    signatures: np.ndarray,
    regulon_indices: list,
    regulon_weights: list,
    method: str = 'gsea',
    abs_score: bool = False,
    normalize: bool = True
) -> np.ndarray:
    """
    Calculate Normalized Enrichment Score (NES) matrix for regulons using Numba.

    Parameters
    ----------
    signatures : np.ndarray
        Gene expression signatures (genes x cells)
    regulon_indices : list
        List of arrays containing indices of genes in each regulon
    regulon_weights : list
        List of arrays containing weights of genes in each regulon
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
    n_regulons = len(regulon_indices)
    n_cells = signatures.shape[1]

    # Initialize NES matrix
    nes_matrix = np.zeros((n_regulons, n_cells), dtype=np.float64)

    # Calculate enrichment for each regulon and cell
    for i in range(n_regulons):
        indices = regulon_indices[i]
        weights = regulon_weights[i]

        if len(indices) == 0:
            continue

        # Use Numba-accelerated function for each cell
        for j in range(n_cells):
            nes_matrix[i, j] = calculate_enrichment_score_numba(
                signatures[:, j],
                indices,
                weights,
                method=method,
                abs_score=abs_score
            )

    # Normalize if requested
    if normalize:
        # Normalize by standard deviation across cells
        for i in range(n_regulons):
            std = np.std(nes_matrix[i, :])
            if std > 0:
                nes_matrix[i, :] = nes_matrix[i, :] / std

    return nes_matrix

def prepare_regulon_data(regulons: List[Regulon], gene_indices: Dict[str, int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Prepare regulon data for Numba-optimized functions.

    Parameters
    ----------
    regulons : list of Regulon
        List of regulons to prepare
    gene_indices : dict
        Mapping from gene names to indices

    Returns
    -------
    tuple
        Lists of arrays containing indices and weights for each regulon
    """
    regulon_indices = []
    regulon_weights = []

    for regulon in regulons:
        # Get genes in the regulon that are also in the data
        common_genes = [gene for gene in regulon.genes if gene in gene_indices]

        if not common_genes:
            # Empty regulon
            regulon_indices.append(np.array([], dtype=np.int64))
            regulon_weights.append(np.array([], dtype=np.float64))
            continue

        # Get indices and weights of common genes
        indices = np.array([gene_indices[gene] for gene in common_genes], dtype=np.int64)
        weights = np.array([regulon.weights.get(gene, 1.0) for gene in common_genes], dtype=np.float64)

        regulon_indices.append(indices)
        regulon_weights.append(weights)

    return regulon_indices, regulon_weights

def viper_scores_numba(
    adata: AnnData,
    regulons: List[Regulon],
    layer: Optional[str] = None,
    signature_method: str = 'rank',
    enrichment_method: str = 'gsea',
    abs_score: bool = False,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Calculate VIPER scores for regulons using Numba acceleration.

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
    if not HAS_NUMBA:
        logger.warning("Numba is not installed. Using slower Python implementation.")
        from .core import viper_scores
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
    elif expr.shape[0] == adata.n_vars and expr.shape[1] == adata.n_obs:
        # Data is already in genes x cells format
        pass
    else:
        raise ValueError(f"Expression matrix shape {expr.shape} does not match AnnData dimensions ({adata.n_vars} genes, {adata.n_obs} cells)")

    # Calculate signatures
    logger.debug("Calculating signatures with Numba")
    signatures = calculate_signature_numba(
        expr,
        method=signature_method
    )

    # Create gene name to index mapping
    gene_names = adata.var_names.tolist()
    gene_indices = {gene: i for i, gene in enumerate(gene_names)}

    # Prepare regulon data for Numba
    logger.debug("Preparing regulon data for Numba")
    regulon_indices, regulon_weights = prepare_regulon_data(regulons, gene_indices)

    # Calculate NES matrix
    logger.debug("Calculating NES matrix with Numba")
    nes_matrix = calculate_nes_matrix_numba(
        signatures,
        regulon_indices,
        regulon_weights,
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

def viper_bootstrap_numba(
    adata: AnnData,
    regulons: List[Regulon],
    n_bootstraps: int = 100,
    sample_fraction: float = 0.8,
    layer: Optional[str] = None,
    signature_method: str = 'rank',
    enrichment_method: str = 'gsea',
    abs_score: bool = False,
    normalize: bool = True,
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate VIPER scores with bootstrapping using Numba acceleration.

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

    Returns
    -------
    tuple of pd.DataFrame
        Mean VIPER scores and standard deviations (regulons x cells)
    """
    if not HAS_NUMBA:
        logger.warning("Numba is not installed. Using slower Python implementation.")
        from .core import viper_bootstrap
        return viper_bootstrap(
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
    elif expr.shape[0] == adata.n_vars and expr.shape[1] == adata.n_obs:
        # Data is already in genes x cells format
        pass
    else:
        raise ValueError(f"Expression matrix shape {expr.shape} does not match AnnData dimensions ({adata.n_vars} genes, {adata.n_obs} cells)")

    # Create gene name to index mapping
    gene_names = adata.var_names.tolist()
    gene_indices = {gene: i for i, gene in enumerate(gene_names)}

    # Prepare regulon data for Numba
    regulon_indices, regulon_weights = prepare_regulon_data(regulons, gene_indices)

    # Initialize arrays for bootstrap results
    n_regulons = len(regulons)
    n_cells = adata.shape[0]
    bootstrap_results = np.zeros((n_bootstraps, n_regulons, n_cells))

    # Run bootstrap iterations
    for i in range(n_bootstraps):
        logger.debug(f"Bootstrap iteration {i+1}/{n_bootstraps}")

        # Sample cells with replacement
        n_samples = int(n_cells * sample_fraction)
        sample_indices = np.random.choice(n_cells, n_samples, replace=True)

        # Create bootstrap sample
        if expr.shape[1] == n_cells:  # genes x cells
            bootstrap_expr = expr[:, sample_indices]
        else:  # cells x genes
            bootstrap_expr = expr[sample_indices, :].T

        # Calculate signatures
        bootstrap_signatures = calculate_signature_numba(
            bootstrap_expr,
            method=signature_method
        )

        # Calculate NES matrix
        bootstrap_results[i] = calculate_nes_matrix_numba(
            bootstrap_signatures,
            regulon_indices,
            regulon_weights,
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

def viper_null_model_numba(
    adata: AnnData,
    regulons: List[Regulon],
    n_permutations: int = 1000,
    layer: Optional[str] = None,
    signature_method: str = 'rank',
    enrichment_method: str = 'gsea',
    abs_score: bool = False,
    normalize: bool = True,
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate VIPER scores with a null model for statistical significance using Numba acceleration.

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

    Returns
    -------
    tuple of pd.DataFrame
        VIPER scores and p-values (regulons x cells)
    """
    if not HAS_NUMBA:
        logger.warning("Numba is not installed. Using slower Python implementation.")
        from .core import viper_null_model
        return viper_null_model(
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

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Calculate actual VIPER scores
    viper_df = viper_scores_numba(
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

    # Calculate signatures
    signatures = calculate_signature_numba(
        expr,
        method=signature_method
    )

    # Create gene name to index mapping
    gene_names = adata.var_names.tolist()
    gene_indices = {gene: i for i, gene in enumerate(gene_names)}

    # Initialize arrays for null model results
    n_regulons = len(regulons)
    n_cells = adata.shape[0]
    null_results = np.zeros((n_permutations, n_regulons, n_cells))

    # Run permutation iterations
    for i in range(n_permutations):
        logger.debug(f"Permutation {i+1}/{n_permutations}")

        # Create permuted regulons
        permuted_regulon_indices = []
        permuted_regulon_weights = []

        for regulon in regulons:
            # Get number of targets
            n_targets = len(regulon.targets)

            if n_targets == 0:
                permuted_regulon_indices.append(np.array([], dtype=np.int64))
                permuted_regulon_weights.append(np.array([], dtype=np.float64))
                continue

            # Randomly sample genes
            random_indices = np.random.choice(len(gene_names), n_targets, replace=False)

            # Get weights from original regulon
            target_list = list(regulon.targets.keys())
            weights = np.array([regulon.likelihood.get(target, 1.0) for target in target_list], dtype=np.float64)

            # Reuse weights (with wrapping if needed)
            if len(weights) < n_targets:
                weights = np.tile(weights, (n_targets // len(weights) + 1))[:n_targets]

            permuted_regulon_indices.append(random_indices)
            permuted_regulon_weights.append(weights[:n_targets])

        # Calculate NES matrix for permuted regulons
        null_results[i] = calculate_nes_matrix_numba(
            signatures,
            permuted_regulon_indices,
            permuted_regulon_weights,
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
