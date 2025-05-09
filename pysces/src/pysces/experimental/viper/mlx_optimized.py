"""
MLX-optimized implementations of VIPER algorithm functions.

This module provides MLX-accelerated implementations of the core VIPER functions
for use on Apple Silicon hardware.
"""

import numpy as np
import pandas as pd
import scipy.stats
import scipy.sparse
import logging
from typing import Dict, List, Tuple, Union, Optional, Set, Iterable, Any
from anndata import AnnData
from .regulons import Regulon, GeneSet

# Check if MLX is available
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    import warnings
    warnings.warn(
        "MLX is not installed. Using slower implementation. "
        "To use the faster MLX implementation on Apple Silicon, install MLX with: pip install mlx"
    )
    # Define dummy mx for type hints
    class DummyMX:
        class array:
            pass
    mx = DummyMX()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_signature_mlx(expr_mx: mx.array, method: str = 'rank') -> mx.array:
    """
    Calculate gene expression signatures using MLX.

    Parameters
    ----------
    expr_mx : mx.array
        Expression matrix (genes x cells)
    method : str, default='rank'
        Method for calculating signatures

    Returns
    -------
    mx.array
        Gene expression signatures (genes x cells)
    """
    if method == 'scale':
        # Center and scale the data
        mean = mx.mean(expr_mx, axis=1, keepdims=True)
        std = mx.std(expr_mx, axis=1, keepdims=True)
        signatures = (expr_mx - mean) / (std + 1e-10)
        return signatures

    elif method == 'rank':
        # For rank method, we need to convert to numpy, rank, and convert back
        # This is because MLX doesn't have a rank function yet
        expr_np = expr_mx.tolist()
        expr_np = np.array(expr_np)  # Convert list to numpy array
        n_genes, n_cells = expr_np.shape

        signatures_np = np.zeros_like(expr_np)
        for i in range(n_cells):
            # Rank transform the column
            ranks = np.argsort(np.argsort(expr_np[:, i])) + 1
            # Scale to [0, 1]
            ranks = ranks / (n_genes + 1)
            # Convert to z-scores
            signatures_np[:, i] = scipy.stats.norm.ppf(ranks)

        # Convert back to MLX
        signatures = mx.array(signatures_np)
        return signatures

    elif method == 'mad':
        # Median absolute deviation normalization
        # Convert to numpy, calculate MAD, and convert back
        expr_np = expr_mx.tolist()
        n_genes, n_cells = expr_np.shape

        signatures_np = np.zeros_like(expr_np)
        for i in range(n_genes):
            median = np.median(expr_np[i, :])
            mad = np.median(np.abs(np.array(expr_np[i, :]) - median)) * 1.4826  # Factor for normal distribution
            if mad > 0:
                signatures_np[i, :] = (np.array(expr_np[i, :]) - median) / mad

        # Convert back to MLX
        signatures = mx.array(signatures_np)
        return signatures

    else:
        raise ValueError(f"Unsupported method: {method}")

def calculate_enrichment_score_mlx(
    signature_mx: mx.array,
    indices: mx.array,
    weights: mx.array,
    method: str = 'gsea',
    abs_score: bool = False
) -> float:
    """
    Calculate enrichment score for a gene set in a signature using MLX.

    Parameters
    ----------
    signature_mx : mx.array
        Gene expression signature (genes,)
    indices : mx.array
        Indices of genes in the gene set
    weights : mx.array
        Weights of genes in the gene set
    method : str, default='gsea'
        Method for calculating enrichment
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
    values = mx.take(signature_mx, indices)

    # Use absolute values if requested
    if abs_score:
        values = mx.abs(values)

    # Calculate enrichment score based on method
    if method == 'mean':
        # Weighted mean
        weight_sum = mx.sum(weights)
        if weight_sum == 0:
            return 0.0
        return mx.sum(values * weights) / weight_sum

    elif method == 'sum':
        # Weighted sum
        return mx.sum(values * weights)

    elif method == 'gsea':
        # For GSEA, we need to convert to numpy for now
        # This is because MLX doesn't have all the operations we need
        values_np = values.tolist()
        weights_np = weights.tolist()

        # Sort genes by signature value
        sorted_indices = np.argsort(values_np)[::-1]
        sorted_values = np.array(values_np)[sorted_indices]
        sorted_weights = np.array(weights_np)[sorted_indices]

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
            return float(max_pos)
        else:
            return float(max_neg)

    else:
        raise ValueError(f"Unsupported method: {method}")

def calculate_nes_matrix_mlx(
    signatures: mx.array,
    regulon_indices: List[mx.array],
    regulon_weights: List[mx.array],
    method: str = 'gsea',
    abs_score: bool = False,
    normalize: bool = True
) -> mx.array:
    """
    Calculate Normalized Enrichment Score (NES) matrix for regulons using MLX.

    Parameters
    ----------
    signatures : mx.array
        Gene expression signatures (genes x cells)
    regulon_indices : list of mx.array
        List of arrays containing indices of genes in each regulon
    regulon_weights : list of mx.array
        List of arrays containing weights of genes in each regulon
    method : str, default='gsea'
        Method for calculating enrichment
    abs_score : bool, default=False
        Whether to use absolute values of the signatures
    normalize : bool, default=True
        Whether to normalize enrichment scores

    Returns
    -------
    mx.array
        NES matrix (regulons x cells)
    """
    n_regulons = len(regulon_indices)
    n_cells = signatures.shape[1]

    # Initialize NES matrix
    nes_matrix = mx.zeros((n_regulons, n_cells))

    # Calculate enrichment for each regulon and cell
    for i in range(n_regulons):
        indices = regulon_indices[i]
        weights = regulon_weights[i]

        if len(indices) == 0:
            continue

        # Calculate enrichment for each cell
        for j in range(n_cells):
            score = calculate_enrichment_score_mlx(
                signatures[:, j],
                indices,
                weights,
                method=method,
                abs_score=abs_score
            )
            # Create a new array with the updated value
            # MLX's update pattern is different from JAX
            temp = nes_matrix.tolist()
            temp[i][j] = score
            nes_matrix = mx.array(temp)

    # Normalize if requested
    if normalize:
        # Normalize by standard deviation across cells
        std = mx.std(nes_matrix, axis=1, keepdims=True)
        # Avoid division by zero
        std = mx.where(std > 0, std, mx.ones_like(std))
        nes_matrix = nes_matrix / std

    return nes_matrix

def prepare_regulon_data_mlx(regulons: List[Regulon], gene_indices: Dict[str, int]) -> Tuple[List[mx.array], List[mx.array]]:
    """
    Prepare regulon data for MLX-optimized functions.

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
            regulon_indices.append(mx.array([], dtype=mx.int32))
            regulon_weights.append(mx.array([], dtype=mx.float32))
            continue

        # Get indices and weights of common genes
        indices = mx.array([gene_indices[gene] for gene in common_genes], dtype=mx.int32)
        weights = mx.array([regulon.weights.get(gene, 1.0) for gene in common_genes], dtype=mx.float32)

        regulon_indices.append(indices)
        regulon_weights.append(weights)

    return regulon_indices, regulon_weights

def viper_scores_mlx(
    adata: AnnData,
    regulons: List[Regulon],
    layer: Optional[str] = None,
    signature_method: str = 'rank',
    enrichment_method: str = 'gsea',
    abs_score: bool = False,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Calculate VIPER scores for regulons using MLX acceleration.

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
    if not HAS_MLX:
        logger.warning("MLX is not installed. Using slower implementation.")
        from .numba_optimized import viper_scores_numba
        return viper_scores_numba(
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

    # Convert to MLX array
    expr_mx = mx.array(expr)

    # Calculate signatures
    logger.debug("Calculating signatures with MLX")
    signatures = calculate_signature_mlx(
        expr_mx,
        method=signature_method
    )

    # Create gene name to index mapping
    gene_names = adata.var_names.tolist()
    gene_indices = {gene: i for i, gene in enumerate(gene_names)}

    # Prepare regulon data for MLX
    logger.debug("Preparing regulon data for MLX")
    regulon_indices, regulon_weights = prepare_regulon_data_mlx(regulons, gene_indices)

    # Calculate NES matrix
    logger.debug("Calculating NES matrix with MLX")
    nes_matrix = calculate_nes_matrix_mlx(
        signatures,
        regulon_indices,
        regulon_weights,
        method=enrichment_method,
        abs_score=abs_score,
        normalize=normalize
    )

    # Convert back to numpy and create DataFrame
    nes_np = nes_matrix.tolist()
    viper_df = pd.DataFrame(
        nes_np,
        index=[r.tf_name for r in regulons],
        columns=adata.obs_names
    )

    return viper_df

def viper_bootstrap_mlx(
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
    Calculate VIPER scores with bootstrapping using MLX acceleration.

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
    if not HAS_MLX:
        logger.warning("MLX is not installed. Using slower implementation.")
        from .numba_optimized import viper_bootstrap_numba
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

    # Convert to MLX array
    expr_mx = mx.array(expr)

    # Create gene name to index mapping
    gene_names = adata.var_names.tolist()
    gene_indices = {gene: i for i, gene in enumerate(gene_names)}

    # Prepare regulon data for MLX
    regulon_indices, regulon_weights = prepare_regulon_data_mlx(regulons, gene_indices)

    # Initialize arrays for bootstrap results
    n_regulons = len(regulons)
    n_cells = adata.shape[0]
    bootstrap_results = []

    # Run bootstrap iterations
    for i in range(n_bootstraps):
        logger.debug(f"Bootstrap iteration {i+1}/{n_bootstraps}")

        # Sample cells with replacement
        n_samples = int(n_cells * sample_fraction)
        sample_indices = np.random.choice(n_cells, n_samples, replace=True)

        # Create bootstrap sample
        if expr_mx.shape[1] == n_cells:  # genes x cells
            bootstrap_expr = expr_mx[:, sample_indices]
        else:  # cells x genes
            bootstrap_expr = expr_mx[sample_indices, :].T

        # Calculate signatures
        bootstrap_signatures = calculate_signature_mlx(
            bootstrap_expr,
            method=signature_method
        )

        # Calculate NES matrix
        bootstrap_nes = calculate_nes_matrix_mlx(
            bootstrap_signatures,
            regulon_indices,
            regulon_weights,
            method=enrichment_method,
            abs_score=abs_score,
            normalize=normalize
        )

        # Add to results
        bootstrap_results.append(bootstrap_nes)

    # Stack results
    bootstrap_stack = mx.stack(bootstrap_results)

    # Calculate mean and standard deviation
    mean_scores = mx.mean(bootstrap_stack, axis=0)
    std_scores = mx.std(bootstrap_stack, axis=0)

    # Convert back to numpy and create DataFrames
    mean_np = mean_scores.tolist()
    std_np = std_scores.tolist()

    mean_df = pd.DataFrame(
        mean_np,
        index=[r.tf_name for r in regulons],
        columns=adata.obs_names
    )

    std_df = pd.DataFrame(
        std_np,
        index=[r.tf_name for r in regulons],
        columns=adata.obs_names
    )

    return mean_df, std_df

def viper_null_model_mlx(
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
    Calculate VIPER scores with a null model for statistical significance using MLX acceleration.

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
    if not HAS_MLX:
        logger.warning("MLX is not installed. Using slower implementation.")
        from .numba_optimized import viper_null_model_numba
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

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Calculate actual VIPER scores
    viper_df = viper_scores_mlx(
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
    signatures = calculate_signature_mlx(
        expr_mx,
        method=signature_method
    )

    # Create gene name to index mapping
    gene_names = adata.var_names.tolist()
    gene_indices = {gene: i for i, gene in enumerate(gene_names)}

    # Initialize arrays for null model results
    n_regulons = len(regulons)
    n_cells = adata.shape[0]
    null_results = []

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
                permuted_regulon_indices.append(mx.array([], dtype=mx.int32))
                permuted_regulon_weights.append(mx.array([], dtype=mx.float32))
                continue

            # Randomly sample genes
            random_indices = np.random.choice(len(gene_names), n_targets, replace=False)

            # Get weights from original regulon
            target_list = list(regulon.targets.keys())
            weights = np.array([regulon.likelihood.get(target, 1.0) for target in target_list], dtype=np.float32)

            # Reuse weights (with wrapping if needed)
            if len(weights) < n_targets:
                weights = np.tile(weights, (n_targets // len(weights) + 1))[:n_targets]

            permuted_regulon_indices.append(mx.array(random_indices, dtype=mx.int32))
            permuted_regulon_weights.append(mx.array(weights[:n_targets], dtype=mx.float32))

        # Calculate NES matrix for permuted regulons
        null_nes = calculate_nes_matrix_mlx(
            signatures,
            permuted_regulon_indices,
            permuted_regulon_weights,
            method=enrichment_method,
            abs_score=abs_score,
            normalize=normalize
        )

        # Add to results
        null_results.append(null_nes)

    # Stack results
    null_stack = mx.stack(null_results)

    # Calculate p-values
    p_values = mx.zeros((n_regulons, n_cells))

    # Convert actual scores to MLX array
    actual_scores = mx.array(viper_df.values)

    # Calculate p-values
    for i in range(n_regulons):
        for j in range(n_cells):
            actual_score = actual_scores[i, j]
            null_scores = null_stack[:, i, j]

            if actual_score >= 0:
                # One-sided p-value for positive scores
                p_value = mx.mean(null_scores >= actual_score)
            else:
                # One-sided p-value for negative scores
                p_value = mx.mean(null_scores <= actual_score)

            p_values = p_values.at[i, j].set(p_value.item())

    # Convert back to numpy and create DataFrame
    p_values_np = p_values.tolist()
    p_value_df = pd.DataFrame(
        p_values_np,
        index=viper_df.index,
        columns=viper_df.columns
    )

    return viper_df, p_value_df
