"""
Validation functions for PySCES.

This module provides functions for validating AnnData objects and their contents
to ensure they are suitable for use with the PySCES pipeline.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse
import logging
from typing import Dict, List, Tuple, Union, Optional, Any

# Set up logging
logger = logging.getLogger(__name__)

def validate_anndata_structure(adata: ad.AnnData) -> Tuple[bool, List[str]]:
    """
    Validate the basic structure of an AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object to validate

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)

    Examples
    --------
    >>> import pysces
    >>> from pysces.utils.validation import validate_anndata_structure
    >>> adata = pysces.read_anndata("path/to/data.h5ad")
    >>> is_valid, issues = validate_anndata_structure(adata)
    >>> if not is_valid:
    >>>     print("Issues found:", issues)
    """
    issues = []

    # Check if adata is an AnnData object
    if not isinstance(adata, ad.AnnData):
        issues.append("Input is not an AnnData object")
        return False, issues

    # Check if adata has observations and variables
    if adata.n_obs == 0:
        issues.append("AnnData object has no observations (cells)")

    if adata.n_vars == 0:
        issues.append("AnnData object has no variables (genes)")

    # Check if expression matrix exists and has the right shape
    if adata.X is None:
        issues.append("AnnData object has no expression matrix (X)")
    elif adata.X.shape != (adata.n_obs, adata.n_vars):
        issues.append(f"Expression matrix shape {adata.X.shape} does not match AnnData dimensions ({adata.n_obs} cells, {adata.n_vars} genes)")

    # Check if obs and var dataframes exist
    if adata.obs is None:
        issues.append("AnnData object has no observation annotations (obs)")

    if adata.var is None:
        issues.append("AnnData object has no variable annotations (var)")

    return len(issues) == 0, issues

def validate_gene_names(adata: ad.AnnData) -> Tuple[bool, List[str]]:
    """
    Validate gene names in an AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object to validate

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)

    Examples
    --------
    >>> import pysces
    >>> from pysces.utils.validation import validate_gene_names
    >>> adata = pysces.read_anndata("path/to/data.h5ad")
    >>> is_valid, issues = validate_gene_names(adata)
    >>> if not is_valid:
    >>>     print("Issues found:", issues)
    """
    issues = []

    # Check if var_names exists
    if adata.var_names is None:
        issues.append("AnnData object has no gene names (var_names)")
        return False, issues

    # Check if var_names is empty
    if len(adata.var_names) == 0:
        issues.append("AnnData object has empty gene names (var_names)")

    # Check if var_names are unique
    if not adata.var_names.is_unique:
        issues.append("Gene names (var_names) are not unique")
        # Count duplicates
        duplicates = adata.var_names[adata.var_names.duplicated()].unique()
        issues.append(f"Found {len(duplicates)} duplicated gene names")

    # Check for missing or empty gene names
    if "" in adata.var_names or None in adata.var_names:
        issues.append("Gene names contain empty strings or None values")

    return len(issues) == 0, issues

def validate_cell_names(adata: ad.AnnData) -> Tuple[bool, List[str]]:
    """
    Validate cell names in an AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object to validate

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)

    Examples
    --------
    >>> import pysces
    >>> from pysces.utils.validation import validate_cell_names
    >>> adata = pysces.read_anndata("path/to/data.h5ad")
    >>> is_valid, issues = validate_cell_names(adata)
    >>> if not is_valid:
    >>>     print("Issues found:", issues)
    """
    issues = []

    # Check if obs_names exists
    if adata.obs_names is None:
        issues.append("AnnData object has no cell names (obs_names)")
        return False, issues

    # Check if obs_names is empty
    if len(adata.obs_names) == 0:
        issues.append("AnnData object has empty cell names (obs_names)")

    # Check if obs_names are unique
    if not adata.obs_names.is_unique:
        issues.append("Cell names (obs_names) are not unique")
        # Count duplicates
        duplicates = adata.obs_names[adata.obs_names.duplicated()].unique()
        issues.append(f"Found {len(duplicates)} duplicated cell names")

    # Check for missing or empty cell names
    if "" in adata.obs_names or None in adata.obs_names:
        issues.append("Cell names contain empty strings or None values")

    return len(issues) == 0, issues

def validate_raw_counts(adata: ad.AnnData, layer: Optional[str] = None) -> Tuple[bool, List[str], float]:
    """
    Check if the data appears to be raw counts.

    Parameters
    ----------
    adata : AnnData
        AnnData object to validate
    layer : str, optional
        Layer to check. If None, checks adata.X

    Returns
    -------
    Tuple[bool, List[str], float]
        (is_raw_counts, list_of_issues, confidence_score)

    Examples
    --------
    >>> import pysces
    >>> from pysces.utils.validation import validate_raw_counts
    >>> adata = pysces.read_anndata("path/to/data.h5ad")
    >>> is_raw, issues, confidence = validate_raw_counts(adata)
    >>> print(f"Raw counts: {is_raw} (confidence: {confidence:.2f})")
    >>> if not is_raw:
    >>>     print("Issues found:", issues)
    """
    issues = []
    confidence_score = 0.0

    # Get the data matrix
    if layer is None:
        X = adata.X
    else:
        if layer not in adata.layers:
            issues.append(f"Layer '{layer}' not found in AnnData object")
            return False, issues, 0.0
        X = adata.layers[layer]

    # Check if X is sparse
    is_sparse = scipy.sparse.issparse(X)

    # Get a sample of the data for checking
    if is_sparse:
        # For sparse matrices, convert a small sample to dense
        if X.shape[0] > 1000:
            sample_idx = np.random.choice(X.shape[0], 1000, replace=False)
            X_sample = X[sample_idx].toarray()
        else:
            X_sample = X.toarray()
    else:
        # For dense matrices, take a sample
        if X.shape[0] > 1000:
            sample_idx = np.random.choice(X.shape[0], 1000, replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X

    # Check 1: Are values integers?
    # For raw counts, values should be integers
    is_integer = np.all(np.equal(np.mod(X_sample, 1), 0))
    if is_integer:
        confidence_score += 0.4
    else:
        issues.append("Values are not integers")

    # Check 2: Are values non-negative?
    # For raw counts, values should be non-negative
    is_non_negative = np.all(X_sample >= 0)
    if is_non_negative:
        confidence_score += 0.3
    else:
        issues.append("Values contain negative numbers")

    # Check 3: Distribution characteristics
    # For raw counts, we expect many zeros and a right-skewed distribution
    zero_fraction = np.sum(X_sample == 0) / X_sample.size
    if zero_fraction > 0.5:
        confidence_score += 0.2

    # Check for right-skewed distribution (mean > median)
    mean_val = np.mean(X_sample)
    median_val = np.median(X_sample)
    if mean_val > median_val:
        confidence_score += 0.1

    # Final determination
    is_raw_counts = confidence_score >= 0.7

    return is_raw_counts, issues, confidence_score

def validate_normalized_data(adata: ad.AnnData, layer: Optional[str] = None) -> Tuple[bool, List[str], float]:
    """
    Check if the data appears to be normalized.

    Parameters
    ----------
    adata : AnnData
        AnnData object to validate
    layer : str, optional
        Layer to check. If None, checks adata.X

    Returns
    -------
    Tuple[bool, List[str], float]
        (is_normalized, list_of_issues, confidence_score)

    Examples
    --------
    >>> import pysces
    >>> from pysces.utils.validation import validate_normalized_data
    >>> adata = pysces.read_anndata("path/to/data.h5ad")
    >>> is_norm, issues, confidence = validate_normalized_data(adata)
    >>> print(f"Normalized: {is_norm} (confidence: {confidence:.2f})")
    >>> if not is_norm:
    >>>     print("Issues found:", issues)
    """
    issues = []
    confidence_score = 0.0

    # Get the data matrix
    if layer is None:
        X = adata.X
    else:
        if layer not in adata.layers:
            issues.append(f"Layer '{layer}' not found in AnnData object")
            return False, issues, 0.0
        X = adata.layers[layer]

    # Check if X is sparse
    is_sparse = scipy.sparse.issparse(X)

    # Get a sample of the data for checking
    if is_sparse:
        # For sparse matrices, convert a small sample to dense
        if X.shape[0] > 1000:
            sample_idx = np.random.choice(X.shape[0], 1000, replace=False)
            X_sample = X[sample_idx].toarray()
        else:
            X_sample = X.toarray()
    else:
        # For dense matrices, take a sample
        if X.shape[0] > 1000:
            sample_idx = np.random.choice(X.shape[0], 1000, replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X

    # Check 1: Are values floating point?
    # For normalized data, values are often floating point
    is_integer = np.all(np.equal(np.mod(X_sample, 1), 0))
    if not is_integer:
        confidence_score += 0.3

    # Check 2: Are values non-negative?
    # For normalized data, values should be non-negative
    is_non_negative = np.all(X_sample >= 0)
    if is_non_negative:
        confidence_score += 0.2
    else:
        issues.append("Values contain negative numbers")

    # Check 3: Distribution characteristics
    # For normalized data, we expect a more uniform distribution
    zero_fraction = np.sum(X_sample == 0) / X_sample.size
    if zero_fraction < 0.5:
        confidence_score += 0.1

    # Check 4: Check for common normalization patterns
    # For CPM normalization, column sums should be close to 1e6
    col_sums = X_sample.sum(axis=0)
    if np.any(np.isclose(col_sums, 1e6, rtol=0.1)):
        confidence_score += 0.2
        logger.info("Data appears to be CPM normalized")
    elif np.any(np.isclose(col_sums, 1e4, rtol=0.1)):
        confidence_score += 0.2
        logger.info("Data appears to be CPT (counts per 10k) normalized")
    elif np.any(np.isclose(col_sums, 1.0, rtol=0.1)):
        confidence_score += 0.2
        logger.info("Data appears to be normalized to sum to 1")

    # Check 5: Check for log-transformation
    # Log-transformed data typically has a more symmetric distribution
    if X_sample.size > 0 and np.max(X_sample) < 30:  # Log values are typically smaller
        # Check for NaN or infinite values
        if not np.any(np.isnan(X_sample)) and not np.any(np.isinf(X_sample)):
            try:
                skewness = np.mean(((X_sample - np.mean(X_sample)) / np.std(X_sample)) ** 3)
                if abs(skewness) < 1.0:  # More symmetric distribution
                    confidence_score += 0.2
                    logger.info("Data appears to be log-transformed")
            except:
                # If there's any error in calculating skewness, just skip this check
                logger.warning("Could not calculate skewness for log-transformation check")

    # Final determination
    is_normalized = confidence_score >= 0.5

    return is_normalized, issues, confidence_score

def validate_sparse_matrix(adata: ad.AnnData, layer: Optional[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate sparse matrix format in an AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object to validate
    layer : str, optional
        Layer to check. If None, checks adata.X

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)

    Examples
    --------
    >>> import pysces
    >>> from pysces.utils.validation import validate_sparse_matrix
    >>> adata = pysces.read_anndata("path/to/data.h5ad")
    >>> is_valid, issues = validate_sparse_matrix(adata)
    >>> if not is_valid:
    >>>     print("Issues found:", issues)
    """
    issues = []

    # Get the data matrix
    if layer is None:
        X = adata.X
    else:
        if layer not in adata.layers:
            issues.append(f"Layer '{layer}' not found in AnnData object")
            return False, issues
        X = adata.layers[layer]

    # Check if X is sparse
    is_sparse = scipy.sparse.issparse(X)

    if not is_sparse:
        # Not a sparse matrix, so no issues to report
        return True, []

    # Check sparse matrix format
    if isinstance(X, scipy.sparse.csr_matrix):
        # CSR format is efficient for row operations
        pass
    elif isinstance(X, scipy.sparse.csc_matrix):
        # CSC format is efficient for column operations
        pass
    elif isinstance(X, scipy.sparse.coo_matrix):
        # COO format is less efficient for many operations
        issues.append("Sparse matrix is in COO format, which may be less efficient for some operations")
    else:
        # Other formats may have specific issues
        issues.append(f"Sparse matrix is in {type(X).__name__} format, which may not be optimal")

    # Check sparsity level
    nnz = X.nnz
    total_elements = X.shape[0] * X.shape[1]
    if total_elements > 0:  # Avoid division by zero
        sparsity = 1.0 - (nnz / total_elements)

        if sparsity < 0.5:
            issues.append(f"Matrix is only {sparsity:.2%} sparse, which may not benefit from sparse storage")
    else:
        issues.append("Matrix has zero elements")

    return len(issues) == 0, issues

def recommend_preprocessing(adata: ad.AnnData) -> List[str]:
    """
    Recommend preprocessing steps based on data characteristics.

    Parameters
    ----------
    adata : AnnData
        AnnData object to analyze

    Returns
    -------
    List[str]
        List of preprocessing recommendations

    Examples
    --------
    >>> import pysces
    >>> from pysces.utils.validation import recommend_preprocessing
    >>> adata = pysces.read_anndata("path/to/data.h5ad")
    >>> recommendations = recommend_preprocessing(adata)
    >>> for rec in recommendations:
    >>>     print(rec)
    """
    recommendations = []

    # Check if data is raw counts
    is_raw, _, raw_confidence = validate_raw_counts(adata)

    # Check if data is normalized
    is_normalized, _, norm_confidence = validate_normalized_data(adata)

    # Make recommendations based on data type
    if is_raw and raw_confidence > 0.7:
        recommendations.append("Data appears to be raw counts. Consider normalizing with pysces.normalize_data().")  # This is the key recommendation for raw counts

        # Check for low-quality cells
        if 'n_genes' not in adata.obs.columns:
            recommendations.append("Calculate quality metrics with scanpy.pp.calculate_qc_metrics().")
        else:
            # Check for cells with few genes
            low_gene_cells = np.sum(adata.obs['n_genes'] < 200)
            if low_gene_cells > 0:
                recommendations.append(f"Found {low_gene_cells} cells with fewer than 200 genes. Consider filtering with pysces.filter_cells().")

        # Check for mitochondrial genes
        if 'mt' not in adata.var.columns:
            recommendations.append("Annotate mitochondrial genes with adata.var['mt'] = adata.var_names.str.startswith('MT-').")
        elif 'pct_counts_mt' in adata.obs.columns:
            high_mt_cells = np.sum(adata.obs['pct_counts_mt'] > 20)
            if high_mt_cells > 0:
                recommendations.append(f"Found {high_mt_cells} cells with >20% mitochondrial reads. Consider filtering with pysces.filter_cells(max_pct_mito=20).")

    elif is_normalized and norm_confidence > 0.7:
        recommendations.append("Data appears to be normalized.")

        # Check if data is log-transformed
        try:
            if adata.X.size > 0 and np.max(adata.X) < 30:
                recommendations.append("Data may be log-transformed.")
            else:
                recommendations.append("Consider log-transforming the data with scanpy.pp.log1p().")
        except:
            # If there's any error in checking max value, just skip this check
            recommendations.append("Consider log-transforming the data with scanpy.pp.log1p().")

    else:
        recommendations.append("Data type is unclear. Consider running pysces.preprocess_data() for standard preprocessing.")

    # Check for sparse matrix
    _, sparse_issues = validate_sparse_matrix(adata)
    if sparse_issues:
        for issue in sparse_issues:
            recommendations.append(f"Sparse matrix issue: {issue}")

    # Check for ARACNe-specific preprocessing
    recommendations.append("For ARACNe, consider rank-transforming the data with pysces.rank_transform().")

    return recommendations
