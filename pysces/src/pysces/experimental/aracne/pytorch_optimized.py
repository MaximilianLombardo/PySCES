"""
PyTorch-accelerated implementation of the ARACNe algorithm.

This module provides GPU-accelerated implementations of the ARACNe algorithm
using PyTorch. It is designed to work efficiently with large-scale single-cell
datasets and supports both batch processing and stratified analysis.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Set up logging
logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch
    HAS_PYTORCH = True
    # Check if CUDA is available
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        logger.info("CUDA is available. Using GPU for acceleration.")
    else:
        logger.info("CUDA is not available. Using CPU for PyTorch operations.")
except ImportError:
    HAS_PYTORCH = False
    HAS_CUDA = False
    logger.warning("PyTorch not available. Install PyTorch with: pip install torch")


class MutualInformationEstimator(nn.Module):
    """
    PyTorch implementation of Mutual Information estimation using kernel density estimation.

    This implementation is based on the approach described in:
    https://github.com/MaximilianLombardo/pytorch-mutual-information

    It has been adapted and optimized for gene expression data and ARACNe requirements.
    """

    def __init__(self, num_bins: int = 10, sigma: float = 0.1, normalize: bool = True, device: Optional[str] = None):
        """
        Initialize the Mutual Information estimator.

        Parameters
        ----------
        num_bins : int, default=10
            Number of bins for discretization
        sigma : float, default=0.1
            Bandwidth for the Gaussian kernel
        normalize : bool, default=True
            Whether to normalize the mutual information
        device : str, optional
            Device to use for computation ('cuda', 'cpu', or None for auto-detection)
        """
        super(MutualInformationEstimator, self).__init__()

        self.num_bins = num_bins
        self.sigma = sigma
        self.normalize = normalize
        self.epsilon = 1e-10

        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Create bins based on the range of gene expression data
        # For gene expression data, the range is typically 0 to max expression value
        # We'll set this dynamically when processing data
        self.bins = None

    def _set_bins(self, data_min: float, data_max: float):
        """
        Set the bins based on the data range.

        Parameters
        ----------
        data_min : float
            Minimum value in the data
        data_max : float
            Maximum value in the data
        """
        # Add a small buffer to the range to avoid edge effects
        buffer = (data_max - data_min) * 0.01
        min_val = max(0, data_min - buffer)  # Ensure non-negative for gene expression
        max_val = data_max + buffer

        # Create bins and register as a parameter (non-trainable)
        bins = torch.linspace(min_val, max_val, self.num_bins, device=self.device)
        self.bins = nn.Parameter(bins, requires_grad=False)

    def marginal_pdf(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the marginal probability density function.

        Parameters
        ----------
        values : torch.Tensor
            Input values of shape (batch_size, num_samples)

        Returns
        -------
        tuple
            (pdf, kernel_values) - Marginal PDF and kernel values
        """
        # Ensure bins are set
        if self.bins is None:
            data_min = values.min().item()
            data_max = values.max().item()
            self._set_bins(data_min, data_max)

        # Reshape for broadcasting
        # values: (batch_size, num_samples) -> (batch_size, num_samples, 1)
        # bins: (num_bins) -> (1, 1, num_bins)
        values_expanded = values.unsqueeze(2)
        bins_expanded = self.bins.unsqueeze(0).unsqueeze(0)

        # Calculate residuals and kernel values
        residuals = values_expanded - bins_expanded
        kernel_values = torch.exp(-0.5 * (residuals / self.sigma).pow(2))

        # Calculate PDF
        pdf = torch.mean(kernel_values, dim=1)  # Average over samples
        normalization = torch.sum(pdf, dim=1, keepdim=True) + self.epsilon
        pdf = pdf / normalization

        return pdf, kernel_values

    def joint_pdf(self, kernel_values1: torch.Tensor, kernel_values2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the joint probability density function.

        Parameters
        ----------
        kernel_values1 : torch.Tensor
            Kernel values for first variable
        kernel_values2 : torch.Tensor
            Kernel values for second variable

        Returns
        -------
        torch.Tensor
            Joint PDF
        """
        # Calculate joint kernel values
        # kernel_values: (batch_size, num_samples, num_bins)
        # joint_kernel_values: (batch_size, num_bins, num_bins)
        joint_kernel_values = torch.bmm(kernel_values1.transpose(1, 2), kernel_values2)

        # Normalize
        normalization = torch.sum(joint_kernel_values, dim=(1, 2), keepdim=True) + self.epsilon
        pdf = joint_kernel_values / normalization

        return pdf

    def calculate_mi(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate mutual information between two variables.

        Parameters
        ----------
        x : torch.Tensor
            First variable of shape (batch_size, num_samples)
        y : torch.Tensor
            Second variable of shape (batch_size, num_samples)

        Returns
        -------
        torch.Tensor
            Mutual information for each batch element
        """
        # Calculate marginal PDFs
        pdf_x, kernel_values1 = self.marginal_pdf(x)
        pdf_y, kernel_values2 = self.marginal_pdf(y)

        # Calculate joint PDF
        pdf_xy = self.joint_pdf(kernel_values1, kernel_values2)

        # Calculate entropies
        H_x = -torch.sum(pdf_x * torch.log2(pdf_x + self.epsilon), dim=1)
        H_y = -torch.sum(pdf_y * torch.log2(pdf_y + self.epsilon), dim=1)
        H_xy = -torch.sum(pdf_xy * torch.log2(pdf_xy + self.epsilon), dim=(1, 2))

        # Calculate mutual information
        mutual_information = H_x + H_y - H_xy

        # Normalize if requested
        if self.normalize:
            mutual_information = 2 * mutual_information / (H_x + H_y + self.epsilon)

        return mutual_information

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to calculate mutual information.

        Parameters
        ----------
        x : torch.Tensor
            First variable of shape (batch_size, num_samples)
        y : torch.Tensor
            Second variable of shape (batch_size, num_samples)

        Returns
        -------
        torch.Tensor
            Mutual information for each batch element
        """
        return self.calculate_mi(x, y)


def calculate_mi_matrix_pytorch(
    expr_matrix: np.ndarray,
    gene_indices1: List[int],
    gene_indices2: List[int],
    batch_size: int = 32,
    num_bins: int = 10,
    sigma: float = 0.1,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Calculate mutual information matrix using PyTorch.

    Parameters
    ----------
    expr_matrix : np.ndarray
        Expression matrix (cells x genes)
    gene_indices1 : list
        Indices of first set of genes
    gene_indices2 : list
        Indices of second set of genes
    batch_size : int, default=32
        Batch size for processing gene pairs
    num_bins : int, default=10
        Number of bins for discretization
    sigma : float, default=0.1
        Bandwidth for the Gaussian kernel
    device : str, optional
        Device to use for computation ('cuda', 'cpu', or None for auto-detection)

    Returns
    -------
    np.ndarray
        Mutual information matrix
    """
    if not HAS_PYTORCH:
        logger.warning("PyTorch not available. Falling back to Numba implementation.")
        from .numba_optimized import calculate_mi_matrix_numba
        return calculate_mi_matrix_numba(expr_matrix, gene_indices1, gene_indices2)

    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create MI estimator
    mi_estimator = MutualInformationEstimator(
        num_bins=num_bins,
        sigma=sigma,
        normalize=True,
        device=device
    ).to(device)

    # Get dimensions
    n_cells = expr_matrix.shape[0]
    n_genes1 = len(gene_indices1)
    n_genes2 = len(gene_indices2)

    # Initialize MI matrix
    mi_matrix = np.zeros((n_genes1, n_genes2))

    # Convert expression matrix to torch tensor
    expr_tensor = torch.tensor(expr_matrix, dtype=torch.float32)

    # Process gene pairs in batches
    for i in range(0, n_genes1, batch_size):
        i_end = min(i + batch_size, n_genes1)
        i_batch_size = i_end - i

        for j in range(0, n_genes2, batch_size):
            j_end = min(j + batch_size, n_genes2)
            j_batch_size = j_end - j

            # Create batch of gene pairs
            batch_x = []
            batch_y = []
            batch_indices = []

            for i_idx in range(i_batch_size):
                for j_idx in range(j_batch_size):
                    i_gene = gene_indices1[i + i_idx]
                    j_gene = gene_indices2[j + j_idx]

                    # Skip self-interactions
                    if i_gene == j_gene:
                        continue

                    batch_x.append(expr_tensor[:, i_gene])
                    batch_y.append(expr_tensor[:, j_gene])
                    batch_indices.append((i + i_idx, j + j_idx))

            if not batch_indices:
                continue

            # Convert to tensors and move to device
            batch_x = torch.stack(batch_x).to(device)
            batch_y = torch.stack(batch_y).to(device)

            # Calculate MI for batch
            with torch.no_grad():
                mi_values = mi_estimator(batch_x, batch_y)

            # Update MI matrix
            mi_values_cpu = mi_values.cpu().numpy()
            for idx, (i_idx, j_idx) in enumerate(batch_indices):
                mi_matrix[i_idx, j_idx] = mi_values_cpu[idx]

    return mi_matrix


def apply_dpi_pytorch(
    mi_matrix: np.ndarray,
    tf_indices: List[int],
    dpi_tolerance: float = 0.1,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Apply Data Processing Inequality using PyTorch.

    Parameters
    ----------
    mi_matrix : np.ndarray
        Mutual information matrix (TFs x genes)
    tf_indices : list
        Indices of TFs in the gene list
    dpi_tolerance : float, default=0.1
        Tolerance for DPI
    device : str, optional
        Device to use for computation ('cuda', 'cpu', or None for auto-detection)

    Returns
    -------
    np.ndarray
        Pruned MI matrix
    """
    if not HAS_PYTORCH:
        logger.warning("PyTorch not available. Falling back to Numba implementation.")
        from .numba_optimized import apply_dpi_numba
        return apply_dpi_numba(mi_matrix, tf_indices, dpi_tolerance)

    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create a copy of the MI matrix
    pruned_matrix = mi_matrix.copy()

    # Get dimensions
    n_tfs, n_genes = mi_matrix.shape

    # Create a TF index lookup dictionary for faster access
    tf_idx_lookup = {tf_idx: i for i, tf_idx in enumerate(tf_indices)}

    # Convert to PyTorch tensor and move to device
    mi_tensor = torch.tensor(mi_matrix, dtype=torch.float32, device=device)
    pruned_tensor = torch.tensor(pruned_matrix, dtype=torch.float32, device=device)

    # Create a mask for self-interactions
    self_mask = torch.zeros((n_tfs, n_genes), dtype=torch.bool, device=device)
    for i, tf_idx in enumerate(tf_indices):
        self_mask[i, tf_idx] = True

    # Apply DPI
    for i in range(n_tfs):
        tf_idx = tf_indices[i]

        # Skip zero or very small values and self-interactions
        non_zero_mask = (pruned_tensor[i] > 1e-10) & ~self_mask[i]
        non_zero_indices = torch.nonzero(non_zero_mask).squeeze(1)

        if len(non_zero_indices) == 0:
            continue

        # Get MI values for this TF
        mi_values = pruned_tensor[i, non_zero_indices]

        # For each target gene
        for idx in range(len(non_zero_indices)):
            j = non_zero_indices[idx].item()
            mi_ij = mi_values[idx].item()

            # Check all possible mediators
            for k in range(n_genes):
                # Skip if k is i or j
                if k == tf_idx or k == j:
                    continue

                # Get MI(i,k) - TF to mediator
                mi_ik = pruned_tensor[i, k].item()

                # Get MI(k,j) - mediator to gene
                # Check if k is a TF
                k_tf_idx = tf_idx_lookup.get(k, -1)

                if k_tf_idx >= 0:
                    # If k is a TF, we can find MI(k,j) in the matrix
                    mi_kj = pruned_tensor[k_tf_idx, j].item()
                else:
                    # If k is not a TF, we can't find MI(k,j) in the matrix
                    # For simplicity, we'll use a placeholder value
                    mi_kj = 0.0

                # Apply DPI
                if mi_ij < min(mi_ik, mi_kj) - dpi_tolerance:
                    pruned_tensor[i, j] = 0.0
                    break

    # Convert back to numpy
    pruned_matrix = pruned_tensor.cpu().numpy()

    return pruned_matrix


def run_aracne_pytorch(
    expr_matrix: np.ndarray,
    gene_list: List[str],
    tf_indices: List[int],
    bootstraps: int = 100,
    consensus_threshold: float = 0.5,
    dpi_tolerance: float = 0.1,
    batch_size: int = 32,
    num_bins: int = 10,
    sigma: float = 0.1,
    device: Optional[str] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Run ARACNe algorithm using PyTorch for acceleration.

    Parameters
    ----------
    expr_matrix : np.ndarray
        Expression matrix (cells x genes)
    gene_list : list
        List of gene names
    tf_indices : list
        Indices of TFs in the gene list
    bootstraps : int, default=100
        Number of bootstrap iterations
    consensus_threshold : float, default=0.5
        Threshold for consensus network
    dpi_tolerance : float, default=0.1
        Tolerance for DPI
    batch_size : int, default=32
        Batch size for processing gene pairs
    num_bins : int, default=10
        Number of bins for discretization
    sigma : float, default=0.1
        Bandwidth for the Gaussian kernel
    device : str, optional
        Device to use for computation ('cuda', 'cpu', or None for auto-detection)

    Returns
    -------
    tuple
        Consensus matrix and regulons dictionary
    """
    if not HAS_PYTORCH:
        logger.warning("PyTorch not available. Falling back to Numba implementation.")
        from .numba_optimized import run_aracne_numba
        return run_aracne_numba(
            expr_matrix,
            gene_list,
            tf_indices,
            bootstraps=bootstraps,
            consensus_threshold=consensus_threshold,
            dpi_tolerance=dpi_tolerance
        )

    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f"Running ARACNe with PyTorch acceleration on {device}")

    # Get dimensions
    n_samples, n_genes = expr_matrix.shape
    n_tfs = len(tf_indices)

    # Initialize consensus matrix
    consensus_matrix = np.zeros((n_tfs, n_genes))

    # Convert expression matrix to torch tensor
    expr_tensor = torch.tensor(expr_matrix, dtype=torch.float32)

    # Run bootstrap iterations
    for b in range(bootstraps):
        logger.debug(f"Bootstrap iteration {b+1}/{bootstraps}")

        # Create bootstrap sample
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_data = expr_matrix[bootstrap_indices]

        # Calculate MI matrix
        mi_matrix = calculate_mi_matrix_pytorch(
            bootstrap_data,
            list(range(n_tfs)),
            list(range(n_genes)),
            batch_size=batch_size,
            num_bins=num_bins,
            sigma=sigma,
            device=device
        )

        # Apply DPI
        pruned_matrix = apply_dpi_pytorch(
            mi_matrix,
            tf_indices,
            dpi_tolerance=dpi_tolerance,
            device=device
        )

        # Add to consensus matrix
        consensus_matrix += (pruned_matrix > 0).astype(np.float32) / bootstraps

    # Apply consensus threshold
    consensus_matrix[consensus_matrix < consensus_threshold] = 0

    # Create regulons dictionary
    regulons = {}
    for i, tf_idx in enumerate(tf_indices):
        # Make sure tf_idx is within bounds
        if tf_idx < len(gene_list):
            tf_name = gene_list[tf_idx]
            targets = {}

            for j in range(min(n_genes, len(gene_list))):
                # Skip self-interactions
                if j == tf_idx:
                    continue

                # Get interaction strength
                strength = consensus_matrix[i, j]

                # Add to targets if non-zero
                if strength > 0 and j < len(gene_list):
                    targets[gene_list[j]] = float(strength)

            regulons[tf_name] = {
                'targets': targets
            }

    return consensus_matrix, regulons


def run_aracne_stratified(
    expr_matrix: np.ndarray,
    gene_list: List[str],
    tf_indices: List[int],
    strata_indices: List[List[int]],
    strata_weights: Optional[List[float]] = None,
    bootstraps: int = 100,
    consensus_threshold: float = 0.5,
    dpi_tolerance: float = 0.1,
    batch_size: int = 32,
    num_bins: int = 10,
    sigma: float = 0.1,
    device: Optional[str] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Run ARACNe algorithm on stratified data.

    This function runs ARACNe separately on each stratum (e.g., cell type)
    and then combines the results using weighted averaging.

    Parameters
    ----------
    expr_matrix : np.ndarray
        Expression matrix (cells x genes)
    gene_list : list
        List of gene names
    tf_indices : list
        Indices of TFs in the gene list
    strata_indices : list of lists
        List of cell indices for each stratum
    strata_weights : list, optional
        Weights for each stratum (default: equal weights)
    bootstraps : int, default=100
        Number of bootstrap iterations
    consensus_threshold : float, default=0.5
        Threshold for consensus network
    dpi_tolerance : float, default=0.1
        Tolerance for DPI
    batch_size : int, default=32
        Batch size for processing gene pairs
    num_bins : int, default=10
        Number of bins for discretization
    sigma : float, default=0.1
        Bandwidth for the Gaussian kernel
    device : str, optional
        Device to use for computation ('cuda', 'cpu', or None for auto-detection)

    Returns
    -------
    tuple
        Consensus matrix and regulons dictionary
    """
    if not HAS_PYTORCH:
        logger.warning("PyTorch not available. Falling back to Numba implementation.")
        from .numba_optimized import run_aracne_numba

        # Run ARACNe on each stratum and combine results
        n_strata = len(strata_indices)
        n_tfs = len(tf_indices)
        n_genes = len(gene_list)

        # Set default weights if not provided
        if strata_weights is None:
            strata_weights = [1.0 / n_strata] * n_strata

        # Initialize combined consensus matrix
        combined_consensus = np.zeros((n_tfs, n_genes))

        # Process each stratum
        for s, indices in enumerate(strata_indices):
            logger.info(f"Processing stratum {s+1}/{n_strata} with {len(indices)} cells")

            # Skip if stratum is empty
            if len(indices) == 0:
                continue

            # Extract data for this stratum
            stratum_data = expr_matrix[indices]

            # Run ARACNe on this stratum
            consensus, _ = run_aracne_numba(
                stratum_data,
                gene_list,
                tf_indices,
                bootstraps=bootstraps,
                consensus_threshold=0.0,  # Don't apply threshold yet
                dpi_tolerance=dpi_tolerance
            )

            # Add to combined consensus with weight
            combined_consensus += consensus * strata_weights[s]

        # Apply consensus threshold
        combined_consensus[combined_consensus < consensus_threshold] = 0

        # Create regulons dictionary
        regulons = {}
        for i, tf_idx in enumerate(tf_indices):
            tf_name = gene_list[tf_idx]
            targets = {}

            for j in range(n_genes):
                # Skip self-interactions
                if j == tf_idx:
                    continue

                # Get interaction strength
                strength = combined_consensus[i, j]

                # Add to targets if non-zero
                if strength > 0:
                    targets[gene_list[j]] = float(strength)

            regulons[tf_name] = {
                'targets': targets
            }

        return combined_consensus, regulons

    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f"Running stratified ARACNe with PyTorch acceleration on {device}")

    # Get dimensions
    n_strata = len(strata_indices)
    n_tfs = len(tf_indices)
    n_genes = len(gene_list)

    # Set default weights if not provided
    if strata_weights is None:
        strata_weights = [1.0 / n_strata] * n_strata

    # Initialize combined consensus matrix
    combined_consensus = np.zeros((n_tfs, n_genes))

    # Process each stratum
    for s, indices in enumerate(strata_indices):
        logger.info(f"Processing stratum {s+1}/{n_strata} with {len(indices)} cells")

        # Skip if stratum is empty
        if len(indices) == 0:
            continue

        # Extract data for this stratum
        stratum_data = expr_matrix[indices]

        # Run ARACNe on this stratum
        consensus, _ = run_aracne_pytorch(
            stratum_data,
            gene_list,
            tf_indices,
            bootstraps=bootstraps,
            consensus_threshold=0.0,  # Don't apply threshold yet
            dpi_tolerance=dpi_tolerance,
            batch_size=batch_size,
            num_bins=num_bins,
            sigma=sigma,
            device=device
        )

        # Add to combined consensus with weight
        combined_consensus += consensus * strata_weights[s]

    # Apply consensus threshold
    combined_consensus[combined_consensus < consensus_threshold] = 0

    # Create regulons dictionary
    regulons = {}
    for i, tf_idx in enumerate(tf_indices):
        tf_name = gene_list[tf_idx]
        targets = {}

        for j in range(n_genes):
            # Skip self-interactions
            if j == tf_idx:
                continue

            # Get interaction strength
            strength = combined_consensus[i, j]

            # Add to targets if non-zero
            if strength > 0:
                targets[gene_list[j]] = float(strength)

        regulons[tf_name] = {
            'targets': targets
        }

    return combined_consensus, regulons
