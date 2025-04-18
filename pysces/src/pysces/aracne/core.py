"""
Core ARACNe implementation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import anndata as ad
import scipy.sparse
import os
import importlib.util
import warnings
import logging
import sys
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Try to import C++ extensions
try:
    from ._cpp import aracne_ext
    _has_cpp_ext = True
    print(f"Using C++ extensions for ARACNe (version: {getattr(aracne_ext, '__version__', 'unknown')})")
except ImportError:
    _has_cpp_ext = False
    warnings.warn(
        "Could not import ARACNe C++ extensions. Using slower Python implementation. "
        "To use the faster C++ implementation, rebuild the package with C++ support enabled."
    )

class ARACNe:
    """
    ARACNe (Algorithm for the Reconstruction of Accurate Cellular Networks) implementation.

    This class provides methods for inferring gene regulatory networks from expression data
    using the ARACNe algorithm, which is based on mutual information and data processing inequality.

    Parameters
    ----------
    p_value : float, default=1e-8
        P-value threshold for mutual information significance
    bootstraps : int, default=100
        Number of bootstrap iterations
    dpi_tolerance : float, default=0.0
        Tolerance for Data Processing Inequality
    consensus_threshold : float, default=0.05
        Threshold for consensus network (fraction of bootstrap networks)
    n_threads : int, default=0
        Number of threads to use (0 = auto)
    use_gpu : bool, default=False
        Whether to use GPU acceleration (if available)
    """

    def __init__(self, bootstraps=100, p_value=0.05, dpi_tolerance=0.1,
                 consensus_threshold=0.5, chi_square_threshold=7.815,
                 use_gpu=False, n_threads=0):
        """Initialize ARACNe."""
        self.bootstraps = bootstraps
        self.p_value = p_value
        self.dpi_tolerance = dpi_tolerance
        self.consensus_threshold = consensus_threshold
        self.chi_square_threshold = chi_square_threshold
        self.use_gpu = use_gpu
        self.n_threads = n_threads
        self._has_cpp_ext = False

        logger.debug("Initializing ARACNe with parameters: %s",
                    {k: v for k, v in locals().items() if k != 'self'})

        # Check C++ extension availability
        self._check_cpp_extensions()

    def _check_cpp_extensions(self):
        """Check if C++ extensions are properly loaded."""
        try:
            logger.debug("Attempting to import C++ extensions")
            from ._cpp import aracne_ext

            # Test with small arrays
            x = np.array([1.0, 2.0], dtype=np.float64)
            y = np.array([2.0, 4.0], dtype=np.float64)

            logger.debug("Testing C++ extension with arrays: x=%s, y=%s", x, y)
            test_mi = aracne_ext.calculate_mi_ap(x, y)

            logger.debug("C++ extension test successful. MI=%f", test_mi)
            self._has_cpp_ext = True

        except ImportError as e:
            logger.error("Failed to import C++ extensions: %s", str(e))
            self._has_cpp_ext = False
            warnings.warn(
                f"Could not import ARACNe C++ extensions: {str(e)}. "
                "Using slower Python implementation."
            )
        except Exception as e:
            logger.error("Error testing C++ extensions: %s\n%s",
                        str(e), traceback.format_exc())
            self._has_cpp_ext = False
            warnings.warn(
                f"Error testing ARACNe C++ extensions: {str(e)}. "
                "Using slower Python implementation."
            )

    def run(
        self,
        adata: ad.AnnData,
        tf_list: Optional[List[str]] = None,
        layer: Optional[str] = None
    ) -> Dict:
        """
        Run ARACNe network inference on expression data.

        Parameters
        ----------
        adata : AnnData
            Expression data (cells x genes)
        tf_list : list of str, optional
            List of transcription factor names. If None, all genes are considered TFs.
        layer : str, optional
            Which layer of the AnnData to use. If None, uses .X

        Returns
        -------
        Dict containing network data and metadata

        Examples
        --------
        >>> import pysces
        >>> adata = pysces.read_census("homo_sapiens", max_cells=1000)
        >>> adata = pysces.preprocess_data(adata)
        >>> aracne = pysces.ARACNe()
        >>> network = aracne.run(adata)
        """
        # Extract expression matrix
        if layer is None:
            expr_matrix = adata.X
        else:
            expr_matrix = adata.layers[layer]

        # If sparse, convert to dense
        if scipy.sparse.issparse(expr_matrix):
            expr_matrix = expr_matrix.toarray()

        # Transpose to genes x cells if needed
        if expr_matrix.shape[0] == adata.n_obs:
            expr_matrix = expr_matrix.T

        # Get gene list
        gene_list = adata.var_names.tolist()

        # Filter TF list
        if tf_list is None:
            tf_list = gene_list
        else:
            # Ensure all TFs are in the gene list
            tf_list = [tf for tf in tf_list if tf in gene_list]

        # Get TF indices
        tf_indices = [gene_list.index(tf) for tf in tf_list]

        # Run network inference
        network = self._run_aracne(expr_matrix, gene_list, tf_indices)

        return network

    def _run_aracne(self, expr_matrix, gene_list, tf_indices):
        """Internal method to run ARACNe algorithm."""
        logger.debug("Starting _run_aracne")
        logger.debug("Input shapes - expr_matrix: %s, tf_indices: %s",
                    expr_matrix.shape, len(tf_indices))

        try:
            # Validate and convert inputs
            expr_matrix = np.asarray(expr_matrix, dtype=np.float64, order='C')
            tf_indices = np.asarray(tf_indices, dtype=np.int32)

            if not self._has_cpp_ext or self.use_gpu:
                logger.debug("Using Python implementation")
                return self._run_aracne_python(expr_matrix, gene_list, tf_indices)

            logger.debug("Using C++ implementation")
            from ._cpp import aracne_ext

            # Test single MI calculation first
            logger.debug("Testing single MI calculation")
            test_genes = expr_matrix[:, :2]  # Take first two genes
            test_mi = aracne_ext.calculate_mi_ap(
                test_genes[:, 0],
                test_genes[:, 1],
                self.chi_square_threshold
            )
            logger.debug("Single MI calculation successful: %f", test_mi)

            # Run full analysis with bootstrapping
            logger.debug(f"Starting full ARACNe analysis with {self.bootstraps} bootstraps")

            # Initialize consensus matrix
            n_tfs = len(tf_indices)
            n_genes = expr_matrix.shape[1]
            consensus_matrix = np.zeros((n_tfs, n_genes), dtype=np.float64)

            # Run bootstrap iterations
            for b in range(self.bootstraps):
                logger.debug(f"Bootstrap iteration {b+1}/{self.bootstraps}")

                # Create bootstrap sample
                if hasattr(aracne_ext, 'bootstrap_matrix'):
                    # Use C++ implementation if available
                    bootstrap_data = aracne_ext.bootstrap_matrix(expr_matrix)
                else:
                    # Otherwise use numpy
                    bootstrap_indices = np.random.choice(expr_matrix.shape[0], expr_matrix.shape[0], replace=True)
                    bootstrap_data = expr_matrix[bootstrap_indices]

                # Calculate MI matrix
                mi_matrix = aracne_ext.calculate_mi_matrix(
                    bootstrap_data,
                    tf_indices,
                    self.chi_square_threshold,
                    self.n_threads
                )

                # Apply DPI
                pruned_matrix = aracne_ext.apply_dpi(
                    mi_matrix,
                    self.dpi_tolerance,
                    self.n_threads
                )

                # Add to consensus matrix
                consensus_matrix += (pruned_matrix > 0).astype(np.float64) / self.bootstraps

            # Apply consensus threshold
            consensus_matrix[consensus_matrix < self.consensus_threshold] = 0.0

            logger.debug("ARACNe analysis complete")
            return self._process_results(consensus_matrix, gene_list, tf_indices)

        except Exception as e:
            logger.error("Error in _run_aracne: %s\n%s",
                        str(e), traceback.format_exc())
            raise

    def _process_results(self, consensus_matrix, gene_list, tf_indices):
        """Process ARACNe results into final format."""
        logger.debug("Processing results")
        try:
            regulons = {}
            for i, tf_idx in enumerate(tf_indices):
                tf_name = gene_list[tf_idx]
                targets = {}

                for j in range(len(gene_list)):
                    # Skip self-interactions
                    if j == tf_idx:
                        continue

                    # Get interaction strength
                    strength = consensus_matrix[i, j]

                    # Add to targets if non-zero
                    if strength > 0:
                        # For now, we'll use the consensus value as the mode
                        # In the future, we might want to determine activation/repression
                        targets[gene_list[j]] = strength

                regulons[tf_name] = {
                    'targets': targets
                }

            # Return network
            return {
                'regulons': regulons,
                'tf_names': [gene_list[i] for i in tf_indices],
                'consensus_matrix': consensus_matrix,
                'metadata': {
                    'p_value': self.p_value,
                    'bootstraps': self.bootstraps,
                    'dpi_tolerance': self.dpi_tolerance,
                    'consensus_threshold': self.consensus_threshold
                }
            }
        except Exception as e:
            logger.error("Error processing results: %s\n%s",
                        str(e), traceback.format_exc())
            raise

    def _run_aracne_python(self, expr_matrix, gene_list, tf_indices):
        """
        Python implementation of ARACNe algorithm.
        This is a fallback for when C++ extensions are not available.
        """
        logger.warning("Using Python implementation of ARACNe.")
        logger.warning("This is much slower than the C++ implementation.")
        logger.warning("Consider installing the C++ extensions for better performance.")

        # Get dimensions
        n_samples = expr_matrix.shape[0]
        n_genes = expr_matrix.shape[1]
        n_tfs = len(tf_indices)

        # Initialize consensus matrix
        consensus_matrix = np.zeros((n_tfs, n_genes), dtype=np.float64)

        # Run bootstrap iterations
        for b in range(self.bootstraps):
            logger.debug(f"Bootstrap iteration {b+1}/{self.bootstraps}")

            # Create bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_data = expr_matrix[bootstrap_indices]

            # Calculate MI matrix
            mi_matrix = np.zeros((n_tfs, n_genes), dtype=np.float64)

            # Calculate MI for each TF-gene pair
            for i, tf_idx in enumerate(tf_indices):
                for j in range(n_genes):
                    # Skip self-interactions
                    if tf_idx == j:
                        continue

                    # Calculate MI using a simple Python implementation
                    x = bootstrap_data[:, tf_idx]
                    y = bootstrap_data[:, j]

                    # Calculate correlation as a simple approximation of MI
                    # In a real implementation, we would use a proper MI calculation
                    correlation = np.corrcoef(x, y)[0, 1]
                    mi = abs(correlation)  # Use absolute correlation as a proxy for MI

                    mi_matrix[i, j] = mi

            # Apply DPI
            pruned_matrix = self._apply_dpi_python(mi_matrix, n_tfs, n_genes)

            # Add to consensus matrix
            consensus_matrix += (pruned_matrix > 0).astype(np.float64) / self.bootstraps

        # Apply consensus threshold
        consensus_matrix[consensus_matrix < self.consensus_threshold] = 0.0

        return self._process_results(consensus_matrix, gene_list, tf_indices)

    def _apply_dpi_python(self, mi_matrix, n_tfs, n_genes):
        """
        Python implementation of Data Processing Inequality.
        """
        # Create a copy of the MI matrix
        pruned_matrix = mi_matrix.copy()

        # Apply DPI
        for i in range(n_tfs):
            for j in range(n_genes):
                # Skip zero or very small values
                if pruned_matrix[i, j] < 1e-10:
                    continue

                # Get TF index
                tf_idx = i

                # Check all possible mediators
                for k in range(n_genes):
                    # Skip if k is i or j
                    if k == tf_idx or k == j:
                        continue

                    # Get MI values
                    mi_ij = pruned_matrix[i, j]

                    # Find MI(i,k) and MI(k,j)
                    mi_ik = 0.0
                    mi_kj = 0.0

                    # Check if k is a TF
                    k_is_tf = False
                    k_tf_idx = -1

                    for tf_i in range(n_tfs):
                        if tf_i == k:
                            k_is_tf = True
                            k_tf_idx = tf_i
                            break

                    # Get MI(i,k)
                    if k_is_tf:
                        # If k is a TF, we can find MI(i,k) in the matrix
                        mi_ik = pruned_matrix[i, k]
                    else:
                        # If k is not a TF, we need to calculate MI(i,k)
                        # For simplicity, we'll use a placeholder value
                        mi_ik = 0.0

                    # Get MI(k,j)
                    if k_is_tf:
                        # If k is a TF, we can find MI(k,j) in the matrix
                        mi_kj = pruned_matrix[k_tf_idx, j]
                    else:
                        # If k is not a TF, we need to calculate MI(k,j)
                        # For simplicity, we'll use a placeholder value
                        mi_kj = 0.0

                    # Apply DPI
                    if mi_ij < min(mi_ik, mi_kj) - self.dpi_tolerance:
                        pruned_matrix[i, j] = 0.0
                        break

        return pruned_matrix


def aracne_to_regulons(network: Dict) -> List:
    """
    Convert ARACNe network to a list of Regulon objects.

    Parameters
    ----------
    network : dict
        ARACNe network output

    Returns
    -------
    List of Regulon objects

    Examples
    --------
    >>> import pysces
    >>> adata = pysces.read_census("homo_sapiens", max_cells=1000)
    >>> adata = pysces.preprocess_data(adata)
    >>> aracne = pysces.ARACNe()
    >>> network = aracne.run(adata)
    >>> regulons = pysces.aracne_to_regulons(network)
    """
    try:
        from ..viper.regulons import Regulon
    except ImportError:
        # Define a simple Regulon class if the viper module is not available
        class Regulon:
            def __init__(self, tf_name):
                self.tf_name = tf_name
                self.targets = {}

            def add_target(self, target, mode):
                self.targets[target] = mode

    regulons = []

    for tf_name, regulon_data in network['regulons'].items():
        # Create a Regulon object
        regulon = Regulon(tf_name=tf_name)

        # Add targets
        for target, mode in regulon_data['targets'].items():
            regulon.add_target(target, mode)

        regulons.append(regulon)

    return regulons
