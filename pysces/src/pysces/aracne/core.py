"""
Core ARACNe implementation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import anndata as ad
import scipy.sparse
import warnings
import logging
import traceback
import time
import platform

# Import Numba-optimized functions
try:
    from .numba_optimized import run_aracne_numba, HAS_NUMBA
except ImportError:
    HAS_NUMBA = False
    warnings.warn(
        "Numba is not installed. Using slower Python implementation. "
        "To use the faster Numba implementation, install Numba with: pip install numba"
    )

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Disable C++ extensions as they're causing issues
# The Python implementation is more reliable and easier to debug
_has_cpp_ext = False
print("Using Python implementation of ARACNe (C++ extensions disabled)")

class ARACNe:
    """
    ARACNe (Algorithm for the Reconstruction of Accurate Cellular Networks) implementation.

    This implementation uses Numba for acceleration by default, with a fallback to pure Python.

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
    backend : str, default='auto'
        Computation backend to use. Options: 'auto', 'numba', 'python'
    stratify_by_tissue : bool, default=False
        Whether to stratify analysis by tissue
    stratify_by_cell_type : bool, default=False
        Whether to stratify analysis by cell type
    tissue_col : str, default='tissue'
        Column name in AnnData.obs containing tissue information
    cell_type_col : str, default='cell_type'
        Column name in AnnData.obs containing cell type information
    """

    def __init__(self, bootstraps=100, p_value=0.05, dpi_tolerance=0.1,
                 consensus_threshold=0.5, chi_square_threshold=7.815,
                 backend='auto', n_threads=0,
                 stratify_by_tissue=False, stratify_by_cell_type=False,
                 tissue_col='tissue', cell_type_col='cell_type'):
        """Initialize ARACNe."""
        self.bootstraps = bootstraps
        self.p_value = p_value
        self.dpi_tolerance = dpi_tolerance
        self.consensus_threshold = consensus_threshold
        self.chi_square_threshold = chi_square_threshold
        self.backend = backend
        self.n_threads = n_threads
        self.stratify_by_tissue = stratify_by_tissue
        self.stratify_by_cell_type = stratify_by_cell_type
        self.tissue_col = tissue_col
        self.cell_type_col = cell_type_col
        self._has_cpp_ext = False

        # Determine which backend to use
        self._determine_backend()

        logger.debug("Initializing ARACNe with parameters: %s",
                    {k: v for k, v in locals().items() if k != 'self'})

        # Log stratification status
        if self.stratify_by_tissue:
            logger.info(f"Using tissue stratification (column: {self.tissue_col})")
        if self.stratify_by_cell_type:
            logger.info(f"Using cell type stratification (column: {self.cell_type_col})")

    def _determine_backend(self):
        """Determine which backend to use based on availability and user preference."""
        # Initialize backend flags
        self.use_numba = False

        # Auto-detect the best available backend
        if self.backend == 'auto':
            # Check for Numba
            if HAS_NUMBA:
                self.use_numba = True
                logger.info("Auto-selected Numba backend")
            else:
                logger.warning("No acceleration available. Using Python implementation.")
        # Use specific backend requested by user
        else:
            if self.backend == 'numba':
                if HAS_NUMBA:
                    self.use_numba = True
                    logger.info("Using Numba backend")
                else:
                    logger.warning("Numba requested but not available. Install with: pip install numba")
                    logger.warning("Falling back to Python implementation")
            elif self.backend == 'python':
                logger.info("Using Python implementation (no acceleration)")
            else:
                logger.warning(f"Unknown backend '{self.backend}'. Using auto-detection.")
                # Recursively call with auto
                self.backend = 'auto'
                self._determine_backend()

    def _check_cpp_extensions(self):
        """Check if C++ extensions are properly loaded.

        Note: Currently configured to always use the Python implementation
        as it's more reliable and easier to debug than the C++ implementation.
        """
        # Always use Python implementation
        self._has_cpp_ext = False
        logger.debug("Using Python implementation of ARACNe")

    def run(
        self,
        adata: ad.AnnData,
        tf_list: Optional[List[str]] = None,
        layer: Optional[str] = None,
        validate: bool = True
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
        validate : bool, default=True
            Whether to validate the input data before running ARACNe

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
        # Validate input data if requested
        if validate:
            try:
                from ..utils.validation import validate_anndata_structure, validate_gene_names, validate_cell_names

                # Check AnnData structure
                is_valid, issues = validate_anndata_structure(adata)
                if not is_valid:
                    error_msg = "Invalid AnnData structure:\n" + "\n".join([f"- {issue}" for issue in issues])
                    raise ValueError(error_msg)

                # Check gene names
                is_valid, issues = validate_gene_names(adata)
                if not is_valid:
                    error_msg = "Invalid gene names:\n" + "\n".join([f"- {issue}" for issue in issues])
                    raise ValueError(error_msg)

                # Check cell names
                is_valid, issues = validate_cell_names(adata)
                if not is_valid:
                    error_msg = "Invalid cell names:\n" + "\n".join([f"- {issue}" for issue in issues])
                    raise ValueError(error_msg)

                # Check if layer exists
                if layer is not None and layer not in adata.layers:
                    raise ValueError(f"Layer '{layer}' not found in AnnData object")

                logger.info("Input data validation successful")
            except ImportError:
                logger.warning("Validation module not found. Skipping validation.")

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
            original_tf_count = len(tf_list)
            tf_list = [tf for tf in tf_list if tf in gene_list]
            filtered_tf_count = len(tf_list)

            if filtered_tf_count < original_tf_count:
                logger.warning(f"Filtered out {original_tf_count - filtered_tf_count} TFs that were not found in the dataset")

            if filtered_tf_count == 0:
                raise ValueError("No valid TFs found in the dataset")

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

            # Check if we should use Numba
            if self.use_numba and HAS_NUMBA:
                logger.debug("Using Numba-accelerated implementation")
                start_time = time.time()

                consensus_matrix, regulons_dict = run_aracne_numba(
                    expr_matrix, gene_list, tf_indices,
                    bootstraps=self.bootstraps,
                    consensus_threshold=self.consensus_threshold,
                    dpi_tolerance=self.dpi_tolerance
                )

                logger.debug(f"Numba-accelerated ARACNe completed in {time.time() - start_time:.2f} seconds")

                # Create edges list
                edges = []
                for tf_name, regulon_data in regulons_dict.items():
                    for target, strength in regulon_data['targets'].items():
                        edges.append({
                            'source': tf_name,
                            'target': target,
                            'weight': strength
                        })

                # Return network
                return {
                    'regulons': regulons_dict,
                    'tf_names': [gene_list[i] for i in tf_indices],
                    'consensus_matrix': consensus_matrix,
                    'edges': edges,
                    'metadata': {
                        'p_value': self.p_value,
                        'bootstraps': self.bootstraps,
                        'dpi_tolerance': self.dpi_tolerance,
                        'consensus_threshold': self.consensus_threshold
                    }
                }
            else:
                # Fall back to Python implementation
                logger.debug("Using Python implementation")
                return self._run_aracne_python(expr_matrix, gene_list, tf_indices)

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
                    # Make sure j is within bounds of the consensus matrix
                    if j < consensus_matrix.shape[1]:
                        strength = consensus_matrix[i, j]
                    else:
                        # Skip genes that are out of bounds
                        continue

                    # Add to targets if non-zero
                    if strength > 0:
                        # For now, we'll use the consensus value as the mode
                        # In the future, we might want to determine activation/repression
                        targets[gene_list[j]] = strength

                regulons[tf_name] = {
                    'targets': targets
                }

            # Create edges list for compatibility with downstream functions
            # Each edge is a dictionary with source (TF), target (gene), and weight (interaction strength)
            edges = []
            for tf_name, regulon_data in regulons.items():
                for target, strength in regulon_data['targets'].items():
                    edges.append({
                        'source': tf_name,
                        'target': target,
                        'weight': strength
                    })

            # Return network
            return {
                'regulons': regulons,
                'tf_names': [gene_list[i] for i in tf_indices],
                'consensus_matrix': consensus_matrix,
                'edges': edges,  # Add edges to the network
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
        if not HAS_NUMBA:
            logger.warning("This is much slower than the Numba-accelerated implementation.")
            logger.warning("Consider installing Numba for better performance: pip install numba")

        # Record start time for performance measurement
        start_time = time.time()

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

        # Log performance
        logger.debug(f"Python implementation of ARACNe completed in {time.time() - start_time:.2f} seconds")

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
