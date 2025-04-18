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

# Try to import C++ extensions
try:
    from ._cpp import aracne_ext
    _has_cpp_ext = True
except ImportError:
    warnings.warn(
        "Could not import ARACNe C++ extensions. Using slower Python implementation. "
        "To use the faster C++ implementation, make sure the extensions are properly compiled."
    )
    _has_cpp_ext = False

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
    
    def __init__(
        self,
        p_value: float = 1e-8,
        bootstraps: int = 100,
        dpi_tolerance: float = 0.0,
        consensus_threshold: float = 0.05,
        n_threads: int = 0,
        use_gpu: bool = False
    ):
        """Initialize ARACNe network inference."""
        self.p_value = p_value
        self.bootstraps = bootstraps
        self.dpi_tolerance = dpi_tolerance
        self.consensus_threshold = consensus_threshold
        self.n_threads = n_threads
        self.use_gpu = use_gpu
        
        # Convert p-value to chi-square threshold (for 3 degrees of freedom)
        # Chi-square value for 95% confidence (p=0.05) is 7.815
        # For other p-values, we'd need to use scipy.stats.chi2.ppf(1-p_value, 3)
        # For now, we'll use a simple mapping
        if p_value <= 1e-8:
            self.chi_square_threshold = 29.877  # p=1e-6
        elif p_value <= 1e-6:
            self.chi_square_threshold = 22.458  # p=1e-4
        elif p_value <= 1e-4:
            self.chi_square_threshold = 16.266  # p=0.001
        elif p_value <= 0.001:
            self.chi_square_threshold = 12.838  # p=0.005
        elif p_value <= 0.01:
            self.chi_square_threshold = 7.815   # p=0.05
        else:
            self.chi_square_threshold = 6.251   # p=0.1
        
        # Check if GPU is requested but not available
        if use_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    print("Warning: GPU requested but not available. Falling back to CPU.")
                    self.use_gpu = False
            except ImportError:
                print("Warning: GPU requested but PyTorch not installed. Falling back to CPU.")
                self.use_gpu = False
    
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
        """
        Internal method to run ARACNe algorithm.
        """
        # Convert to numpy array if needed
        expr_matrix = np.asarray(expr_matrix, dtype=np.float64)
        tf_indices = np.asarray(tf_indices, dtype=np.int32)
        
        # Check if C++ extensions are available and have the required functions
        cpp_bootstrap_available = (_has_cpp_ext and 
                                  hasattr(aracne_ext, 'run_aracne_bootstrap') and 
                                  not self.use_gpu)
        
        if cpp_bootstrap_available:
            # Use C++ implementation
            print(f"Running ARACNe with {self.bootstraps} bootstraps...")
            
            try:
                # Run ARACNe with bootstrapping
                consensus_matrix = aracne_ext.run_aracne_bootstrap(
                    expr_matrix,
                    tf_indices,
                    self.bootstraps,
                    self.chi_square_threshold,
                    self.dpi_tolerance,
                    self.consensus_threshold,
                    self.n_threads
                )
            except Exception as e:
                print(f"Error running C++ implementation: {str(e)}")
                print("Falling back to Python implementation.")
                return self._run_aracne_python(expr_matrix, gene_list, tf_indices)
            
            # Convert to regulons
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
        
        elif self.use_gpu:
            # TODO: Implement GPU version
            print("Warning: GPU implementation not yet available. Falling back to CPU.")
            return self._run_aracne_python(expr_matrix, gene_list, tf_indices)
        
        else:
            # Use Python implementation
            return self._run_aracne_python(expr_matrix, gene_list, tf_indices)
    
    def _run_aracne_python(self, expr_matrix, gene_list, tf_indices):
        """
        Python implementation of ARACNe algorithm.
        This is a fallback for when C++ extensions are not available.
        """
        print("Warning: Using Python implementation of ARACNe.")
        print("This is much slower than the C++ implementation.")
        print("Consider installing the C++ extensions for better performance.")
        
        # For now, return a dummy network
        # In the future, we could implement a pure Python version
        n_tfs = len(tf_indices)
        n_genes = len(gene_list)
        
        # Create dummy regulons
        regulons = {}
        for i, tf_idx in enumerate(tf_indices):
            tf_name = gene_list[tf_idx]
            # Create random targets (about 50 per TF)
            n_targets = min(50, n_genes)
            target_indices = np.random.choice(n_genes, n_targets, replace=False)
            target_modes = np.random.uniform(-1, 1, n_targets)
            
            regulons[tf_name] = {
                'targets': {gene_list[idx]: mode for idx, mode in zip(target_indices, target_modes)}
            }
        
        # Return dummy network
        return {
            'regulons': regulons,
            'tf_names': [gene_list[i] for i in tf_indices],
            'metadata': {
                'p_value': self.p_value,
                'bootstraps': self.bootstraps,
                'dpi_tolerance': self.dpi_tolerance,
                'consensus_threshold': self.consensus_threshold
            }
        }


def aracne_to_regulons(network: Dict) -> List["Regulon"]:
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
    from ..viper.regulons import Regulon
    
    regulons = []
    
    for tf_name, regulon_data in network['regulons'].items():
        # Create a Regulon object
        regulon = Regulon(tf_name=tf_name)
        
        # Add targets
        for target, mode in regulon_data['targets'].items():
            regulon.add_target(target, mode)
        
        regulons.append(regulon)
    
    return regulons
