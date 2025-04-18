"""
Integration between ARACNe and VIPER.

This module provides functions for converting ARACNe networks to VIPER-compatible regulons.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from .regulons import Regulon, GeneSet, create_regulon_from_network

# Configure logging
logger = logging.getLogger(__name__)

def aracne_to_regulons(
    network: Dict,
    min_targets: int = 10,
    max_targets: Optional[int] = None,
    keep_top_fraction: Optional[float] = None,
    infer_mode: bool = True,
    mode_method: str = 'correlation',
    expression_data: Optional[Union[pd.DataFrame, np.ndarray]] = None
) -> List[Regulon]:
    """
    Convert ARACNe network to a list of Regulon objects.
    
    Parameters
    ----------
    network : dict
        ARACNe network output
    min_targets : int, default=10
        Minimum number of targets required to keep a regulon
    max_targets : int, optional
        Maximum number of targets to keep for each regulon
    keep_top_fraction : float, optional
        Fraction of top targets to keep (by absolute mode value)
    infer_mode : bool, default=True
        Whether to infer activation/repression modes
    mode_method : str, default='correlation'
        Method for inferring modes. Options:
        - 'correlation': Use correlation between TF and target
        - 'random': Assign random modes
        - 'positive': Assign all positive modes
        - 'negative': Assign all negative modes
    expression_data : DataFrame or ndarray, optional
        Expression data for inferring modes (required if mode_method='correlation')
        
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
    >>> # Basic conversion
    >>> regulons = pysces.aracne_to_regulons(network)
    >>> # With mode inference
    >>> regulons = pysces.aracne_to_regulons(network, infer_mode=True, mode_method='correlation', expression_data=adata.X)
    >>> # With target pruning
    >>> regulons = pysces.aracne_to_regulons(network, min_targets=20, max_targets=100)
    """
    # Check if network has the expected structure
    if 'regulons' not in network:
        raise ValueError("Network does not have 'regulons' key")
    
    # Check if expression data is provided when needed
    if infer_mode and mode_method == 'correlation' and expression_data is None:
        logger.warning("Expression data is required for correlation-based mode inference. Using random modes instead.")
        mode_method = 'random'
    
    # Convert expression data to numpy array if it's a DataFrame
    if expression_data is not None and isinstance(expression_data, pd.DataFrame):
        gene_names = expression_data.index.tolist()
        expression_data = expression_data.values
    
    # Initialize list for regulons
    regulons = []
    
    # Process each TF in the network
    for tf_name, regulon_data in network['regulons'].items():
        # Get targets and their weights
        targets = regulon_data['targets']
        
        # Skip if not enough targets
        if len(targets) < min_targets:
            continue
        
        # Sort targets by weight
        sorted_targets = sorted(
            targets.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Determine how many targets to keep
        if keep_top_fraction is not None:
            n_keep = max(min_targets, int(len(sorted_targets) * keep_top_fraction))
        elif max_targets is not None:
            n_keep = min(len(sorted_targets), max_targets)
        else:
            n_keep = len(sorted_targets)
        
        # Take top n_keep targets
        sorted_targets = sorted_targets[:n_keep]
        
        # Extract target names and weights
        target_names = [t[0] for t in sorted_targets]
        target_weights = [t[1] for t in sorted_targets]
        
        # Infer modes if requested
        if infer_mode:
            if mode_method == 'correlation' and expression_data is not None:
                # Get indices of TF and targets in expression data
                try:
                    tf_idx = gene_names.index(tf_name)
                    target_indices = [gene_names.index(t) for t in target_names]
                    
                    # Calculate correlation between TF and targets
                    tf_expr = expression_data[tf_idx, :]
                    target_expr = expression_data[target_indices, :]
                    
                    # Calculate correlation
                    modes = []
                    for i in range(len(target_indices)):
                        corr = np.corrcoef(tf_expr, target_expr[i, :])[0, 1]
                        # Convert correlation to mode (-1 to 1)
                        modes.append(np.clip(corr, -0.999, 0.999))
                except (ValueError, IndexError):
                    logger.warning(f"Could not calculate correlation for {tf_name}. Using random modes.")
                    modes = np.random.uniform(-0.8, 0.8, len(target_names))
            elif mode_method == 'random':
                # Assign random modes
                modes = np.random.uniform(-0.8, 0.8, len(target_names))
            elif mode_method == 'positive':
                # Assign all positive modes
                modes = np.random.uniform(0.2, 0.8, len(target_names))
            elif mode_method == 'negative':
                # Assign all negative modes
                modes = np.random.uniform(-0.8, -0.2, len(target_names))
            else:
                raise ValueError(f"Unknown mode_method: {mode_method}")
        else:
            # Use weights as modes
            modes = target_weights
        
        # Create regulon
        regulon = create_regulon_from_network(
            tf_name=tf_name,
            targets=target_names,
            modes=modes,
            likelihoods=target_weights,
            description=f"ARACNe-inferred regulon for {tf_name}"
        )
        
        regulons.append(regulon)
    
    return regulons

def aracne_to_viper(
    network: Dict,
    expression_data: Union[pd.DataFrame, np.ndarray],
    min_targets: int = 10,
    max_targets: Optional[int] = None,
    keep_top_fraction: Optional[float] = None,
    infer_mode: bool = True,
    mode_method: str = 'correlation',
    method: str = 'gsea',
    signature_method: str = 'rank',
    abs_score: bool = False,
    normalize: bool = True,
    use_gpu: bool = False
) -> pd.DataFrame:
    """
    Convert ARACNe network to VIPER activity scores in one step.
    
    Parameters
    ----------
    network : dict
        ARACNe network output
    expression_data : DataFrame or ndarray
        Expression data for VIPER analysis
    min_targets : int, default=10
        Minimum number of targets required to keep a regulon
    max_targets : int, optional
        Maximum number of targets to keep for each regulon
    keep_top_fraction : float, optional
        Fraction of top targets to keep (by absolute mode value)
    infer_mode : bool, default=True
        Whether to infer activation/repression modes
    mode_method : str, default='correlation'
        Method for inferring modes
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
        
    Returns
    -------
    DataFrame with VIPER activity scores
    
    Examples
    --------
    >>> import pysces
    >>> adata = pysces.read_census("homo_sapiens", max_cells=1000)
    >>> adata = pysces.preprocess_data(adata)
    >>> aracne = pysces.ARACNe()
    >>> network = aracne.run(adata)
    >>> activity = pysces.aracne_to_viper(network, adata.X)
    """
    # Import here to avoid circular imports
    from .activity import viper
    import anndata as ad
    
    # Convert expression data to AnnData if it's not already
    if not isinstance(expression_data, ad.AnnData):
        if isinstance(expression_data, pd.DataFrame):
            adata = ad.AnnData(X=expression_data.values, var=pd.DataFrame(index=expression_data.columns), obs=pd.DataFrame(index=expression_data.index))
        else:
            adata = ad.AnnData(X=expression_data)
    else:
        adata = expression_data
    
    # Convert network to regulons
    regulons = aracne_to_regulons(
        network=network,
        min_targets=min_targets,
        max_targets=max_targets,
        keep_top_fraction=keep_top_fraction,
        infer_mode=infer_mode,
        mode_method=mode_method,
        expression_data=expression_data
    )
    
    # Run VIPER
    activity = viper(
        adata=adata,
        regulons=regulons,
        method=method,
        signature_method=signature_method,
        abs_score=abs_score,
        normalize=normalize,
        use_gpu=use_gpu
    )
    
    return activity
