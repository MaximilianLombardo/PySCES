"""
Regulon handling functionality for VIPER.
"""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np

class Regulon:
    """
    Container for a transcription factor regulon.
    
    A regulon represents a transcription factor and its target genes,
    along with the mode of regulation (activation or repression).
    
    Parameters
    ----------
    tf_name : str
        Name of the transcription factor
    targets : dict, optional
        Dictionary mapping target gene names to regulation modes (float)
        Positive values indicate activation, negative values indicate repression
    """
    
    def __init__(self, tf_name: str, targets: Optional[Dict[str, float]] = None):
        """Initialize a regulon."""
        self.tf_name = tf_name
        self.targets = targets or {}
    
    def add_target(self, target_name: str, mode: float):
        """
        Add a target to the regulon.
        
        Parameters
        ----------
        target_name : str
            Name of the target gene
        mode : float
            Mode of regulation (positive for activation, negative for repression)
        """
        self.targets[target_name] = mode
    
    def remove_target(self, target_name: str):
        """
        Remove a target from the regulon.
        
        Parameters
        ----------
        target_name : str
            Name of the target gene to remove
        """
        if target_name in self.targets:
            del self.targets[target_name]
    
    def get_targets(self) -> Dict[str, float]:
        """
        Get all targets and their modes.
        
        Returns
        -------
        Dictionary mapping target gene names to regulation modes
        """
        return self.targets.copy()
    
    def __len__(self):
        """Number of targets in the regulon."""
        return len(self.targets)
    
    def __repr__(self):
        """String representation."""
        return f"Regulon({self.tf_name}, targets={len(self.targets)})"


def prune_regulons(
    regulons: List[Regulon],
    min_targets: int = 10,
    max_targets: Optional[int] = None,
    keep_top_fraction: Optional[float] = None
) -> List[Regulon]:
    """
    Prune regulons to control size.
    
    Parameters
    ----------
    regulons : list of Regulon
        List of regulons to prune
    min_targets : int, default=10
        Minimum number of targets required to keep a regulon
    max_targets : int, optional
        Maximum number of targets to keep for each regulon
    keep_top_fraction : float, optional
        Fraction of top targets to keep (by absolute mode value)
        
    Returns
    -------
    List of pruned Regulon objects
    
    Examples
    --------
    >>> import pysces
    >>> # Prune regulons to have between 10 and 100 targets
    >>> pruned_regulons = pysces.prune_regulons(regulons, min_targets=10, max_targets=100)
    >>> # Keep only the top 20% of targets by mode strength
    >>> pruned_regulons = pysces.prune_regulons(regulons, keep_top_fraction=0.2)
    """
    pruned = []
    
    for regulon in regulons:
        if len(regulon) < min_targets:
            continue
        
        # Sort targets by absolute mode value
        sorted_targets = sorted(
            regulon.targets.items(), 
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
        
        # Create new regulon
        new_regulon = Regulon(regulon.tf_name)
        for target, mode in sorted_targets:
            new_regulon.add_target(target, mode)
        
        pruned.append(new_regulon)
    
    return pruned
