"""
Regulon handling functionality for VIPER.

This module provides classes and functions for working with gene sets and regulons
for the VIPER (Virtual Inference of Protein-activity by Enriched Regulon analysis) algorithm.
"""

from typing import Dict, List, Tuple, Union, Optional, Set, Iterable
import numpy as np
import pandas as pd

class GeneSet:
    """
    Container for a gene set.

    A gene set represents a collection of genes that share a common property,
    such as being targets of a transcription factor or members of a pathway.

    Parameters
    ----------
    name : str
        Name of the gene set
    genes : set or list, optional
        Set of gene names in the gene set
    weights : dict, optional
        Dictionary mapping gene names to weights (float)
    description : str, optional
        Description of the gene set
    """

    def __init__(self,
                 name: str,
                 genes: Optional[Union[Set[str], List[str]]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 description: Optional[str] = None):
        """Initialize a gene set."""
        self.name = name
        self.genes = set(genes) if genes is not None else set()
        self.weights = weights or {}
        self.description = description or ""

        # Ensure all genes have weights
        for gene in self.genes:
            if gene not in self.weights:
                self.weights[gene] = 1.0

    def add_gene(self, gene_name: str, weight: float = 1.0):
        """
        Add a gene to the gene set.

        Parameters
        ----------
        gene_name : str
            Name of the gene
        weight : float, default=1.0
            Weight of the gene in the set
        """
        self.genes.add(gene_name)
        self.weights[gene_name] = weight

    def remove_gene(self, gene_name: str):
        """
        Remove a gene from the gene set.

        Parameters
        ----------
        gene_name : str
            Name of the gene to remove
        """
        if gene_name in self.genes:
            self.genes.remove(gene_name)
            if gene_name in self.weights:
                del self.weights[gene_name]

    def get_genes(self) -> Set[str]:
        """
        Get all genes in the set.

        Returns
        -------
        Set of gene names
        """
        return self.genes.copy()

    def get_weights(self) -> Dict[str, float]:
        """
        Get all gene weights.

        Returns
        -------
        Dictionary mapping gene names to weights
        """
        return self.weights.copy()

    def intersect(self, other: 'GeneSet') -> 'GeneSet':
        """
        Create a new gene set containing genes present in both this set and another.

        Parameters
        ----------
        other : GeneSet
            Another gene set to intersect with

        Returns
        -------
        GeneSet
            A new gene set containing the intersection
        """
        name = f"{self.name}_intersect_{other.name}"
        genes = self.genes.intersection(other.genes)

        # Combine weights (average them)
        weights = {}
        for gene in genes:
            weights[gene] = (self.weights.get(gene, 0) + other.weights.get(gene, 0)) / 2

        return GeneSet(name, genes, weights)

    def union(self, other: 'GeneSet') -> 'GeneSet':
        """
        Create a new gene set containing genes present in either this set or another.

        Parameters
        ----------
        other : GeneSet
            Another gene set to union with

        Returns
        -------
        GeneSet
            A new gene set containing the union
        """
        name = f"{self.name}_union_{other.name}"
        genes = self.genes.union(other.genes)

        # Combine weights (use max)
        weights = {}
        for gene in genes:
            weights[gene] = max(self.weights.get(gene, 0), other.weights.get(gene, 0))

        return GeneSet(name, genes, weights)

    def filter_genes(self, gene_list: Iterable[str]) -> 'GeneSet':
        """
        Create a new gene set containing only genes present in the provided list.

        Parameters
        ----------
        gene_list : iterable of str
            List of genes to keep

        Returns
        -------
        GeneSet
            A new filtered gene set
        """
        gene_set = set(gene_list)
        filtered_genes = self.genes.intersection(gene_set)
        filtered_weights = {gene: self.weights[gene] for gene in filtered_genes}

        return GeneSet(self.name, filtered_genes, filtered_weights, self.description)

    def __len__(self):
        """Number of genes in the set."""
        return len(self.genes)

    def __repr__(self):
        """String representation."""
        return f"GeneSet({self.name}, genes={len(self.genes)})"

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the gene set to a DataFrame.

        Returns
        -------
        DataFrame with gene names and weights
        """
        df = pd.DataFrame({
            'gene': list(self.genes),
            'weight': [self.weights.get(gene, 1.0) for gene in self.genes]
        })
        return df


class Regulon(GeneSet):
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
    likelihood : dict, optional
        Dictionary mapping target gene names to likelihood/confidence scores (float)
    description : str, optional
        Description of the regulon
    """

    def __init__(self,
                 tf_name: str,
                 targets: Optional[Dict[str, float]] = None,
                 likelihood: Optional[Dict[str, float]] = None,
                 description: Optional[str] = None):
        """Initialize a regulon."""
        super().__init__(tf_name, genes=targets.keys() if targets else None, description=description)
        self.tf_name = tf_name
        self.targets = targets or {}
        self.likelihood = likelihood or {}

        # Ensure all targets have likelihood scores
        for target in self.targets:
            if target not in self.likelihood:
                self.likelihood[target] = 1.0

        # Set weights to likelihood values
        self.weights = self.likelihood.copy()

    def add_target(self, target_name: str, mode: float, likelihood: float = 1.0):
        """
        Add a target to the regulon.

        Parameters
        ----------
        target_name : str
            Name of the target gene
        mode : float
            Mode of regulation (positive for activation, negative for repression)
        likelihood : float, default=1.0
            Likelihood/confidence score for this target
        """
        self.targets[target_name] = mode
        self.likelihood[target_name] = likelihood
        self.genes.add(target_name)
        self.weights[target_name] = likelihood

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
            if target_name in self.likelihood:
                del self.likelihood[target_name]
            if target_name in self.genes:
                self.genes.remove(target_name)
            if target_name in self.weights:
                del self.weights[target_name]

    def get_targets(self) -> Dict[str, float]:
        """
        Get all targets and their modes.

        Returns
        -------
        Dictionary mapping target gene names to regulation modes
        """
        return self.targets.copy()

    def get_likelihood(self) -> Dict[str, float]:
        """
        Get all targets and their likelihood scores.

        Returns
        -------
        Dictionary mapping target gene names to likelihood scores
        """
        return self.likelihood.copy()

    def get_positive_targets(self) -> Dict[str, float]:
        """
        Get targets with positive regulation modes.

        Returns
        -------
        Dictionary mapping target gene names to positive regulation modes
        """
        return {target: mode for target, mode in self.targets.items() if mode > 0}

    def get_negative_targets(self) -> Dict[str, float]:
        """
        Get targets with negative regulation modes.

        Returns
        -------
        Dictionary mapping target gene names to negative regulation modes
        """
        return {target: mode for target, mode in self.targets.items() if mode < 0}

    def normalize_likelihood(self):
        """
        Normalize likelihood scores to sum to 1.

        This is useful for ensuring that regulons with different numbers of targets
        have comparable influence in enrichment calculations.
        """
        total = sum(self.likelihood.values())
        if total > 0:
            for target in self.likelihood:
                self.likelihood[target] /= total
                self.weights[target] = self.likelihood[target]

    def __repr__(self):
        """String representation."""
        return f"Regulon({self.tf_name}, targets={len(self.targets)})"

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the regulon to a DataFrame.

        Returns
        -------
        DataFrame with target names, modes, and likelihood scores
        """
        df = pd.DataFrame({
            'target': list(self.targets.keys()),
            'mode': list(self.targets.values()),
            'likelihood': [self.likelihood.get(target, 1.0) for target in self.targets]
        })
        return df


def prune_regulons(
    regulons: List[Regulon],
    min_targets: int = 10,
    max_targets: Optional[int] = None,
    keep_top_fraction: Optional[float] = None,
    sort_by: str = 'mode'
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
    sort_by : str, default='mode'
        How to sort targets for pruning. Options:
        - 'mode': Sort by absolute mode value
        - 'likelihood': Sort by likelihood score
        - 'combined': Sort by product of absolute mode and likelihood

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
    >>> # Sort by likelihood instead of mode
    >>> pruned_regulons = pysces.prune_regulons(regulons, sort_by='likelihood')
    """
    pruned = []

    for regulon in regulons:
        if len(regulon) < min_targets:
            continue

        # Get targets, modes, and likelihoods
        targets = list(regulon.targets.keys())
        modes = [regulon.targets[t] for t in targets]
        likelihoods = [regulon.likelihood.get(t, 1.0) for t in targets]

        # Sort targets based on specified criterion
        if sort_by == 'mode':
            # Sort by absolute mode value
            sorted_indices = np.argsort([-abs(m) for m in modes])
        elif sort_by == 'likelihood':
            # Sort by likelihood
            sorted_indices = np.argsort([-l for l in likelihoods])
        elif sort_by == 'combined':
            # Sort by product of absolute mode and likelihood
            combined_scores = [abs(m) * l for m, l in zip(modes, likelihoods)]
            sorted_indices = np.argsort([-s for s in combined_scores])
        else:
            raise ValueError(f"Unknown sort_by value: {sort_by}. Must be 'mode', 'likelihood', or 'combined'.")

        # Reorder targets, modes, and likelihoods
        targets = [targets[i] for i in sorted_indices]
        modes = [modes[i] for i in sorted_indices]
        likelihoods = [likelihoods[i] for i in sorted_indices]

        # Determine how many targets to keep
        if keep_top_fraction is not None:
            n_keep = max(min_targets, int(len(targets) * keep_top_fraction))
        elif max_targets is not None:
            n_keep = min(len(targets), max_targets)
        else:
            n_keep = len(targets)

        # Take top n_keep targets
        targets = targets[:n_keep]
        modes = modes[:n_keep]
        likelihoods = likelihoods[:n_keep]

        # Create new regulon
        new_regulon = Regulon(regulon.tf_name, description=regulon.description)
        for target, mode, likelihood in zip(targets, modes, likelihoods):
            new_regulon.add_target(target, mode, likelihood)

        pruned.append(new_regulon)

    return pruned


def create_regulon_from_network(tf_name: str,
                               targets: List[str],
                               modes: Optional[List[float]] = None,
                               likelihoods: Optional[List[float]] = None,
                               description: Optional[str] = None) -> Regulon:
    """
    Create a regulon from network data.

    Parameters
    ----------
    tf_name : str
        Name of the transcription factor
    targets : list of str
        List of target gene names
    modes : list of float, optional
        List of regulation modes (positive for activation, negative for repression)
        If None, all modes will be set to 1.0
    likelihoods : list of float, optional
        List of likelihood/confidence scores
        If None, all likelihoods will be set to 1.0
    description : str, optional
        Description of the regulon

    Returns
    -------
    Regulon object

    Examples
    --------
    >>> import pysces
    >>> # Create a regulon with all positive modes
    >>> regulon = pysces.create_regulon_from_network('STAT1', ['IRF1', 'CXCL10', 'IFIT1'])
    >>> # Create a regulon with custom modes and likelihoods
    >>> regulon = pysces.create_regulon_from_network(
    ...     'STAT1',
    ...     ['IRF1', 'CXCL10', 'IFIT1'],
    ...     modes=[1.0, 1.0, -0.5],
    ...     likelihoods=[0.9, 0.8, 0.7]
    ... )
    """
    # Create a new regulon
    regulon = Regulon(tf_name, description=description)

    # Set default modes and likelihoods if not provided
    if modes is None:
        modes = [1.0] * len(targets)
    if likelihoods is None:
        likelihoods = [1.0] * len(targets)

    # Check that lengths match
    if len(targets) != len(modes) or len(targets) != len(likelihoods):
        raise ValueError("Length of targets, modes, and likelihoods must match.")

    # Add targets to the regulon
    for target, mode, likelihood in zip(targets, modes, likelihoods):
        regulon.add_target(target, mode, likelihood)

    return regulon
