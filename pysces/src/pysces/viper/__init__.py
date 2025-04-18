"""
VIPER (Virtual Inference of Protein-activity by Enriched Regulon analysis) module.

This module provides functionality for inferring protein activity from gene expression data
using the VIPER algorithm. It includes classes for representing regulons and gene sets,
functions for calculating enrichment scores, and utilities for integrating with ARACNe.
"""

from .regulons import (
    Regulon,
    GeneSet,
    prune_regulons,
    create_regulon_from_network
)

from .activity import (
    viper,
    metaviper,
    viper_bootstrap_wrapper as viper_bootstrap,
    viper_null_model_wrapper as viper_null_model,
    viper_similarity_wrapper as viper_similarity,
    viper_cluster_wrapper as viper_cluster
)

from .aracne_integration import (
    aracne_to_regulons,
    aracne_to_viper
)

__all__ = [
    # Core classes
    "Regulon",
    "GeneSet",

    # Regulon utilities
    "prune_regulons",
    "create_regulon_from_network",

    # Core VIPER functions
    "viper",
    "metaviper",
    "viper_bootstrap",
    "viper_null_model",
    "viper_similarity",
    "viper_cluster",

    # ARACNe integration
    "aracne_to_regulons",
    "aracne_to_viper"
]
