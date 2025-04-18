"""
VIPER (Virtual Inference of Protein-activity by Enriched Regulon analysis) module.
"""

from .regulons import Regulon, prune_regulons
from .activity import viper, metaviper

__all__ = ["Regulon", "prune_regulons", "viper", "metaviper"]
