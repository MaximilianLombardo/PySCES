"""
PySCES: Python Single-Cell Expression System
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pysces")
except PackageNotFoundError:
    __version__ = "0.2.0-dev"

# Import key functions for top-level API
from .data import read_anndata, read_census, read_census_direct, preprocess_data, rank_transform
from .aracne import ARACNe, aracne_to_regulons
from .viper import viper, metaviper
from .analysis import cluster_activity, identify_mrs
