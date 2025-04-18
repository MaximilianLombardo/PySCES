"""
CELLxGENE Census data loading functionality.

This module provides functions for loading data from CELLxGENE Census into AnnData format.
It offers two approaches:
1. read_census_direct: Uses get_anndata for direct loading of datasets (recommended)
2. read_census: Uses batch processing for memory-efficient loading (experimental)

Note: The batch processing approach (read_census) is currently experimental and has
compatibility issues with the latest cellxgene-census API. It will be updated in a
future release to use the more stable TileDB-SOMA-ML API.
"""

import anndata as ad
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List, Any
import importlib.util
import warnings

# Check if required packages are available
_has_census = importlib.util.find_spec("cellxgene_census") is not None
_has_soma = importlib.util.find_spec("tiledbsoma") is not None

def read_census(
    experiment_id: str,
    measurement: str = "RNA",
    obs_query: Optional[str] = None,
    batch_size: int = 128,
    max_cells: Optional[int] = None,
    census_version: Optional[str] = None
) -> ad.AnnData:
    """
    Load data from CELLxGENE Census into AnnData format using batch processing.
    
    WARNING: This function is experimental and may not work with the latest
    cellxgene-census API. It will be updated in a future release to use the
    more stable TileDB-SOMA-ML API. For now, consider using read_census_direct
    instead.
    
    This function is designed for large datasets by processing data in batches.
    
    Parameters
    ----------
    experiment_id : str
        The ID of the experiment in Census (e.g., "homo_sapiens")
    measurement : str, default="RNA"
        The measurement to extract
    obs_query : str, optional
        Filter expression for cell selection
    batch_size : int, default=128
        Size of batches for loading
    max_cells : int, optional
        Maximum number of cells to load. If None, load all cells.
    census_version : str, optional
        Census version to use for data consistency
        
    Returns
    -------
    AnnData object with loaded data
    
    Examples
    --------
    >>> import pysces
    >>> # Load blood cells from human census
    >>> adata = pysces.read_census("homo_sapiens", obs_query="tissue_general == 'blood'")
    """
    # Check if required packages are available
    if not _has_census or not _has_soma:
        missing = []
        if not _has_census:
            missing.append("cellxgene-census")
        if not _has_soma:
            missing.append("tiledb-soma")
        
        error_msg = (
            f"Cannot load data from CELLxGENE Census. "
            f"Missing required packages: {', '.join(missing)}. "
            f"Please install them with: pip install {' '.join(missing)}"
        )
        raise ImportError(error_msg)
    
    # Import required packages
    import cellxgene_census
    import tiledbsoma as soma
    import cellxgene_census.experimental.ml as census_ml
    
    try:
        # Connect to Census
        census = cellxgene_census.open_soma(census_version=census_version)
        experiment = census["census_data"][experiment_id]
        
        # Create query
        query_obj = None
        if obs_query:
            query_obj = soma.AxisQuery(value_filter=obs_query)
        
        # Create data pipe
        experiment_datapipe = census_ml.ExperimentDataPipe(
            experiment,
            measurement_name=measurement,
            X_name="raw",
            obs_query=query_obj,
            obs_column_names=["soma_joinid", "assay", "cell_type", "tissue", "tissue_general", "sex", "disease"],  # Specify columns explicitly
            batch_size=batch_size,
            shuffle=False,
        )
        
        # Load data into memory (for now, we'll implement streaming later)
        # This is a simplification - for large datasets, we'll need chunking
        X_data = []
        obs_data = []
        var_data = None
        
        cell_count = 0
        for batch in experiment_datapipe:
            X, obs_batch = batch
            X_data.append(X)
            obs_data.append(obs_batch)
            
            # Store var_names from first batch
            if var_data is None and hasattr(experiment_datapipe, 'var_names'):
                var_data = pd.DataFrame(index=experiment_datapipe.var_names)
            
            cell_count += X.shape[0]
            if max_cells is not None and cell_count >= max_cells:
                # Truncate the last batch if needed
                excess = cell_count - max_cells
                if excess > 0:
                    X_data[-1] = X_data[-1][:-excess]
                    obs_data[-1] = obs_data[-1][:-excess]
                break
        
        # Concatenate batches
        X_combined = np.vstack(X_data)
        obs_combined = pd.concat(obs_data)
        
        # Create AnnData object
        adata = ad.AnnData(
            X=X_combined, 
            obs=obs_combined,
            var=var_data
        )
        
        return adata
    
    except Exception as e:
        raise RuntimeError(f"Error loading data from CELLxGENE Census: {str(e)}")
    finally:
        # Ensure the Census connection is closed
        if 'census' in locals():
            census.close()


def read_census_direct(
    organism: str,
    obs_value_filter: Optional[str] = None,
    var_value_filter: Optional[str] = None,
    obs_column_names: Optional[List[str]] = None,
    var_column_names: Optional[List[str]] = None,
    measurement: str = "RNA",
    X_name: str = "raw",
    census_version: Optional[str] = None
) -> ad.AnnData:
    """
    Load data from CELLxGENE Census directly into AnnData format using get_anndata.
    
    This function provides a simpler interface for smaller datasets. For larger
    datasets, consider using read_census() which uses batch processing.
    
    Parameters
    ----------
    organism : str
        The organism name (e.g., "Homo sapiens") or ID (e.g., "homo_sapiens")
    obs_value_filter : str, optional
        Filter expression for cell selection
    var_value_filter : str, optional
        Filter expression for gene selection
    obs_column_names : list of str, optional
        Cell metadata columns to include
    var_column_names : list of str, optional
        Gene metadata columns to include
    measurement : str, default="RNA"
        The measurement to extract
    X_name : str, default="raw"
        The matrix to extract (e.g., "raw", "normalized")
    census_version : str, optional
        Census version to use for data consistency
        
    Returns
    -------
    AnnData object with loaded data
    
    Examples
    --------
    >>> import pysces
    >>> # Load specific genes for female microglial cells and neurons
    >>> adata = pysces.read_census_direct(
    ...     "homo_sapiens",
    ...     obs_value_filter="sex == 'female' and cell_type in ['microglial cell', 'neuron']",
    ...     var_value_filter="feature_id in ['ENSG00000161798', 'ENSG00000188229']",
    ...     obs_column_names=["assay", "cell_type", "tissue", "tissue_general"]
    ... )
    """
    # Check if required packages are available
    if not _has_census or not _has_soma:
        missing = []
        if not _has_census:
            missing.append("cellxgene-census")
        if not _has_soma:
            missing.append("tiledb-soma")
        
        error_msg = (
            f"Cannot load data from CELLxGENE Census. "
            f"Missing required packages: {', '.join(missing)}. "
            f"Please install them with: pip install {' '.join(missing)}"
        )
        raise ImportError(error_msg)
    
    # Import required packages
    import cellxgene_census
    import tiledbsoma as soma
    
    # Normalize organism name/ID
    if organism.lower() in ["homo sapiens", "human"]:
        organism_id = "homo_sapiens"
    elif organism.lower() in ["mus musculus", "mouse"]:
        organism_id = "mus_musculus"
    else:
        organism_id = organism.lower().replace(" ", "_")
    
    # Prepare column names
    column_names = {}
    if obs_column_names is not None:
        column_names["obs"] = obs_column_names
    if var_column_names is not None:
        column_names["var"] = var_column_names
    
    try:
        # Connect to Census
        with cellxgene_census.open_soma(census_version=census_version) as census:
            # Create obs query
            obs_query = None
            if obs_value_filter:
                obs_query = soma.AxisQuery(value_filter=obs_value_filter)
            
            # Create var query
            var_query = None
            if var_value_filter:
                var_query = soma.AxisQuery(value_filter=var_value_filter)
            
            # Get AnnData
            adata = cellxgene_census.get_anndata(
                census=census,
                organism=organism_id,
                measurement_name=measurement,
                X_name=X_name,
                obs_query=obs_query,
                var_query=var_query,
                column_names=column_names if column_names else None
            )
            
            return adata
    
    except Exception as e:
        raise RuntimeError(f"Error loading data from CELLxGENE Census: {str(e)}")
