"""
Master regulator analysis functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from scipy import stats

def identify_mrs(
    activity_matrix: pd.DataFrame,
    groups: pd.Series,
    group_of_interest: str,
    min_cells: int = 10,
    n_perms: int = 1000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Identify master regulators specific to a cell group.
    
    Parameters
    ----------
    activity_matrix : DataFrame
        Protein activity matrix (proteins x cells)
    groups : Series
        Cell group assignments
    group_of_interest : str
        Group to compare against others
    min_cells : int, default=10
        Minimum cells required in each group
    n_perms : int, default=1000
        Number of permutations for null model
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    DataFrame with MR statistics
    
    Examples
    --------
    >>> import pysces
    >>> adata = pysces.read_census("homo_sapiens", max_cells=1000)
    >>> adata = pysces.preprocess_data(adata)
    >>> aracne = pysces.ARACNe()
    >>> network = aracne.run(adata)
    >>> regulons = pysces.aracne_to_regulons(network)
    >>> activity = pysces.viper(adata, regulons)
    >>> adata = pysces.cluster_activity(adata, activity)
    >>> mrs = pysces.identify_mrs(activity, adata.obs["activity_clusters"], "0")
    """
    # Check cell counts
    group_cells = groups[groups == group_of_interest].index
    other_cells = groups[groups != group_of_interest].index
    
    if len(group_cells) < min_cells or len(other_cells) < min_cells:
        raise ValueError(f"Insufficient cells for comparison: {len(group_cells)} vs {len(other_cells)}")
    
    # Extract activities
    group_act = activity_matrix[group_cells]
    other_act = activity_matrix[other_cells]
    
    # Set random seed
    np.random.seed(random_state)
    
    # Calculate statistics
    results = []
    for tf in activity_matrix.index:
        # Calculate t-test
        t_stat, p_val = stats.ttest_ind(
            group_act.loc[tf], 
            other_act.loc[tf],
            equal_var=False
        )
        
        # Calculate effect size
        mean_diff = group_act.loc[tf].mean() - other_act.loc[tf].mean()
        
        # Calculate permutation p-value
        if n_perms > 0:
            # Combine data
            all_values = np.concatenate([group_act.loc[tf].values, other_act.loc[tf].values])
            n_group = len(group_act.loc[tf])
            n_other = len(other_act.loc[tf])
            
            # Calculate observed difference
            obs_diff = np.mean(all_values[:n_group]) - np.mean(all_values[n_group:])
            
            # Permutation test
            perm_diffs = np.zeros(n_perms)
            for i in range(n_perms):
                np.random.shuffle(all_values)
                perm_diffs[i] = np.mean(all_values[:n_group]) - np.mean(all_values[n_group:])
            
            # Calculate p-value
            perm_p_val = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
        else:
            perm_p_val = np.nan
        
        results.append({
            'tf': tf,
            't_stat': t_stat,
            'p_value': p_val,
            'perm_p_value': perm_p_val,
            'mean_diff': mean_diff
        })
    
    # Create DataFrame and sort by significance
    result_df = pd.DataFrame(results)
    result_df['adj_p_value'] = stats.multipletests(
        result_df['p_value'], method='fdr_bh'
    )[1]
    
    result_df = result_df.sort_values('adj_p_value')
    
    return result_df

def cluster_mrs(
    activity_matrix: pd.DataFrame,
    clusters: pd.Series,
    top_n: int = 10,
    min_cells: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Identify master regulators for each cluster.
    
    Parameters
    ----------
    activity_matrix : DataFrame
        Protein activity matrix (proteins x cells)
    clusters : Series
        Cluster assignments for cells
    top_n : int, default=10
        Number of top MRs to return for each cluster
    min_cells : int, default=10
        Minimum cells required in each cluster
        
    Returns
    -------
    Dictionary mapping cluster names to DataFrames of MRs
    
    Examples
    --------
    >>> import pysces
    >>> adata = pysces.read_census("homo_sapiens", max_cells=1000)
    >>> adata = pysces.preprocess_data(adata)
    >>> aracne = pysces.ARACNe()
    >>> network = aracne.run(adata)
    >>> regulons = pysces.aracne_to_regulons(network)
    >>> activity = pysces.viper(adata, regulons)
    >>> adata = pysces.cluster_activity(adata, activity)
    >>> cluster_mrs = pysces.cluster_mrs(activity, adata.obs["activity_clusters"])
    """
    # Get unique clusters
    unique_clusters = clusters.unique()
    
    # Initialize results
    results = {}
    
    # Identify MRs for each cluster
    for cluster in unique_clusters:
        try:
            mrs = identify_mrs(activity_matrix, clusters, cluster, min_cells=min_cells)
            results[cluster] = mrs.head(top_n)
        except ValueError as e:
            print(f"Skipping cluster {cluster}: {e}")
    
    return results
