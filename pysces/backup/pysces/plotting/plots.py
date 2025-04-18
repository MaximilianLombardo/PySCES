"""
Visualization functions for PySCES.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
from typing import Dict, List, Tuple, Union, Optional

def plot_activity_umap(
    adata: ad.AnnData,
    activity_matrix: pd.DataFrame,
    tfs: Union[str, List[str]],
    cluster_key: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 10),
    cmap: str = 'viridis',
    ncols: int = 3,
    save: Optional[str] = None
) -> plt.Figure:
    """
    Plot protein activity on UMAP embedding.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with UMAP coordinates
    activity_matrix : DataFrame
        Protein activity matrix (proteins x cells)
    tfs : str or list of str
        Transcription factor(s) to plot
    cluster_key : str, optional
        Key in adata.obs for cluster assignments
    figsize : tuple, default=(12, 10)
        Figure size
    cmap : str, default='viridis'
        Colormap for activity values
    ncols : int, default=3
        Number of columns in the plot grid
    save : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib Figure
    
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
    >>> fig = pysces.plot_activity_umap(adata, activity, ["STAT1", "IRF1", "GATA3"])
    """
    # Check if UMAP coordinates are available
    if 'X_umap' not in adata.obsm:
        raise ValueError("UMAP coordinates not found in adata.obsm['X_umap']")
    
    # Convert single TF to list
    if isinstance(tfs, str):
        tfs = [tfs]
    
    # Check if all TFs are in the activity matrix
    missing_tfs = [tf for tf in tfs if tf not in activity_matrix.index]
    if missing_tfs:
        raise ValueError(f"TFs not found in activity matrix: {missing_tfs}")
    
    # Get UMAP coordinates
    umap_coords = adata.obsm['X_umap']
    
    # Create figure
    nrows = int(np.ceil(len(tfs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each TF
    for i, tf in enumerate(tfs):
        ax = axes[i]
        
        # Get activity values
        activity_values = activity_matrix.loc[tf]
        
        # Plot UMAP with activity values
        scatter = ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=activity_values,
            cmap=cmap,
            s=5,
            alpha=0.7
        )
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Activity')
        
        # Add title
        ax.set_title(tf)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
    
    # If cluster_key is provided, add a plot with cluster assignments
    if cluster_key is not None and cluster_key in adata.obs:
        if i + 1 < len(axes):
            ax = axes[i + 1]
            
            # Get cluster assignments
            clusters = adata.obs[cluster_key]
            
            # Plot UMAP with cluster assignments
            scatter = ax.scatter(
                umap_coords[:, 0],
                umap_coords[:, 1],
                c=clusters.astype('category').cat.codes,
                cmap='tab20',
                s=5,
                alpha=0.7
            )
            
            # Add legend
            handles, labels = scatter.legend_elements()
            ax.legend(handles, clusters.unique(), title=cluster_key, loc='center left', bbox_to_anchor=(1, 0.5))
            
            # Add title
            ax.set_title(f'Clusters ({cluster_key})')
            ax.set_xlabel('UMAP1')
            ax.set_ylabel('UMAP2')
    
    # Hide empty subplots
    for j in range(i + 1 + (1 if cluster_key is not None else 0), len(axes)):
        axes[j].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    return fig

def plot_activity_heatmap(
    activity_matrix: pd.DataFrame,
    clusters: pd.Series,
    top_n: int = 10,
    figsize: Tuple[float, float] = (12, 10),
    cmap: str = 'viridis',
    cluster_cmap: str = 'tab20',
    save: Optional[str] = None
) -> plt.Figure:
    """
    Plot heatmap of protein activity across clusters.
    
    Parameters
    ----------
    activity_matrix : DataFrame
        Protein activity matrix (proteins x cells)
    clusters : Series
        Cluster assignments for cells
    top_n : int, default=10
        Number of top differentially active proteins per cluster to show
    figsize : tuple, default=(12, 10)
        Figure size
    cmap : str, default='viridis'
        Colormap for activity values
    cluster_cmap : str, default='tab20'
        Colormap for cluster annotations
    save : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib Figure
    
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
    >>> fig = pysces.plot_activity_heatmap(activity, adata.obs["activity_clusters"])
    """
    from pysces.analysis.master_regulators import cluster_mrs
    
    # Get top MRs for each cluster
    cluster_mrs_dict = cluster_mrs(activity_matrix, clusters, top_n=top_n)
    
    # Get unique TFs across all clusters
    top_tfs = set()
    for cluster, mrs in cluster_mrs_dict.items():
        top_tfs.update(mrs['tf'].values)
    
    # Create a new activity matrix with only the top TFs
    top_activity = activity_matrix.loc[list(top_tfs)]
    
    # Calculate mean activity per cluster
    cluster_means = pd.DataFrame(index=top_activity.index)
    for cluster in clusters.unique():
        cluster_cells = clusters[clusters == cluster].index
        cluster_means[cluster] = top_activity[cluster_cells].mean(axis=1)
    
    # Z-score normalize
    cluster_means = (cluster_means - cluster_means.mean(axis=1).values.reshape(-1, 1)) / \
                    cluster_means.std(axis=1).values.reshape(-1, 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cluster_means,
        cmap=cmap,
        center=0,
        ax=ax,
        cbar_kws={'label': 'Z-score'}
    )
    
    # Add title
    ax.set_title('Top Master Regulators by Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Transcription Factor')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    return fig

def plot_master_regulators(
    mrs_df: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[float, float] = (10, 8),
    save: Optional[str] = None
) -> plt.Figure:
    """
    Plot master regulator analysis results.
    
    Parameters
    ----------
    mrs_df : DataFrame
        Master regulator analysis results from identify_mrs
    top_n : int, default=20
        Number of top MRs to show
    figsize : tuple, default=(10, 8)
        Figure size
    save : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib Figure
    
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
    >>> fig = pysces.plot_master_regulators(mrs)
    """
    # Sort by significance
    mrs_df = mrs_df.sort_values('adj_p_value')
    
    # Take top N
    top_mrs = mrs_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot as bar chart
    bars = ax.barh(
        top_mrs['tf'][::-1],
        top_mrs['mean_diff'][::-1],
        color=np.where(top_mrs['mean_diff'][::-1] > 0, 'red', 'blue'),
        alpha=0.7
    )
    
    # Add significance markers
    for i, (_, row) in enumerate(top_mrs[::-1].iterrows()):
        if row['adj_p_value'] < 0.001:
            ax.text(row['mean_diff'] * (1.1 if row['mean_diff'] > 0 else 0.9), i, '***', ha='center', va='center')
        elif row['adj_p_value'] < 0.01:
            ax.text(row['mean_diff'] * (1.1 if row['mean_diff'] > 0 else 0.9), i, '**', ha='center', va='center')
        elif row['adj_p_value'] < 0.05:
            ax.text(row['mean_diff'] * (1.1 if row['mean_diff'] > 0 else 0.9), i, '*', ha='center', va='center')
    
    # Add labels
    ax.set_xlabel('Mean Activity Difference')
    ax.set_ylabel('Transcription Factor')
    ax.set_title('Top Master Regulators')
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add zero line
    ax.axvline(0, color='black', linestyle='-', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    return fig
