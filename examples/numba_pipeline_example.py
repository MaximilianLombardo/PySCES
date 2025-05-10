"""
Simple end-to-end ARACNe+VIPER pipeline example using the Numba backend.

This script demonstrates the complete workflow from ARACNe network inference
to VIPER enrichment analysis using a small synthetic dataset.
"""
import numpy as np
import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt

# Import PySCES components
from pysces.aracne.core import ARACNe, aracne_to_regulons
from pysces.viper.core import viper_scores

def create_synthetic_dataset(n_genes=100, n_cells=200, n_tfs=20, seed=42):
    """Create a synthetic dataset for demonstration."""
    np.random.seed(seed)
    
    # Create expression matrix with some structure
    X = np.zeros((n_cells, n_genes))
    
    # Create 5 different cell types with different expression patterns
    cell_types = np.random.choice(5, n_cells)
    
    # Create base expression
    for i in range(n_cells):
        # Base expression for this cell type
        base_expr = np.random.normal(loc=cell_types[i], scale=0.5, size=n_genes)
        
        # Add some noise
        noise = np.random.normal(scale=0.2, size=n_genes)
        
        # Combined expression
        X[i, :] = base_expr + noise
    
    # Make sure values are positive
    X = np.exp(X)
    
    # Create AnnData object
    obs = pd.DataFrame({
        'cell_type': [f'type_{cell_types[i]}' for i in range(n_cells)]
    }, index=[f'cell_{i}' for i in range(n_cells)])
    
    var = pd.DataFrame({
        'gene_type': ['TF' if i < n_tfs else 'target' for i in range(n_genes)]
    }, index=[f'gene_{i}' for i in range(n_genes)])
    
    adata = AnnData(X=X, obs=obs, var=var)
    
    # Create TF list
    tf_list = [f'gene_{i}' for i in range(n_tfs)]
    
    return adata, tf_list

def run_pipeline():
    """Run the complete ARACNe+VIPER pipeline."""
    print("Creating synthetic dataset...")
    adata, tf_list = create_synthetic_dataset()
    print(f"Created dataset with {adata.n_obs} cells and {adata.n_vars} genes")
    print(f"Using {len(tf_list)} TFs")
    
    # Initialize ARACNe with Numba backend
    print("\nInitializing ARACNe with Numba backend...")
    aracne = ARACNe(
        bootstraps=10,  # Use 10 bootstraps for this example
        backend='numba',
        consensus_threshold=0.3
    )
    
    # Run ARACNe
    print("Running ARACNe network inference...")
    network = aracne.run(adata, tf_list=tf_list)
    
    # Print network summary
    n_edges = len(network['edges'])
    n_regulons = len(network['regulons'])
    print(f"Network has {n_edges} edges and {n_regulons} regulons")
    
    # Convert to regulons
    print("\nConverting network to regulons...")
    regulons = aracne_to_regulons(network)
    print(f"Created {len(regulons)} regulons")
    
    # Print first regulon
    regulon = regulons[0]
    print(f"\nFirst regulon ({regulon.tf_name}) has {len(regulon.targets)} targets:")
    for target, mode in list(regulon.targets.items())[:5]:
        print(f"  {target}: {mode:.3f}")
    if len(regulon.targets) > 5:
        print(f"  ... and {len(regulon.targets) - 5} more targets")
    
    # Run VIPER
    print("\nRunning VIPER analysis...")
    viper_df = viper_scores(
        adata,
        regulons,
        signature_method='rank',
        enrichment_method='gsea',
        use_numba=True
    )
    
    print(f"VIPER result shape: {viper_df.shape}")
    
    # Calculate average activity by cell type
    print("\nCalculating average activity by cell type...")
    cell_types = adata.obs['cell_type'].unique()
    avg_activity = {}
    
    for cell_type in cell_types:
        cell_indices = adata.obs[adata.obs['cell_type'] == cell_type].index
        avg_activity[cell_type] = viper_df[cell_indices].mean(axis=1)
    
    # Plot results
    print("\nPlotting results...")
    plt.figure(figsize=(12, 8))
    
    for i, cell_type in enumerate(cell_types):
        plt.subplot(2, 3, i+1)
        avg_activity[cell_type].sort_values().tail(10).plot.barh()
        plt.title(f"Top 10 active TFs in {cell_type}")
        plt.tight_layout()
    
    plt.savefig("viper_activity_by_cell_type.png")
    print("Saved plot to viper_activity_by_cell_type.png")
    
    print("\nPipeline completed successfully!")
    
if __name__ == "__main__":
    run_pipeline()
