"""
Example of using ARACNe with CELLxGENE Census data.

This script demonstrates how to:
1. Load data from Census using the recommended direct loading method
2. Preprocess the data
3. Run ARACNe to infer the regulatory network
4. Convert the network to regulons
5. Visualize the results

Note: This example uses the read_census_direct function, which is the recommended
approach for loading Census data. The batch processing approach (read_census) is
currently experimental and may not work with the latest cellxgene-census API.
"""

# Standard imports
import os
import sys

# Import from the pysces package
import pysces
from pysces import read_census_direct, preprocess_data, ARACNe, aracne_to_regulons

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import time

def main():
    # Check if Census is available
    try:
        import cellxgene_census
        import tiledbsoma
    except ImportError:
        print("CELLxGENE Census dependencies not found.")
        print("This example requires cellxgene-census and tiledb-soma packages.")
        print("Using a small synthetic dataset instead.")
        use_census = False
    else:
        use_census = True
    
    # Load data
    if use_census:
        print("\n=== Census Data Loading ===")
        print("Loading testis cells from Census using direct loading...")
        start_time = time.time()
        adata = read_census_direct(
            "homo_sapiens",
            obs_value_filter="tissue_general == 'testis'",
            obs_column_names=["assay", "cell_type", "tissue", "tissue_general", "sex", "disease"],
            # Limit to specific genes for faster processing
            var_value_filter="feature_name in ['SOX2', 'SOX9', 'FOXP1', 'GATA1', 'PAX6', 'NEUROD1', 'NEUROG2', 'ASCL1', 'OLIG2', 'NKX2-2']",
            census_version="latest"  # Specify the census version
        )
        print(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes in {time.time() - start_time:.2f} seconds.")
        print(f"Metadata columns: {list(adata.obs.columns)}")
        print(f"Genes: {list(adata.var_names)}")
    else:
        # Create synthetic data if Census is not available
        print("Creating synthetic dataset...")
        n_genes = 200
        n_cells = 500
        
        # Generate random expression data
        np.random.seed(42)
        expr_matrix = np.random.negative_binomial(5, 0.3, (n_genes, n_cells))
        
        # Create gene names
        gene_names = [f"gene_{i}" for i in range(n_genes)]
        
        # Create AnnData object
        adata = sc.AnnData(
            X=expr_matrix.T,  # Transpose to cells x genes
            var=pd.DataFrame(index=gene_names)
        )
        print(f"Created synthetic dataset with {adata.n_obs} cells and {adata.n_vars} genes.")
    
    # Preprocess data
    print("\n=== Data Preprocessing ===")
    print("Preprocessing data...")
    adata = preprocess_data(adata)
    
    # Get list of transcription factors
    if use_census:
        # Load a list of human TFs
        # For this example, we'll use a simple approach - in practice, you'd want a curated list
        tf_list = [gene for gene in adata.var_names if gene.startswith("SOX") or 
                                                      gene.startswith("FOX") or 
                                                      gene.startswith("GATA") or
                                                      gene.startswith("PAX")]
        print(f"Using {len(tf_list)} transcription factors.")
    else:
        # For synthetic data, use the first 20 genes as TFs
        tf_list = [f"gene_{i}" for i in range(20)]
    
    # Run ARACNe
    print("\n=== ARACNe Network Inference ===")
    print("Running ARACNe...")
    aracne = ARACNe(
        p_value=0.01,        # Less stringent for this example
        bootstraps=10,       # Fewer bootstraps for faster processing
        dpi_tolerance=0.1,   # Tolerance for DPI
        n_threads=0          # Auto-detect number of threads
    )
    network = aracne.run(adata, tf_list=tf_list)
    
    # Convert to regulons
    print("Converting network to regulons...")
    regulons = aracne_to_regulons(network)
    
    # Print some statistics
    print(f"Inferred {len(regulons)} regulons.")
    total_targets = sum(len(regulon.targets) for regulon in regulons)
    print(f"Total targets: {total_targets}")
    print(f"Average targets per TF: {total_targets / len(regulons):.1f}")
    
    # Print the top 5 regulons by number of targets
    print("\nTop 5 regulons by number of targets:")
    top_regulons = sorted(regulons, key=lambda r: len(r.targets), reverse=True)[:5]
    for i, regulon in enumerate(top_regulons):
        print(f"{i+1}. {regulon.tf_name}: {len(regulon.targets)} targets")
    
    # Visualize the network
    print("\n=== Network Visualization ===")
    
    # Create a heatmap of the top 10 TFs and their top 20 targets
    if len(regulons) >= 10:
        top_tfs = sorted(regulons, key=lambda r: len(r.targets), reverse=True)[:10]
        
        # Create a matrix of TF-target interactions
        tf_names = [regulon.tf_name for regulon in top_tfs]
        all_targets = set()
        for regulon in top_tfs:
            if len(regulon.targets) > 0:
                # Get top 20 targets by absolute mode value
                top_targets = sorted(regulon.targets.items(), 
                                    key=lambda x: abs(x[1]), 
                                    reverse=True)[:20]
                all_targets.update(target for target, _ in top_targets)
        
        target_names = sorted(list(all_targets))
        
        # Create interaction matrix
        interaction_matrix = np.zeros((len(tf_names), len(target_names)))
        for i, regulon in enumerate(top_tfs):
            for j, target in enumerate(target_names):
                if target in regulon.targets:
                    interaction_matrix[i, j] = regulon.targets[target]
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(interaction_matrix, cmap="coolwarm", center=0,
                    xticklabels=target_names, yticklabels=tf_names)
        plt.title("Top TF-Target Interactions")
        plt.xlabel("Targets")
        plt.ylabel("Transcription Factors")
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        # Save figure
        plt.savefig("aracne_network_heatmap.png")
        print("Saved network visualization to aracne_network_heatmap.png")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
