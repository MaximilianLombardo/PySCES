"""
Example of using VIPER with ARACNe and CELLxGENE Census data.

This script demonstrates how to:
1. Load data from Census
2. Preprocess the data
3. Run ARACNe to infer the regulatory network
4. Convert the network to regulons
5. Run VIPER to infer protein activity
6. Visualize the results
"""

# Standard imports
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import from the pysces package
import sys
import os

# Import directly from the package
from pysces.src.pysces.utils.preprocessing import preprocess_data
from pysces.src.pysces.aracne.core import ARACNe
from pysces.src.pysces.viper import (
    aracne_to_regulons,
    viper,
    viper_bootstrap_wrapper as viper_bootstrap,
    viper_null_model_wrapper as viper_null_model,
    viper_similarity_wrapper as viper_similarity,
    viper_cluster_wrapper as viper_cluster
)

def main():
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)

    # Load data from local Tabula Sapiens dataset
    print("Loading data from local Tabula Sapiens dataset...")
    import scanpy as sc
    import os

    # Check if the file exists
    data_path = "tabula_sapiens_testis.h5ad"
    if not os.path.exists(data_path):
        # Try alternative paths
        if os.path.exists("./tabula_sapiens_testis.h5ad"):
            data_path = "./tabula_sapiens_testis.h5ad"
        else:
            raise FileNotFoundError("Could not find tabula_sapiens_testis.h5ad file")

    # Load the data
    adata = sc.read_h5ad(data_path)

    # Subsample to a very small dataset for testing
    import numpy as np
    np.random.seed(42)  # For reproducibility
    # Use only 100 cells and 500 genes to make it run faster
    if adata.n_obs > 100:
        cell_indices = np.random.choice(adata.n_obs, 100, replace=False)
        adata = adata[cell_indices]
    if adata.n_vars > 500:
        gene_indices = np.random.choice(adata.n_vars, 500, replace=False)
        adata = adata[:, gene_indices]

    # Preprocess the data
    print("Preprocessing data...")
    adata = preprocess_data(adata)

    # Get a list of transcription factors
    print("Loading transcription factors...")
    # Since we don't have get_transcription_factors, let's use a simple approach
    # Just use a small random subset of genes as potential TFs
    import numpy as np
    np.random.seed(42)
    tf_list = np.random.choice(adata.var_names, size=20, replace=False).tolist()  # Use fewer TFs for faster execution
    print(f"Using {len(tf_list)} transcription factors")

    # Run ARACNe
    print("Running ARACNe...")
    aracne = ARACNe(p_value=1e-5, bootstraps=5, use_cpp=False)  # Use more bootstraps in practice
    print("Using Python/NumPy implementation for MI calculation")
    network = aracne.run(adata, tf_list=tf_list)

    # Convert network to regulons
    print("Converting network to regulons...")
    regulons = aracne_to_regulons(
        network,
        min_targets=10,
        max_targets=100,
        infer_mode=True,
        mode_method='correlation',
        expression_data=adata.X
    )

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

        # Print some targets and their modes
        print("  Sample targets:")
        for j, (target, mode) in enumerate(list(regulon.targets.items())[:5]):
            print(f"    - {target}: mode={mode:.3f}, likelihood={regulon.likelihood.get(target, 1.0):.3f}")

    # Run VIPER
    print("\nRunning VIPER...")
    activity = viper(
        adata,
        regulons,
        method='gsea',
        signature_method='rank',
        normalize=True
    )

    # Print some statistics
    print(f"VIPER activity matrix shape: {activity.shape}")
    print("Top 5 TFs by average activity:")
    avg_activity = activity.mean(axis=1).sort_values(ascending=False)
    for i, (tf, act) in enumerate(avg_activity.head(5).items()):
        print(f"{i+1}. {tf}: {act:.3f}")

    # Run VIPER with bootstrapping
    print("\nRunning VIPER with bootstrapping...")
    mean_activity, std_activity = viper_bootstrap(
        adata,
        regulons,
        n_bootstraps=10,  # Use more bootstraps in practice
        sample_fraction=0.8
    )

    # Run VIPER with null model
    print("\nRunning VIPER with null model...")
    activity_null, p_values = viper_null_model(
        adata,
        regulons,
        n_permutations=10  # Use more permutations in practice
    )

    # Calculate similarity between cells
    print("\nCalculating cell similarity...")
    similarity = viper_similarity(activity)

    # Cluster cells
    print("\nClustering cells...")
    clusters = viper_cluster(activity, n_clusters=5)

    # Add clusters to AnnData
    adata.obs['viper_clusters'] = pd.Categorical(clusters)

    # Visualize results
    print("\nVisualizing results...")

    # Plot heatmap of top TFs
    plt.figure(figsize=(12, 8))
    top_tfs = avg_activity.head(20).index
    sns.heatmap(
        activity.loc[top_tfs].T,
        cmap='RdBu_r',
        center=0,
        xticklabels=True,
        yticklabels=False
    )
    plt.title('VIPER Activity of Top 20 TFs')
    plt.xlabel('Transcription Factors')
    plt.ylabel('Cells')
    plt.tight_layout()
    plt.savefig('viper_activity_heatmap.png')

    # Plot cluster distribution
    plt.figure(figsize=(8, 6))
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    cluster_counts.plot(kind='bar')
    plt.title('Cell Distribution by VIPER Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Cells')
    plt.tight_layout()
    plt.savefig('viper_clusters.png')

    # Plot similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity.iloc[:100, :100],  # Just show a subset for visualization
        cmap='viridis',
        xticklabels=False,
        yticklabels=False
    )
    plt.title('Cell Similarity Based on VIPER Activity (Subset)')
    plt.xlabel('Cells')
    plt.ylabel('Cells')
    plt.tight_layout()
    plt.savefig('viper_similarity.png')

    print("\nDone! Results saved to:")
    print("- viper_activity_heatmap.png")
    print("- viper_clusters.png")
    print("- viper_similarity.png")

if __name__ == "__main__":
    main()
