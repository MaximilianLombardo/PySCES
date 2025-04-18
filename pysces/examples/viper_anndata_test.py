"""
Test of the VIPER implementation with a real AnnData object from Tabula Sapiens.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the VIPER implementation
from pysces.viper import Regulon, GeneSet, viper

def main():
    # Load the real Tabula Sapiens dataset
    print("Loading Tabula Sapiens dataset...")
    data_path = "tabula_sapiens_testis.h5ad"
    if not os.path.exists(data_path):
        # Try alternative paths
        if os.path.exists("./tabula_sapiens_testis.h5ad"):
            data_path = "./tabula_sapiens_testis.h5ad"
        else:
            raise FileNotFoundError("Could not find tabula_sapiens_testis.h5ad file")
    
    # Load the data
    adata = sc.read_h5ad(data_path)
    print(f"Original dataset shape: {adata.shape}")
    
    # Aggressively subsample to make it run faster
    np.random.seed(42)  # For reproducibility
    # Use only 100 cells and 500 genes
    if adata.n_obs > 100:
        cell_indices = np.random.choice(adata.n_obs, 100, replace=False)
        adata = adata[cell_indices]
    if adata.n_vars > 500:
        gene_indices = np.random.choice(adata.n_vars, 500, replace=False)
        adata = adata[:, gene_indices]
    
    print(f"Subsampled dataset shape: {adata.shape}")
    
    # Create some simple regulons from the dataset
    print("Creating regulons...")
    # Select 20 random genes as TFs
    tf_indices = np.random.choice(adata.n_vars, 20, replace=False)
    tf_names = [adata.var_names[i] for i in tf_indices]
    
    # Create regulons
    regulons = []
    for tf in tf_names:
        # Select random targets (excluding the TF itself)
        n_targets = np.random.randint(10, 30)
        potential_targets = [g for g in adata.var_names if g != tf]
        if len(potential_targets) > n_targets:
            targets = np.random.choice(potential_targets, n_targets, replace=False)
            
            # Create random modes and likelihoods
            modes = np.random.uniform(-1, 1, n_targets)
            likelihoods = np.random.uniform(0, 1, n_targets)
            
            # Create regulon
            regulon = Regulon(
                tf_name=tf,
                targets={t: m for t, m in zip(targets, modes)},
                likelihood={t: l for t, l in zip(targets, likelihoods)}
            )
            
            regulons.append(regulon)
    
    print(f"Created {len(regulons)} regulons")
    
    # Run VIPER
    print("Running VIPER...")
    activity = viper(
        adata,
        regulons,
        method='gsea',
        signature_method='rank',
        normalize=True
    )
    
    # Print results
    print(f"VIPER activity matrix shape: {activity.shape}")
    print("Top 5 TFs by average activity:")
    avg_activity = activity.mean(axis=1).sort_values(ascending=False)
    for i, (tf, act) in enumerate(avg_activity.head(5).items()):
        print(f"{i+1}. {tf}: {act:.3f}")
    
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(activity.values, cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='VIPER Activity')
    plt.title('VIPER Activity on Tabula Sapiens Data')
    plt.xlabel('Cells')
    plt.ylabel('Transcription Factors')
    plt.xticks(range(0, activity.shape[1], 10))
    plt.yticks(range(activity.shape[0]), activity.index)
    plt.tight_layout()
    plt.savefig('viper_anndata_test.png')
    
    print("Done! Results saved to viper_anndata_test.png")

if __name__ == "__main__":
    main()
