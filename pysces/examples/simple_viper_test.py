"""
Simple test of the VIPER implementation.
"""

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the VIPER implementation
from pysces.viper import Regulon, GeneSet, viper

def main():
    # Create a simple test dataset
    print("Creating test dataset...")
    n_genes = 100
    n_cells = 50
    n_tfs = 10
    
    # Create gene names
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    
    # Create TF names (subset of genes)
    tf_indices = np.random.choice(n_genes, n_tfs, replace=False)
    tf_names = [gene_names[i] for i in tf_indices]
    
    # Create expression data
    expr = np.random.normal(0, 1, (n_cells, n_genes))
    
    # Create AnnData object
    adata = ad.AnnData(
        X=expr,
        var=pd.DataFrame(index=gene_names),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    )
    
    # Create regulons
    print("Creating regulons...")
    regulons = []
    for tf in tf_names:
        # Select random targets
        n_targets = np.random.randint(10, 30)
        targets = np.random.choice(
            [g for g in gene_names if g != tf],
            n_targets,
            replace=False
        )
        
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
    plt.title('VIPER Activity')
    plt.xlabel('Cells')
    plt.ylabel('Transcription Factors')
    plt.xticks(range(0, n_cells, 10))
    plt.yticks(range(n_tfs), activity.index)
    plt.tight_layout()
    plt.savefig('simple_viper_test.png')
    
    print("Done! Results saved to simple_viper_test.png")

if __name__ == "__main__":
    main()
