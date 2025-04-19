"""
Test script for the ARACNe and VIPER pipeline with validation.
"""

import pysces
import numpy as np
import pandas as pd
import anndata as ad
import logging
import scipy.sparse

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    # Create a synthetic dataset
    print("Creating synthetic dataset...")
    n_cells = 100
    n_genes = 200

    # Create random count data
    X = np.random.randint(0, 10, size=(n_cells, n_genes))

    # Create gene names with some TFs
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    tf_names = [f"TF_{i}" for i in range(20)]
    gene_names[:20] = tf_names  # First 20 genes are TFs

    # Create cell names
    cell_names = [f"cell_{i}" for i in range(n_cells)]

    # Create AnnData object
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=gene_names)
    )

    print(f"Dataset shape: {adata.shape}")

    # Validate the dataset
    print("\n=== Validating dataset ===")
    is_valid, issues = pysces.validate_anndata_structure(adata)
    if not is_valid:
        print("❌ AnnData structure has issues:")
        for issue in issues:
            print(f"  - {issue}")
        return
    else:
        print("✅ AnnData structure is valid")

    # Run the ARACNe algorithm with validation
    print("\n=== Running ARACNe with validation ===")
    try:
        aracne = pysces.ARACNe(
            bootstraps=2,  # Small number for quick testing
            p_value=0.05,  # Less stringent for testing
            dpi_tolerance=0.1,
            use_gpu=True  # Force Python implementation
        )
        network = aracne.run(adata, tf_list=tf_names, validate=True)
        print(f"✅ ARACNe completed successfully")
        print(f"Network has {len(network['edges'])} edges")
    except Exception as e:
        print(f"❌ ARACNe failed: {str(e)}")
        return

    # Convert network to regulons
    print("\n=== Converting network to regulons ===")
    regulons = pysces.aracne_to_regulons(network)
    print(f"Created {len(regulons)} regulons")

    # Run VIPER with validation
    print("\n=== Running VIPER with validation ===")
    try:
        activity = pysces.viper(
            adata,
            regulons,
            method='gsea',
            validate=True
        )
        print(f"✅ VIPER completed successfully")
        print(f"Activity matrix shape: {activity.shape}")
    except Exception as e:
        print(f"❌ VIPER failed: {str(e)}")
        return

    # Test with invalid data
    print("\n=== Testing with invalid data ===")

    # Create an AnnData with duplicate gene names
    gene_names_dup = gene_names.copy()
    gene_names_dup[0] = gene_names_dup[1]  # Create a duplicate
    adata_invalid = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=gene_names_dup)
    )

    print("Testing ARACNe with duplicate gene names...")
    try:
        network = aracne.run(adata_invalid, tf_list=tf_names, validate=True)
        print("❌ ARACNe did not catch the invalid data")
    except Exception as e:
        print(f"✅ ARACNe correctly caught the error: {str(e)}")

    print("\nTesting VIPER with duplicate gene names...")
    try:
        activity = pysces.viper(adata_invalid, regulons, validate=True)
        print("❌ VIPER did not catch the invalid data")
    except Exception as e:
        print(f"✅ VIPER correctly caught the error: {str(e)}")

    print("\n=== Pipeline validation test complete ===")

if __name__ == "__main__":
    main()
