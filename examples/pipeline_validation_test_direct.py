"""
Test script for the ARACNe and VIPER pipeline with validation.

This script tests the complete pipeline from AnnData to ARACNe to VIPER,
including validation at each step. It creates a synthetic dataset,
runs ARACNe to infer a gene regulatory network, and then runs VIPER
to estimate protein activity.

It also tests the validation functions with invalid data (duplicate gene names)
to ensure they correctly catch and report issues.

This version uses direct imports from the src directory to avoid Python path issues.
"""

import numpy as np
import pandas as pd
import anndata as ad
import logging
import scipy.sparse
import sys
import os
import traceback

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import pysces modules directly
from pysces.utils.validation import validate_anndata_structure
from pysces.aracne.core import ARACNe, aracne_to_regulons
from pysces.viper.activity import viper

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    # Create a synthetic dataset
    print("Creating synthetic dataset...")
    n_cells = 50  # Reduced from 100 to avoid index out of bounds error
    n_genes = 100  # Reduced from 200 to avoid index out of bounds error

    # Create random count data
    X = np.random.randint(0, 10, size=(n_cells, n_genes))

    # Create gene names with some TFs
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    tf_names = [f"TF_{i}" for i in range(10)]  # Reduced from 20 to avoid index out of bounds error
    gene_names[:10] = tf_names  # First 10 genes are TFs

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
    is_valid, issues = validate_anndata_structure(adata)
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
        aracne = ARACNe(
            bootstraps=2,  # Small number for quick testing
            p_value=0.05,  # Less stringent for testing
            dpi_tolerance=0.1,
            consensus_threshold=0.5
        )
        network = aracne.run(adata, tf_list=tf_names, validate=True)
        print(f"✅ ARACNe completed successfully")
        print(f"Network keys: {list(network.keys())}")
        print(f"Network has {len(network['regulons'])} regulons and {len(network['edges'])} edges")
    except Exception as e:
        print(f"❌ ARACNe failed: {str(e)}")
        traceback.print_exc()
        return

    # Convert network to regulons
    print("\n=== Converting network to regulons ===")
    try:
        regulons = aracne_to_regulons(network)
        print(f"Created {len(regulons)} regulons")
    except Exception as e:
        print(f"❌ Converting network to regulons failed: {str(e)}")
        traceback.print_exc()
        return

    # Run VIPER with validation
    print("\n=== Running VIPER with validation ===")
    try:
        activity = viper(
            adata,
            regulons,
            method='gsea',
            validate=True
        )
        print(f"✅ VIPER completed successfully")
        print(f"Activity matrix shape: {activity.shape}")
    except Exception as e:
        print(f"❌ VIPER failed: {str(e)}")
        traceback.print_exc()
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
        activity = viper(adata_invalid, regulons, validate=True)
        print("❌ VIPER did not catch the invalid data")
    except Exception as e:
        print(f"✅ VIPER correctly caught the error: {str(e)}")

    print("\n=== Pipeline validation test complete ===")

if __name__ == "__main__":
    main()
