"""
Debug script for the validation module.

This script creates a minimal test case to debug the validation functions.
"""

import pysces
import numpy as np
import pandas as pd
import anndata as ad
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    print("\n=== Creating synthetic dataset ===")
    # Create minimal test data
    n_cells = 50
    n_genes = 100
    
    # Create random count data
    X = np.random.randint(0, 10, size=(n_cells, n_genes))
    
    # Create gene names with some TFs
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    
    # Create cell names
    cell_names = [f"cell_{i}" for i in range(n_cells)]
    
    # Create AnnData object
    adata = ad.AnnData(
        X=X, 
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=gene_names)
    )
    
    print(f"Dataset shape: {adata.shape}")
    
    # Test validation functions
    print("\n=== Testing validation functions ===")
    
    # Test validate_anndata_structure
    print("\n--- Testing validate_anndata_structure ---")
    try:
        is_valid, issues = pysces.validate_anndata_structure(adata)
        print(f"AnnData structure is valid: {is_valid}")
        if not is_valid:
            print("Issues:")
            for issue in issues:
                print(f"  - {issue}")
    except Exception as e:
        print(f"Error in validate_anndata_structure: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test validate_gene_names
    print("\n--- Testing validate_gene_names ---")
    try:
        is_valid, issues = pysces.validate_gene_names(adata)
        print(f"Gene names are valid: {is_valid}")
        if not is_valid:
            print("Issues:")
            for issue in issues:
                print(f"  - {issue}")
    except Exception as e:
        print(f"Error in validate_gene_names: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test validate_cell_names
    print("\n--- Testing validate_cell_names ---")
    try:
        is_valid, issues = pysces.validate_cell_names(adata)
        print(f"Cell names are valid: {is_valid}")
        if not is_valid:
            print("Issues:")
            for issue in issues:
                print(f"  - {issue}")
    except Exception as e:
        print(f"Error in validate_cell_names: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test validate_raw_counts
    print("\n--- Testing validate_raw_counts ---")
    try:
        is_raw, issues, confidence = pysces.validate_raw_counts(adata)
        print(f"Data is raw counts: {is_raw} (confidence: {confidence:.2f})")
        if issues:
            print("Issues:")
            for issue in issues:
                print(f"  - {issue}")
    except Exception as e:
        print(f"Error in validate_raw_counts: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test validate_normalized_data
    print("\n--- Testing validate_normalized_data ---")
    try:
        is_norm, issues, confidence = pysces.validate_normalized_data(adata)
        print(f"Data is normalized: {is_norm} (confidence: {confidence:.2f})")
        if issues:
            print("Issues:")
            for issue in issues:
                print(f"  - {issue}")
    except Exception as e:
        print(f"Error in validate_normalized_data: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test recommend_preprocessing
    print("\n--- Testing recommend_preprocessing ---")
    try:
        recommendations = pysces.recommend_preprocessing(adata)
        print("Preprocessing recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    except Exception as e:
        print(f"Error in recommend_preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
    
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
    
    print("\n--- Testing validate_gene_names with duplicate names ---")
    try:
        is_valid, issues = pysces.validate_gene_names(adata_invalid)
        print(f"Gene names are valid: {is_valid}")
        if not is_valid:
            print("Issues:")
            for issue in issues:
                print(f"  - {issue}")
    except Exception as e:
        print(f"Error in validate_gene_names: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Validation debug complete ===")

if __name__ == "__main__":
    main()
