"""
Example script demonstrating the use of validation functions in PySCES.
"""

import pysces
import numpy as np
import pandas as pd
import anndata as ad
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    # Load or create sample data
    try:
        # Try to load the Tabula Sapiens testis dataset if available
        adata = pysces.read_anndata("data/tabula_sapiens_testis.h5ad")
        print("Loaded Tabula Sapiens testis dataset")
    except (FileNotFoundError, TypeError):
        # Create a synthetic dataset if the real one is not available
        print("Creating synthetic dataset")
        n_cells = 100
        n_genes = 200
        
        # Create random count data
        X = np.random.randint(0, 10, size=(n_cells, n_genes))
        
        # Create gene names with some mitochondrial genes
        gene_names = [f"gene_{i}" for i in range(n_genes)]
        gene_names[0:10] = [f"MT-{i}" for i in range(10)]  # Add some mitochondrial genes
        
        # Create cell names
        cell_names = [f"cell_{i}" for i in range(n_cells)]
        
        # Create AnnData object
        adata = ad.AnnData(
            X=X, 
            obs=pd.DataFrame(index=cell_names),
            var=pd.DataFrame(index=gene_names)
        )
    
    print(f"Dataset shape: {adata.shape}")
    
    # Validate AnnData structure
    print("\n=== Validating AnnData structure ===")
    is_valid, issues = pysces.validate_anndata_structure(adata)
    if is_valid:
        print("✅ AnnData structure is valid")
    else:
        print("❌ AnnData structure has issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Validate gene names
    print("\n=== Validating gene names ===")
    is_valid, issues = pysces.validate_gene_names(adata)
    if is_valid:
        print("✅ Gene names are valid")
    else:
        print("❌ Gene names have issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Validate cell names
    print("\n=== Validating cell names ===")
    is_valid, issues = pysces.validate_cell_names(adata)
    if is_valid:
        print("✅ Cell names are valid")
    else:
        print("❌ Cell names have issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Check if data is raw counts
    print("\n=== Checking if data is raw counts ===")
    is_raw, issues, confidence = pysces.validate_raw_counts(adata)
    if is_raw:
        print(f"✅ Data appears to be raw counts (confidence: {confidence:.2f})")
    else:
        print(f"❌ Data does not appear to be raw counts (confidence: {confidence:.2f})")
        if issues:
            print("Issues:")
            for issue in issues:
                print(f"  - {issue}")
    
    # Check if data is normalized
    print("\n=== Checking if data is normalized ===")
    is_norm, issues, confidence = pysces.validate_normalized_data(adata)
    if is_norm:
        print(f"✅ Data appears to be normalized (confidence: {confidence:.2f})")
    else:
        print(f"❌ Data does not appear to be normalized (confidence: {confidence:.2f})")
        if issues:
            print("Issues:")
            for issue in issues:
                print(f"  - {issue}")
    
    # Validate sparse matrix
    print("\n=== Validating sparse matrix ===")
    is_valid, issues = pysces.validate_sparse_matrix(adata)
    if is_valid:
        print("✅ Sparse matrix format is valid")
    else:
        print("❌ Sparse matrix format has issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Get preprocessing recommendations
    print("\n=== Preprocessing recommendations ===")
    recommendations = pysces.recommend_preprocessing(adata)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # Demonstrate integration with pipeline
    print("\n=== Demonstrating integration with pipeline ===")
    print("1. Validating input data")
    is_valid, issues = pysces.validate_anndata_structure(adata)
    if not is_valid:
        print("❌ Cannot proceed with invalid AnnData structure")
        return
    
    print("2. Preprocessing data")
    adata = pysces.preprocess_data(adata)
    
    print("3. Validating preprocessed data")
    is_norm, issues, confidence = pysces.validate_normalized_data(adata)
    print(f"Data is normalized: {is_norm} (confidence: {confidence:.2f})")
    
    print("4. Ready for ARACNe and VIPER")
    print("Pipeline complete!")

if __name__ == "__main__":
    main()
