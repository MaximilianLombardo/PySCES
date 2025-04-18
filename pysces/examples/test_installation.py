#!/usr/bin/env python
"""
Simple script to test the PySCES installation.
"""

import sys
import numpy as np
import pandas as pd
import anndata as ad

def test_pysces_imports():
    """Test importing PySCES modules."""
    try:
        import pysces
        print("✅ Successfully imported pysces")
        
        from pysces.data import preprocessing
        print("✅ Successfully imported pysces.data.preprocessing")
        
        from pysces.aracne import ARACNe
        print("✅ Successfully imported pysces.aracne.ARACNe")
        
        from pysces.viper import viper
        print("✅ Successfully imported pysces.viper.viper")
        
        from pysces.analysis import cluster_activity
        print("✅ Successfully imported pysces.analysis.cluster_activity")
        
        from pysces.plotting import plot_activity_umap
        print("✅ Successfully imported pysces.plotting.plot_activity_umap")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without requiring external data."""
    try:
        import pysces
        
        # Create a more robust test dataset that won't be filtered out
        n_cells = 50
        n_genes = 100
        
        # Create expression matrix with enough non-zero values to pass filtering
        X = np.random.randint(1, 20, size=(n_cells, n_genes))
        
        # Ensure each cell has at least 200 genes expressed (min_genes default)
        # Since we only have 100 genes, we'll make sure all genes are expressed in all cells
        X = np.maximum(X, 1)  # Ensure no zeros
        
        # Create gene names with some mitochondrial genes
        gene_names = [f"gene_{i}" for i in range(n_genes)]
        gene_names[0:5] = [f"MT-{i}" for i in range(5)]  # Add some mitochondrial genes
        
        # Create cell names
        cell_names = [f"cell_{i}" for i in range(n_cells)]
        
        # Create AnnData object
        adata = ad.AnnData(
            X=X, 
            var=pd.DataFrame(index=gene_names),
            obs=pd.DataFrame(index=cell_names)
        )
        
        # Test preprocessing with lower min_genes since our test dataset only has 100 genes
        processed = pysces.preprocess_data(adata, min_genes=10)
        print(f"✅ Successfully preprocessed data: {processed.shape}")
        
        # Test rank transform
        ranked = pysces.rank_transform(processed)
        print(f"✅ Successfully rank-transformed data")
        
        # Create dummy regulons for testing
        from pysces.viper.regulons import Regulon
        regulons = []
        for i in range(5):
            # Create a Regulon object
            regulon = Regulon(tf_name=f"TF_{i}")
            # Add targets
            for j in range(10):
                regulon.add_target(f"gene_{j}", 1.0)
            regulons.append(regulon)
        
        # Test VIPER
        activity = pysces.viper(ranked, regulons)
        print(f"✅ Successfully ran VIPER: {activity.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False

if __name__ == "__main__":
    print("Testing PySCES installation...")
    imports_ok = test_pysces_imports()
    
    if imports_ok:
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n✅ All tests passed! PySCES is installed correctly.")
        else:
            print("\n❌ Basic functionality tests failed.")
            sys.exit(1)
    else:
        print("\n❌ Import tests failed.")
        sys.exit(1)
