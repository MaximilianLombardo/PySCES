#!/usr/bin/env python
"""
Simple script to test the PySCES installation.
"""

import sys
import numpy as np
import pandas as pd
import anndata as ad

def test_pysces_imports():
    """Test that all required packages can be imported."""
    import pysces
    import numpy as np
    import pandas as pd
    import anndata as ad
    
    assert all(sys.modules.get(mod) for mod in ['pysces', 'numpy', 'pandas', 'anndata'])

def test_basic_functionality():
    """Test basic package functionality."""
    import pysces
    import numpy as np
    
    # Create test data
    n_cells = 100
    n_genes = 100
    X = np.random.randint(0, 100, size=(n_cells, n_genes))
    adata = ad.AnnData(X=X)
    
    # Test preprocessing
    processed = pysces.preprocess_data(adata, min_genes=10)
    assert processed.shape[0] > 0
    
    # Test rank transform
    ranked = pysces.rank_transform(processed)
    assert 'rank' in ranked.layers

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
