import numpy as np
import pandas as pd
import anndata as ad
import pysces


def test_sctransform_normalization():
    n_cells = 10
    n_genes = 20
    X = np.random.poisson(1, (n_cells, n_genes))
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]))
    processed = pysces.preprocess_data(
        adata,
        min_genes=0,
        min_cells=0,
        normalize=True,
        norm_method="sctransform",
    )
    assert processed.shape == (n_cells, n_genes)

