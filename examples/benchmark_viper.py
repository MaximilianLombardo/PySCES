"""
Benchmark script for VIPER implementation with and without Numba acceleration.
"""

import numpy as np
import pandas as pd
import time
import logging
import sys
import os
import importlib.util
from anndata import AnnData

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pysces', 'src'))

# Import VIPER modules
from pysces.viper.regulons import Regulon, GeneSet
from pysces.viper.core import viper_scores

# Import Numba-optimized VIPER
try:
    from pysces.viper.numba_optimized import viper_scores_numba, HAS_NUMBA
except ImportError:
    HAS_NUMBA = False
    logger.warning("Numba is not installed. Only benchmarking Python implementation.")

def create_synthetic_dataset(n_cells, n_genes, n_tfs):
    """Create a synthetic dataset for benchmarking."""
    # Create random expression data
    expr_matrix = np.random.rand(n_cells, n_genes)
    
    # Create gene list
    gene_list = [f"gene_{i}" for i in range(n_genes)]
    
    # Create AnnData object
    adata = AnnData(X=expr_matrix)
    adata.var_names = gene_list
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    
    # Create TF list
    tf_list = [f"gene_{i}" for i in range(n_tfs)]
    
    return adata, tf_list

def create_synthetic_regulons(gene_list, n_tfs, targets_per_tf=20):
    """Create synthetic regulons for benchmarking."""
    regulons = []
    
    for i in range(n_tfs):
        tf_name = f"gene_{i}"
        regulon = Regulon(tf_name)
        
        # Add random targets
        n_genes = len(gene_list)
        target_indices = np.random.choice(n_genes, targets_per_tf, replace=False)
        
        for idx in target_indices:
            target_gene = gene_list[idx]
            mode = 1 if np.random.rand() > 0.5 else -1
            likelihood = np.random.rand()
            regulon.add_target(target_gene, mode, likelihood)
        
        regulons.append(regulon)
    
    return regulons

def benchmark_viper(n_cells, n_genes, n_tfs, targets_per_tf=20, use_numba=False):
    """Benchmark VIPER with and without Numba acceleration."""
    # Create synthetic dataset
    logger.info(f"Creating dataset with {n_cells} cells, {n_genes} genes, {n_tfs} TFs")
    adata, tf_list = create_synthetic_dataset(n_cells, n_genes, n_tfs)
    
    # Create synthetic regulons
    logger.info(f"Creating {n_tfs} regulons with {targets_per_tf} targets each")
    regulons = create_synthetic_regulons(adata.var_names.tolist(), n_tfs, targets_per_tf)
    
    # Run VIPER and measure time
    logger.info(f"Running VIPER with {'Numba' if use_numba else 'Python'}")
    start_time = time.time()
    
    if use_numba:
        activity = viper_scores_numba(
            adata,
            regulons,
            signature_method='rank',
            enrichment_method='gsea'
        )
    else:
        activity = viper_scores(
            adata,
            regulons,
            signature_method='rank',
            enrichment_method='gsea',
            use_numba=False
        )
    
    elapsed_time = time.time() - start_time
    
    return elapsed_time, activity

def run_benchmarks():
    """Run benchmarks with different dataset sizes."""
    # Define dataset sizes
    dataset_sizes = [
        (100, 100, 10, 20),   # Small dataset: 100 cells, 100 genes, 10 TFs, 20 targets per TF
        (500, 200, 20, 30),   # Medium dataset: 500 cells, 200 genes, 20 TFs, 30 targets per TF
        (1000, 500, 50, 50),  # Large dataset: 1000 cells, 500 genes, 50 TFs, 50 targets per TF
    ]
    
    # Run benchmarks
    results = []
    
    for n_cells, n_genes, n_tfs, targets_per_tf in dataset_sizes:
        logger.info(f"\nBenchmarking dataset: {n_cells} cells, {n_genes} genes, {n_tfs} TFs, {targets_per_tf} targets per TF")
        
        # Benchmark Python implementation
        logger.info("Benchmarking Python implementation...")
        python_time, _ = benchmark_viper(
            n_cells=n_cells,
            n_genes=n_genes,
            n_tfs=n_tfs,
            targets_per_tf=targets_per_tf,
            use_numba=False
        )
        logger.info(f"Python implementation: {python_time:.2f} seconds")
        
        # Benchmark Numba implementation if available
        if HAS_NUMBA:
            # Warm-up run for Numba compilation
            logger.info("Performing warm-up run for Numba...")
            _, _ = benchmark_viper(
                n_cells=50,
                n_genes=50,
                n_tfs=5,
                targets_per_tf=10,
                use_numba=True
            )
            
            logger.info("Benchmarking Numba implementation...")
            numba_time, _ = benchmark_viper(
                n_cells=n_cells,
                n_genes=n_genes,
                n_tfs=n_tfs,
                targets_per_tf=targets_per_tf,
                use_numba=True
            )
            logger.info(f"Numba implementation: {numba_time:.2f} seconds")
            
            # Calculate speedup
            speedup = python_time / numba_time
            logger.info(f"Speedup: {speedup:.2f}x")
        else:
            numba_time = None
            speedup = None
            logger.warning("Numba not available. Skipping Numba benchmarks.")
        
        results.append({
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_tfs": n_tfs,
            "targets_per_tf": targets_per_tf,
            "python_time": python_time,
            "numba_time": numba_time,
            "speedup": speedup
        })
    
    # Print summary
    logger.info("\nBenchmark Results:")
    logger.info("=================")
    for result in results:
        logger.info(f"Dataset: {result['n_cells']}x{result['n_genes']} ({result['n_tfs']} TFs, {result['targets_per_tf']} targets per TF)")
        logger.info(f"  Python: {result['python_time']:.2f} seconds")
        
        if result['numba_time'] is not None:
            logger.info(f"  Numba:  {result['numba_time']:.2f} seconds")
            logger.info(f"  Speedup: {result['speedup']:.2f}x")
        else:
            logger.info("  Numba:  Not available")
    
    return results

if __name__ == "__main__":
    run_benchmarks()
