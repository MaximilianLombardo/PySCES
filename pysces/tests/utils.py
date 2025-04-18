"""
Test utilities for PySCES.

This module contains common utilities for testing PySCES functionality,
particularly for the ARACNe algorithm.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_test_data(n_samples=100, n_genes=20, n_tfs=5, seed=42):
    """
    Generate synthetic test data for ARACNe testing.
    
    Parameters
    ----------
    n_samples : int, default=100
        Number of samples (cells)
    n_genes : int, default=20
        Number of genes
    n_tfs : int, default=5
        Number of transcription factors
    seed : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    expr_matrix : numpy.ndarray
        Expression matrix (samples x genes)
    gene_names : list
        List of gene names
    tf_indices : numpy.ndarray
        Array of TF indices
    """
    np.random.seed(seed)
    
    # Generate random expression data
    expr_matrix = np.random.normal(0, 1, (n_samples, n_genes)).astype(np.float64)
    
    # Create gene names
    gene_names = [f"Gene{i+1}" for i in range(n_genes)]
    
    # Create TF indices
    tf_indices = np.array(list(range(n_tfs)), dtype=np.int32)
    
    return expr_matrix, gene_names, tf_indices

def verify_mi_calculation(mi_func):
    """
    Verify mutual information calculation with various test cases.
    
    Parameters
    ----------
    mi_func : callable
        Function to calculate mutual information
        
    Returns
    -------
    dict
        Dictionary of test results
    """
    results = {}
    
    # Test perfect correlation
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([2.0, 4.0, 6.0], dtype=np.float64)
    results['perfect_correlation'] = mi_func(x, y)
    
    # Test perfect anti-correlation
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([3.0, 2.0, 1.0], dtype=np.float64)
    results['perfect_anticorrelation'] = mi_func(x, y)
    
    # Test independent variables
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    results['independent'] = mi_func(x, y)
    
    # Test constant arrays
    x = np.ones(10, dtype=np.float64)
    y = np.ones(10, dtype=np.float64)
    results['constant_arrays'] = mi_func(x, y)
    
    # Test arrays with NaN
    x = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    y = np.array([2.0, 4.0, 6.0], dtype=np.float64)
    try:
        results['nan_arrays'] = mi_func(x, y)
    except Exception as e:
        results['nan_arrays'] = np.nan
        logger.warning(f"NaN test raised exception: {str(e)}")
    
    # Test arrays with different sizes
    x = np.array([1.0, 2.0], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    try:
        results['different_sizes'] = mi_func(x, y)
    except Exception as e:
        results['different_sizes'] = "error"
        logger.warning(f"Different sizes test raised exception: {str(e)}")
    
    return results

def benchmark_mi_calculation(mi_func, n_samples=10000, n_repeats=10):
    """
    Benchmark mutual information calculation.
    
    Parameters
    ----------
    mi_func : callable
        Function to calculate mutual information
    n_samples : int, default=1000
        Number of samples in test vectors
    n_repeats : int, default=10
        Number of times to repeat the calculation
        
    Returns
    -------
    float
        Average time per calculation in seconds
    """
    import time
    
    # Generate random data
    np.random.seed(42)
    x = np.random.normal(0, 1, n_samples).astype(np.float64)
    y = np.random.normal(0, 1, n_samples).astype(np.float64)
    
    # Warm-up
    mi_func(x, y)
    
    # Benchmark
    start_time = time.time()
    for _ in range(n_repeats):
        mi_func(x, y)
    end_time = time.time()
    
    # Calculate average time
    avg_time = (end_time - start_time) / n_repeats
    
    return avg_time
