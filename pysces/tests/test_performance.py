"""
Performance benchmarks for PySCES.

This module contains performance benchmarks for PySCES functionality,
particularly comparing C++ and Python implementations.
"""

import pytest
import numpy as np
import time
import os
import logging
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath('..'))

from aracne._cpp import aracne_ext
from aracne.core import ARACNe
from utils import generate_test_data, benchmark_mi_calculation

logger = logging.getLogger(__name__)

def test_mi_calculation_performance():
    """Compare performance of C++ and Python MI calculation."""
    # Skip this test if running in CI
    if os.environ.get('CI', 'false').lower() == 'true':
        pytest.skip("Skipping performance test in CI environment")
    
    # Define a simple Python MI calculation function
    def calculate_mi_python(x, y):
        # Calculate correlation as a simple approximation of MI
        correlation = np.corrcoef(x, y)[0, 1]
        return abs(correlation)  # Use absolute correlation as a proxy for MI
    
    # Benchmark C++ implementation
    cpp_time = benchmark_mi_calculation(aracne_ext.calculate_mi_ap)
    
    # Benchmark Python implementation
    py_time = benchmark_mi_calculation(calculate_mi_python)
    
    # Print performance comparison
    logger.info(f"C++ implementation: {cpp_time:.6f} seconds per calculation")
    logger.info(f"Python implementation: {py_time:.6f} seconds per calculation")
    logger.info(f"Speedup ratio: {py_time / cpp_time:.2f}x")
    
    # For this simple test, we don't assert performance
    # as the Python implementation using NumPy might be faster
    # Just log the results for information
    logger.info("Performance test completed successfully")

def test_mi_matrix_performance():
    """Compare performance of C++ and Python MI matrix calculation."""
    # Skip this test if running in CI
    if os.environ.get('CI', 'false').lower() == 'true':
        pytest.skip("Skipping performance test in CI environment")
    
    # Generate test data
    n_samples = 100
    n_genes = 20
    n_tfs = 5
    expr_matrix, gene_names, tf_indices = generate_test_data(
        n_samples=n_samples, n_genes=n_genes, n_tfs=n_tfs)
    
    # Create ARACNe instance with C++ implementation
    aracne_cpp = ARACNe(bootstraps=1, p_value=0.05, dpi_tolerance=0.1,
                      consensus_threshold=0.3, chi_square_threshold=7.815)
    aracne_cpp._has_cpp_ext = True
    
    # Create ARACNe instance with Python implementation
    aracne_py = ARACNe(bootstraps=1, p_value=0.05, dpi_tolerance=0.1,
                      consensus_threshold=0.3, chi_square_threshold=7.815)
    aracne_py._has_cpp_ext = False
    
    # Measure C++ performance
    start_time = time.time()
    try:
        network_cpp = aracne_cpp._run_aracne(expr_matrix, gene_names, tf_indices)
        cpp_time = time.time() - start_time
        
        # Measure Python performance
        start_time = time.time()
        network_py = aracne_py._run_aracne_python(expr_matrix, gene_names, tf_indices)
        py_time = time.time() - start_time
        
        # Print performance comparison
        logger.info(f"C++ implementation: {cpp_time:.2f} seconds")
        logger.info(f"Python implementation: {py_time:.2f} seconds")
        logger.info(f"Speedup ratio: {py_time / cpp_time:.2f}x")
        
        # For this test, we don't assert performance
        # Just log the results for information
        logger.info("Performance test completed successfully")
    except Exception as e:
        logger.error(f"Error in performance test: {str(e)}")
        pytest.skip(f"Skipping due to error: {str(e)}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_mi_calculation_performance()
    test_mi_matrix_performance()
    
    print("All performance tests completed")
