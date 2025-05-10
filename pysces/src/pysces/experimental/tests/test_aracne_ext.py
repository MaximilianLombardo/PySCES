"""
Tests for ARACNe C++ extensions.

This module contains tests specifically for the C++ extensions used in ARACNe,
focusing on validating the mutual information calculation and related algorithms.
"""

import pytest
import os
import numpy as np
import logging
import time
from pysces.aracne._cpp import aracne_ext
from pysces.aracne.core import ARACNe
from utils import generate_test_data, verify_mi_calculation, benchmark_mi_calculation

logger = logging.getLogger(__name__)

class TestMICalculation:
    """Test mutual information calculation."""
    
    def test_mi_basic(self):
        """Test basic MI calculation."""
        results = verify_mi_calculation(aracne_ext.calculate_mi_ap)
        
        # Check perfect correlation
        assert results['perfect_correlation'] > 0, "MI for perfect correlation should be positive"
        
        # Check perfect anti-correlation
        assert results['perfect_anticorrelation'] > 0, "MI for perfect anti-correlation should be positive"
        
        # Check constant arrays
        assert results['constant_arrays'] == 0, "MI for constant arrays should be zero"
        
        # Check that perfect correlation has higher MI than independent variables
        assert results['perfect_correlation'] >= results['independent'], \
            "MI for perfect correlation should be higher than for independent variables"
    
    def test_mi_edge_cases(self):
        """Test MI calculation edge cases."""
        # Test constant arrays
        x = np.ones(10, dtype=np.float64)
        y = np.ones(10, dtype=np.float64)
        mi = aracne_ext.calculate_mi_ap(x, y)
        assert mi == 0, f"MI should be 0 for constant arrays, got {mi}"
        
        # Test arrays with NaN
        x = np.array([1.0, np.nan, 3.0], dtype=np.float64)
        y = np.array([2.0, 4.0, 6.0], dtype=np.float64)
        mi = aracne_ext.calculate_mi_ap(x, y)
        assert np.isnan(mi), f"MI should be NaN for arrays with NaN, got {mi}"
    
    def test_mi_different_sizes(self):
        """Test MI calculation with arrays of different sizes."""
        x = np.array([1.0, 2.0], dtype=np.float64)
        y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with pytest.raises(RuntimeError):
            aracne_ext.calculate_mi_ap(x, y)

class TestMIMatrix:
    """Test mutual information matrix calculation."""
    
    def test_mi_matrix_small(self):
        """Test MI matrix calculation with small dataset."""
        n_samples = 10
        n_genes = 3
        data = np.random.normal(0, 1, (n_samples, n_genes)).astype(np.float64)
        tf_indices = np.array([0], dtype=np.int32)
        
        try:
            mi_matrix = aracne_ext.calculate_mi_matrix(
                data, tf_indices, chi_square_threshold=7.815, n_threads=1)
            assert mi_matrix.shape == (1, n_genes)
        except Exception as e:
            pytest.fail(f"MI matrix calculation failed: {str(e)}")
    
    def test_mi_matrix_large(self):
        """Test MI matrix calculation with larger dataset."""
        n_samples = 100
        n_genes = 20
        n_tfs = 5
        
        # Generate test data
        expr_matrix, gene_names, tf_indices = generate_test_data(
            n_samples=n_samples, n_genes=n_genes, n_tfs=n_tfs)
        
        try:
            mi_matrix = aracne_ext.calculate_mi_matrix(
                expr_matrix, tf_indices, chi_square_threshold=7.815, n_threads=1)
            assert mi_matrix.shape == (n_tfs, n_genes)
            
            # Check that diagonal elements are zero (self-interactions)
            for i in range(n_tfs):
                assert mi_matrix[i, tf_indices[i]] == 0, "Self-interactions should have zero MI"
        except Exception as e:
            pytest.fail(f"MI matrix calculation failed: {str(e)}")

class TestDPI:
    """Test Data Processing Inequality algorithm."""
    
    def test_dpi_basic(self):
        """Test basic DPI functionality."""
        n_tfs = 2
        n_genes = 3
        
        # Create a simple MI matrix
        mi_matrix = np.array([
            [0.0, 0.5, 0.3],  # TF1 -> Gene1, Gene2, Gene3
            [0.4, 0.0, 0.6]   # TF2 -> Gene1, Gene2, Gene3
        ], dtype=np.float64)
        
        # Apply DPI
        pruned_matrix = aracne_ext.apply_dpi(mi_matrix, tolerance=0.0, n_threads=1)
        
        # Check shape
        assert pruned_matrix.shape == (n_tfs, n_genes)
        
        # Check that diagonal elements are still zero
        assert pruned_matrix[0, 0] == 0
        assert pruned_matrix[1, 1] == 0

class TestBootstrap:
    """Test bootstrap sampling."""
    
    def test_bootstrap_matrix(self):
        """Test bootstrap matrix generation."""
        n_samples = 10
        n_genes = 5
        data = np.random.normal(0, 1, (n_samples, n_genes)).astype(np.float64)
        
        # Generate bootstrap sample
        bootstrap_data = aracne_ext.bootstrap_matrix(data)
        
        # Check shape
        assert bootstrap_data.shape == data.shape

class TestPerformance:
    """Test performance of C++ extensions."""
    
    def test_mi_calculation_performance(self):
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
        logger.info(f"Speedup: {py_time / cpp_time:.2f}x")
        
        # For this simple test, we don't assert performance
        # as the Python implementation using NumPy might be faster
        # Just log the results for information
        logger.info("Performance test completed successfully")

if __name__ == "__main__":
    # Run tests manually
    test_mi = TestMICalculation()
    test_mi.test_mi_basic()
    test_mi.test_mi_edge_cases()
    
    try:
        test_mi.test_mi_different_sizes()
    except Exception as e:
        print(f"Expected error for different sizes: {str(e)}")
    
    test_matrix = TestMIMatrix()
    test_matrix.test_mi_matrix_small()
    test_matrix.test_mi_matrix_large()
    
    test_dpi = TestDPI()
    test_dpi.test_dpi_basic()
    
    test_bootstrap = TestBootstrap()
    test_bootstrap.test_bootstrap_matrix()
    
    print("All tests completed")
