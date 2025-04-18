"""
Tests for ARACNe C++ extensions.

This module contains tests specifically for the C++ extensions used in ARACNe,
focusing on validating the mutual information calculation and related algorithms.
"""

import pytest
import numpy as np
import logging
from pysces.aracne._cpp import aracne_ext

logger = logging.getLogger(__name__)

class TestMICalculation:
    @pytest.fixture
    def chi_square_threshold(self):
        return 7.815

    @pytest.mark.parametrize("test_case", [
        (np.array([1.0, 2.0, 3.0], dtype=np.float64),
         np.array([2.0, 4.0, 6.0], dtype=np.float64),
         True),  # Perfect correlation
        (np.array([1.0, 2.0, 3.0], dtype=np.float64),
         np.array([3.0, 2.0, 1.0], dtype=np.float64),
         True),  # Perfect anti-correlation
        (np.array([1.0, 2.0, 3.0], dtype=np.float64),
         np.array([4.0, 5.0, 6.0], dtype=np.float64),
         True),  # Linear relationship
        (np.array([1.0, 2.0, 3.0], dtype=np.float64),
         np.array([1.0, 1.0, 1.0], dtype=np.float64),
         False),  # No relationship (constant y)
        (np.random.normal(0, 1, 100),
         np.random.normal(0, 1, 100),
         False),  # Random relationship
    ])
    def test_mi_basic(self, test_case, chi_square_threshold):
        x, y, expect_positive = test_case
        mi = aracne_ext.calculate_mi_ap(x, y, chi_square_threshold)
        
        if expect_positive:
            assert mi > 0, f"Expected positive MI for correlated data, got {mi}"
        else:
            assert mi >= 0, f"MI should never be negative, got {mi}"

    def test_mi_edge_cases(self, chi_square_threshold):
        # Test constant arrays
        x = np.ones(10, dtype=np.float64)
        y = np.ones(10, dtype=np.float64)
        mi = aracne_ext.calculate_mi_ap(x, y, chi_square_threshold)
        assert mi == 0, f"MI should be 0 for constant arrays, got {mi}"

        # Test arrays with NaN
        x = np.array([1.0, np.nan, 3.0], dtype=np.float64)
        y = np.array([2.0, 4.0, 6.0], dtype=np.float64)
        mi = aracne_ext.calculate_mi_ap(x, y, chi_square_threshold)
        assert np.isnan(mi), f"MI should be NaN for arrays with NaN, got {mi}"

    def test_mi_different_sizes(self, chi_square_threshold):
        x = np.array([1.0, 2.0], dtype=np.float64)
        y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        with pytest.raises(RuntimeError):
            aracne_ext.calculate_mi_ap(x, y, chi_square_threshold)

class TestMIMatrix:
    def test_mi_matrix_small(self):
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

if __name__ == "__main__":
    # Run tests manually
    test_mi = TestMICalculation()
    for case in MI_TEST_CASES:
        test_mi.test_mi_basic(case, 7.815)
    
    test_mi.test_mi_edge_cases(7.815)
    
    try:
        test_mi.test_mi_different_sizes()
    except Exception as e:
        print(f"Expected error for different sizes: {str(e)}")
    
    test_matrix = TestMIMatrix()
    test_matrix.test_mi_matrix_small(7.815)
    
    print("All tests completed")
