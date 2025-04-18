#!/usr/bin/env python
"""
Test script for the fixed ARACNe C++ extensions.
"""

import numpy as np
import sys
import os
import logging

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import the fixed extension
from pysces.aracne._cpp import aracne_ext_fixed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_perfect_correlation():
    """Test MI calculation for perfect correlation."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([2.0, 4.0, 6.0], dtype=np.float64)
    
    logger.info("Testing perfect correlation")
    logger.info(f"x: {x}")
    logger.info(f"y: {y}")
    
    mi = aracne_ext_fixed.calculate_mi_ap(x, y)
    logger.info(f"MI result: {mi}")
    
    assert mi > 0, f"Expected positive MI for perfect correlation, got {mi}"
    logger.info("Perfect correlation test passed")

def test_perfect_anticorrelation():
    """Test MI calculation for perfect anti-correlation."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([3.0, 2.0, 1.0], dtype=np.float64)
    
    logger.info("Testing perfect anti-correlation")
    logger.info(f"x: {x}")
    logger.info(f"y: {y}")
    
    mi = aracne_ext_fixed.calculate_mi_ap(x, y)
    logger.info(f"MI result: {mi}")
    
    assert mi > 0, f"Expected positive MI for perfect anti-correlation, got {mi}"
    logger.info("Perfect anti-correlation test passed")

def test_constant_arrays():
    """Test MI calculation for constant arrays."""
    x = np.ones(10, dtype=np.float64)
    y = np.ones(10, dtype=np.float64)
    
    logger.info("Testing constant arrays")
    logger.info(f"x: {x}")
    logger.info(f"y: {y}")
    
    mi = aracne_ext_fixed.calculate_mi_ap(x, y)
    logger.info(f"MI result: {mi}")
    
    assert mi == 0, f"MI should be 0 for constant arrays, got {mi}"
    logger.info("Constant arrays test passed")

def test_mi_matrix():
    """Test MI matrix calculation."""
    n_samples = 10
    n_genes = 3
    data = np.random.normal(0, 1, (n_samples, n_genes)).astype(np.float64)
    tf_indices = np.array([0], dtype=np.int32)
    
    logger.info("Testing MI matrix calculation")
    logger.info(f"data shape: {data.shape}")
    logger.info(f"tf_indices: {tf_indices}")
    
    mi_matrix = aracne_ext_fixed.calculate_mi_matrix(data, tf_indices)
    logger.info(f"MI matrix shape: {mi_matrix.shape}")
    
    assert mi_matrix.shape == (1, n_genes), f"Expected shape (1, {n_genes}), got {mi_matrix.shape}"
    logger.info("MI matrix test passed")

def test_edge_cases():
    """Test edge cases."""
    # Empty array
    x = np.array([], dtype=np.float64)
    y = np.array([], dtype=np.float64)
    
    logger.info("Testing empty arrays")
    mi = aracne_ext_fixed.calculate_mi_ap(x, y)
    logger.info(f"MI result for empty arrays: {mi}")
    assert mi == 0, f"MI should be 0 for empty arrays, got {mi}"
    
    # Single value array
    x = np.array([1.0], dtype=np.float64)
    y = np.array([2.0], dtype=np.float64)
    
    logger.info("Testing single value arrays")
    mi = aracne_ext_fixed.calculate_mi_ap(x, y)
    logger.info(f"MI result for single value arrays: {mi}")
    assert mi == 0, f"MI should be 0 for single value arrays, got {mi}"
    
    # NaN values
    x = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    y = np.array([2.0, 4.0, 6.0], dtype=np.float64)
    
    logger.info("Testing arrays with NaN values")
    mi = aracne_ext_fixed.calculate_mi_ap(x, y)
    logger.info(f"MI result for arrays with NaN values: {mi}")
    assert np.isnan(mi), f"MI should be NaN for arrays with NaN values, got {mi}"
    
    logger.info("Edge cases test passed")

if __name__ == "__main__":
    try:
        test_perfect_correlation()
    except AssertionError as e:
        logger.error(f"Perfect correlation test failed: {str(e)}")
    
    try:
        test_perfect_anticorrelation()
    except AssertionError as e:
        logger.error(f"Perfect anti-correlation test failed: {str(e)}")
    
    try:
        test_constant_arrays()
    except AssertionError as e:
        logger.error(f"Constant arrays test failed: {str(e)}")
    
    try:
        test_mi_matrix()
    except AssertionError as e:
        logger.error(f"MI matrix test failed: {str(e)}")
    
    try:
        test_edge_cases()
    except AssertionError as e:
        logger.error(f"Edge cases test failed: {str(e)}")
