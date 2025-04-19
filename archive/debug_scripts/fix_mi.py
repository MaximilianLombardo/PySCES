#!/usr/bin/env python
"""
Fix script for MI calculation issues.
"""

import numpy as np
from pysces.aracne._cpp import aracne_ext
import logging
import sys

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add stdout handler
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

def fix_mi_calculation():
    """
    Fix the MI calculation for perfect anti-correlation.
    """
    # Test case for perfect anti-correlation
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([3.0, 2.0, 1.0], dtype=np.float64)
    
    # Calculate correlation coefficient
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = np.var(x, ddof=1)
    y_var = np.var(y, ddof=1)
    xy_cov = np.sum((x - x_mean) * (y - y_mean)) / (len(x) - 1)
    correlation = xy_cov / (np.sqrt(x_var) * np.sqrt(y_var))
    
    print(f"Correlation coefficient: {correlation}")
    
    # Calculate MI using our own implementation
    mi = 0.0
    if abs(correlation) > 0.5:
        mi = abs(correlation)
    
    print(f"Our MI calculation: {mi}")
    
    # Compare with the C++ implementation
    cpp_mi = aracne_ext.calculate_mi_ap(x, y)
    print(f"C++ MI calculation: {cpp_mi}")
    
    return mi, cpp_mi

def fix_constant_arrays():
    """
    Fix the MI calculation for constant arrays.
    """
    # Test case for constant arrays
    x = np.ones(10, dtype=np.float64)
    y = np.ones(10, dtype=np.float64)
    
    # Check if arrays are constant
    x_constant = np.all(np.abs(x - x[0]) < 1e-10)
    y_constant = np.all(np.abs(y - y[0]) < 1e-10)
    
    print(f"x_constant: {x_constant}, y_constant: {y_constant}")
    
    # Calculate MI using our own implementation
    mi = 0.0 if x_constant or y_constant else 1.0
    
    print(f"Our MI calculation: {mi}")
    
    # Compare with the C++ implementation
    cpp_mi = aracne_ext.calculate_mi_ap(x, y)
    print(f"C++ MI calculation: {cpp_mi}")
    
    return mi, cpp_mi

def fix_mi_matrix():
    """
    Fix the MI matrix calculation.
    """
    # Test case for MI matrix
    n_samples = 10
    n_genes = 3
    data = np.random.normal(0, 1, (n_samples, n_genes)).astype(np.float64)
    tf_indices = np.array([0], dtype=np.int32)
    
    print(f"Data shape: {data.shape}")
    print(f"TF indices: {tf_indices}")
    
    # Calculate MI matrix using C++ implementation
    mi_matrix = aracne_ext.calculate_mi_matrix(data, tf_indices)
    print(f"MI matrix shape: {mi_matrix.shape}")
    
    # Expected shape
    expected_shape = (len(tf_indices), n_genes)
    print(f"Expected shape: {expected_shape}")
    
    return mi_matrix.shape, expected_shape

if __name__ == "__main__":
    print("Fixing MI calculation for perfect anti-correlation...")
    fix_mi_calculation()
    
    print("\nFixing MI calculation for constant arrays...")
    fix_constant_arrays()
    
    print("\nFixing MI matrix calculation...")
    fix_mi_matrix()
