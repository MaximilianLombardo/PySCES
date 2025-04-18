#!/usr/bin/env python
"""
Debug script for MI calculation issues.
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

def test_perfect_correlation():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([2.0, 4.0, 6.0], dtype=np.float64)

    logger.debug("Testing perfect correlation")
    logger.debug(f"x: {x}")
    logger.debug(f"y: {y}")

    mi = aracne_ext.calculate_mi_ap(x, y)
    logger.debug(f"MI result: {mi}")

    assert mi > 0, f"Expected positive MI for perfect correlation, got {mi}"

def test_perfect_anticorrelation():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([3.0, 2.0, 1.0], dtype=np.float64)

    logger.debug("Testing perfect anti-correlation")
    logger.debug(f"x: {x}")
    logger.debug(f"y: {y}")

    mi = aracne_ext.calculate_mi_ap(x, y)
    logger.debug(f"MI result: {mi}")

    assert mi > 0, f"Expected positive MI for perfect anti-correlation, got {mi}"

def test_constant_arrays():
    x = np.ones(10, dtype=np.float64)
    y = np.ones(10, dtype=np.float64)

    logger.debug("Testing constant arrays")
    logger.debug(f"x: {x}")
    logger.debug(f"y: {y}")

    mi = aracne_ext.calculate_mi_ap(x, y)
    logger.debug(f"MI result: {mi}")

    assert mi == 0, f"MI should be 0 for constant arrays, got {mi}"

def test_mi_matrix():
    n_samples = 10
    n_genes = 3
    data = np.random.normal(0, 1, (n_samples, n_genes)).astype(np.float64)
    tf_indices = np.array([0], dtype=np.int32)

    logger.debug("Testing MI matrix calculation")
    logger.debug(f"data shape: {data.shape}")
    logger.debug(f"tf_indices: {tf_indices}")

    mi_matrix = aracne_ext.calculate_mi_matrix(data, tf_indices)
    logger.debug(f"MI matrix shape: {mi_matrix.shape}")

    assert mi_matrix.shape == (1, n_genes), f"Expected shape (1, {n_genes}), got {mi_matrix.shape}"

if __name__ == "__main__":
    try:
        test_perfect_correlation()
        print("Perfect correlation test passed")
    except AssertionError as e:
        print(f"Perfect correlation test failed: {str(e)}")

    try:
        test_perfect_anticorrelation()
        print("Perfect anti-correlation test passed")
    except AssertionError as e:
        print(f"Perfect anti-correlation test failed: {str(e)}")

    try:
        test_constant_arrays()
        print("Constant arrays test passed")
    except AssertionError as e:
        print(f"Constant arrays test failed: {str(e)}")

    try:
        test_mi_matrix()
        print("MI matrix test passed")
    except AssertionError as e:
        print(f"MI matrix test failed: {str(e)}")
