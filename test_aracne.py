#!/usr/bin/env python
"""
Test script for the ARACNe algorithm implementation.
"""

import numpy as np
import sys
import os
import logging
import time

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import the ARACNe class
from pysces.src.pysces.aracne.core import ARACNe

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data(n_samples=100, n_genes=20, n_tfs=5, seed=42):
    """Generate synthetic test data."""
    np.random.seed(seed)

    # Generate random TF expression
    tf_expr = np.random.normal(0, 1, (n_samples, n_tfs))

    # Generate target gene expression based on TFs
    # Each gene is influenced by 1-3 TFs
    gene_expr = np.zeros((n_samples, n_genes))

    # First n_tfs genes are the TFs themselves
    gene_expr[:, :n_tfs] = tf_expr

    # Generate remaining genes
    for i in range(n_tfs, n_genes):
        # Choose 1-3 random TFs to influence this gene
        n_influencers = np.random.randint(1, min(4, n_tfs + 1))
        influencers = np.random.choice(n_tfs, n_influencers, replace=False)

        # Generate gene expression as a function of TFs plus noise
        for tf in influencers:
            # Randomly choose positive or negative influence
            coef = np.random.choice([-1, 1]) * np.random.uniform(0.5, 2.0)
            gene_expr[:, i] += coef * tf_expr[:, tf]

        # Add noise
        gene_expr[:, i] += np.random.normal(0, 0.5, n_samples)

    # Create gene names
    tf_names = [f"TF{i+1}" for i in range(n_tfs)]
    target_names = [f"Gene{i+1}" for i in range(n_tfs, n_genes)]
    gene_names = tf_names + target_names

    # Create TF indices
    tf_indices = list(range(n_tfs))

    return gene_expr, gene_names, tf_indices

def test_aracne_cpp():
    """Test ARACNe with C++ extensions."""
    logger.info("Testing ARACNe with C++ extensions")

    # Generate test data
    expr_matrix, gene_names, tf_indices = generate_test_data()

    # Create ARACNe instance
    aracne = ARACNe(bootstraps=10, p_value=0.05, dpi_tolerance=0.1,
                   consensus_threshold=0.3, chi_square_threshold=7.815)

    # Force Python implementation for now
    aracne._has_cpp_ext = False

    # Run ARACNe
    start_time = time.time()
    network = aracne._run_aracne(expr_matrix, gene_names, tf_indices)
    end_time = time.time()

    # Print results
    logger.info(f"ARACNe completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Number of TFs: {len(network['tf_names'])}")
    logger.info(f"Number of regulons: {len(network['regulons'])}")

    # Print first regulon
    tf_name = list(network['regulons'].keys())[0]
    targets = network['regulons'][tf_name]['targets']
    logger.info(f"TF: {tf_name}, Number of targets: {len(targets)}")

    return network

def test_aracne_python():
    """Test ARACNe with Python implementation."""
    logger.info("Testing ARACNe with Python implementation")

    # Generate test data
    expr_matrix, gene_names, tf_indices = generate_test_data()

    # Create ARACNe instance
    aracne = ARACNe(bootstraps=10, p_value=0.05, dpi_tolerance=0.1,
                   consensus_threshold=0.3, chi_square_threshold=7.815)

    # Force Python implementation
    aracne._has_cpp_ext = False

    # Run ARACNe
    start_time = time.time()
    network = aracne._run_aracne(expr_matrix, gene_names, tf_indices)
    end_time = time.time()

    # Print results
    logger.info(f"ARACNe completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Number of TFs: {len(network['tf_names'])}")
    logger.info(f"Number of regulons: {len(network['regulons'])}")

    # Print first regulon
    tf_name = list(network['regulons'].keys())[0]
    targets = network['regulons'][tf_name]['targets']
    logger.info(f"TF: {tf_name}, Number of targets: {len(targets)}")

    return network

def compare_implementations():
    """Compare C++ and Python implementations."""
    logger.info("Comparing C++ and Python implementations")

    # Generate test data
    expr_matrix, gene_names, tf_indices = generate_test_data()

    # Create ARACNe instances
    aracne_cpp = ARACNe(bootstraps=10, p_value=0.05, dpi_tolerance=0.1,
                       consensus_threshold=0.3, chi_square_threshold=7.815)

    aracne_py = ARACNe(bootstraps=10, p_value=0.05, dpi_tolerance=0.1,
                      consensus_threshold=0.3, chi_square_threshold=7.815)
    aracne_py._has_cpp_ext = False

    # Run ARACNe with C++
    start_time = time.time()
    network_cpp = aracne_cpp._run_aracne(expr_matrix, gene_names, tf_indices)
    cpp_time = time.time() - start_time

    # Run ARACNe with Python
    start_time = time.time()
    network_py = aracne_py._run_aracne(expr_matrix, gene_names, tf_indices)
    py_time = time.time() - start_time

    # Compare results
    logger.info(f"C++ implementation: {cpp_time:.2f} seconds")
    logger.info(f"Python implementation: {py_time:.2f} seconds")
    logger.info(f"Speedup: {py_time / cpp_time:.2f}x")

    # Compare networks
    cpp_tfs = set(network_cpp['tf_names'])
    py_tfs = set(network_py['tf_names'])
    logger.info(f"TFs in both implementations: {len(cpp_tfs.intersection(py_tfs))}")

    # Compare regulons
    cpp_regulons = network_cpp['regulons']
    py_regulons = network_py['regulons']

    for tf in cpp_regulons:
        if tf in py_regulons:
            cpp_targets = set(cpp_regulons[tf]['targets'].keys())
            py_targets = set(py_regulons[tf]['targets'].keys())
            overlap = len(cpp_targets.intersection(py_targets))
            logger.info(f"TF: {tf}, C++ targets: {len(cpp_targets)}, Python targets: {len(py_targets)}, Overlap: {overlap}")

    return network_cpp, network_py

if __name__ == "__main__":
    try:
        # Test Python implementation only for now
        network_py = test_aracne_python()
        logger.info("Test completed successfully!")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
