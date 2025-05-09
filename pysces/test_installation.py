#!/usr/bin/env python
"""
Test script to verify PySCES installation with C++ extensions.
"""

import sys
import numpy as np
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cpp_extensions():
    """Test if C++ extensions are properly installed."""
    logger.info("Testing C++ extensions...")

    try:
        from pysces.aracne._cpp import aracne_ext
        logger.info(f"C++ extensions loaded successfully (version: {getattr(aracne_ext, '__version__', 'unknown')})")

        # Test with small arrays
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        y = np.array([2.0, 4.0, 6.0], dtype=np.float64)

        logger.info("Testing mutual information calculation...")
        mi = aracne_ext.calculate_mi_ap(x, y)
        logger.info(f"Mutual information: {mi}")

        # Test with larger arrays
        logger.info("Testing with larger arrays...")
        n_samples = 1000
        x = np.random.normal(0, 1, n_samples).astype(np.float64)
        y = np.random.normal(0, 1, n_samples).astype(np.float64)

        start_time = time.time()
        mi = aracne_ext.calculate_mi_ap(x, y)
        elapsed = time.time() - start_time

        logger.info(f"Mutual information: {mi}")
        logger.info(f"Calculation time: {elapsed:.6f} seconds")

        # Test MI matrix calculation
        logger.info("Testing MI matrix calculation...")
        n_samples = 100
        n_genes = 10
        data = np.random.normal(0, 1, (n_samples, n_genes)).astype(np.float64)
        tf_indices = np.array([0, 1], dtype=np.int32)

        start_time = time.time()
        mi_matrix = aracne_ext.calculate_mi_matrix(data, tf_indices)
        elapsed = time.time() - start_time

        logger.info(f"MI matrix shape: {mi_matrix.shape}")
        logger.info(f"Calculation time: {elapsed:.6f} seconds")

        logger.info("All tests passed!")
        return True

    except ImportError as e:
        logger.error(f"Failed to import C++ extensions: {str(e)}")
        logger.info("PySCES will use the slower Python implementation.")
        return False

    except Exception as e:
        logger.error(f"Error testing C++ extensions: {str(e)}")
        logger.info("PySCES will use the slower Python implementation.")
        return False

def test_python_implementation():
    """Test the Python implementation."""
    logger.info("Testing Python implementation...")

    try:
        from pysces.aracne.core import ARACNe

        # Create ARACNe instance with Python implementation
        aracne = ARACNe(backend='python')
        aracne._has_cpp_ext = False

        # Generate test data
        n_samples = 100
        n_genes = 10
        expr_matrix = np.random.normal(0, 1, (n_samples, n_genes)).astype(np.float64)
        gene_names = [f"Gene{i+1}" for i in range(n_genes)]
        tf_indices = list(range(2))

        logger.info("Running ARACNe with Python implementation...")
        start_time = time.time()
        network = aracne._run_aracne_python(expr_matrix, gene_names, tf_indices)
        elapsed = time.time() - start_time

        logger.info(f"Network contains {len(network['regulons'])} regulons")
        logger.info(f"Calculation time: {elapsed:.2f} seconds")

        logger.info("Python implementation test passed!")
        return True

    except Exception as e:
        logger.error(f"Error testing Python implementation: {str(e)}")
        return False

def main():
    """Main function."""
    logger.info("PySCES Installation Test")
    logger.info("=======================")

    # Test Python version
    logger.info(f"Python version: {sys.version}")

    # Test NumPy
    logger.info(f"NumPy version: {np.__version__}")

    # Test C++ extensions
    cpp_success = test_cpp_extensions()

    # Test Python implementation
    python_success = test_python_implementation()

    # Summary
    logger.info("\nSummary:")
    logger.info(f"C++ extensions: {'Available' if cpp_success else 'Not available'}")
    logger.info(f"Python implementation: {'Working' if python_success else 'Not working'}")

    if cpp_success:
        logger.info("\nPySCES is installed correctly with C++ extensions!")
    elif python_success:
        logger.info("\nPySCES is installed with Python implementation only.")
        logger.info("For better performance, try reinstalling with C++ extensions.")
    else:
        logger.error("\nPySCES installation has issues. Please check the error messages above.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
