#!/usr/bin/env python
"""
Test PySCES installation with a focus on the Numba implementation.

This script verifies that the key components of PySCES can be imported
and that the Numba backend is properly detected.
"""
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test importing key modules."""
    logger.info("Testing imports...")

    # Try importing core modules
    try:
        import pysces
        logger.info("✓ Successfully imported pysces")

        from pysces.aracne.core import ARACNe
        logger.info("✓ Successfully imported ARACNe")

        from pysces.viper.core import viper_scores
        logger.info("✓ Successfully imported viper_scores")

        from pysces.viper.regulons import Regulon
        logger.info("✓ Successfully imported Regulon")

        logger.info("\nAll imports successful!")
        return True
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False

def test_numba_backend():
    """Test that Numba backend is properly detected."""
    logger.info("\nTesting Numba backend...")

    try:
        from pysces.aracne.core import ARACNe

        # Initialize ARACNe with auto backend
        aracne = ARACNe(backend='auto')
        logger.info(f"✓ ARACNe initialized with auto backend")
        logger.info(f"✓ Using Numba: {aracne.use_numba}")

        # Initialize ARACNe with explicit Numba backend
        aracne_numba = ARACNe(backend='numba')
        logger.info(f"✓ ARACNe initialized with Numba backend")
        logger.info(f"✓ Using Numba: {aracne_numba.use_numba}")

        logger.info("\nNumba backend test completed!")
        return aracne.use_numba or aracne_numba.use_numba
    except Exception as e:
        logger.error(f"✗ Numba backend error: {e}")
        return False

def main():
    """Main function."""
    logger.info("=" * 50)
    logger.info("PySCES Installation Test")
    logger.info("=" * 50)

    imports_ok = test_imports()
    numba_ok = test_numba_backend()

    logger.info("\n" + "=" * 50)
    if imports_ok and numba_ok:
        logger.info("✓ All tests passed! PySCES is properly installed.")
        sys.exit(0)
    else:
        logger.error("✗ Some tests failed. Please check the messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
