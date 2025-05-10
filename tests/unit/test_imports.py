"""
Tests that all key modules can be imported correctly.
"""
import pytest

def test_import_core_modules():
    """Test importing all core modules."""
    # Core package
    import pysces

    # ARACNe modules
    from pysces.src.pysces.aracne.core import ARACNe, aracne_to_regulons
    from pysces.src.pysces.aracne.numba_optimized import run_aracne_numba

    # VIPER modules
    from pysces.src.pysces.viper.core import viper_scores, viper_bootstrap, viper_null_model
    from pysces.src.pysces.viper.regulons import Regulon, GeneSet, prune_regulons

    # Utils
    from pysces.src.pysces.utils.validation import validate_anndata_structure

    # Test successful if all imports complete without errors
    assert True

def test_backend_availability():
    """Test that Numba backend is available."""
    try:
        from pysces.src.pysces.aracne.numba_optimized import HAS_NUMBA
        from pysces.src.pysces.viper.numba_optimized import HAS_NUMBA as VIPER_HAS_NUMBA

        # Print helpful message but don't fail if Numba not available
        # It's acceptable to run with Python backend if Numba isn't installed
        if not HAS_NUMBA:
            print("Warning: Numba not available for ARACNe. Using Python backend.")
        if not VIPER_HAS_NUMBA:
            print("Warning: Numba not available for VIPER. Using Python backend.")

    except ImportError:
        # Failed to import at all - this is a problem
        pytest.fail("Failed to import Numba modules")

if __name__ == "__main__":
    # Run tests manually
    test_import_core_modules()
    test_backend_availability()
    print("All tests passed!")
