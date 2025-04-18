import pytest

def test_basic_imports():
    import pysces
    assert hasattr(pysces, '__version__')
    assert isinstance(pysces.__version__, str)
    
    # Test core functionality imports
    from pysces import ARACNe
    from pysces import preprocess_data
    from pysces import viper
    
    assert all(callable(f) for f in [ARACNe, preprocess_data, viper])

def test_submodule_structure():
    import pysces.data
    import pysces.aracne
    import pysces.viper
    import pysces.analysis