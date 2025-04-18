"""
C++ extensions for ARACNe algorithm.
"""

try:
    from . import aracne_ext_fixed
except ImportError:
    pass

try:
    from . import aracne_ext
except ImportError:
    pass
