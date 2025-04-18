"""
C++ extensions for ARACNe algorithm.
"""

# The aracne_ext_fixed module has been archived
# It was an earlier version of the ARACNe C++ extension
# See pysces/archive/aracne_cpp/ for the archived files

try:
    from . import aracne_ext
except ImportError:
    # If the C++ extension is not available, the Python fallback will be used
    pass
