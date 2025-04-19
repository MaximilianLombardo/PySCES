#!/usr/bin/env python
"""
Test if we can import the ARACNe class.
"""

import sys
import os

# Print Python path
print("Python path:")
for p in sys.path:
    print(f"  {p}")

# Try different import paths
print("\nTrying different import paths:")

try:
    from pysces.aracne.core import ARACNe
    print("Successfully imported from pysces.aracne.core")
except ImportError as e:
    print(f"Failed to import from pysces.aracne.core: {e}")

try:
    sys.path.append('pysces')
    from aracne.core import ARACNe
    print("Successfully imported from aracne.core")
except ImportError as e:
    print(f"Failed to import from aracne.core: {e}")

try:
    sys.path.append('pysces/src')
    from pysces.aracne.core import ARACNe
    print("Successfully imported from pysces.aracne.core (with pysces/src in path)")
except ImportError as e:
    print(f"Failed to import from pysces.aracne.core (with pysces/src in path): {e}")

# Check if the file exists
print("\nChecking if core.py exists:")
paths_to_check = [
    'pysces/aracne/core.py',
    'pysces/src/pysces/aracne/core.py',
    'src/pysces/aracne/core.py',
]

for path in paths_to_check:
    if os.path.exists(path):
        print(f"File exists: {path}")
    else:
        print(f"File does not exist: {path}")
