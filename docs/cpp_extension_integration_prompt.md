# Task: Integrate Fixed ARACNe C++ Extensions into PySCES

## Background
I've been working on PySCES (Python Single-Cell Expression System), specifically improving the ARACNe algorithm implementation. The critical C++ extensions for Mutual Information (MI) calculation have been successfully fixed with:
- Correct handling of correlations and edge cases
- Robust error handling for numerical stability
- Cross-platform compatibility including OpenMP support

## Current Status
The fixed C++ extensions have been implemented and tested:
- `fix_mi.py` contains implementations for MI calculation fixes
- `debug_mi.py` includes comprehensive tests for the fixed functionality
- `setup_fixed.py` includes platform-specific build improvements
- The fixed extensions have been successfully compiled (`aracne_ext_fixed.cpython-310-darwin.so`)

## Current Issue
While the C++ extensions have been fixed and compiled, they need to be properly integrated into the main package:
1. The main package still uses the original extensions, not the fixed ones
2. The build system needs to be updated to use the fixed extensions by default
3. The core.py file needs to be updated to use the fixed extensions

## Next Task
Please help me:
1. Integrate the fixed extensions into the main package
2. Update the package setup files (setup.py, pyproject.toml) to use the fixed extensions
3. Update the core.py file to use the fixed extensions
4. Create a test utilities module and expand the test suite
5. Update documentation to reflect the fixed implementation
6. Add performance metrics comparing C++ and Python implementations

## Specific Technical Requirements
- Replace the old extension with the fixed one entirely (consolidate to a single extension name)
- Add explicit error handling for compilation failures in setup.py
- Add version detection for compilers to handle different compiler requirements
- Simplify the import strategy in core.py to a single try-except block
- Create a common test utilities module for both debug scripts and formal tests
- Add API documentation for the C++ functions
- Ensure backward compatibility with existing code

## Files to Consider
- `pysces/setup.py` (needs to be updated with fixes from `setup_fixed.py`)
- `pysces/pyproject.toml` (may need updates for build requirements)
- `pysces/src/pysces/aracne/core.py` (needs to use fixed extensions)
- `fix_mi.py` and `debug_mi.py` (contain fixes and tests to be integrated)
- `tests/test_aracne_ext.py` (should incorporate tests from `debug_mi.py`)
- `tests/utils.py` (new file for common test utilities)

## Success Criteria
- Running `import pysces` should successfully import the fixed C++ extensions
- The warning about falling back to Python implementation should not appear
- All tests should pass, including those that use the fixed extensions
- The package should install and run on different platforms
- Performance metrics should show significant improvement with C++ extensions

## Detailed Steps

### 1. Consolidate Extensions
Replace the old extension with the fixed one entirely:

```bash
# Replace the old extension source with the fixed one
cp pysces/src/pysces/aracne/_cpp/aracne_ext_fixed.cpp pysces/src/pysces/aracne/_cpp/aracne_ext.cpp

# If the compiled extension exists, replace it too
cp pysces/aracne/_cpp/aracne_ext_fixed.cpython-310-darwin.so pysces/src/pysces/aracne/_cpp/aracne_ext.cpython-310-darwin.so
```

### 2. Update Package Configuration
Update the main setup.py with the improvements from setup_fixed.py:

```python
# In setup.py
import platform
import sys
import os
from setuptools import setup, Extension, find_packages
import pybind11

# Get the pybind11 include directory
pybind11_include = pybind11.get_include()

# Set compiler flags based on platform
extra_compile_args = ['-std=c++11']
extra_link_args = []

# Add OpenMP flags if not on macOS
if platform.system() != 'Darwin':
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')

# Add compiler version detection
def get_compiler_version():
    """Get the compiler version to handle different compiler requirements."""
    import subprocess
    try:
        if platform.system() == 'Darwin':
            # Check for clang version
            result = subprocess.run(['clang', '--version'], capture_output=True, text=True)
            # Parse version from output
            return result.stdout
        elif platform.system() == 'Linux':
            # Check for gcc version
            result = subprocess.run(['gcc', '--version'], capture_output=True, text=True)
            # Parse version from output
            return result.stdout
        else:
            # Windows or other
            return "Unknown"
    except Exception:
        return "Unknown"

# Define the extension module with error handling
try:
    ext_modules = [
        Extension(
            'pysces.aracne._cpp.aracne_ext',
            ['src/pysces/aracne/_cpp/aracne_ext.cpp'],
            include_dirs=[
                pybind11_include,
                'src/pysces/aracne/_cpp/include'
            ],
            language='c++',
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ]
except Exception as e:
    print(f"Error setting up C++ extensions: {str(e)}")
    print("Falling back to Python-only installation")
    ext_modules = []

# Setup
setup(
    name='pysces',
    version='0.1.0',
    description='Python Single-Cell Expression System',
    author='PySCES Team',
    author_email='example@example.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=ext_modules,
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'anndata>=0.8.0',
        'pybind11>=2.6.0',
    ],
    python_requires='>=3.8',
)
```

### 3. Update Core Implementation
Simplify the import strategy in core.py:

```python
# In core.py
try:
    from ._cpp import aracne_ext
    _has_cpp_ext = True
    print(f"Using C++ extensions for ARACNe (version: {getattr(aracne_ext, '__version__', 'unknown')})")
except ImportError:
    _has_cpp_ext = False
    warnings.warn(
        "Could not import ARACNe C++ extensions. Using slower Python implementation. "
        "To use the faster C++ implementation, rebuild the package with C++ support enabled."
    )
```

### 4. Create Test Utilities Module
Create a common test utilities module for both debug scripts and formal tests:

```python
# In tests/utils.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_test_data():
    """Generate test data for MI calculation tests."""
    test_cases = {
        'perfect_correlation': {
            'x': np.array([1.0, 2.0, 3.0], dtype=np.float64),
            'y': np.array([2.0, 4.0, 6.0], dtype=np.float64),
            'expected': 'positive'
        },
        'perfect_anticorrelation': {
            'x': np.array([1.0, 2.0, 3.0], dtype=np.float64),
            'y': np.array([3.0, 2.0, 1.0], dtype=np.float64),
            'expected': 'positive'
        },
        'constant_arrays': {
            'x': np.ones(10, dtype=np.float64),
            'y': np.ones(10, dtype=np.float64),
            'expected': 'zero'
        },
        'independent': {
            'x': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            'y': np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64),
            'expected': 'zero'
        }
    }
    return test_cases

def verify_mi_calculation(mi_func):
    """Verify MI calculation function with standard test cases."""
    test_cases = generate_test_data()
    results = {}
    
    for name, case in test_cases.items():
        x = case['x']
        y = case['y']
        expected = case['expected']
        
        logger.debug(f"Testing {name}")
        logger.debug(f"x: {x}")
        logger.debug(f"y: {y}")
        
        mi = mi_func(x, y)
        logger.debug(f"MI result: {mi}")
        
        if expected == 'positive':
            assert mi > 0, f"Expected positive MI for {name}, got {mi}"
        elif expected == 'zero':
            assert mi == 0 or np.isclose(mi, 0), f"Expected zero MI for {name}, got {mi}"
        
        results[name] = mi
    
    return results
```

### 5. Update Formal Test Suite
Update the formal test suite to use the test utilities:

```python
# In tests/test_aracne_ext.py
import pytest
import numpy as np
import logging
from pysces.aracne._cpp import aracne_ext
from tests.utils import generate_test_data, verify_mi_calculation

logger = logging.getLogger(__name__)

def test_mi_calculation():
    """Test MI calculation with various test cases."""
    results = verify_mi_calculation(aracne_ext.calculate_mi_ap)
    
    # Additional assertions specific to the formal test suite
    assert results['perfect_correlation'] > results['independent'], \
        "MI for perfect correlation should be higher than for independent variables"

def test_mi_matrix():
    """Test MI matrix calculation."""
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

def test_performance():
    """Test performance of C++ vs Python implementation."""
    import time
    from pysces.aracne.core import ARACNe
    
    # Generate test data
    n_samples = 100
    n_genes = 20
    data = np.random.normal(0, 1, (n_samples, n_genes)).astype(np.float64)
    tf_indices = np.array([0, 1], dtype=np.int32)
    
    # Test C++ implementation
    start_time = time.time()
    mi_matrix_cpp = aracne_ext.calculate_mi_matrix(data, tf_indices)
    cpp_time = time.time() - start_time
    
    # Test Python implementation
    # This would require access to the Python implementation
    # For now, just log the C++ time
    logger.info(f"C++ implementation time: {cpp_time:.4f} seconds")
    
    # In a real test, we would compare the two times
    # assert cpp_time < py_time, "C++ implementation should be faster than Python"
```

### 6. Update Documentation
Add API documentation for the C++ functions:

```markdown
# In docs/API_Reference.md
## ARACNe C++ Extensions

The ARACNe C++ extensions provide high-performance implementations of the core ARACNe algorithms.

### `calculate_mi_ap(x, y, chi_square_threshold=7.815)`

Calculate mutual information between two vectors using adaptive partitioning.

**Parameters:**
- `x` (numpy.ndarray): First vector (1D array of float64)
- `y` (numpy.ndarray): Second vector (1D array of float64)
- `chi_square_threshold` (float, optional): Chi-square threshold for adaptive partitioning. Default: 7.815 (p=0.05)

**Returns:**
- `mi` (float): Mutual information value

**Example:**
```python
import numpy as np
from pysces.aracne._cpp import aracne_ext

x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
y = np.array([2.0, 4.0, 6.0], dtype=np.float64)

mi = aracne_ext.calculate_mi_ap(x, y)
print(f"Mutual information: {mi}")
```

### `calculate_mi_matrix(data, tf_indices, chi_square_threshold=7.815, n_threads=0)`

Calculate mutual information matrix for all gene pairs.

**Parameters:**
- `data` (numpy.ndarray): Expression data matrix (genes x samples, float64)
- `tf_indices` (numpy.ndarray): Indices of transcription factors (1D array of int32)
- `chi_square_threshold` (float, optional): Chi-square threshold for adaptive partitioning. Default: 7.815 (p=0.05)
- `n_threads` (int, optional): Number of threads to use. Default: 0 (auto)

**Returns:**
- `mi_matrix` (numpy.ndarray): Mutual information matrix (TFs x genes, float64)

### `apply_dpi(mi_matrix, tolerance=0.0, n_threads=0)`

Apply Data Processing Inequality to mutual information matrix.

**Parameters:**
- `mi_matrix` (numpy.ndarray): Mutual information matrix (TFs x genes, float64)
- `tolerance` (float, optional): Tolerance for DPI. Default: 0.0
- `n_threads` (int, optional): Number of threads to use. Default: 0 (auto)

**Returns:**
- `pruned_matrix` (numpy.ndarray): Pruned mutual information matrix (TFs x genes, float64)

### `run_aracne_bootstrap(data, tf_indices, n_bootstraps=100, chi_square_threshold=7.815, dpi_tolerance=0.0, consensus_threshold=0.05, n_threads=0)`

Run ARACNe with bootstrapping.

**Parameters:**
- `data` (numpy.ndarray): Expression data matrix (genes x samples, float64)
- `tf_indices` (numpy.ndarray): Indices of transcription factors (1D array of int32)
- `n_bootstraps` (int, optional): Number of bootstrap iterations. Default: 100
- `chi_square_threshold` (float, optional): Chi-square threshold for adaptive partitioning. Default: 7.815 (p=0.05)
- `dpi_tolerance` (float, optional): Tolerance for DPI. Default: 0.0
- `consensus_threshold` (float, optional): Threshold for consensus network. Default: 0.05
- `n_threads` (int, optional): Number of threads to use. Default: 0 (auto)

**Returns:**
- `consensus_matrix` (numpy.ndarray): Consensus matrix (TFs x genes, float64)
```

## Expected Outcome
After completing these steps, the PySCES package should:
- Successfully use the fixed C++ extensions by default
- Build correctly on different platforms with appropriate compiler detection
- Pass all tests, including those that use the fixed extensions
- Provide clear API documentation for the C++ functions
- Show significant performance improvements compared to the Python implementation
