# Repository Cleanup

This document summarizes the cleanup and organization of the PySCES repository.

## Changes Made

### 1. Archived Debug and Test Scripts

Moved outdated debug and test scripts related to the C++ extensions to the `archive/debug_scripts/` directory:

- `debug_mi.py`: Debug script for mutual information calculation issues
- `fix_mi.py`: Script for fixing mutual information calculation issues
- `setup_fixed.py`: Setup script for compiling the fixed ARACNe C++ extensions
- `test_fixed.py`: Test script for the fixed ARACNe C++ extensions
- `test_import.py`: Script for testing import paths for the ARACNe class

Added a README.md file to the archive directory to explain what these scripts are.

### 2. Moved Test Scripts to Tests Directory

Moved test scripts to the appropriate `tests/` directory:

- `test_aracne.py`: Test script for the ARACNe algorithm implementation
- `test_aracne_real_data.py`: Test script for the ARACNe algorithm with real data

### 3. Organized Documentation

Moved documentation files to the `docs/` directory:

- `debugging_pipeline_validation.md`: Detailed account of the debugging process for pipeline validation issues

Created a new documentation file:

- `pipeline_validation_fixes.md`: Summary of the changes made to fix the pipeline validation issues

### 4. Organized Images

Created a `docs/images/` directory and moved image files there:

- `simple_viper_test.png`: Visualization of VIPER test results
- `viper_anndata_test.png`: Visualization of VIPER results with AnnData

### 5. Updated Documentation

Updated the following documentation files to reflect the changes:

- `README.md`: Updated to reflect the current state of the project and the new organization
- `docs/README.md`: Updated to include the new documentation files
- `archive/debug_scripts/README.md`: Added to explain the archived scripts

### 6. Disabled C++ Extensions

Modified the ARACNe implementation to always use the Python implementation:

- Updated `pysces/src/pysces/aracne/core.py` to disable C++ extensions
- Updated `pysces/src/pysces/aracne/_cpp/__init__.py` to reflect this change
- Updated `pysces/src/pysces/aracne/_cpp/README.md` to indicate the C++ extensions are archived

### 7. Added 'edges' Key to Network Dictionary

Modified the ARACNe implementation to include an 'edges' key in the network dictionary:

- Updated `pysces/src/pysces/aracne/core.py` to create an 'edges' list in the `_process_results` method
- Removed the redundant `add_edges_to_network` function from the pipeline validation test script

### 8. Created Direct Import Version of Pipeline Validation Test

Created a version of the pipeline validation test script that uses direct imports:

- Created `examples/pipeline_validation_test_direct.py` with direct imports
- Updated the original script to reference the new direct import version

## Results

The repository is now more organized and easier to navigate:

- Debug and test scripts are properly archived
- Documentation is centralized in the `docs/` directory
- Test scripts are in the `tests/` directory
- Images are organized in the `docs/images/` directory
- The ARACNe implementation is more reliable and easier to debug
- The pipeline validation test now runs successfully

## Next Steps

With the repository now cleaned up and organized, we can focus on the next steps:

1. Performance optimization of the Python implementation
2. MLX/GPU acceleration
3. Comprehensive testing
4. Enhanced documentation and examples
5. Additional features
6. User feedback and community building
