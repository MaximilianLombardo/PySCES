# Archived ARACNe C++ Files

This directory contains archived C++ files from the ARACNe implementation that are no longer actively used but are kept for reference.

## Files

- `aracne_ext_fixed.cpp`: An earlier version of the ARACNe C++ extension with fixes for various issues. This file has been superseded by the current implementation in `pysces/src/pysces/aracne/_cpp/aracne_ext.cpp`.

## Notes

The C++ implementation of ARACNe is being maintained for backward compatibility and performance reasons, but we are exploring alternatives such as MLX and GPU acceleration for future versions. The Python fallback implementation is now the recommended approach for most users due to its improved maintainability and compatibility across platforms.

If you need to use the C++ extensions, please refer to the current implementation in `pysces/src/pysces/aracne/_cpp/`.
