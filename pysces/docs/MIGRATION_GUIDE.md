# PySCES Migration Guide

This document provides guidance for migrating from PySCES 0.1.x to PySCES 0.2.0, which introduces a new package directory structure.

## Package Structure Changes

In version 0.2.0, we've restructured the package to use a modern "src-layout" approach:

**Old structure (v0.1.x):**
```
pysces/
├── pysces/                      # Main package
│   ├── data/                    # Data handling
│   ├── aracne/                  # ARACNe implementation
│   ├── viper/                   # VIPER implementation
│   ├── analysis/                # Analysis tools
│   └── plotting/                # Visualization
```

**New structure (v0.2.0+):**
```
pysces/
├── src/                         # Source directory
│   └── pysces/                  # Main package
│       ├── data/                # Data handling
│       ├── aracne/              # ARACNe implementation
│       ├── viper/               # VIPER implementation
│       ├── analysis/            # Analysis tools
│       └── plotting/            # Visualization
```

## What This Means for Users

### No Changes to Import Statements

If you were using the recommended import approach, your code should continue to work without changes:

```python
# This still works in v0.2.0
import pysces
from pysces import ARACNe, viper, preprocess_data
```

### Changes for Direct Imports

If you were using direct imports from submodules, you may need to update your code:

```python
# Old approach (v0.1.x)
from pysces.pysces.data import preprocessing
from pysces.pysces.aracne import ARACNe

# New approach (v0.2.0+)
from pysces.data import preprocessing
from pysces.aracne import ARACNe
```

### Changes for Development

If you're developing PySCES or extending it:

1. All package code now goes in `src/pysces/`
2. Use `pip install -e .` for development installation
3. Import the package as `import pysces` (not `import pysces.pysces`)

## Benefits of the New Structure

The new src-layout structure provides several benefits:

1. **Cleaner imports**: Eliminates confusion between `pysces` and `pysces.pysces`
2. **Development isolation**: Prevents accidental imports from the development directory
3. **Installation reliability**: Makes the installation process more robust
4. **Best practices**: Follows modern Python packaging recommendations

## Need Help?

If you encounter any issues migrating to the new version, please:

1. Check the [examples](../examples/) directory for updated usage examples
2. File an issue on our GitHub repository
