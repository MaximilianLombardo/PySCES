# PySCES Tests

This directory contains tests for the PySCES package, focused on the Numba-accelerated implementation.

## Directory Structure

- `unit/`: Unit tests for individual components
  - `test_aracne.py`: Tests for ARACNe implementation
  - `test_viper.py`: Tests for VIPER implementation
  - `test_imports.py`: Tests for package imports
  - `test_validation.py`: Tests for data validation utilities

- `integration/`: Integration tests for end-to-end workflows

## Running Tests

Run all tests:
```
python -m pytest tests/
```

Run specific test file:
```
python -m pytest tests/unit/test_aracne.py
```

Run with verbose output:
```
python -m pytest -v tests/
```

## Notes

- Tests are focused on the Numba implementation, which is the primary backend
- Python fallback implementation is also tested when Numba is not available
- Alternative implementations (PyTorch, MLX) have been moved to the experimental directory
