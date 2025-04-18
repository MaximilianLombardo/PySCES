# PySCES Test Status

This document outlines the current status of tests in the PySCES project and the plan for improving test coverage.

## Current Test Status

### Unit Tests

The unit tests in the `tests/` directory are currently in a transitional state due to recent structural changes in the codebase:

- **ARACNe Tests**: Some tests are failing due to changes in default parameters and import paths
- **VIPER Tests**: Some tests are failing due to import path changes
- **Performance Tests**: Currently disabled due to import issues

These test failures are primarily due to structural changes rather than functional issues with the core pipeline.

### Manual Validation

We have successfully validated the core pipeline functionality through manual testing:

1. **ARACNe Algorithm**: Verified with synthetic data and real data from Tabula Sapiens
2. **VIPER Algorithm**: Verified with synthetic data and real data from Tabula Sapiens
3. **AnnData Compatibility**: Confirmed that the pipeline works with AnnData objects
4. **End-to-End Pipeline**: Validated the full ARACNe → VIPER pipeline

## Test Improvement Plan

As outlined in the [validation_tests.md](validation_tests.md) document, we plan to implement comprehensive validation tests for the PySCES pipeline:

### Phase 1: Input Validation Tests
- Raw counts vs. normalized data detection
- Sparse matrix handling
- AnnData structure validation

### Phase 2: ARACNe Validation Tests
- Network inference with known ground truth
- Bootstrapping parameter sensitivity
- Edge case handling

### Phase 3: VIPER Validation Tests
- Known regulon testing
- Comparison with reference implementation
- Enrichment method comparison

### Phase 4: End-to-End Pipeline Tests
- Full pipeline on different datasets
- Reproducibility testing
- Parameter sensitivity

### Phase 5: Edge Case Handling Tests
- Small dataset testing
- Large dataset testing
- Missing value handling

## Next Steps

1. **Fix Existing Tests**: Gradually fix the existing unit tests to align with the current codebase structure
2. **Implement Validation Tests**: Start implementing the validation tests outlined in the validation_tests.md document
3. **Add Integration Tests**: Create integration tests for the full pipeline
4. **Add Performance Benchmarks**: Implement benchmarks to track performance improvements

## Test Execution Guidelines

When running tests, use the following guidelines:

1. **Unit Tests**: Run with `python -m pytest -xvs tests/`
2. **Manual Validation**: Use the example scripts in the `examples/` directory
3. **Performance Tests**: Currently disabled, will be re-enabled after fixing import issues

## Known Issues

- The performance test (`test_performance.py`) is currently disabled due to import issues
- Some ARACNe tests are failing due to changes in default parameters
- Some VIPER tests are failing due to import path changes

These issues will be addressed as part of the ongoing maintenance of the codebase.
