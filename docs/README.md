# PySCES Documentation

This directory contains documentation for the PySCES (Python Single-Cell Expression System) project. The documentation is organized to provide a comprehensive overview of the project, its implementation details, and usage guidelines.

## Contents

### PySCES_Project_Status.md

A comprehensive overview of the project's current status, implementation details, and future plans. This document serves as the primary reference for understanding the current state of the project and its roadmap.

Key sections:
- Current Status: Completed, in-progress, and pending components
- Implementation Challenges: Challenges encountered and solutions implemented
- Project Structure: Organization of the codebase
- Next Steps: Short-term, medium-term, and long-term goals

### census-data-loaders-docs

Documentation for the CELLxGENE Census data loaders, including usage examples, API reference, and best practices. This document provides detailed information about integrating with the Census API and working with Census data.

### engineer_prompt.md

Engineering guidelines and best practices for contributing to the PySCES project. This document provides information about coding standards, architecture decisions, and development workflows.

### pysces_implementation_plan.md

The original implementation plan for the PySCES project, including the system architecture, project setup, and implementation phases. This document serves as a reference for the initial design and planning of the project.

### pysces_implementation_plan_update.md

Updates to the implementation plan based on progress and new insights. This document provides information about changes to the original plan and adjustments to the project roadmap.

### validation_tests.md

A comprehensive plan for validating the PySCES pipeline, including input data validation, algorithm validation, and edge case handling. This document outlines the tests that should be implemented to ensure the correctness and robustness of the pipeline.

### test_status.md

A document outlining the current status of tests in the PySCES project and the plan for improving test coverage. This document provides information about the current state of unit tests, manual validation, and the plan for implementing comprehensive validation tests.

### implementation_gameplan.md

A detailed implementation plan for the next phase of the PySCES project, focusing on three key areas: input data structure validation, comprehensive pipeline testing, and MLX/GPU acceleration for VIPER. This document outlines the objectives, implementation plans, and expected outcomes for each focus area.

### pipeline_validation_fixes.md

A summary of the changes made to fix the pipeline validation issues in the PySCES project. This document describes the issues encountered, the changes made to resolve them, and the results of these changes.

### debugging_pipeline_validation.md

A detailed account of the debugging process for the pipeline validation issues. This document provides context, describes the issues encountered, the attempted solutions, and the final resolution.

### repository_cleanup.md

A summary of the cleanup and organization of the PySCES repository. This document describes the changes made to organize the repository, including archiving outdated scripts, moving test scripts to the appropriate directory, organizing documentation, and updating documentation files.

### performance_optimizations.md

A comprehensive documentation of the performance optimizations implemented in PySCES, including Numba acceleration for both ARACNe and VIPER algorithms, MLX integration for Apple Silicon, and detailed benchmarking results. This document explains how the optimizations work, why speedup increases with dataset size, and the practical impact for scientists analyzing large-scale single-cell datasets.

### images/

A directory containing images used in the documentation and examples, including:
- **simple_viper_test.png**: Visualization of VIPER test results
- **viper_anndata_test.png**: Visualization of VIPER results with AnnData
- **aracne_benchmark.png**: Benchmark comparison of Python vs. Numba implementations for ARACNe
- **aracne_speedup.png**: Speedup factors for ARACNe across different dataset sizes
- **viper_benchmark.png**: Benchmark comparison of Python vs. Numba implementations for VIPER
- **viper_speedup.png**: Speedup factors for VIPER across different dataset sizes
- **speedup_comparison.png**: Comparison of speedup factors between ARACNe and VIPER

## Additional Documentation

Additional documentation can be found in the following locations:

- **API Documentation**: Generated from docstrings in the code (coming soon)
- **Example Notebooks**: Located in the `examples` directory
- **README.md**: Project overview and quick start guide
- **CONTRIBUTING.md**: Guidelines for contributing to the project
- **ROADMAP.md**: Detailed roadmap for the project

## Contributing to Documentation

Contributions to the documentation are welcome! When contributing, please follow these guidelines:

1. Use clear and concise language
2. Provide examples where appropriate
3. Keep the documentation up-to-date with the code
4. Follow the existing structure and formatting
5. Add cross-references to related documentation

## Building Documentation

(Coming soon) Instructions for building the documentation locally and generating API documentation.
