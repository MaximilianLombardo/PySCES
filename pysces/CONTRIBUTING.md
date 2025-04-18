# Contributing to PySCES

Thank you for your interest in contributing to PySCES! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with the following information:

- A clear, descriptive title
- A detailed description of the bug
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Environment information (OS, Python version, package versions)

### Suggesting Features

If you have an idea for a new feature, please create an issue on GitHub with the following information:

- A clear, descriptive title
- A detailed description of the feature
- Any relevant background information or use cases
- If possible, a sketch of how the feature might be implemented

### Pull Requests

We welcome pull requests! Here's how to submit one:

1. Fork the repository
2. Create a new branch for your changes
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Submit a pull request

Please include the following in your pull request:

- A clear, descriptive title
- A detailed description of the changes
- Any relevant issue numbers
- Tests for new functionality

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/pysces.git
   cd pysces
   ```

2. Create and activate a conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate pysces
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Run tests:
   ```bash
   pytest
   ```

## Package Structure

PySCES uses a modern "src-layout" package structure:

```
pysces/
├── src/                         # Source directory
│   └── pysces/                  # Main package
│       ├── data/                # Data handling
│       ├── aracne/              # ARACNe implementation
│       ├── viper/               # VIPER implementation
│       ├── analysis/            # Analysis tools
│       └── plotting/            # Visualization
├── tests/                       # Test suite
├── examples/                    # Example notebooks
```

This structure has several advantages:
- Prevents import confusion during development
- Makes the installation process more reliable
- Clearly separates package code from project files

When developing, remember:
- All package code goes in `src/pysces/`
- Import the package as `import pysces` (not `import pysces.pysces`)
- Tests should import from the installed package, not using relative imports

## Coding Standards

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings in NumPy format
- Write tests for new functionality
- Keep functions and methods focused on a single responsibility

## Git Workflow

1. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   ```

3. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Submit a pull request to the main repository

## Testing

- Write tests for new functionality
- Run tests before submitting a pull request
- Tests should be placed in the `tests/` directory

## Documentation

- Update documentation for any changes to the API
- Write clear, concise docstrings
- Include examples where appropriate

## License

By contributing to PySCES, you agree that your contributions will be licensed under the project's MIT license.
