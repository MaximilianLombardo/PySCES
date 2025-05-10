from setuptools import setup, find_packages

# No C++ extensions are needed for the Numba-accelerated implementation
ext_modules = []

setup(
    name="pysces",
    version="0.3.0",  # Increment version for Numba-focused architecture
    description="Python Single-Cell Expression System",
    author="PySCES Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,

    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.8",
        "pandas>=1.4",
        "scikit-learn>=1.0",
        "anndata>=0.8",
        "scanpy>=1.9",
        "matplotlib>=3.5",
        "seaborn>=0.12",
        "numba>=0.56",  # Required for performance optimization
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "sphinx>=5.0",
            "black",
            "isort",
            "flake8",
        ],
        "census": [
            "cellxgene-census[experimental]>=0.2.1",
            "tiledb-soma",
        ],
    },
)
