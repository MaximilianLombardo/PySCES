import os
import sys
import platform
from setuptools import setup, find_packages, Extension

# Detect the platform
system = platform.system()
is_windows = (system == "Windows")
is_mac = (system == "Darwin")
is_linux = (system == "Linux")

# Set compiler flags based on platform
extra_compile_args = ["-O3", "-std=c++17"]
extra_link_args = []

# Add OpenMP support
if is_windows:
    extra_compile_args.extend(["/openmp"])
elif is_mac:
    # macOS requires special handling for OpenMP
    # Apple Clang doesn't support OpenMP by default
    # We'll try to use libomp from conda-forge or homebrew
    if os.path.exists("/usr/local/opt/libomp"):
        # Homebrew libomp location
        extra_compile_args.extend(["-Xpreprocessor", "-fopenmp"])
        extra_link_args.extend(["-lomp", "-L/usr/local/opt/libomp/lib"])
        os.environ["CPPFLAGS"] = "-I/usr/local/opt/libomp/include"
    elif os.path.exists(os.path.join(sys.prefix, "include", "omp.h")):
        # Conda libomp location
        extra_compile_args.extend(["-Xpreprocessor", "-fopenmp"])
        extra_link_args.extend(["-lomp", f"-L{os.path.join(sys.prefix, 'lib')}"])
    else:
        print("Warning: OpenMP not found. ARACNe will run in single-threaded mode.")
else:  # Linux and other Unix-like systems
    extra_compile_args.extend(["-fopenmp"])
    extra_link_args.extend(["-fopenmp"])

# Define the extension module with error handling
try:
    import pybind11
    pybind11_include = pybind11.get_include()

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
    print("C++ extensions enabled.")
except Exception as e:
    print(f"Error setting up C++ extensions: {str(e)}")
    print("Falling back to Python-only installation")
    ext_modules = []

setup(
    name="pysces",
    version="0.2.0",  # Increment version for directory structure change
    description="Python Single-Cell Expression System",
    author="PySCES Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    # Temporarily disable build_ext
    # cmdclass={"build_ext": build_ext},

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
        "numba>=0.56",  # For performance optimization
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
