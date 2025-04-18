#!/usr/bin/env python
"""
Setup script for compiling the fixed ARACNe C++ extensions.
"""

from setuptools import setup, Extension
import pybind11
import os
import sys
import platform

# Get the pybind11 include directory
pybind11_include = pybind11.get_include()

# Set compiler flags based on platform
extra_compile_args = ['-std=c++11']
extra_link_args = []

# Add OpenMP flags if not on macOS
if platform.system() != 'Darwin':
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')

# Define the extension module
ext_modules = [
    Extension(
        'pysces.aracne._cpp.aracne_ext_fixed',
        ['pysces/src/pysces/aracne/_cpp/aracne_ext_fixed.cpp'],
        include_dirs=[
            pybind11_include,
            'pysces/src/pysces/aracne/_cpp/include'
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

# Setup
setup(
    name='pysces-aracne-fixed',
    version='0.1',
    description='Fixed ARACNe C++ extensions for PySCES',
    ext_modules=ext_modules,
)
