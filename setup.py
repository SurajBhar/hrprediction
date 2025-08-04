#!/usr/bin/env python
"""
Setup script for the hrprediction package.

This module uses setuptools to configure the packaging and distribution
of the hrprediction project. It reads project metadata and dependencies
and invokes setuptools.setup() to handle installation.

Author: Suraj Bhardwaj
License: MIT
"""
import io
import os
from setuptools import setup, find_packages

# Determine the project root directory
here = os.path.abspath(os.path.dirname(__file__))

def read_file(filename):
    """
    Read and return the contents of a text file from the project root.

    Args:
        filename (str): Name of the file to read.

    Returns:
        str: File contents.
    """
    filepath = os.path.join(here, filename)
    with io.open(filepath, encoding="utf-8") as f:
        return f.read()

# Project metadata
PACKAGE_NAME = "hrprediction"
VERSION = "0.1.1"
DESCRIPTION = "Hotel reservation prediction package."
LONG_DESCRIPTION = read_file("README.md")
AUTHOR = "Suraj Bhardwaj"
AUTHOR_EMAIL = "suraj.unisiegen@gmail.com"
URL = "https://github.com/SurajBhar/hrprediction"
LICENSE = "MIT"
REQUIRES_PYTHON = ">=3.8"

# Load dependencies from requirements.txt, ignoring comments and empty lines
install_requires = [
    line.strip()
    for line in read_file("requirements.txt").splitlines()
    if line and not line.strip().startswith("#")
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    license=LICENSE,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=[]),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
