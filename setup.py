#!/usr/bin/env python3
"""
Setup configuration for the anemll_bench package
"""

from setuptools import setup, find_packages
import os
import re


def get_version():
    """Extract version from __init__.py"""
    init_py = os.path.join('anemll_bench', '__init__.py')
    with open(init_py, 'r', encoding='utf-8') as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def get_long_description():
    """Get long description from README"""
    with open('README.md', encoding='utf-8') as f:
        return f.read()


def get_requirements():
    """Get requirements from requirements.txt"""
    with open('requirements.txt', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


setup(
    name="anemll-bench",
    version=get_version(),
    description="Benchmarking tools for Apple Neural Engine performance",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="ANEMLL Team",
    author_email="contact@anemll.org",
    url="https://github.com/anemll/anemll-bench",
    packages=find_packages(),
    include_package_data=True,
    install_requires=get_requirements(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: MacOS :: MacOS X",
    ],
    keywords="machine learning, benchmarking, apple neural engine, ML, ANE, CoreML",
) 