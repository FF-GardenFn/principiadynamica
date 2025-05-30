#!/usr/bin/env python3
"""
Setup script for Constitutional Dynamics package
"""

from setuptools import setup, find_packages
import os

# Get version from constitutional_dynamics/__init__.py
with open(os.path.join(os.path.dirname(__file__), "constitutional_dynamics", "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break
    else:
        version = "0.1.0"
with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as f:
    long_description = f.read()

setup(
    name="constitutional-dynamics",
    version=version,
    description="Constitutional Dynamics: An application for AI alignment analysis using State Transition Calculus principles from the PrincipiaDynamica project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="FF-GardenFn",
    author_email="faycal.farhat@mail.mcgill.ca",
    url="https://github.com/FF-GardenFn/principiadynamica",
    packages=find_packages(exclude=["legacy", "legacy.*", "*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    package_data={
        "constitutional_dynamics.cfg": ["*.yaml"],
    },
    install_requires=[
        "numpy",
        "scipy",
        "pyyaml",
        "rich",
    ],
    extras_require={
        "graph": ["neo4j"],
        "quantum": ["dwave-ocean-sdk"],
        "dev": ["pytest", "black", "ruff"],
    },
    entry_points={
        "console_scripts": [
            "constitutional-dynamics=constitutional_dynamics.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
