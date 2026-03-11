"""
Holeyfield v2.0 — Byzantine Stability
======================================
Unitary Regulator for topological phase-transition inference.
"""

from setuptools import setup, find_packages

setup(
    name="unitarity-lab",
    version="2.0.0",
    description=(
        "Holeyfield v2.0: Unitary Regulator with Byzantine Kill-Switch, "
        "Precision Alignment, and Adaptive Gossip Epoch for distributed "
        "topological inference."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="holeyfield33-art",
    url="https://github.com/holeyfield33-art/unitarity-lab",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "pyzmq>=25.0",
        "safetensors>=0.4",
        "msgpack>=1.0",
        "rich>=13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
        ],
        "community": [
            "transformers>=4.38",
            "pynvml",
        ],
    },
    entry_points={
        "console_scripts": [
            "holeyfield-node=start_node:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
