"""unitarity-labs — version shim for pyproject.toml dynamic versioning."""

from setuptools import setup, find_packages

# Internal version metadata — the human-readable release name.
_META = dict(version="3.1.3-Singularity")

if __name__ == "__main__":
    setup(packages=find_packages(exclude=["tests", "tests.*", "benchmarks", "benchmarks.*"]))
