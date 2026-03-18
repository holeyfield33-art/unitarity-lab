"""unitarity-labs — version shim for pyproject.toml dynamic versioning."""

from setuptools import setup

# Internal version metadata — the human-readable release name.
_META = dict(version="3.1.2-Singularity")

if __name__ == "__main__":
    setup()
