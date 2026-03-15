"""unitarity-labs — version shim for pyproject.toml dynamic versioning."""

from setuptools import setup

# Internal version metadata — the human-readable release name.
_META = dict(version="3.0.0-Singularity")

if __name__ == "__main__":
    setup()
