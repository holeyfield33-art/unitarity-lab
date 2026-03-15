"""unitarity-lab — version shim for pyproject.toml dynamic versioning."""

import re
from setuptools import setup


def get_version():
    with open("core/version.py") as f:
        return re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', f.read()).group(1)


if __name__ == "__main__":
    setup(version="3.0.0-Singularity")
