#!/usr/bin/env python3
"""scripts/validate_notebooks.py — guard against notebooks that ship broken.

The notebooks under ``notebooks/`` target Colab: their install cells pull from
git and download model weights, so *executing* them in CI would mostly test
GitHub's network. What CI can check cheaply is that a notebook cannot be
committed in a state that fails on contact, which is how the previous starter
notebook stayed green for so long while emitting placeholder numbers.

Three failure modes are checked, each one previously shipped:

``REPLACE_WITH_*``
    An unresolved placeholder token. ``exp1_prompt_separation.ipynb`` shipped
    with ``COMMIT_SHA = "REPLACE_WITH_COMMIT_SHA"``, so its very first cell
    failed for anyone who opened it.

``random.randn`` in a code cell
    A placeholder data generator. The old ``gemma4_starter.ipynb`` caught every
    exception and fell through to ``np.random.randn``, so a completely failed
    run still printed a plausible "coherence score" — a different one each
    time.

Wrong package name in an install command
    The repo is ``unitarity-lab`` but the distribution is ``unitarity-labs``,
    and ``pip install unitarity-lab`` therefore cannot succeed. Likewise VAR
    installs as ``var-spectral``; plain ``var`` is an unrelated project.

Only executable lines are scanned for install mistakes — a ``#`` comment or
markdown prose is free to name the wrong spelling in order to warn about it.

Usage::

    python scripts/validate_notebooks.py [notebook_dir]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

import nbformat

#: Tokens that must never appear in a code cell.
FORBIDDEN_IN_CODE = {
    "REPLACE_WITH": "unresolved placeholder token",
    "random.randn": "placeholder data generator masquerading as a measurement",
    "random.rand(": "placeholder data generator masquerading as a measurement",
}

# The lookbehind excludes `/`, `.` and `-` so that a *URL* containing the repo
# name is not flagged: `git+https://github.com/holeyfield33-art/unitarity-lab`
# is correct precisely because the repository, unlike the distribution, has no
# trailing `s`.

#: `pip install unitarity-lab` with no trailing `s` — not a real distribution.
BAD_UNITARITY_INSTALL = re.compile(
    r"pip\s+install\b[^\n]*?(?<![\w./-])unitarity-lab(?![s\w-])"
)
#: `pip install var` — the unrelated Value-at-Risk project.
BAD_VAR_INSTALL = re.compile(
    r"pip\s+install\b[^\n]*?(?<![\w./-])var(?![-_\w])"
)


def _executable_lines(source: str) -> List[str]:
    """Lines that actually run, excluding comments.

    A comment naming the wrong spelling is documentation, not a defect — this
    file and several notebook cells deliberately do exactly that.
    """
    out = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(line)
    return out


def validate_notebook(path: Path) -> List[str]:
    """Return a list of human-readable failures for one notebook."""
    failures: List[str] = []
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        source = "".join(cell.source)

        for token, why in FORBIDDEN_IN_CODE.items():
            if token in source:
                failures.append(f"{path}:cell{index}: contains {token!r} — {why}")

        for line in _executable_lines(source):
            if BAD_UNITARITY_INSTALL.search(line):
                failures.append(
                    f"{path}:cell{index}: installs 'unitarity-lab' — the "
                    "distribution is 'unitarity-labs' (trailing s); the repo "
                    "name is what drops it"
                )
            if BAD_VAR_INSTALL.search(line):
                failures.append(
                    f"{path}:cell{index}: installs 'var' — VAR's distribution "
                    "is 'var-spectral'; plain 'var' is an unrelated project"
                )

    return failures


def main(argv: List[str]) -> int:
    directory = Path(argv[1]) if len(argv) > 1 else Path("notebooks")
    notebooks = sorted(directory.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {directory}", file=sys.stderr)
        return 1

    all_failures: List[str] = []
    for path in notebooks:
        failures = validate_notebook(path)
        status = "FAIL" if failures else "OK"
        print(f"[{status}] {path}")
        all_failures.extend(failures)

    if all_failures:
        print("\nNotebook validation failed:", file=sys.stderr)
        for failure in all_failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"\nAll {len(notebooks)} notebook(s) validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
