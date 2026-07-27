"""test_validate_notebooks.py — the notebook guard must catch what shipped.

Every "must reject" case below is a state this repo actually committed at some
point, and every "must accept" case is a correct line the guard has to leave
alone. Without these, a slightly-too-greedy regex would either wave the real
defects through or block the correct install command.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# validate_notebooks imports nbformat, which ships in the optional `dev` extra.
pytest.importorskip("nbformat")

from validate_notebooks import (  # noqa: E402
    BAD_UNITARITY_INSTALL,
    BAD_VAR_INSTALL,
    FORBIDDEN_IN_CODE,
    _executable_lines,
)


class TestUnitarityInstallPattern:
    @pytest.mark.parametrize("line", [
        "!pip install unitarity-lab",
        "!pip install -q unitarity-lab",
        '!pip install "unitarity-lab[dist]"',
        "pip install 'unitarity-lab[dist]'",
    ])
    def test_rejects_missing_trailing_s(self, line):
        assert BAD_UNITARITY_INSTALL.search(line), line

    @pytest.mark.parametrize("line", [
        "!pip install unitarity-labs",
        "!pip install -q unitarity-labs==3.2.0",
        '!pip install "unitarity-labs[bench,spectral]"',
        # The repo genuinely has no trailing `s`, so a git URL is correct.
        '!pip install "unitarity-labs @ git+https://github.com/x/unitarity-lab@main"',
        "!pip install -q 'unitarity-labs @ git+https://github.com/a/unitarity-lab.git@abc'",
    ])
    def test_accepts_correct_forms(self, line):
        assert not BAD_UNITARITY_INSTALL.search(line), line


class TestVarInstallPattern:
    @pytest.mark.parametrize("line", [
        "!pip install var",
        "!pip install -q var",
    ])
    def test_rejects_bare_var(self, line):
        assert BAD_VAR_INSTALL.search(line), line

    @pytest.mark.parametrize("line", [
        "!pip install var-spectral",
        "!pip install -q var-spectral>=1.1.0",
        '!pip install "var-spectral @ git+https://github.com/x/VAR@main"',
        "!pip install variance-tools",
    ])
    def test_accepts_correct_forms(self, line):
        assert not BAD_VAR_INSTALL.search(line), line


class TestExecutableLines:
    def test_comments_are_ignored(self):
        source = (
            "# pip install unitarity-lab fails -- the name has no trailing s\n"
            "!pip install unitarity-labs\n"
        )
        lines = _executable_lines(source)
        assert len(lines) == 1
        assert not any(BAD_UNITARITY_INSTALL.search(l) for l in lines)

    def test_blank_lines_dropped(self):
        assert _executable_lines("\n\n  \n") == []


class TestForbiddenTokens:
    @pytest.mark.parametrize("token", ["REPLACE_WITH", "random.randn"])
    def test_known_placeholders_are_listed(self, token):
        assert token in FORBIDDEN_IN_CODE


def test_repository_notebooks_pass():
    """The notebooks actually committed here must satisfy the guard."""
    from validate_notebooks import validate_notebook

    notebooks = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
    assert notebooks, "no notebooks found to validate"
    for path in notebooks:
        assert validate_notebook(path) == [], path
