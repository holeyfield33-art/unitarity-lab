# Contributing to unitarity-lab

Thanks for your interest in contributing.

## Ways to contribute

- Report bugs and edge cases
- Propose features and API improvements
- Improve documentation and examples
- Add tests and benchmark scenarios
- Improve performance or numerical stability

## Development setup

1. Fork and clone the repository.
2. Create a feature branch from `main`.
3. Create and activate a Python environment.
4. Install dependencies in editable mode.

```bash
git clone https://github.com/holeyfield33-art/unitarity-lab.git
cd unitarity-lab
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quality checks

Run tests before opening a pull request:

```bash
pytest -q tests
```

If you changed the audit site or front-end behavior, run browser smoke tests:

```bash
npm run test:e2e
```

If you changed notebooks, ensure the notebook executes locally or in CI.

## Pull request guidelines

- Keep PRs focused and small when possible.
- Add or update tests for behavior changes.
- Update docs for user-facing changes.
- Include clear rationale in the PR description.
- Link related issues.

## Commit style

Use clear, imperative commit messages:

- `feat: add new spectral monitor hook`
- `fix: prevent null input in audit classifier`
- `docs: clarify active mode behavior`

## Reporting security issues

Do not open public issues for vulnerabilities.

Follow the process in `SECURITY.md`.
