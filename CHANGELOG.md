# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## v3.2.1 — Packaging fixes (fresh checkout installs and tests cleanly)

Fixed the "every fresh clone is broken" class of bug: a base `pip install` +
`pytest`/`audit_suite` now works without silent failures.

- **`reedsolo` moved from the `dist` extra to core dependencies.** It is used by
  `core.chronos_lock`, which `benchmarks.audit_suite` exercises in single-node
  mode — so a base install running the (reproducibility-claim) audit suite was
  erroring 5 of its 16 checks with `ModuleNotFoundError: reedsolo`. The audit
  suite now runs complete on a plain `pip install`.
- **Optional-feature tests now skip instead of erroring collection.**
  `test_dual_link` (zmq), `test_passive_hook` (var_spectral), `test_chronos_lock`
  (reedsolo), `test_validate_notebooks` (nbformat), and the dual-link case in
  `test_bf16_paths` guard their imports with `pytest.importorskip`. A core
  install reports `399 passed, 6 skipped` (was: 4 collection errors); a full
  `[all]` install plus VAR reports `450 passed`.
- **Added `dev` extra** (`pytest`, `nbformat`) and an **`all` extra** bundling
  `dist`+`spectral`+`bench`+`dev`. README documents the core-vs-full install.
- Cross-repo dependency `var-spectral>=1.1.0` now resolves: the VAR repo gained
  a `var_spectral` import alias at v1.1.0 (previously it only published `var`,
  so the `spectral` extra could never import).

## v4.0.0-slim — Phase 2 checkpoint (depend on VAR, don't duplicate)

- Added `var` (https://github.com/holeyfield33-art/VAR) as a real dependency,
  pinned to commit `3123455` (VAR's Phase 1 checkpoint) — pinned to a commit
  rather than a tag because this session's git proxy rejects tag-ref pushes;
  swap in `@v1.0.1` once that tag exists upstream.
- Added `unitarity_labs/core/passive_hook.py`: `PassiveTelemetryHook`, a
  single read-only wrapper around a passive-mode `UniversalHookWrapper`
  exposing exactly `{zeta_raw, spectral_gap, flagged}`. `flagged` is driven
  by VAR's `SpectralRuptureDetector` (calibrated median/MAD baseline +
  hysteresis) fed with the wrapper's per-step `spectral_gap` — a real
  rupture check, not a hardcoded value.
- Audited `spectral_monitor.py`, `bocpd.py`, `pll_monitor.py`, and
  `validator.py`'s `log_audit` for logic that literally duplicates VAR's
  rolling W2 eigen-tracker / Merkle chain. None found: they solve adjacent
  but distinct problems (GUE r-ratio criticality checks used by
  ghost_layer/virtual_layer13/mirror, and a plain non-chained JSON audit
  trail) with their own test coverage, so nothing was deleted. The new
  dependency point is the passive hook above.
- No changes to `universal_hook.py`'s forward/hook logic — passive mode
  remains byte-identical (`tests/test_hardening.py::TestPassiveMode` and
  the new `tests/test_passive_hook.py` both green: 427 passed, 2 skipped).
- No changes to package version (stays 3.1.7 in `pyproject.toml`) — this
  is a checkpoint label, not a package release.

## [3.1.7] - 2026-04-14

### Added

- Security policy document with threat model and reporting process.
- Notebook validation and execution workflow in GitHub Actions.
- Self-Serve Audit Hub guidance for HuggingFace model audit flow.
- Expanded educational method cards for GUE r-statistic and Berry-Keating k-value.
- Community health files and contribution workflow documents.

### Changed

- External text proxy flow now requires explicit opt-in consent.
- CSP policy was tightened on the audit landing page.
- Release version metadata updated to 3.1.7 for PyPI publication.

## [3.1.6] - 2026-04-12

### Maintenance

- Colab active-mode runtime tensor placement fixes.

## [3.1.5] - 2026-04-11

### Packaging

- CLI entry point packaging compatibility improvements for PyPI and Colab.
