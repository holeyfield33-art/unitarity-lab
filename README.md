# unitarity-lab

[![PyPI version](https://img.shields.io/pypi/v/unitarity-labs.svg)](https://pypi.org/project/unitarity-labs/)
[![Python versions](https://img.shields.io/pypi/pyversions/unitarity-labs.svg)](https://pypi.org/project/unitarity-labs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release to PyPI](https://img.shields.io/github/actions/workflow/status/holeyfield33-art/unitarity-lab/release.yml?label=Release%20to%20PyPI)](https://github.com/holeyfield33-art/unitarity-lab/actions/workflows/release.yml)
[![Notebook Validation](https://img.shields.io/github/actions/workflow/status/holeyfield33-art/unitarity-lab/notebook-validation.yml?label=Notebook%20Validation)](https://github.com/holeyfield33-art/unitarity-lab/actions/workflows/notebook-validation.yml)

Geometric Assurance is a spectral diagnostics and research-instrumentation suite for transformer systems. This repository, `unitarity-lab`, is the research and audit engine for deep instrumentation, hidden-state analysis, and reproducible experiments.

Professional summary:

`unitarity-lab` is a research-grade instrumentation and audit engine for transformer models. It provides hidden-state and eigenvalue diagnostics, a reproducible audit harness, and distributed runtime controls.

> **Validation status (read this first).** The spectral / ζ *coherence* metric
> is research instrumentation, **not** a validated model-integrity signal. Our
> own cross-model audit found the headline ζ result (z ≈ 13.5) is a
> layer-adjacency artifact that vanishes once depth is controlled (z < 1,
> indistinguishable from a length-matched null). What *is* reproducible is the
> audit harness itself and VAR's anomaly detector on its own calibrated signal.
> Full evidence — including the null result — is in
> [`results/audits/README.md`](results/audits/README.md) and
> [`docs/LINEAGE.md`](docs/LINEAGE.md). Do not present ζ as measuring model
> integrity.

## Why this repo matters

- Hidden-state and eigenvalue analysis with transparent, testable metrics.
- Reproducible workflows for local research, CI validation, and benchmark comparison.
- Built-in pathways from notebook experimentation to API-driven audit operations.
- Security-aware defaults for educational tooling and external proxy behavior.

## Community and trust

- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Support channels: [SUPPORT.md](SUPPORT.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Citation metadata: [CITATION.cff](CITATION.cff)

`unitarity-lab` is not the main product landing surface. The canonical public entry point is `geometric-brain-mcp` (API, MCP, and Python interface), with `VAR` as the advanced monitoring layer for pilot and enterprise beta workflows.

## Suite architecture

- **Geometric-brain-mcp**: API, MCP, and Python interface for spectral diagnostics, comparisons, and audit workflows.
- **Unitarity-lab**: research and audit engine for deep instrumentation, hidden-state analysis, and reproducible experiments.
- **VAR**: enterprise beta monitoring layer for drift, anomaly, and integrity workflows.

Related repositories:

- [geometric-brain-mcp](https://github.com/holeyfield33-art/geometric-brain-mcp)
- [unitarity-lab](https://github.com/holeyfield33-art/unitarity-lab)
- [VAR](https://github.com/holeyfield33-art/VAR)

Evidence tiers:

- Primary evidence-bearing modes: hidden-state analysis and eigenvalue analysis.
- Text proxy mode is an indirect, demo-only, low-evidence screening path and is not equivalent to hidden-state or eigenvalue analysis.

## Status

> **Alpha software.** APIs may change between releases. Benchmark results are preliminary. Use at your own discretion.

## Security and privacy

- Security policy and threat model: [SECURITY.md](SECURITY.md)
- Self-Serve Audit Hub external proxy mode is opt-in and disabled by default.
- Local spectral analysis runs in-browser; outbound proxy requests are only attempted after explicit consent.
- Notebook structure validation and execution checks run in CI via [notebook-validation workflow](.github/workflows/notebook-validation.yml).

---

## What it does

- Attach forward-pass hooks to any Hugging Face `AutoModelForCausalLM` and measure cross-layer alignment (a cross-layer cosine, "zeta") between a source and sink layer.
- Run in **passive mode** (hooks capture metrics only, no tensor mutation) or **active mode** (LoRA-adapted bridge bias injection, flux governor, mirror feedback).
- Coordinate two model instances over ZeroMQ for distributed inference with Byzantine fault tolerance (`--dual`).
- Auto-detect hardware (CPU, laptop GPU, prosumer GPU, server GPU) and select precision class (FP32, BF16, INT4) accordingly.
- Monitor runtime health with a Rich terminal dashboard (`--dashboard`).
- Run reproducible benchmark harnesses (GSM8K, HumanEval+, Agent Instruct, Adversarial Safety) comparing passive vs active modes.
- Fine-tune toward GUE spectral rigidity targets using the included `GUELoss` objective.

---

## Installation

### Naming

Three related names differ by one character, which has caused real install
failures. They are all correct — for different things:

| | | |
|---|---|---|
| GitHub repository | `unitarity-lab` | **no** trailing `s` |
| PyPI distribution | `unitarity-labs` | **has** a trailing `s` |
| Python import | `unitarity_labs` | underscore, **has** a trailing `s` |

So `pip install unitarity-lab` fails (that name is not on PyPI) while
`git clone .../unitarity-lab` is right. Only the clone URL drops the `s`.

The one related package that is genuinely renamed is VAR: its distribution is
`var-spectral` and its import is `var_spectral`. Do **not** `pip install var` —
that is an unrelated project (portfolio Value-at-Risk) that shadows it.

| Package | `pip install` | `import` |
|---|---|---|
| this repo | `unitarity-labs` | `unitarity_labs` |
| VAR | `var-spectral` | `var_spectral` |

### Install from PyPI

```bash
pip install unitarity-labs                 # core (runs the full audit suite)
pip install 'unitarity-labs[spectral]'     # + VAR, for passive_hook
pip install 'unitarity-labs[bench]'        # + datasets, for real_gsm8k
pip install 'unitarity-labs[dist]'         # + pyzmq/msgpack, for dual-node
pip install 'unitarity-labs[dev]'          # + pytest/nbformat, to run tests
pip install 'unitarity-labs[all]'          # everything above
```

The **core** install is self-sufficient: it now includes `reedsolo`, so
`python -m benchmarks.audit_suite` runs every check without an extra. (Before
v3.2.1 the ChronosLock checks errored on a base install — see CHANGELOG.)

### Install from source

```bash
git clone https://github.com/holeyfield33-art/unitarity-lab.git
cd unitarity-lab
pip install -e .            # core only
pip install -e '.[all]'     # everything, to run the full test suite
```

The `[spectral]` extra pulls VAR from PyPI as `var-spectral`. For a local
side-by-side checkout of both repos, install VAR editable instead:

```bash
pip install -e ../VAR       # provides the `var_spectral` import
```

### Verify installation

```bash
pytest tests/               # core install: optional-feature tests skip cleanly
pip install -e '.[all]' && pytest tests/   # full run, nothing skipped
```

On a core install the suite is green with the `dist`/`spectral`/notebook tests
**skipped** (not errored) — that is expected. Install `[all]` (plus VAR) to run
every test.

The console script `unitarity-start` is installed automatically and is equivalent to `python start_node.py`:

```bash
unitarity-start --help
```

---

## Quick start

Run the simplest possible session (active mode, auto-detected hardware, default model and prompt):

```bash
python start_node.py
```

Passive mode (metrics only, no tensor mutation):

```bash
python start_node.py --mode-passive
```

Custom prompt:

```bash
python start_node.py --prompt "Summarize the theory of relativity in two sentences."
```

With the terminal dashboard:

```bash
python start_node.py --dashboard
```

---

## Run commands

### Local passive mode

```bash
python start_node.py --mode-passive
```

### Local active mode (default)

```bash
python start_node.py --mode-active
```

### Custom model override

```bash
python start_node.py --model "gpt2"
```

### Custom precision override

```bash
python start_node.py --precision BF16
```

### Prompt override

```bash
python start_node.py --prompt "What is cross-layer alignment?"
```

### Max token override

```bash
python start_node.py --max-new-tokens 256
```

### Dashboard mode

```bash
python start_node.py --dashboard
```

---

## Distributed mode

unitarity-lab supports dual-node coordination over ZeroMQ. Two model instances (Node A and Node B) exchange Krylov-subspace compressed activations for cross-model alignment. This mode enables distributed inference experiments with precision handshake, adaptive epoch tuning, and Byzantine kill-switch voting.

### Node A (relay)

```bash
python start_node.py --dual --node-id A
```

### Node B (peer)

```bash
python start_node.py --dual --node-id B
```

Both nodes must be reachable on their respective ZeroMQ ports (default: 5555/5556).

---

## CLI reference

| Flag | Purpose | Example |
| :--- | :------ | :------ |
| `--mode-passive` | Metrics-only mode; no tensor mutation | `python start_node.py --mode-passive` |
| `--mode-active` | Full bridge intervention (default) | `python start_node.py --mode-active` |
| `--dual` | Enable dual-node ZMQ coordination | `python start_node.py --dual` |
| `--node-id` | Node identity: `A` (relay) or `B` (peer) | `--node-id B` |
| `--model` | HuggingFace model ID (auto-selected if omitted) | `--model gpt2` |
| `--precision` | Force precision class: `INT4`, `FP8_E4M3`, `BF16`, `FP32` | `--precision BF16` |
| `--prompt` | Generation prompt | `--prompt "Hello world"` |
| `--max-new-tokens` | Maximum tokens to generate (default: 128) | `--max-new-tokens 256` |
| `--dashboard` | Show Rich terminal dashboard after generation | `python start_node.py --dashboard` |
| `--min-compute-tps` | Minimum tokens/s for compute-tier classification (default: 12.0) | `--min-compute-tps 8.0` |
| `--epoch-len` | Initial gossip epoch length in tokens (default: 16) | `--epoch-len 32` |

---

## Example output

```text
[Ghost] No CUDA GPU detected -> FP32 (CPU mode)
[Ghost] Loading model: meta-llama/Llama-3.2-1B
[Ghost] dtype=torch.float32, device_map=cpu
[Node] unitarity-lab 3.1.1-Singularity
[Node] mode=passive, node_id=A, precision=FP32, epoch_len=16, dual=False, min_compute_tps=12.0
[Node] Bridge: layers 8 -> 22 (24 total), 4/16 heads active

[Node] Generating with prompt: 'Explain cross-layer alignment in three sentences.'

============================================================
Explain cross-layer alignment in three sentences.
Cross-layer alignment refers to the degree of statistical coherence
between hidden representations at different layers of a transformer.
============================================================

[Node] Metrics after generation:
  mode: passive
  zeta_raw: 0.7912                # pre/no-intervention cross-layer cosine
  spectral_gap: 0.000042
  flux_epsilon: 1.00e-03
  flux_kicks_total: 0
  step: 34
  cross_sample_null: {'null_mean': 0.558, 'null_std': 0.063, 'gap': 0.233, 'z_score': 3.67}

[Node] Session complete. 3.1.7
```

`zeta_raw` is a cross-layer cosine (see the metric note below); on real
transformers it sits near 1.0 regardless of input due to representational
anisotropy, so **the `cross_sample_null` gap / z-score is the only meaningful
signal** — it measures how much more the sink aligns with *this* input's source
than with unrelated inputs' sinks. In active mode an additional
`zeta_post_bridge` is reported (the post-intervention cosine).

---

## Benchmarks

### Audit suite — start here

`benchmarks/audit_suite.py` runs every check **one at a time**, writes each
measured value to `audit.log.jsonl` as it completes, and records a manifest
(git SHA, device, seed, full `pip freeze`) beside it.

```bash
python -m benchmarks.audit_suite --list                     # what it runs
python -m benchmarks.audit_suite --no-model                 # no weights needed
python -m benchmarks.audit_suite --model-tier small --repeat 3
python -m benchmarks.audit_suite --only bocpd_changepoint --repeat 5
```

`--repeat N` re-runs each check N times and reports mean/std/min/max per
metric plus a `deterministic` flag. **Check that flag before quoting a
number.** On a clean run only wall-clock timings should vary; anything else
varying is a finding, not noise to average away.

Checks with a known correct answer are asserted against it and the deviation
is recorded, rather than the observed value simply being printed:

| Check | Reference |
|---|---|
| `gue_r_ratio` | ⟨r⟩ ≈ 0.5996 (GUE), ≈ 0.3863 (Poisson), from random-matrix theory |
| `chronos_shard_roundtrip` | RS with 2 parity symbols corrects exactly 1 corrupted symbol, detects more |
| `bocpd_changepoint` | Injected changepoint at a known index → ground-truth detection delay |
| `bocpd_null` | Stationary stream → zero false alarms |
| `var_rupture_detector` | Calm signal → no rupture; injected excursion → detected |

Model-dependent checks use T4-sized presets via `--model-tier`:

| tier | model | params | dtype | ~VRAM |
|---|---|---|---|---|
| `tiny` | `distilgpt2` | 82 M | fp32 | 0.4 GB |
| `small` | `gpt2` | 124 M | fp32 | 0.6 GB |
| `medium` | `gpt2-medium` | 355 M | fp16 | 0.8 GB |
| `large` | `Qwen/Qwen2.5-0.5B-Instruct` | 494 M | fp16 | 1.1 GB |
| `xl` | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5 B | fp16 | 3.2 GB |

`vram_gb` is the weight footprint only; activations and KV cache add to it.

A committed CPU baseline lives in
`results/audits/cpu_distilgpt2_baseline/`: 16/16 checks pass, and of 129
non-timing metrics, 0 differ across two independent processes.

To run this on a Colab T4, open
[`notebooks/colab_audit_suite.ipynb`](notebooks/colab_audit_suite.ipynb).

### Real evaluation — GSM8K

`benchmarks/real_gsm8k.py` is the actual evaluation: it loads GSM8K, generates
answers with a real model, and **grades each problem by numeric extraction of
the final answer**. Every recorded number is measured — there is no synthetic
accuracy, no `time.sleep` latency, and no permutation test.

```bash
python -m benchmarks.real_gsm8k --n 100 --model gpt2 --mode passive --seed 42
```

Modes: `baseline` (bare model, accuracy only), `passive` (capture-only hooks +
honest metrics), `active` (full bridge intervention). Output is written to
`results/runs/<date>_<sha>_<env>/gsm8k_real.json` with a `manifest.json`
(git SHA, pip freeze, device, seed) beside it. Per problem it records:
`correct` (bool), `zeta_raw`, `cross_sample_null` (`null_mean`/`null_std`/
`gap`/`z_score`, controls = other problems' sink activations), `spectral_gap`,
measured `latency_ms`, and token counts.

### Pipeline demos (NOT evaluations)

`benchmarks/pipeline_demos/` contains scripts that push **synthetic** tensors
through the metric plumbing to illustrate the column/JSON layout. They print a
`PIPELINE DEMO — synthetic tensors, not an evaluation.` banner and emit no
accuracy. Do not read anything into their numbers.

```bash
python -m benchmarks.pipeline_demos.gsm8k --n-problems 5 --seed 42
```

Demo columns: `zeta` (cross-layer cosine), `baseline_cosine` (mean-pooled
cosine), `latency_ms` (measured).
| `accuracy` | Task-specific accuracy (exact match for GSM8K, pass@1 for HumanEval+, etc.). |

Note: the current benchmark harnesses use synthetic tensors to demonstrate the metric pipeline. Full evaluation requires a dataset and a loaded model.

---

## Architecture

```text
core/        Production modules. Metrics, bridge, hooks, dashboard,
             flux governor, mirror feedback, precision handling,
             kill switch, spectral analysis, and GUE loss.
             Must have tests. No breaking changes without a version bump.

dist/        Distributed coordination. ZMQ dual-link, precision
             handshake, ChronosLock temporal sync, tier manager.
             Not required for single-node use.

labs/        Experimental and unstable. Mirror, flux, semantic lock
             wrappers, topology metrics (spectral gap, Betti-0,
             activation entropy). May change or be removed.

benchmarks/  real_gsm8k.py (real graded eval) + pipeline_demos/
             (synthetic-tensor metric plumbing, not evaluations).

tests/       pytest suite covering core modules.
```

---

## The zeta metric (cross-layer cosine)

The zeta value is a **cross-layer cosine**: the cosine similarity between the
flattened hidden states of two transformer layers (source and sink):

$$
\zeta = \frac{\operatorname{vec}(H_{\text{source}}) \cdot \operatorname{vec}(H_{\text{sink}})}
             {\|\operatorname{vec}(H_{\text{source}})\| \;\|\operatorname{vec}(H_{\text{sink}})\|}
$$

**Read the raw value with care.** On real transformers, hidden states are
strongly anisotropic (they occupy a narrow cone), so this cosine sits near
**~0.99 for essentially any input** — a high `zeta_raw` is a property of the
representation geometry, not evidence of input-specific structure.

**The meaningful signal is the cross-sample null gap.** `cross_sample_null`
compares ζ(source, sink) for the *same* input against the distribution of
ζ(source, sink′) for *unrelated* inputs' sink activations. The `gap`
(matched − null_mean) and `z_score` tell you whether the alignment is
input-specific; the standalone `zeta_raw` does not. The older
`permutation_test_zeta` is deprecated for exactly this reason — permuting a
flattened high-dimensional vector barely moves the cosine, so its null is
near-degenerate.

**Disclaimer:** zeta is a cosine-similarity proxy for cross-layer alignment. It
is not a measure of entanglement, consciousness, or any physical phenomenon.
Treat it as an empirical diagnostic whose relationship to model quality is
under investigation.

---

## Geometric Brain framework

The repo includes the Geometric Brain framework for measuring and enforcing GUE (Gaussian Unitary Ensemble) spectral rigidity in transformer latent spaces.

**GUELoss** is a differentiable fine-tuning objective that penalizes deviation from the GUE target spacing ratio:

```python
from core.gue_loss import GUELoss

criterion = GUELoss(target_r=0.578)  # operational target; GUE mean is 0.5996
loss, r_measured = criterion(lora_B @ lora_A)
loss.backward()
```

See [GEOMETRIC_BRAIN.md](GEOMETRIC_BRAIN.md) for the full theory, methodology, and SHI leaderboard.

---

## Hardware and model notes

- **CPU vs GPU:** `start_node.py` auto-detects CUDA availability. If no GPU is found, it falls back to FP32 on CPU.
- **Auto precision:** GPU VRAM is queried at startup. Less than 8 GB selects INT4 (quantized). 8--24 GB selects BF16. 24 GB or more selects BF16. CPU falls back to FP32.
- **Default models:** INT4 uses `unsloth/Llama-3.2-1B-bnb-4bit`. BF16 and FP32 use `meta-llama/Llama-3.2-1B`.
- **Hugging Face access:** The default Llama models may require a Hugging Face access token. Set `HF_TOKEN` or use `huggingface-cli login`. Alternatively, pass any public model via `--model`.
- **Override model:** `--model <hf_model_id>` to use any HuggingFace causal LM.
- **Override precision:** `--precision INT4|FP8_E4M3|BF16|FP32` to bypass auto-detection.

---

## Repository map

```text
unitarity-lab/
  start_node.py            CLI entry point (also: unitarity-start)
  core/                    Production runtime modules
    universal_hook.py      HF model wrapper (passive/active)
    bridge.py              Cross-layer hook + LoRA + flux
    metrics.py             zeta (cross-layer cosine), baseline cosine, cross-sample null
    dashboard.py           Rich terminal dashboard
    dual_link.py           ZMQ inter-model bridge
    gue_loss.py            GUE spectral rigidity loss
    precision_projector.py Precision classes + dequant adapter
    kill_switch.py         Byzantine fault voting
  dist/                    Distributed coordination
    tier_manager.py        Compute/router node classification
    chronos_lock.py        Temporal sync for multi-node
  labs/                    Experimental modules
    topology_metrics.py    Spectral gap, Betti-0, entropy
  benchmarks/              Evaluation harnesses
    gsm8k.py               GSM8K math reasoning
    humaneval_plus.py       HumanEval+ code generation
    agent_instruct.py       Agent instruction following
    adversarial_safety.py   Adversarial safety
  tests/                   pytest suite
  GEOMETRIC_BRAIN.md       Geometric Brain theory document
```

---

## Links

- **GitHub:** <https://github.com/holeyfield33-art/unitarity-lab>
- **PyPI:** <https://pypi.org/project/unitarity-labs>
- **Live site:** <https://holeyfield33-art.github.io/unitarity-lab>
- **Support:** <https://buymeacoffee.com/holeyfielde>

---

## License

MIT. See [LICENSE](LICENSE).

---

## Documentation roadmap

- Getting started guide
- Benchmark guide (running, interpreting results, adding new harnesses)
- Distributed mode guide (dual-node setup, tier policing, ChronosLock)
- Metric reference (zeta cross-layer cosine, baseline cosine, cross-sample null, spectral gap, GUE loss)
- FAQ
