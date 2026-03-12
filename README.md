# unitarity-lab

**v3.0.0-Singularity**

An experimental multi-model runtime for transformer instrumentation,
latent alignment tracing, distributed coordination, and optional
intervention.

> **Status: Alpha.** This project is under active development. APIs may
> change between minor releases. Benchmark results are preliminary.

---

## What It Does

unitarity-lab hooks into Hugging Face transformer models at the forward-pass
level.  It can:

1. **Measure** cross-layer alignment between a source and sink layer
   (Manifold Coherence ζ).
2. **Optionally intervene** by injecting a LoRA-adapted bias from the
   source eigenvectors into the sink layer (active mode).
3. **Coordinate** multiple model instances over ZeroMQ for distributed
   inference with Byzantine fault tolerance (dist tier).
4. **Monitor** runtime health via a Rich terminal dashboard.

---

## Manifold Coherence ζ

The primary metric is **Manifold Coherence ζ** — the cosine similarity
between flattened hidden states of the source and sink layers:

$$
\zeta = \frac{\operatorname{vec}(H_{\text{source}}) \cdot \operatorname{vec}(H_{\text{sink}})}
             {\|\operatorname{vec}(H_{\text{source}})\| \;\|\operatorname{vec}(H_{\text{sink}})\|}
$$

**Disclaimer:** ζ is a cosine-similarity proxy for cross-layer alignment.
It is not a measure of entanglement, consciousness, or physical phenomena.
Treat it as an empirical diagnostic whose relationship to model quality
is under investigation.

A permutation test (`permutation_test_zeta`) is provided to evaluate
statistical significance.

---

## Runtime Modes

| Mode | Flag | Behaviour |
| :--- | :--- | :-------- |
| **Passive** | `--mode-passive` | Hooks capture metrics only. No tensor is mutated. Baseline for A/B comparison. |
| **Active** | `--mode-active` | Full bridge intervention: LoRA bias injection, flux governor kicks, mirror feedback. |

Default is **active** for backward compatibility.

---

## Repository Structure (Three Tiers)

```text
unitarity-lab/
├── core/                   # Tier 1 — production-grade, tested
│   ├── version.py          #   Canonical version string
│   ├── metrics.py          #   manifold_coherence_zeta, baseline_cosine, permutation_test
│   ├── diversity_snapshot.py  # Drift detection (solo inference windows)
│   ├── bridge.py           #   Cross-layer entanglement hook + LoRA + flux
│   ├── universal_hook.py   #   HF model wrapper (passive/active modes)
│   ├── dashboard.py        #   Rich heartbeat dashboard
│   ├── dual_link.py        #   ZMQ inter-model bridge
│   ├── precision_projector.py # DequantAdapter + PrecisionClass
│   ├── handshake.py        #   Precision handshake protocol
│   ├── kill_switch.py      #   Byzantine voting (β_TB trust)
│   ├── flux.py             #   Hawking Flux Governor (GOE kicks)
│   ├── mirror.py           #   Proprioceptive feedback
│   ├── horizons.py         #   PageCurveHook + Lanczos spectral analysis
│   ├── pll_monitor.py      #   Phase-locked loop monitor
│   ├── casimir_opt.py      #   Casimir pressure optimizer
│   ├── unitary_regulator.py  # Aggregated diagnostics
│   ├── ghost_layer.py      #   RecursiveMirror
│   ├── virtual_layer13.py  #   VirtualLayer13
│   ├── safety_head.py      #   SafetyHead
│   ├── chronos_lock.py     #   Temporal sync (distributed-only, see below)
│   └── semantic_lock.py    #   Semantic anchor consensus
├── dist/                   # Tier 2 — distributed coordination
│   ├── dual_link.py        #   Re-export of core.dual_link
│   ├── handshake.py        #   Re-export of core.handshake
│   ├── chronos_lock.py     #   Re-export of core.chronos_lock
│   └── tier_manager.py     #   Node classification (compute/router tiers)
├── labs/                   # Tier 3 — EXPERIMENTAL, unstable
│   ├── mirror.py           #   Experimental mirror wrappers
│   ├── flux.py             #   Experimental flux wrappers
│   ├── semantic_lock.py    #   Experimental semantic lock wrappers
│   └── topology_metrics.py #   Spectral gap, Betti-0, activation entropy
├── benchmarks/             # Evaluation harness
│   ├── _harness.py         #   Shared helpers (seed, metrics, JSON output)
│   ├── gsm8k.py            #   GSM8K math reasoning
│   ├── humaneval_plus.py   #   HumanEval+ code generation
│   ├── agent_instruct.py   #   Agent instruction following
│   └── adversarial_safety.py  # Adversarial safety
├── tests/                  # pytest suite
├── start_node.py           # CLI entry point
├── run_community.py        # Legacy community example
├── setup.py
├── manifesto.md
└── README.md
```

### Tier Rules

- **core/** — Production. Must have tests. No breaking changes without a
  version bump.
- **dist/** — Distributed coordination utilities. Depends on core/.
  ChronosLock lives here (see below). Not required for single-node use.
- **labs/** — Experimental. May change or be removed at any time. No
  stability guarantees.

---

## ChronosLock (Distributed Only)

`ChronosLock` is a temporal synchronization subsystem for multi-node
inference.  It is **not required for single-node operation**.  The
canonical import path for distributed code is:

```python
from dist.chronos_lock import ChronosLock
```

Single-node `UniversalHookWrapper` does not import or use ChronosLock
when `enable_dual=False`.

---

## Tier Policing

When running in distributed mode, `dist.tier_manager.TierManager` classifies
nodes as **Compute** (≥ 12 tok/s sustained) or **Router** (relay only).

- Nodes self-attest TPS at handshake; false high-performance claims are
  detected and flagged.
- Compute nodes that fail to sustain throughput are demoted to Router
  after a penalty window.
- Promotion back to Compute requires a cooldown and re-attestation.
- Router nodes contribute relay bandwidth but do not participate in
  quorum votes for bridge intervention.

---

## Diversity Snapshots

`core.diversity_snapshot.DiversitySnapshotMonitor` runs periodic solo
inference windows (bridge disabled) to detect coherence collapse — when
ζ_bridged ≈ ζ_solo, the bridge is no longer contributing meaningful
alignment.

Constants: `SNAPSHOT_INTERVAL_TOKENS=4096`, `SOLO_WINDOW_TOKENS=128`,
`COLLAPSE_THRESHOLD_RATIO=0.08`, `CONSECUTIVE_TRIGGERS_REQUIRED=2`.

---

## Quick Start

### Install

```bash
pip install -e .
```

### Run Tests

```bash
pytest tests/ -v
```

### Start a Node

```bash
# Passive mode (metrics only, no tensor mutation)
python start_node.py --mode-passive

# Active mode (default — full intervention)
python start_node.py

# Dual-node distributed mode
python start_node.py --dual --node-id A
python start_node.py --dual --node-id B  # on second machine
```

### Run Benchmarks

```bash
python -m benchmarks.gsm8k --mode passive --seed 42 --output passive.json
python -m benchmarks.gsm8k --mode active  --seed 42 --output active.json
```

Each benchmark outputs JSON with columns: `zeta`, `baseline_cosine`,
`permutation_p`, `latency_ms`, `accuracy`.

---

## Benchmark Columns

| Column | Description |
| :----- | :---------- |
| `zeta` | Manifold Coherence ζ (cosine similarity, source↔sink) |
| `baseline_cosine` | Mean-pooled cosine baseline |
| `permutation_p` | p-value from permutation test (H₀: ζ is random) |
| `latency_ms` | Wall-clock latency per sample |
| `accuracy` | Task-specific accuracy (exact match, pass@1, etc.) |

---

## License

MIT. See [LICENSE](LICENSE).
