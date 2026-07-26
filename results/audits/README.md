# Audit suite baselines

Committed runs of `benchmarks/audit_suite.py`. Every directory holds three
files, and they only mean anything together:

| file | contents |
|---|---|
| `audit.json` | aggregated report — each metric with mean/min/max and a `deterministic` flag |
| `audit.log.jsonl` | one line per individual run, written as it completed |
| `manifest.json` | git SHA, device, seed, effective dtype, full `pip freeze` |

Reproduce any of them with:

```bash
python -m benchmarks.audit_suite --model-tier <tier> --repeat 3
```

## Reproducibility

On `cpu_tiny_distilgpt2` (16 checks, 3 repeats): **129 non-timing metrics, 0 of
which differ across two independent processes.** The only metrics that vary are
`load_seconds` and `mean_forward_latency_ms`, both wall-clock.

Getting there required fixing three defects, all of which had been silently
producing different numbers on every run:

- `_lanczos_tridiagonal` drew its Lanczos start vector from the **global**
  torch RNG, so `spectral_gap()` returned a different value on every call — and
  consumed global RNG draws as a side effect, perturbing any sampled generation
  elsewhere in the process. Any `spectral_gap` in `results/runs/` predating
  commit `006a3b1` cannot be reproduced and should be regenerated.
- `ChronosLock.compute_τ_hash` hashed only the last two τ values, so a
  rewritten history validated clean.
- `test_heisenberg_scaling` passed on roughly 60% of unseeded runs by luck (see
  below).

## Cross-model results (CPU, float32, seed 42)

| tier | model | params | layers | source→sink | gap | ζ mean | ζ std | null z |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `tiny` | `distilgpt2` | 81.9 M | 6 | 3→4 | **1** | 0.992812 | 0.001124 | **13.473** |
| `small` | `gpt2` | 124.4 M | 12 | 6→10 | 4 | 0.987063 | 0.000752 | 1.965 |
| `medium` | `gpt2-medium` | 354.8 M | 24 | 12→22 | 10 | 0.739682 | 0.005988 | 0.813 |
| `large` | `Qwen/Qwen2.5-0.5B-Instruct` | 494.0 M | 24 | 12→22 | 10 | 0.132993 | 0.024376 | 0.380 |

Greedy decoding was byte-identical across repeats for all four.

### What the null z column means

`null z` is ζ measured between a prompt's own source and sink activations,
scored against a length-matched null built from *other* prompts' sink
activations. It answers: how far from chance is this? Roughly, z ≈ 2 is
marginal and z < 1 is indistinguishable from mismatched controls.

### The headline number is a layer-adjacency artifact

`UniversalHookWrapper` picks its layers proportionally — source = `n // 2`,
sink = `n - 2`. The **gap between them therefore grows with depth**: in
6-layer distilgpt2 the two layers are *adjacent* (3 and 4), while in a 24-layer
model they are 10 apart.

So z = 13.5 on distilgpt2 is close to comparing a layer with itself. It is not
evidence that ζ detects anything; it is evidence that adjacent layers have
similar activations, which is not in question.

**Controlling for depth kills the effect.** `gpt2-medium` and
`Qwen2.5-0.5B-Instruct` have identical layer counts (24) and identical gaps
(10). Both give z < 1 — that is, ζ on both is **statistically indistinguishable
from the length-matched null**. Neither 24-layer model shows a detectable
signal.

Two further observations:

- Depth is not the whole story. At *matched* depth and gap, ζ still differs
  sharply between the two (0.740 vs 0.133), so architecture matters
  independently. The two effects are not separated by these runs.
- ζ across models of different depth is **not a like-for-like comparison** and
  should not be tabulated as though it were. `layer_gap` is now recorded next
  to every ζ so the confound is visible in the data rather than needing to be
  remembered.

### Limits of these runs — read before citing

- n = 4 models, 2 architecture families. Not a survey.
- 6 prompts per model for the null, 4 for ζ. Small.
- All runs are CPU **float32**. The `medium`/`large`/`xl` tiers request
  float16, but torch's CPU float16 kernels are incomplete so the suite coerces
  to float32 and records `dtype_effective` accordingly. **The dtype confound is
  untested here** and needs a GPU run to settle.
- Layer-gap and architecture are confounded across the GPT-2 → Qwen boundary.
  Separating them needs models sampled at matched depth across families.
- Nothing here evaluates task accuracy. For a graded evaluation see
  `benchmarks/real_gsm8k.py`.

## Non-model checks

All 12 pass and are fully deterministic. Those with a known correct answer are
asserted against theory rather than merely reported:

| check | result |
|---|---|
| `gue_r_ratio` | ⟨r⟩ = 0.5953 GUE vs reference 0.5996; 0.3849 Poisson vs 0.3863; separation 0.210 |
| `bocpd_null` | 0 false alarms in 500 post-warmup steps on a stationary stream |
| `bocpd_changepoint` | changepoint at index 300 detected at 300, delay 0, 0 pre-changepoint alarms |
| `bocpd_return_range` | all outputs in [0, 1] under adversarial input (±1e6), no NaN/inf |
| `chronos_shard_roundtrip` | clean round-trip; 1 corrupted symbol repaired; 3 corrupted raise rather than return wrong data |
| `chronos_tau_chain` | tamper deep in history detected (this failed before the hash-chain fix) |
| `var_rupture_detector` | 0 ruptures on a calm signal; injected excursion at 300 detected at 302 |

## A note on `test_heisenberg_scaling`

The old test asserted `ratio > 1.1 or rate_large < rate_small`. Across a sweep
of 40 seeds the first clause held **0 times**; the second is satisfied by any
noise excursion below 1.0, so the test passed about 60% of the time and was
being read as confirming Heisenberg 1/√N scaling.

The measured ratio is **1.00 ± 0.03** over 12 seeds — standard quantum limit,
not Heisenberg. Heisenberg would predict ≈ 2.0. The test is now seeded,
averages over seeds, and asserts what is actually measurable.
