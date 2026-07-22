# results/

Evaluation outputs live under `results/runs/`.

## Layout

Each run gets its own directory:

```
results/runs/<date>_<sha>_<env>/
    gsm8k_real.json             # real_gsm8k.py: per-problem records + summary accuracy
    exp1_prompt_separation.json # exp1_prompt_separation.py: per-prompt records + within/cross-category null summary
    manifest.json               # git SHA, pip freeze, device name, seed, model, mode
```

- `<date>` — `YYYYMMDD`
- `<sha>`  — short git SHA of the commit the run was produced from
- `<env>`  — `cpu` or `cuda`

Produced by `python -m benchmarks.real_gsm8k` (see `benchmarks/real_gsm8k.py`)
and `python -m benchmarks.exp1_prompt_separation` (see
`benchmarks/exp1_prompt_separation.py`).

## What's recorded (per problem)

`real_gsm8k.py`: `correct` (graded numeric match), `zeta_raw`,
`cross_sample_null` (`null_mean`/`null_std`/`gap`/`z_score`, controls = other
problems' sink activations), `spectral_gap`, measured `latency_ms`,
`prompt_tokens`, `completion_tokens`. No synthetic accuracy, no `time.sleep`
latency.

`exp1_prompt_separation.py`: the same per-prompt columns, plus
`null_within_category` and `null_cross_category` -- the cross-sample null
computed against controls drawn from the same vs. a different prompt
category, and a top-level `summary.separation` (mean within-category gap
minus mean cross-category gap) that answers whether the zeta metric is
sensitive to prompt category at all.

## Version control

Run artifacts (`results/runs/*/`) **are committed**. Without a committed
baseline there is nothing to diff against, so drift between runs is
undetectable -- a `pip freeze` embedded in an uncommitted file proves
nothing to a reader who never sees it. Every commit that changes a metric,
a model default, or the hook/bridge code should be accompanied by a fresh
run directory so the diff is visible in the PR.

At least one baseline run per benchmark is committed as a reference point;
regenerate a comparison run with the commands above and diff against it.
