"""benchmarks — TMRP-13 evaluation harness for unitarity-lab v3.0.0-Singularity.

Each benchmark script supports:
  --mode passive|active   (passive = metrics only, active = full intervention)
  --seed INT              (deterministic reproducibility)
  --output FILE           (JSON results path)

Reported columns: zeta, baseline_cosine, permutation_p, latency_ms, accuracy.
"""
