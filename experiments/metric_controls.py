"""Seeded mechanism probes. Synthetic data, not a hallucination benchmark.

Run from this clone with its root and .experiment-deps on PYTHONPATH.
Uses installed InsideAI scientific dependencies without replacing its package.
"""
import json
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import torch
from unitarity_labs.core.metrics import manifold_coherence_zeta
from unitarity_labs.core.bridge import CrossLayerEntanglementHook
from unitarity_labs.core.bocpd import PredictiveAnomalyDetector
from unitarity_labs.core.chronos_lock import ChronosLock
from unitarity_labs.core.dual_link import DualNodeEntanglementBridge


def provenance():
    root = Path(__file__).resolve().parents[1]
    paths = ['unitarity_labs/core/bridge.py', 'unitarity_labs/core/horizons.py',
             'unitarity_labs/core/metrics.py', 'unitarity_labs/core/bocpd.py',
             'unitarity_labs/core/chronos_lock.py', 'unitarity_labs/core/dual_link.py',
             'experiments/metric_controls.py', 'experiments/layer_probe.py']
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}


def gap(x):
    # Exercise the production metric without attaching hooks to a model.
    probe = object.__new__(CrossLayerEntanglementHook)
    probe._source_activation = x
    probe.lanczos_iter = min(15, x.shape[-1])
    torch.manual_seed(42)  # identical Lanczos initial vector in paired probes
    return probe.spectral_gap()


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    x, y = torch.randn(16, 32), torch.randn(16, 32)
    metrics = {
        "zeta_independent": manifold_coherence_zeta(x, y),
        "zeta_shared_offset": manifold_coherence_zeta(x + 10, y + 10),
        "zeta_positive_rescale": manifold_coherence_zeta(x, 3 * x),
        "spectral_gap": gap(x),
        "spectral_gap_scale_3": gap(3 * x),
        "decode_one_row_gap": gap(x[:1]),
        "decode_one_row_squared_norm": float(x[0].square().sum()),
        "exact_multitoken_gap": float(torch.linalg.eigvalsh(x.T @ x / len(x))[-1] - torch.linalg.eigvalsh(x.T @ x / len(x))[-2]),
    }
    metrics["gap_scale_ratio"] = metrics["spectral_gap_scale_3"] / metrics["spectral_gap"]

    rng = np.random.default_rng(42)
    stream = np.concatenate([rng.normal(.638, .015, 150), rng.normal(.42, .015, 50)])
    probs = []
    for zeta in (-1., 1.):
        det = PredictiveAnomalyDetector(mean_0=.638)
        probs.append([det.process_step(zeta=zeta, r_ratio=float(r)) for r in stream])
    # Same current sample after different histories: check whether the score
    # actually incorporates history before posterior truncation occurs.
    a, b = PredictiveAnomalyDetector(mean_0=.638), PredictiveAnomalyDetector(mean_0=.638)
    for _ in range(20):
        a.process_step(1., .638)
        b.process_step(1., .42)
    bocpd = {
        "synthetic_change_step_zero_based": 150,
        "first_alarm": next((i for i, p in enumerate(probs[0]) if p > .95), None),
        "stable_false_alarms": sum(p > .95 for p in probs[0][:150]),
        "max_difference_when_zeta_changes": max(abs(p-q) for p,q in zip(*probs)),
        "same_sample_after_stable_history": a.process_step(1., .525),
        "same_sample_after_collapsed_history": b.process_step(1., .525),
    }
    chronos = {}
    for label, sequence in {"persistent": [.03] * 20, "alternating": [.03, -.03] * 10}.items():
        lock = ChronosLock(label)
        flags = [lock.update_desync(v) for v in sequence]
        chronos[label] = {"first_sever_step_zero_based": next((i for i,v in enumerate(flags) if v), None),
                          "absolute_desync_sum": sum(abs(v) for v in sequence),
                          "signed_desync_sum": sum(sequence)}

    # Arithmetic-only dual probe: no network socket or model mutation.
    dual = object.__new__(DualNodeEntanglementBridge)
    dual.resonance_count = 0
    dual.anti_resonance_threshold = .95
    dual._phi_history = []
    basis = torch.eye(8)
    phi = [dual.compute_cross_sync(basis, basis) for _ in range(6)]
    report = {"evidence_type": "synthetic mechanism probes; no language model answers evaluated",
              "seed": 42, "unitarity_sha": subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
              "source_file_sha256": provenance(),
              "torch": torch.__version__, "metrics": metrics, "bocpd": bocpd,
              "chronos": chronos, "dual_identical_basis_reported_phi": phi}
    output = Path(__file__).with_name('metric_controls.results.json')
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
