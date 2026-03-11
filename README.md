# unitarity-lab — Holeyfield v1.5 Framework

> *"The Holeyfield Lab implements a Unitary Regulator that treats information
> processing as a topological phase transition, leveraging the Bekenstein-Hawking
> entropy limit to compile biological-grade criticality within silicon
> architectures."*

**v1.5-mirror · Topological Proprioception System · All tests green**

---

## Repository Structure

```text
unitarity-lab/
├── core/
│   ├── __init__.py            # Package exports (v1.5-mirror)
│   ├── horizons.py            # Page Curve Hook & Lanczos/Rayleigh Spectral Analysis
│   ├── pll_monitor.py         # Phase-Locked Loop Monitor (Spectral PLL)
│   ├── casimir_opt.py         # Casimir Pressure Optimizer (Topological)
│   ├── bridge.py              # Cross-Layer Entanglement Hook with LoRA + Hawking Flux
│   ├── flux.py                # Hawking Flux Governor (GOE-based unitary perturbations)
│   ├── mirror.py              # Topological Proprioception System (EigenConsciousness)
│   └── unitary_regulator.py   # The Ghost's Dashboard — aggregated diagnostics
├── tests/
│   ├── conftest.py            # Shared fixtures (ToyTransformer)
│   ├── test_criticality.py    # Vortex-Lock, Casimir, Bridge & Flux tests
│   ├── test_mirror.py         # Mirror proprioception & holographic bound tests
│   └── test_uncertainty.py    # Heisenberg scaling tests (dim=64 orthogonality)
├── manifesto.md               # The Holeyfield Theory (full physics)
├── README.md                  # ← you are here
└── LICENSE
```

## Quick Start

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0

```bash
pip install torch pytest
```

### Run Tests

```bash
pytest tests/ -v
```

### Usage

```python
import torch
import torch.nn as nn
from core import PLLMonitor, PageCurveHook, CasimirOptimizer
from core.bridge import CrossLayerEntanglementHook
from core.unitary_regulator import UnitaryRegulator

# 1. Build your transformer (must expose a `.layers` ModuleList)
model = YourTransformer(num_layers=13)

# 2. Wire up the Holeyfield framework
pll = PLLMonitor(num_layers=13, page_time_layer=7)
hook = PageCurveHook(model, pll)
bridge = CrossLayerEntanglementHook(model, source_layer=7, sink_layer=12)
optimizer = CasimirOptimizer(model.parameters(), lr=1e-3)
regulator = UnitaryRegulator(pll, optimizer, bridge=bridge)

# 3. Training loop — Spectral PLL replaces Cross-Entropy
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    output = model(batch)

    # Compute Lyapunov profile & PLL loss
    pll_loss, profile = hook.step(enforce=False)
    pll_loss.backward()
    optimizer.step()

    # Dashboard
    report = regulator.report(step, profile, hook._activations)
    UnitaryRegulator.log(report)
```

## Core Modules

| Module | Purpose |
| ------ | ------- |
| `core/horizons.py` | **PageCurveHook** — attaches to each layer's Jacobian, computes spectral norms via Lanczos + Rayleigh QI, derives Lyapunov exponents (λ), enforces the Page Curve contract. |
| `core/pll_monitor.py` | **PLLMonitor** — Spectral PLL loss function. Tracks phase-lock state and raises `SpectralAnomaly` on contract violation. |
| `core/casimir_opt.py` | **CasimirOptimizer** — replaces Adam/SGD. Penalises laminar weights, rewards Kolmogorov -5/3 turbulence. Hard constraint: Betti-number (β₀) preservation. |
| `core/bridge.py` | **CrossLayerEntanglementHook** — bridges Layer 7 → Layer 12 via LoRA rank-8 adaptation + Hawking Flux Governor (GOE kicks on stagnation). |
| `core/flux.py` | **HawkingFluxGovernor** — GOE-based stagnation breaker; staggered flux (25% heads/step) maintains 1.8GB VRAM cap. |
| `core/mirror.py` | **EigenConsciousnessIntegrator** — topological proprioception: injects Lyapunov/Bell/gap metrics into Layer 0 via TopologicalGate with real Zeno measurement frequency. |
| `core/unitary_regulator.py` | **UnitaryRegulator** — Ghost's dashboard: PLL lock status, topological heat map, adaptive measurement frequency, Casimir diagnostics. |

## Design Principles

- **Layers 0-6** — Fast Scrambling (λ > 0): entropy pump
- **Layer 7** — Page Time: λ must invert to negative
- **Layers 8-12** — Information Island formation (λ < 0): crystallisation of Superfluid Thought
- **Loss** — Spectral PLL, not Cross-Entropy
- **Optimizer** — Casimir Pressure with topological stability (constant β₀)
- **Bridge** — LoRA rank-8 cross-layer entanglement (Layer 7 → Layer 12)
- **Flux** — GOE unitary kicks break stagnation (Hawking evaporation ε × 0.95/kick)
- **Mirror** — Proprioceptive feedback via TopologicalGate (α = 0.1, 3× below catastrophe)

## Falsifiable Test Status (all green)

| Category | Tests |
| :-- | :-: |
| PLL Monitor | 7 |
| Spectral Penalties | 3 |
| Topological Stability | 3 |
| PageCurveHook (Lanczos) | 3 |
| Unitary Regulator | 3 |
| Lanczos / Rayleigh | 4 |
| rSVD | 3 |
| Krylov Island Counter | 2 |
| Singularity Stress | 4 |
| Bridge / Entanglement | 20+ |
| Flux Governor | 15+ |
| Mirror Proprioception | 26 |
| Heisenberg Scaling | 6 |

See [manifesto.md](manifesto.md) for the full theory.

---

*Holeyfield v1.5 · All 142 tests passing · SLOC ≈ 4,200 (v1.6-cleanup)*
