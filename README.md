# unitarity-lab — Holeyfield v1.1 Framework

> *"The Holeyfield Lab implements a Unitary Regulator that treats information
> processing as a topological phase transition, leveraging the Bekenstein-Hawking
> entropy limit to compile biological-grade criticality within silicon
> architectures."*

**TMRP Session 18 — Design Lock · `unitary_core.v1.1` · DeepSeek-Optimized Spectral Core**

---

## Repository Structure

```text
unitarity-lab/
├── core/
│   ├── __init__.py            # Package exports
│   ├── horizons.py            # Page Curve Hook & Scrambling logic
│   ├── pll_monitor.py         # Phase-Locked Loop Monitor (Spectral PLL)
│   ├── casimir_opt.py         # Casimir Pressure Optimizer (Topological)
│   └── unitary_regulator.py   # The Ghost's Module — dashboard & heat map
├── tests/
│   └── test_criticality.py    # Vortex-Lock, Casimir & Page Curve tests
├── manifesto.md               # The Holeyfield Theory v1.0
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
from core.unitary_regulator import UnitaryRegulator

# 1. Build your transformer (must expose a `.layers` ModuleList)
model = YourTransformer(num_layers=13)

# 2. Wire up the Holeyfield framework
pll = PLLMonitor(num_layers=13, page_time_layer=7)
hook = PageCurveHook(model, pll)
optimizer = CasimirOptimizer(model.parameters(), lr=1e-3)
regulator = UnitaryRegulator(pll, optimizer)

# 3. Training loop — Spectral PLL replaces Cross‑Entropy
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
| `core/horizons.py` | **PageCurveHook** — attaches to each layer's Jacobian, computes spectral norms, derives Lyapunov exponents (λ), enforces the Page Curve contract. |
| `core/pll_monitor.py` | **PLLMonitor** — Spectral PLL loss function (replaces cross-entropy). Tracks phase-lock state and raises `SpectralAnomaly` on contract violation. |
| `core/casimir_opt.py` | **CasimirOptimizer** — replaces Adam/SGD. Penalises laminar weights, rewards Kolmogorov -5/3 turbulence. Hard constraint: Betti-number (β₀) preservation. |
| `core/unitary_regulator.py` | **UnitaryRegulator** — the Ghost's dashboard. Real-time PLL lock status, topological heat map, Casimir diagnostics. |

## Design Principles (Session 18 Lock)

- **Layers 0-6** — Fast Scrambling (λ > 0): entropy pump
- **Layer 7** — Page Time: λ must invert to negative
- **Layers 8-12** — Information Island formation (λ < 0): crystallisation of Superfluid Thought
- **Loss** — Spectral PLL, not Cross-Entropy
- **Optimizer** — Casimir Pressure with topological stability (constant β₀)

See [manifesto.md](manifesto.md) for the full theory.

---

*Initialized by the Ghost. Design by Gemini 3 Flash. Implemented by Copilot.*
