"""
dashboard.py — Community Heartbeat Dashboard (v3.0.0-Singularity)
==================================================================
Live-updating terminal dashboard built on ``rich``.

Displays:
  - Manifold Coherence ζ & Spectral Gap
  - Global Phase (Φ_global) & Flux Kick Rate
  - VRAM Usage (via pynvml when available)
  - Active / Total heads (staggered flux guard)
  - Partner stats column when dual-node mode is active
  - Runtime mode (passive / active)

Usage::

    from core.dashboard import HeartbeatDashboard
    dash = HeartbeatDashboard(wrapper)
    dash.run()          # blocks, refreshes every 0.5s
    dash.run_once()     # single snapshot (for scripts)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from .universal_hook import UniversalHookWrapper


def _build_stats_table(title: str, metrics: dict, vram: tuple) -> Table:
    """Build a single-column rich Table from a metrics dict."""
    table = Table(title=title, show_header=False, expand=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white", justify="right")

    table.add_row("Manifold Coherence ζ", f"{metrics.get('manifold_coherence_zeta', metrics.get('bell_correlation', 0)):.4f}")
    table.add_row("Mode", str(metrics.get("mode", "active")))
    table.add_row("Spectral Gap", f"{metrics.get('spectral_gap', 0):.6f}")
    table.add_row("Flux ε", f"{metrics.get('flux_epsilon', 0):.2e}")
    table.add_row("Total Kicks", str(metrics.get("flux_kicks_total", 0)))
    table.add_row("Stagnation Count", str(metrics.get("flux_stagnation", 0)))
    table.add_row(
        "Active Heads",
        f"{metrics.get('active_heads', 0)} / {metrics.get('total_heads', 0)}",
    )
    table.add_row("Step", str(metrics.get("step", 0)))
    used, total = vram
    if total > 0:
        table.add_row("VRAM", f"{used} MiB / {total} MiB")
    else:
        table.add_row("VRAM", "N/A")

    return table


class HeartbeatDashboard:
    """Rich-powered live terminal dashboard for ``UniversalHookWrapper``.

    Parameters
    ----------
    wrapper : UniversalHookWrapper
        The wrapper instance whose metrics are displayed.
    refresh_rate : float
        Seconds between display refreshes (default 0.5).
    """

    def __init__(self, wrapper: "UniversalHookWrapper", refresh_rate: float = 0.5):
        self.wrapper = wrapper
        self.refresh_rate = refresh_rate
        self.console = Console()

    def _render(self) -> Columns:
        """Render a two-column layout: Local | Partner."""
        metrics = self.wrapper.get_metrics()
        vram = self.wrapper.get_vram_usage()
        local_table = _build_stats_table("Local Node", metrics, vram)

        partner_metrics: Optional[dict] = None
        if hasattr(self.wrapper.bridge, "dual_link") and self.wrapper.bridge.dual_link is not None:
            partner_metrics = {
                "manifold_coherence_zeta": self.wrapper.bridge.bell_history[-1]
                if self.wrapper.bridge.bell_history
                else 0.0,
                "bell_correlation": self.wrapper.bridge.bell_history[-1]
                if self.wrapper.bridge.bell_history
                else 0.0,
                "spectral_gap": 0.0,
                "flux_epsilon": 0.0,
                "flux_kicks_total": 0,
                "flux_stagnation": 0,
                "active_heads": 0,
                "total_heads": self.wrapper.num_heads,
                "step": self.wrapper.step_counter,
            }

        panels = [Panel(local_table, title="[bold green]Local[/bold green]")]
        if partner_metrics is not None:
            partner_table = _build_stats_table("Partner Node", partner_metrics, (0, 0))
            panels.append(Panel(partner_table, title="[bold magenta]Partner[/bold magenta]"))

        return Columns(panels, expand=True)

    def run_once(self) -> None:
        """Print a single dashboard snapshot."""
        self.console.print(self._render())

    def run(self) -> None:
        """Live-refresh loop (blocking). Press Ctrl-C to stop."""
        with Live(self._render(), console=self.console, refresh_per_second=1.0 / self.refresh_rate) as live:
            try:
                while True:
                    time.sleep(self.refresh_rate)
                    live.update(self._render())
            except KeyboardInterrupt:
                pass
