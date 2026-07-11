"""benchmarks/pipeline_demos — metric-plumbing demonstrations.

These scripts feed **synthetic** source/sink tensors through the metric
pipeline to show the column layout and JSON shape. They are NOT an evaluation:
there is no dataset, no model generation, and no graded accuracy. For a real,
graded run see ``benchmarks/real_gsm8k.py``.
"""

BANNER = "PIPELINE DEMO — synthetic tensors, not an evaluation."


def print_banner() -> None:
    bar = "=" * len(BANNER)
    print(bar)
    print(BANNER)
    print(bar)
