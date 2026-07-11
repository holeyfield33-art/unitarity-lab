"""benchmarks — evaluation and metric-plumbing scripts for unitarity-lab.

Contents
--------
- ``real_gsm8k.py``      : a real, graded GSM8K run (loads the dataset, generates
                          answers, grades by numeric extraction, records honest
                          per-problem metrics). This is the actual evaluation.
- ``pipeline_demos/``    : metric-plumbing DEMOS over synthetic tensors. They
                          print a "PIPELINE DEMO — synthetic tensors, not an
                          evaluation." banner and emit no accuracy. Use them to
                          inspect the metric column/JSON layout, nothing more.
"""
