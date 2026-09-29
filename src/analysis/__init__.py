"""Turning stored ``reward``/``done`` arrays into learning curves and metrics.

:mod:`analysis.curves` works on each run over time; :mod:`analysis.stats`
aggregates across runs. Both take plain arrays and never read the store or draw.
"""
