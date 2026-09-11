"""The weekly benchmark's own workflow, built on the experiment harness.

:mod:`benchmarking.schedule` decides when a run is due, :mod:`benchmarking.weekly`
dispatches it and finishes it, and :mod:`benchmarking.report` renders the result.
None of it is experiment infrastructure - it is this repo's policy for running one
particular suite on a cadence, so it lives here rather than in ``experiment``.
"""
