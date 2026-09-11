"""Turning this project's stored runs into learning curves and summary numbers.

:mod:`analysis.plotting` reads a component's store through the harness and works
in this project's vocabulary - per-timestep ``reward`` and ``done`` curves,
episodes, returns - which is why it sits here rather than
in ``experiment``.
"""
