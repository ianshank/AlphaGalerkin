"""AlphaGalerkin: Resolution-independent Go AI using Continuous Operator Learning."""

# Deploy-bundle mirror of ``src/__init__.py``. It cannot read the installed
# distribution's metadata (the Space installs no ``alphagalerkin`` package), so
# the version is a literal -- kept in step with ``pyproject.toml`` by
# ``tests/docs/test_version_consistency.py``, which previously would have caught
# this file sitting at "0.1.0" through the 0.4.0-dev bump.
__version__ = "0.4.0.dev0"
