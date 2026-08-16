"""AlphaGalerkin: Resolution-independent Go AI using Continuous Operator Learning."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    # Single source of truth is pyproject.toml's [project] version; reading it
    # here means a release bump cannot leave this file behind.
    __version__ = _pkg_version("alphagalerkin")
except PackageNotFoundError:  # pragma: no cover - only when not installed
    __version__ = "0.0.0+unknown"
