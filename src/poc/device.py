"""Identity re-export of :mod:`src.device`.

The implementation lives in ``src/device.py`` (a root module, like
``src/seeding.py``) so research baselines can resolve a device without
importing the PoC package. This module re-exports the same objects so
existing ``from src.poc.device import resolve_device`` call sites keep
working — including the frozen codec track, which must not be edited in
the same changeset as core solver work.

The re-export is an identity (``src.poc.device.resolve_device is
src.device.resolve_device``). Wrapping here would pass the import and
break the mutation-killed identity assertion.
"""

from __future__ import annotations

from src.device import DevicePreference, resolve_device

__all__ = ["DevicePreference", "resolve_device"]
