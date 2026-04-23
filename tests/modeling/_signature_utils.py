"""Shared signature introspection utilities for the ADR contract framework.

Both the golden-file generator and the contract test use identical logic
for extracting constructor and forward parameter entries. Keeping it here
ensures that the two stay in sync automatically: any change to the
extraction rules is a single-file edit.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

EMPTY = "<empty>"
FACTORY_UNKNOWN = "<factory:unknown>"
_TOPLEVEL_PACKAGE = "src.modeling"


def resolve_class(class_name: str) -> type | None:
    """Look up *class_name* as a top-level re-export of ``src.modeling``.

    Returns ``None`` if the attribute is absent — the contract test turns
    that into an assertion failure with remediation guidance, rather than
    letting an ``ImportError`` derail pytest collection before any
    meaningful message can be emitted.
    """
    pkg = importlib.import_module(_TOPLEVEL_PACKAGE)
    obj = getattr(pkg, class_name, None)
    if isinstance(obj, type):
        return obj
    return None


def _factory_identifier(factory: Any) -> str:
    """Serialise a ``default_factory`` callable to a stable string.

    Named functions and classes serialise to ``module.qualname`` so a
    swap from (say) ``default_factory=list`` to
    ``default_factory=OrderedDict`` is caught by the golden comparison.
    Lambdas serialise to ``module.<containing-class>.<lambda>`` — stable
    enough to detect refactors that move a lambda between classes, but
    a lambda-body edit will slip through unless the body is extracted
    to a named helper (recommended regardless). Unknown or
    un-introspectable callables fall back to ``<factory:unknown>``.
    """
    module = getattr(factory, "__module__", None)
    qualname = getattr(factory, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return FACTORY_UNKNOWN


def param_entries_from_signature(sig: inspect.Signature) -> list[dict[str, str]]:
    """Extract stable parameter entries from an ``inspect.Signature``.

    ``self`` is dropped; defaults are serialised via ``repr`` so JSON
    can carry tuples / ``None`` / bools uniformly.
    """
    entries: list[dict[str, str]] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        default_repr = EMPTY if param.default is inspect.Parameter.empty else repr(param.default)
        entries.append(
            {
                "name": name,
                "kind": param.kind.name,
                "default": default_repr,
            }
        )
    return entries


def pydantic_init_entries(cls: type[BaseModel]) -> list[dict[str, str]]:
    """Synthesise init-surface entries for a Pydantic model from ``model_fields``.

    Pydantic v2 rewrites ``__init__`` to ``(**data)``; the true surface
    is the field set (including inherited). Iterating ``model_fields``
    preserves declaration order (MRO-aware). ``default_factory`` values
    are serialised via :func:`_factory_identifier` so swapping the
    factory function is caught.
    """
    entries: list[dict[str, str]] = []
    for field_name, field_info in cls.model_fields.items():
        if field_info.default is not PydanticUndefined:
            default_repr = repr(field_info.default)
        elif field_info.default_factory is not None:
            default_repr = _factory_identifier(field_info.default_factory)
        else:
            default_repr = EMPTY
        entries.append(
            {
                "name": field_name,
                "kind": "PYDANTIC_FIELD",
                "default": default_repr,
            }
        )
    return entries


def init_entries(cls: type) -> list[dict[str, str]]:
    """Return init parameter entries for any class (Pydantic or regular)."""
    if isinstance(cls, type) and issubclass(cls, BaseModel):
        return pydantic_init_entries(cls)
    return param_entries_from_signature(inspect.signature(cls.__init__))


def forward_entries(cls: type) -> list[dict[str, str]] | None:
    """Return forward parameter entries, or ``None`` if there is no ``forward``."""
    fwd = getattr(cls, "forward", None)
    if fwd is None:
        return None
    return param_entries_from_signature(inspect.signature(fwd))
