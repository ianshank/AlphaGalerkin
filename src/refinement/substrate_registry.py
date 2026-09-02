"""Registry for concrete ``RefinementSubstrate`` implementations.

Mirrors ``src.refinement.registry`` (which registers ``RefinementGame``
implementations) using the same ``src.templates.registry`` factory, so a
concrete substrate — ``TensorGridSubstrate``, ``SkfemTriSubstrate`` — can be
looked up by a string key with the ``@register_refinement_substrate("name")``
decorator, without callers needing to import the concrete module directly.

``RefinementSubstrate`` is a ``Protocol``, not an ABC: registration validates
structural compliance (``issubclass`` against a ``@runtime_checkable``
Protocol checks that the registered class defines every protocol method by
name), so a concrete substrate need not — and should not — inherit from it.
"""

from __future__ import annotations

from src.refinement.substrate import RefinementSubstrate
from src.templates.registry import create_registry

RefinementSubstrateRegistry, register_refinement_substrate = create_registry(
    "RefinementSubstrate",
    RefinementSubstrate,  # type: ignore[type-abstract]
)

__all__ = ["RefinementSubstrateRegistry", "register_refinement_substrate"]
