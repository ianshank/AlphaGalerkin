"""Name-lock helper for compare-scenario configs.

Kept torch-free so ``load_config_from_dict`` can import a compare *config*
without pulling ``CompareScenarioBase`` / CUDA teardown. The lifecycle
helpers live in :mod:`src.poc.scenarios._compare_common`, which re-exports
this function.

Name-lock is a **function**, not a config field: adding a field would change
``BaseScenarioConfig.compute_hash()``.
"""

from __future__ import annotations


def lock_scenario_name(
    expected: str,
    value: str,
    *,
    yaml_dispatch_note: bool = False,
) -> str:
    """Reject a scenario ``name`` that is not the registry dispatch key.

    Used by each compare config's ``@field_validator("name")``.

    Args:
        expected: Canonical ``SCENARIO_NAME`` for the config class.
        value: Incoming ``name`` field.
        yaml_dispatch_note: When True, the error mentions ``YAML dispatch key``
            (stochastic Galerkin's historical wording; its tests match it).

    Returns:
        ``value`` when it equals ``expected``.

    Raises:
        ValueError: If ``value`` is not the locked dispatch key.

    """
    if value != expected:
        if yaml_dispatch_note:
            msg = f"name must be {expected!r} (YAML dispatch key); got {value!r}"
            raise ValueError(msg)
        raise ValueError(f"name must be {expected!r}, got {value!r}")
    return value


# Stochastic config historically named the validator ``_name_locked``. Tests
# that assert the symbol keep working via this alias of the same function.
_name_locked = lock_scenario_name

__all__ = ["lock_scenario_name", "_name_locked"]
