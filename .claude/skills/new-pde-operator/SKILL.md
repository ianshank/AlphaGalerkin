---
name: new-pde-operator
description: Add a new PDE operator to AlphaGalerkin end-to-end. Use when introducing a PDE family (e.g. a new elliptic/parabolic/vector operator) so it registers in the operator registry, the PDEType enum, the canonical PDE_TYPE_MAP, and every dependent Literal enum — following the documented checklist so scenarios auto-pick it up.
---

# new-pde-operator — register a PDE operator across the stack

Adding a PDE operator touches several coupled registration points. Missing one leaves the
operator invisible to scenarios or breaks a `Literal` validator. Follow this checklist (it mirrors
the CLAUDE.md guidance for Helmholtz/Biharmonic).

## Checklist

1. **Implement the operator** in `src/pde/operators.py` (or `operators_picogk.py` for SDF/helical
   geometries), subclassing `PDEOperator`. Provide `source_term`, `boundary_value`,
   `exact_solution` (manufactured solution), and the residual. No hardcoded coefficients — surface
   them as Pydantic fields on `PDEConfig`/subclass, with a named module constant default
   (e.g. `DEFAULT_HELMHOLTZ_WAVENUMBER`).
2. **Register** it with `@register_pde_operator("<name>")` so `PDEOperatorRegistry` can find it.
3. **Add to `PDEType`** enum in `src/pde/config.py`.
4. **Add to `PDE_TYPE_MAP`** in `src/poc/scenarios/_centaur_common.py` (the canonical name→PDEType
   map shared by every centaur scenario).
5. **Extend dependent `Literal` enums** as needed: `ood_pde` (`llm_prior_config.py`),
   `ScalingLawConfig.pde` (`scaling_law_config.py`), `ResearchPDEName` (agents research loop).
6. **Tests**: `tests/pde/test_ood_operators.py`-style — properties/order, manufactured-solution
   analytics, residual-vanishes-on-exact (≤1e-3, incl. a Hypothesis parameter sweep), registry
   round-trip, and `BasisSelectionGame` construction with finite initial error.
7. **Regression surface**: add/extend the CLAUDE.md row for the operator family.

## Guardrails

- Additive only — never change an existing `PDEType` value or registry key.
- `ruff` + `mypy --strict` clean; residual property test must pass on CPU.
