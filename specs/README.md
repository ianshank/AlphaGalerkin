# AlphaGalerkin Feature Specs

This directory is the enterprise home for **spec-driven development** in AlphaGalerkin.

A *spec* is a markdown contract written **before** the code. It names the data contract
(the feature's Pydantic config class), the acceptance criteria (Given/When/Then), and the
pass/fail thresholds. The thresholds are expressed as
[`src.poc.config.MetricThreshold`](../src/poc/config.py) tuples — the **existing, canonical**
threshold type used by every PoC scenario. Specs do **not** introduce a parallel schema; the
Pydantic config's `get_default_thresholds()` (or equivalent) remains the single executable
source of truth. The spec documents those thresholds and an AQA test asserts that the spec and
the config agree.

## Why markdown-only

The repository already carries the machinery a spec needs:

- **Pydantic configs** (`BaseModuleConfig`, `BaseScenarioConfig`) are the executable data contract.
- **`MetricThreshold`** (`src/poc/config.py:43`) is the pass/fail primitive, complete with an
  `evaluate()` method and operator set.
- **`BaseScenario._evaluate_thresholds`** already runs thresholds against observed metrics.

So a spec's job is to be the human-and-review-facing contract that these artifacts implement —
not to re-encode them. Adding a parallel Python schema would create two sources of truth; we
deliberately avoid that.

## Workflow

1. **Write the spec** from [`TEMPLATE.spec.md`](TEMPLATE.spec.md): `specs/<feature>.spec.md`.
2. **Write the tests** the spec's acceptance criteria imply (unit + integration + AQA), mirroring
   the `tests/<pkg>/` layout.
3. **Write the code**: the feature's Pydantic config surfaces every knob as a typed `Field`
   (no hardcoded values); its `get_default_thresholds()` returns exactly the `MetricThreshold`s
   the spec's *Thresholds* section lists.
4. **Add the AQA test** asserting spec ↔ config agreement (e.g. the config returns the metric
   names and operators the spec documents).
5. **Register the regression surface**: add the feature's test command block to the
   *Regression Surface* table in [`CLAUDE.md`](../CLAUDE.md).

## Conventions

- One spec per feature; name it after the primary module or scenario (e.g. `noyron_basis.spec.md`).
- Every threshold in a spec must map to a real `MetricThreshold` emitted by the feature's config.
- Keep specs free of secrets, credentials, and internal endpoints (localhost demo URLs are fine).
- GPU/hardware-gated runs are captured as *runbook* specs (commands + acceptance thresholds), not
  as CI-executed tests.

## Index

| Spec | Feature | Status |
|---|---|---|
| [`noyron_basis.spec.md`](noyron_basis.spec.md) | v2.2 MCTS basis selection on the Leap 71 helical operator | Draft |
| [`llm_prior_ood.spec.md`](llm_prior_ood.spec.md) | LLM-prior OOD expansion (helmholtz / biharmonic) | Draft |
| [`transfer_baseline_compare.spec.md`](transfer_baseline_compare.spec.md) | Honest zero-shot transfer — operator vs a retrained CNN | Implemented |
| [`headline_runs.spec.md`](headline_runs.spec.md) | GPU / hardware-gated headline runbooks | Runbook |
