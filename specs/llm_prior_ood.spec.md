# Spec: llm_prior_ood — Helmholtz / Biharmonic OOD expansion

> **Status:** Implemented (CPU wiring) + Runbook (GPU headline)
> **Primary module(s):** `src/poc/scenarios/llm_prior_ablation.py`, `src/pde/operators.py`
> **Config class:** `src.poc.scenarios.llm_prior_config.LLMPriorAblationConfig`
> **Tracking:** CLAUDE.md Next-Steps row "LLM-prior OOD coverage expansion"

## Context

The `llm_prior_ablation` scenario compares random / trained / LLM MCTS evaluators on an
in-distribution PDE and an out-of-distribution PDE. The trained `FNetEvaluator` never saw the
Helmholtz (oscillatory zeroth-order term, ∇²u + k²u = f) or Biharmonic (∇⁴u = f) residual
structures, so they are stronger held-out generalisation tests than Burgers. Both operators already
exist in the registry, `PDEType`, `PDE_TYPE_MAP`, and the `ood_pde` Literal; this spec covers using
them as OOD families.

## Data Contract

`LLMPriorAblationConfig.ood_pde ∈ {poisson, heat, advection_diffusion, burgers, helmholtz, biharmonic}`.
Shipped configs: `config/scenarios/llm_prior_helmholtz.yaml`, `config/scenarios/llm_prior_biharmonic.yaml`.

## Acceptance Criteria

### AC1: OOD family is registered and usable (CPU, no GPU)
- **Given** `ood_pde ∈ {helmholtz, biharmonic}`
- **When** the operator + basis-selection game are built via `_centaur_common`
- **Then** the game has a finite positive initial error and an 8-action space (proving a
  non-degenerate OOD target the FNet was not trained on).

### AC2: Demo YAML validates
- **Given** the shipped `llm_prior_<family>.yaml`
- **When** parsed via `load_config_from_dict`
- **Then** it yields an `LLMPriorAblationConfig` with the expected `ood_pde`.

### AC3 (Runbook): GPU headline retains the LLM advantage
- **Given** CUDA + LM Studio serving Qwen-14B
- **When** the scenario runs on the OOD family
- **Then** the headline thresholds hold: `ood_llm_residual ≤ 1e-2`, `ood_trained_residual > 1e-1`,
  `id_rollout_reduction_pct ≥ 25%`, `llm_call_p95_latency_ms ≤ 3000`.

## Thresholds

Same `MetricThreshold`s as `LLMPriorAblationConfig.get_default_thresholds()` (unchanged by the OOD
family). See `llm_prior_config.py:255-283`.

## Regression Surface

```bash
pytest tests/poc/test_llm_prior_ood_expansion.py -v            # CPU wiring
pytest tests/pde/test_ood_operators.py -v                       # operator analytics
```

## Runbook (GPU, manual)

```bash
pip install -e '.[lm-studio]'
LM_STUDIO_URL=http://127.0.0.1:1234/v1 \
  python -m src.poc.cli run --config config/scenarios/llm_prior_helmholtz.yaml
python -m src.poc.cli run --config config/scenarios/llm_prior_biharmonic.yaml
```

## Out of Scope

- New OOD operators beyond helmholtz/biharmonic (follow the `new-pde-operator` skill).
