# Spec: headline_runs — GPU / hardware-gated runbooks

> **Status:** Runbook (manual, hardware-gated)
> **Tracking:** CLAUDE.md Next-Steps table (GPU headline rows) + `docs/PR86_HEADLINE_RUNS.md`

## Context

Several results require CUDA (and, for LLM arms, a running OpenAI-compatible server). CI runs the
CPU/mocked paths only; the headline numbers are a manual reviewer step. This spec collects the
runbooks so they are reproducible, records baselines via the existing
`python -m src.poc.cli record-baseline` / `diff` harness, and states the acceptance thresholds.

**Environment note.** These do not run in a CPU-only container (no `torch` CUDA kernels, no
`nvidia-smi`, no LM Studio). Attempting them there skips cleanly via the root `conftest.py`
`gpu_required` hook. Run them on a CUDA host.

## Runbooks

### R1 — Noyron basis selection (v2.2)
```bash
python -m src.poc.cli run --config config/scenarios/noyron_basis_demo.yaml
```
Acceptance: `error_reduction_pct ≥ 0` (monotone), `final_residual` bounded. The magnitude target is
a research item (see `specs/noyron_basis.spec.md`).

### R2 — Scaling law + research loop
```bash
python -m src.poc.cli run --config config/scenarios/scaling_law_demo.yaml
python -m src.agents.cli research --config config/agents/research_loop_demo.yaml
```
Acceptance: `residual_scaling_exponent` clearly negative, `residual_fit_r2 ≥ 0.5`;
`solved_fraction ≥ 0.5` across the manifest.

### R3 — LLM-prior (incl. OOD helmholtz / biharmonic)
```bash
LM_STUDIO_URL=http://127.0.0.1:1234/v1 \
  python -m src.poc.cli run --config config/scenarios/llm_prior_helmholtz.yaml
```
Acceptance: `id_rollout_reduction_pct ≥ 25`, `ood_llm_residual ≤ 1e-2`,
`ood_trained_residual > 1e-1`, `llm_call_p95_latency_ms ≤ 3000`.

### R4 — SBIR P40 benchmark
```bash
python -u -m scripts.run_sbir_p40
```
Acceptance: `mean_sm_util_pct` populated for every `pinn_p40` row in `outputs/sbir_p40/results.json`.

## Baseline gating

```bash
python -m src.poc.cli record-baseline --run-id <id> --out baselines/<name>.json --tolerance-pct 10
python -m src.poc.cli diff --baseline baselines/<name>.json --run-id <later-id>   # exit 1 on regression
```

## Out of Scope

Any change to CI (these stay manual). No secrets/endpoints beyond documented localhost.
