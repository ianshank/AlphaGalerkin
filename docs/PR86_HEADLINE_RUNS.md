# PR #86 Headline Runs — Scaling Law + Centaur Research Loop

PR #86 ("AI-for-Physics scaling themes") merged the infrastructure for three
Adam Brown themes (held-out OOD generalisation, MCTS-budget scaling curves, the
"billions of cloneable Einsteins" research loop). The machinery shipped fully
wired but had only ever been exercised on **mocked CPU** in CI. This runbook is
the bridge from "merged" to "result": it records the CPU baseline (runnable
anywhere) and gives the turnkey commands for the GPU headline runs that produce
the publishable numbers.

## TL;DR

| Run | Hardware | Config | Produces |
|---|---|---|---|
| Scaling-law (CPU smoke) | CPU, no LLM | `config/scenarios/scaling_law_cpu.yaml` | Random-arm null: flat residual, ~0 scaling exponent |
| Scaling-law (headline) | CUDA + LM Studio | `config/scenarios/scaling_law_demo.yaml` | Trained/LLM scaling curves vs budget |
| Research-loop (CPU smoke) | CPU, no LLM | `config/agents/research_loop_cpu.yaml` | Random-arm discovery ledger over poisson/helmholtz/biharmonic |
| Research-loop (headline) | CUDA + LM Studio | `config/agents/research_loop_demo.yaml` | Per-problem arm ranking (random vs trained vs LLM) |

## 1. CPU baseline (established, reproducible without a GPU)

These two CPU-only configs run the **full, non-mocked pipeline** on the random
arm. They establish the null against which the GPU arms must show an advantage,
and they are safe to run on a laptop or in CI.

```bash
python -m src.poc.cli run --config config/scenarios/scaling_law_cpu.yaml
python -m src.agents.cli research --config config/agents/research_loop_cpu.yaml
```

### Recorded CPU result — scaling law (random arm, Poisson, budgets [4,8,16,32], 3 seeds)

| Metric | Value |
|---|---|
| `random_residual_median_b4` | 0.4905 |
| `random_residual_median_b8` | 0.4955 |
| `random_residual_median_b16` | 0.4902 |
| `random_residual_median_b32` | 0.4972 |
| `random_residual_scaling_exponent` | 0.0043 |
| `random_residual_fit_r2` | 0.294 |
| Acceptance | **FAIL (expected)** |

**Interpretation:** the residual is flat across a 8× budget range and the
scaling exponent is ~0. This is the *correct* null — random basis selection
does not convert MCTS compute into lower residual. The acceptance FAIL on a
random-only sweep confirms the threshold logic (`min_residual_decay`,
`min_fit_r2`) discriminates a real scaling curve from a flat one. The headline
run must show the **trained** and **LLM** arms producing a meaningfully negative
exponent with `r2 >= 0.5`.

### Recorded CPU result — research loop (random arm, manifest = poisson_id / helmholtz_ood / biharmonic_ood, 3 seeds)

| Metric | Value |
|---|---|
| `poisson_id_random_median_residual` | 0.4923 |
| `helmholtz_ood_random_median_residual` | 0.4927 |
| `biharmonic_ood_random_median_residual` | 0.4850 |
| `arm_wins_random` | 3 |
| `solved_fraction` | 0.000 |
| Acceptance | **FAIL (expected)** |

**Interpretation:** the new OOD operators (`helmholtz`, `biharmonic`) drive
cleanly through the full `BasisSelectionGame` → `PDEGameAdapter` → MCTS →
discovery-ledger path. Random search solves none of the three problems
(`solved_fraction = 0`), establishing the held-out difficulty. The headline run
must show the trained and especially the **LLM** arm lifting `solved_fraction`
on the OOD problems the FNet evaluator never trained on.

## 2. GPU headline runs (manual; require CUDA + LM Studio)

Hardware/endpoints required:

- A CUDA GPU (the scenarios call `resolve_device('cuda')` and **hard-fail** on
  CPU by design — there is no silent fallback).
- LM Studio (or any OpenAI-compatible server) serving `qwen2.5-14b-instruct`
  reachable at `lm_studio.base_url` (default `http://127.0.0.1:1234/v1`), with
  `>= min_free_vram_gib` (default 10 GiB) free.
- Optional: a trained FNet checkpoint to enable the `trained` arm — set
  `trained_checkpoint_path` in the config (left `null` skips that arm cleanly).

```bash
pip install -e '.[lm-studio]'                 # adds the openai SDK
export LM_STUDIO_URL=http://127.0.0.1:1234/v1 # used by the GPU smoke tests

# Scaling-law headline (random / trained / LLM arms)
python -m src.poc.cli run --config config/scenarios/scaling_law_demo.yaml

# Research-loop headline (OOD manifest, random / trained / LLM arms)
python -m src.agents.cli research --config config/agents/research_loop_demo.yaml
```

### Acceptance criteria to capture

Scaling-law (`scaling_law_demo.yaml`, primary arm = first in `arms`):

- `residual_scaling_exponent` clearly negative (residual decays with budget).
- `residual_fit_r2 >= min_fit_r2` (0.5) — a clean log-log power law.
- Arm-vs-arm Mann–Whitney comparison favours trained/LLM over random.

Research-loop (`research_loop_demo.yaml`):

- `solved_fraction >= min_solved_fraction` (0.5) across the 3-problem manifest.
- `arm_wins_llm` / `arm_wins_trained` dominate `arm_wins_random` on the OOD
  rows (`helmholtz_ood`, `biharmonic_ood`).

LM-Studio latency (from the LLM-prior surface, recalibrated for Qwen-14B Q4):

- `llm_call_p95_latency_ms <= 3000`.

### Where outputs land

- PoC scenarios: `outputs/poc/results/<run_id>/scaling_law_*.json` +
  `outputs/poc/summaries/summary_<run_id>.json` (+ HTML report when artifacts
  are emitted).
- Per-GPU latency baselines for the LLM arm should be recorded under
  `outputs/poc/llm_prior_ablation/` per the CLAUDE.md roadmap.

## 2b. Switching the LLM backend (vLLM / llama.cpp / LM Studio)

The LLM arm runs against any OpenAI-wire-compatible server. Select it with the
`backend:` field under the scenario's `lm_studio:` block — endpoint, model, and
the local free-VRAM policy auto-resolve from the backend profile, and any field
you set explicitly always wins:

```yaml
lm_studio:
  enabled: true
  backend: vllm          # lm_studio (default) | vllm | llama_cpp
  model: Qwen/Qwen2.5-14B-Instruct   # optional; backend has a sensible default
  # base_url / vram_check_mode auto-fill from the backend profile when unset.
```

| `backend:` | Default endpoint | `vram_check_mode` default |
|---|---|---|
| `lm_studio` | `http://127.0.0.1:1234/v1` | `local` (colocated) |
| `vllm` | `http://127.0.0.1:8000/v1` | `off` (remote) |
| `llama_cpp` | `http://127.0.0.1:8080/v1` | `off` (remote) |

See `src/integrations/AGENT.md` for the full tested-server matrix. The `openai`
SDK from the `[lm-studio]` extra serves all three — no extra install.

## 2c. Recording & regression-guarding headline numbers

Once a GPU headline run completes, capture its metrics as a baseline and gate
later runs against it (the diff exits non-zero on regression, so it is CI-usable):

```bash
# Record a baseline from a completed run's metrics (run id = the outputs/poc/results/<id> dir).
python -m src.poc.cli record-baseline --run-id <id> --out config/baselines/headline.json \
    --tolerance-pct 10 --hardware-tag "RTX 5060 Ti 16GiB" --llm-backend vllm

# Later: fail if any metric regressed beyond its recorded tolerance.
python -m src.poc.cli diff --baseline config/baselines/headline.json --run-id <new_id>
```

Metric direction is recorded in the document (higher-better for
`*_fit_r2` / `solved_fraction` / `*_reduction_pct` / `accept_rate`, lower-better
otherwise); extend with `--higher-better` / `--higher-better-suffix`. The
research loop persists its result the same way — pass `--output-dir` to
`python -m src.agents.cli research` and feed the written `result.json` to the
same recorder.

## 3. Pointers

- Operator definitions: `src/pde/operators.py` (`HelmholtzOperator`,
  `BiharmonicOperator`); registry in `src/pde/registry.py`; `PDEType` in
  `src/pde/config.py`.
- Canonical PDE map + shared rollout: `src/poc/scenarios/_centaur_common.py`
  (`PDE_TYPE_MAP`, `build_arm_evaluator`, `run_basis_selection_cell`).
- Scenario: `src/poc/scenarios/scaling_law.py` / `scaling_law_config.py`.
- Research loop: `src/agents/research_loop.py` / `src/agents/config.py`.
- Device policy (GPU-preferred, fail-loud): `src/poc/device.py::resolve_device`.
