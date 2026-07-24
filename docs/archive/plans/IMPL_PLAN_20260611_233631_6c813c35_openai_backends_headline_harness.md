# Implementation Plan — Operationalizing the PR #86 Headline Runs

- **Plan ID:** `20260611_233631_6c813c35`
- **Created:** 2026-06-11
- **Author:** Claude Code (deep-reasoning planning pass; grounded by two read-only `Explore` gap-analysis sub-agents)
- **Branch:** `claude/alphagalerkin-pr86-next-eYkVF`
- **Predecessor:** PR #86 (AI-for-Physics scaling themes — merged), PR #87 (CPU configs + runbook — open draft)
- **Status:** PROPOSED — not yet implemented. This document is the blueprint.

---

## 0. Why this, and why now

PR #86 shipped the three Adam-Brown-theme deliverables (OOD operators, `ScalingLawScenario`,
centaur `ResearchLoopOrchestrator`) **fully wired but only ever exercised on mocked CPU**.
PR #87 added CPU-runnable configs + a runbook, establishing the random-arm null baseline.

The single most logical *implementable* next step that the research surfaced is to **turn the
merged infrastructure into reproducible results across heterogeneous hardware**. That decomposes
into three code workstreams plus a docs workstream, all **additive and backwards-compatible**:

| WS | Title | Roadmap origin | Implementable on CPU CI? |
|----|-------|----------------|--------------------------|
| **WS1** | Generic OpenAI-compatible LLM backends (vLLM, llama.cpp-server) | CLAUDE.md "LLM-prior alternative backends" | ✅ (mocked SDK) |
| **WS2** | Reproducible headline-run + baseline-recording harness | New (gap: no PoC baseline registry; research-loop result unpersisted) | ✅ |
| **WS3** | Gap / redundancy / coverage hardening on the PR #86 surface | CLAUDE.md "OOD coverage expansion" + DRY | ✅ |
| **WS4** | Docs: tested-server matrix, runbook integration, CLAUDE.md milestone | CLAUDE.md "Document tested-server matrix in src/integrations/AGENT.md" | ✅ |

The GPU headline *numbers* themselves remain hardware-gated (CUDA + a live LLM server) and are
out of scope for code; WS2 makes capturing and regression-guarding them a one-command step on the
user's rig.

---

## 1. Governing constraints (non-negotiable, applied to every WS)

These are lifted directly from the project conventions (`src/integrations/AGENT.md`, CLAUDE.md):

1. **No hardcoded values.** Every tunable is a typed Pydantic field with a default and a
   `description`. New numeric constants surface as module-level named constants or config fields.
2. **Backwards compatible.** No existing import path, config field, CLI flag, or public symbol may
   break. New behaviour is opt-in; old `lm_studio:` configs and `LMStudio*` imports keep working
   verbatim. Back-compat shims are explicit aliases with a comment, not silent re-exports.
3. **Reusable / DRY.** Extract shared logic once (e.g. `_median`), mirror existing patterns
   (`BaselineRegistry`, `ResultCollector`, the `@register_*` registry decorators) rather than
   re-inventing. Prefer Protocol/ABC seams over concrete coupling.
4. **Best testing practices.** Unit + property-based (Hypothesis) + integration, CPU-safe via the
   `FakeOpenAIModule` mock, GPU/real-server paths behind `@pytest.mark.gpu_required` +
   `LM_STUDIO_URL`/equivalent env gating. Per-module coverage ≥ 85% (branch). Full suite stays green.
5. **Structured logging + debugging.** `structlog.get_logger(__name__)` module loggers; bound
   context (`scenario`/`run_id`/`arm`/`backend`); one event per network call carrying
   `latency_ms`, `parse_ok`, `retries_used`, `prompt_hash` (mirror existing `lm_studio_call`).
6. **Fail loud on GPU.** Reuse `src/poc/device.py::resolve_device`; never silently fall back to CPU
   for a `cuda` request.
7. **Gap-analysis discipline.** No dead code, no duplicated logic left behind, no unused parameters.
   Every new branch has a test that exercises it.

---

## 2. Current-state facts (from gap analysis — file:line anchored)

### 2.1 LLM backend seam (WS1 inputs)
- Client factory is **already injectable**: `_centaur_common.py:224-237` builds the LLM arm and
  `gate_llm_client(client_factory=LMStudioClient)` accepts any factory callable.
- Standard OpenAI SDK only: `client.py:187-194` (`chat.completions.create` with
  `response_format`/`seed`); preflight hits the standard `/v1/models` (`preflight.py:141-261`).
- VRAM check is **local CUDA** (`preflight.py:108-138`, `torch.cuda.mem_get_info`) — meaningful only
  when the model is colocated with the solver; must be conditioned for remote backends.
- Lazy SDK import: `client.py:48-61` (`_import_openai`). Tests patch exactly this:
  `tests/integrations/conftest.py:138-154` (`fake_openai` fixture → `FakeOpenAIModule`).
- Config nested identically in three places: `llm_prior_config.py:143-146`,
  `scaling_law_config.py:147-150`, `agents/config.py:634-637` — all `lm_studio: LMStudioConfig =
  Field(default_factory=LMStudioConfig, ...)`.
- `[lm-studio]` extra installs `openai>=1.40,<2.0` (pyproject). `src/integrations/AGENT.md` exists
  (33 lines) but has **no tested-server matrix**.

### 2.2 Baseline-recording prior art (WS2 inputs)
- `src/video_compression/perf/baseline.py::BaselineRegistry` — `load()`/`save()`/`compare_report()`,
  `_migrate_baseline_document()` (unversioned→v1), `_METRIC_DEFS` metric-direction framework,
  schema constants `PERF_BASELINE_DOCUMENT_SCHEMA_VERSION`/`PERF_BASELINE_ENTRY_SCHEMA_VERSION`,
  `extra="ignore"` forward-compat. **This is the pattern to mirror.**
- PoC results: `src/poc/results.py::ResultCollector` writes
  `outputs/poc/results/{run_id}/{scenario}_{hash}.json` + `summaries/summary_{run_id}.json`;
  `compare_runs()` compares two *historical runs*, **not a recorded baseline**.
- `ScenarioResult` (`src/poc/config.py`) already carries `metrics`, `threshold_results`,
  `artifacts`, `status`, `passed`, `device`, `torch_version`.
- **Gap:** `src/agents/cli.py::research` does **not** persist its `ExecutionResult` to JSON
  (`ExecutionResult.to_dict()`/`from_dict` exist in `src/templates/base.py` but are unused on disk).

### 2.3 Redundancy / coverage (WS3 inputs)
- `_median()` duplicated: `scaling_law.py:54` and `research_loop.py:50` → extract to `_centaur_common`.
- Coverage-sensitive branches (untested error paths): `_centaur_common.py` arm-gating raises
  (217-237), `gate_llm_client` recoverable-error returns (332-354), `gate_trained_model`
  (381-385), `run_basis_selection_cell` early-exit (271-291).
- No dead code, no TODO/FIXME found.

### 2.4 Logging + config dispatch (cross-cutting)
- Convention: `structlog.get_logger(__name__)`; class loggers bind `scenario`; `ResultCollector`
  binds `run_id`.
- PoC dispatch: `src/poc/config.py::load_config_from_dict` (372-429) infers from `data["name"]` and
  lazy-imports heavy configs. Agents: `load_config_file(path, ConfigClass)` (explicit class).

---

## 3. WS1 — Generic OpenAI-compatible LLM backends

### 3.1 Design (mirror the registry + Protocol patterns already in the repo)

Introduce a backend-agnostic layer **under** the existing `lm_studio` package, keeping `LMStudio*`
as the canonical reference backend (and back-compat surface).

New module: `src/integrations/openai_compat/` (sibling of `lm_studio/`), containing:

| File | Contents | Reuse |
|------|----------|-------|
| `config.py` | `OpenAICompatibleConfig(BaseModel)` — the generic superset of today's `LMStudioConfig` fields, **plus** `backend: Literal["lm_studio","vllm","llama_cpp"] = "lm_studio"` and `vram_check_mode: Literal["local","off"] = "local"` (off = remote server, skip `mem_get_info`). | Promote existing fields verbatim |
| `client.py` | `OpenAICompatibleClient` — exactly today's `LMStudioClient` body (SDK calls are already generic). `_import_openai` moved here. | Move, don't rewrite |
| `preflight.py` | `check_openai_compatible_server(config)` — today's logic with the VRAM step gated on `vram_check_mode == "local"`. | Move + condition |
| `registry.py` | `@register_backend("vllm")` decorator + `BackendRegistry` mirroring `src/games/registry.py` / `@register_runtime`. Maps backend name → `(ConfigClass, client_factory, preflight_fn)`. | Mirror existing registry template |
| `__init__.py` | Public exports + the three registered backends. | — |

`src/integrations/lm_studio/` becomes thin back-compat:
```python
# src/integrations/lm_studio/config.py  (after)
from src.integrations.openai_compat.config import OpenAICompatibleConfig

class LMStudioConfig(OpenAICompatibleConfig):
    """Back-compat alias; defaults backend='lm_studio'. Pre-existing import path."""
    backend: Literal[...] = "lm_studio"
```
Same for `LMStudioClient(OpenAICompatibleClient)`, `check_lm_studio_server = check_openai_compatible_server`.
**Every existing import keeps resolving.**

vLLM / llama.cpp are *configuration*, not new clients — they register the same
`OpenAICompatibleClient` with backend-specific defaults (e.g. llama.cpp default `model` token,
`vram_check_mode="off"` for a remote box). This is the "zero-code-change beyond a model rename"
claim, now *enforced by a registry* instead of asserted.

### 3.2 Config wiring (backwards compatible)
- Add an optional discriminator at the three embed sites. Keep the field **named `lm_studio`** for
  back-compat, but widen its type to `OpenAICompatibleConfig` (superclass) so existing YAML parses
  unchanged and new YAML can set `backend: vllm`. Add a deprecation-free alias property
  `llm_backend` returning the same object, so new configs read naturally. No field is removed.
- The arm builder (`_centaur_common.build_arm_evaluator` / `gate_llm_client`) resolves the
  client factory + preflight from `BackendRegistry.get(config.backend)` instead of the hardcoded
  `LMStudioClient`. Default `backend="lm_studio"` ⇒ identical behaviour to today.

### 3.3 No hardcoded values
- Backend-specific defaults (base_url ports, default model strings, `vram_check_mode`) live in the
  registered `ConfigClass` defaults — never inline. The `1234`/`8000`/`8080` ports become named
  defaults on each backend config.

### 3.4 Logging / debugging
- Bind `backend=<name>` into every `lm_studio_call`/`lm_studio_retry` event (rename event to the
  neutral `llm_call`/`llm_retry`, keeping the old names as duplicate emissions for one release if
  any dashboard greps them — decide via gap check; likely safe to just rename since they're internal).
- Add a `backend_resolved` debug event at construction.

### 3.5 Testing
- Generalize `tests/integrations/conftest.py`: the `fake_openai` fixture already patches
  `_import_openai`; move the patch target to `openai_compat.client._import_openai` and keep a
  `lm_studio`-named alias fixture. `FakeOpenAIModule` is already OpenAI-contract-shaped → reused.
- New `tests/integrations/test_backend_registry.py`: registration round-trip, unknown-backend
  raises, each backend resolves to a working client factory, `vram_check_mode="off"` skips
  `mem_get_info` (assert not called), preflight `/v1/models` parity across backends.
- New `tests/integrations/test_backcompat.py`: `LMStudioConfig`/`LMStudioClient`/
  `check_lm_studio_server` still importable and behave identically; old YAML (no `backend` key)
  parses and defaults to `lm_studio`.
- Real-server smoke (`@pytest.mark.gpu_required`, env-gated): parametrize over
  `{LM_STUDIO_URL, VLLM_URL, LLAMA_CPP_URL}` — each runs only when its env var is set.
- Coverage gate ≥ 85% branch on `src/integrations/openai_compat` and the lm_studio shims.

### 3.6 Files touched (WS1)
- **New:** `src/integrations/openai_compat/{__init__,config,client,preflight,registry}.py`;
  `tests/integrations/test_backend_registry.py`, `test_backcompat.py`.
- **Modified (additive):** `src/integrations/lm_studio/*` → re-export shims; `_centaur_common.py`
  arm builder to use the registry; the 3 config embed sites (widen type, add alias); `pyproject.toml`
  optional extras (`[vllm]`/`[llama-cpp]` can alias `[lm-studio]` since the SDK is the same `openai`
  package — likely just document that `[lm-studio]` covers all three); `tests/integrations/conftest.py`.

---

## 4. WS2 — Reproducible headline-run + baseline-recording harness

### 4.1 Design (mirror `BaselineRegistry`)

New subpackage `src/poc/baselines/` (PoC-scenario analogue of the perf harness):

| File | Contents |
|------|----------|
| `schema.py` | `ScenarioBaselineEntry` (scenario_name, metric_name, value, direction `Literal["higher_better","lower_better"]`, tolerance_pct) + `ScenarioBaselineDocument` (schema_version, hardware_tag, git_sha, llm_backend, entries). Constants `POC_BASELINE_DOCUMENT_SCHEMA_VERSION=1`, `..._ENTRY_SCHEMA_VERSION=1`, `extra="ignore"`. |
| `registry.py` | `ScenarioBaselineRegistry.load/save/compare(ScenarioResult|dict)` → `ScenarioRegressionReport` (per-metric ok/regressed/improved/skipped). `_migrate_*` for unversioned docs. Direct port of `perf/baseline.py` generalized off perf-specific metric names. |

The metric *direction* + tolerance lives in the recorded baseline document (data, not code), so the
same registry handles `residual` (lower better), `id_rollout_reduction_pct` (higher better),
`solved_fraction` (higher better), `llm_call_p95_latency_ms` (lower better) with no metric-specific
code — this is the reusable generalization of `_METRIC_DEFS`.

### 4.2 Close the research-loop persistence gap
- Add `--output-dir` to `src/agents/cli.py::research`; on completion write
  `outputs/agents/research/{run_id}/result.json` via the **existing** `ExecutionResult.to_dict()`.
  Mirrors `ResultCollector._save_result`. Pure addition; default off-path unchanged if flag omitted
  (defaults to `outputs/agents/research`).

### 4.3 CLI surface (no breaking changes)
- `python -m src.poc.cli record-baseline --run-id <id> --out <path>` — reads a completed run's
  result JSON(s), emits a `ScenarioBaselineDocument` (tolerances from config thresholds or a
  `--tolerance-pct` flag, default surfaced as a constant).
- `python -m src.poc.cli diff --baseline <path> --run-id <id>` — `ScenarioRegressionReport`,
  non-zero exit on regression (for CI use). Mirrors `scripts/benchmark_codec.py diff`.
- Existing `compare` subcommand untouched.

### 4.4 No hardcoded values
- Default tolerance %, the `outputs/...` roots, schema version — all named constants / config fields.

### 4.5 Logging / debugging
- `baseline_recorded` / `regression_detected` / `regression_clean` structlog events bound to
  `run_id`, `baseline_path`, `hardware_tag`. Debug event per metric diff with observed/baseline/pct.

### 4.6 Testing
- `tests/poc/test_scenario_baselines.py`: schema validation, JSON+migration round-trip
  (Hypothesis property test for unversioned→v1, mirroring the zoo manifest tests), compare
  direction logic (higher/lower better), tolerance boundary cases, regression exit code.
- `tests/agents/test_research_persistence.py`: `research --output-dir` writes a `from_dict`-loadable
  JSON; absent flag uses default; round-trip equality.
- Coverage ≥ 85% branch on `src/poc/baselines`.

### 4.7 Files touched (WS2)
- **New:** `src/poc/baselines/{__init__,schema,registry}.py`; CLI subcommands in `src/poc/cli.py`;
  `tests/poc/test_scenario_baselines.py`, `tests/agents/test_research_persistence.py`.
- **Modified (additive):** `src/agents/cli.py` (research `--output-dir`).
- **Sample baseline:** `config/baselines/poc_headline.example.json` (documented, not asserted in CI).

---

## 5. WS3 — Gap / redundancy / coverage hardening

1. **Extract `_median`** to `src/poc/scenarios/_centaur_common.py` (single definition + tests);
   `scaling_law.py` and `research_loop.py` import it. Delete both local copies. Regression: run the
   scaling-law + research-loop + centaur-common surfaces.
2. **Cover the arm-gating error paths** flagged in §2.3 with explicit unit tests
   (unknown arm → ValueError, trained-without-model/device → RuntimeError, llm disabled → None,
   preflight-fail → None, early-target / terminal-state early exits). Lifts `_centaur_common` (89%),
   `scaling_law` (86%), `research_loop` (86%) toward 90%+ and removes the "uncovered error path"
   risk.
3. **OOD coverage expansion (run-level):** add `helmholtz`/`biharmonic` to the *shipped demo manifest
   assertions* — already in the enums (PR #86); WS2's harness makes recording their headline deltas
   trivial. No new operator code.

### Files touched (WS3)
- **Modified:** `_centaur_common.py` (+`_median`), `scaling_law.py`/`research_loop.py` (import it),
  `tests/poc/test_centaur_common.py`, `tests/poc/test_scaling_law_scenario.py`,
  `tests/agents/test_research_loop.py`.

---

## 6. WS4 — Documentation

1. **`src/integrations/AGENT.md`**: add the **tested-server matrix** table (Backend × OpenAI-SDK
   version × status `tested|untested` × notes), per the roadmap.
2. **`docs/PR86_HEADLINE_RUNS.md`** (created in PR #87): add a "Recording & regression-guarding
   headline numbers" section pointing at the WS2 CLI, and a "Switching LLM backend" section pointing
   at WS1.
3. **`CLAUDE.md`**: new milestone line; new Regression-Surface rows for `openai_compat`,
   `poc/baselines`, research-loop persistence; verification commands; update the Directory Structure
   tree; mark the two roadmap rows DONE.

---

## 7. Phased execution playbook (sub-agents + skills + tooling)

Each phase is independently shippable, lands on `claude/alphagalerkin-pr86-next-eYkVF` (or a child
branch), and ends green. Sequential thinking: WS1 and WS2 are independent and can parallelize;
WS3 depends on nothing; WS4 trails each.

| Phase | Work | Sub-agent / skill / tool | Exit gate |
|-------|------|--------------------------|-----------|
| **P0** | Confirm seams, write failing tests first (TDD) | `Plan` agent to lock file-level design; `Explore` (done) | Test stubs compile, red |
| **P1** | WS1 backend abstraction + shims | `general-purpose` agent (isolation: worktree) implements; `claude-code-guide` if SDK questions | `pytest tests/integrations -m "not gpu_required"` green; back-compat tests pass |
| **P2** | WS2 baseline harness + research persistence | `general-purpose` agent (parallel worktree) | `pytest tests/poc/test_scenario_baselines.py tests/agents/test_research_persistence.py` green |
| **P3** | WS3 DRY + coverage | direct edits (small) | Per-module coverage ≥ 85% branch on all touched modules |
| **P4** | Lint/type/format gate | `Bash`: `ruff check`, `ruff format --check`, `mypy --strict` on changed surface | All clean |
| **P5** | Review | `/code-review` skill (high effort) then `/security-review` skill | Findings triaged/fixed |
| **P6** | Full suite | `Bash`: `pytest tests/ -q` | Green; no skips beyond known-skip list |
| **P7** | Verify behaviour | `/verify` skill — run both CPU demo configs + a `record-baseline`→`diff` round-trip | Commands succeed; diff exits 0 against self-baseline |
| **P8** | Docs + CLAUDE.md | direct edits | Regression-surface rows added |
| **P9** | Ship | commit per WS, push w/ retry, open **draft** PR, then `subscribe_pr_activity` | CI green |

**Tooling note:** use the GitHub MCP tools (not `gh`) for the PR; create as **draft**; keep the
model identifier out of all commits/PR text.

---

## 8. Test strategy (the "full test suite" bar)

- **Unit** — every new public function; every arm-gating branch; registry round-trips.
- **Property-based (Hypothesis)** — baseline doc migration (unversioned→v1), tolerance comparison
  invariants (a value within tolerance never flags; just outside always flags), config seed
  derivation. Mirrors existing zoo/loss-balancing Hypothesis suites.
- **Integration** — CPU, real-interface: WS1 arm built through the registry drives a real
  `BasisSelectionGame` micro-run with the mocked SDK; WS2 record→diff against a live `ScenarioResult`.
- **Backwards-compat** — dedicated suite asserting old imports/configs/CLI flags still work.
- **GPU / real-server** — `@pytest.mark.gpu_required`, env-gated per backend; auto-skip on CPU CI
  via the root `conftest.py` hook.
- **Coverage gates** — add CLAUDE.md Regression-Surface rows with explicit
  `--cov=... --cov-branch --cov-fail-under=85` commands for `openai_compat` and `poc/baselines`.
- **Full suite** — `pytest tests/ -q` green before each PR; the centaur test pyramid
  (sanity/integration/e2e/regression/AQA) must stay green since WS1/WS3 touch `_centaur_common`.

### Verification commands (to land in CLAUDE.md)
```bash
ruff check src/integrations/openai_compat src/poc/baselines
ruff format --check src/integrations/openai_compat src/poc/baselines
mypy --strict src/integrations/openai_compat src/poc/baselines
pytest tests/integrations -m "not gpu_required" -v
pytest tests/poc/test_scenario_baselines.py tests/agents/test_research_persistence.py -v
pytest tests/integrations tests/poc tests/agents -m "not gpu_required" \
  --cov=src/integrations/openai_compat --cov=src/poc/baselines --cov-branch --cov-fail-under=85
# behaviour
python -m src.poc.cli run --config config/scenarios/scaling_law_cpu.yaml
python -m src.poc.cli record-baseline --run-id <id> --out /tmp/base.json
python -m src.poc.cli diff --baseline /tmp/base.json --run-id <id>   # exits 0
python -m src.agents.cli research --config config/agents/research_loop_cpu.yaml --output-dir /tmp/rl
```

---

## 9. Risk register & mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Widening `lm_studio` field type breaks a strict consumer | Low | Superclass widening is covariant for reads; dedicated back-compat test suite; keep field name |
| Renaming structlog events breaks a dashboard grep | Low | Events are internal; grep repo first; if any external consumer, emit both names one release |
| `mypy --strict` on the discriminated-union config | Med | Use `Literal` discriminator + `Field(discriminator=...)`; covered by existing zoo precedent |
| Baseline registry over-couples to perf harness | Low | New subpackage; share *pattern* not code (perf metrics are domain-specific) |
| Coverage churn on `_centaur_common` from `_median` move | Low | Move + tests in same commit; run all three dependent surfaces |
| Scope creep onto PR #87 | Med | Ship WS1–WS4 as their **own** PR(s); keep #87 to CPU configs |

---

## 10. Out of scope (explicit)

- Producing the *actual* GPU headline numbers (hardware-gated; WS2 makes capture turnkey).
- The Noyron/Leap-71 roadmap (v2.1→v3.2), octree AMR, PicoGK STL ingestion — separate track.
- New PDE operators (Helmholtz/Biharmonic already landed in #86).
- Non-OpenAI-compatible LLM transports (e.g. raw gRPC) — the abstraction targets the OpenAI wire
  protocol the roadmap specifies.

---

## 11. Definition of done

- [ ] WS1: vLLM + llama.cpp selectable via `backend:` config; `LMStudio*` imports/configs unchanged;
      registry + back-compat tests green; ≥85% branch coverage.
- [ ] WS2: `record-baseline`/`diff` CLI + research-loop JSON persistence; migration + tolerance tests
      green; ≥85% branch coverage.
- [ ] WS3: single `_median`; arm-gating error paths covered; three centaur surfaces green.
- [ ] WS4: tested-server matrix in AGENT.md; runbook + CLAUDE.md updated (milestone, surfaces, dirs).
- [ ] `ruff` + `ruff format` + `mypy --strict` clean on changed surface.
- [ ] `pytest tests/ -q` green; centaur pyramid green.
- [ ] Draft PR(s) opened via GitHub MCP; CI green; PR activity subscribed.
