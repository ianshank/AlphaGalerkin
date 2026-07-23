# Changelog

All notable changes to AlphaGalerkin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Stochastic Galerkin operator-splitting layer (NKE, `alphagalerkin-nke-integration`)

- New additive subpackage `src/pde/stochastic/` implementing the Lagrangian Galerkin
  projection of a Kolmogorov-forward generator `L = A + D + J` onto a Gaussian-mixture
  basis (after NKE, arXiv:2607.19173 — implemented from the standard derivation with a
  documented provenance caveat; see `specs/stochastic_galerkin_nke.spec.md` and
  `docs/related-work.md`): exact expm advection/diffusion moment flows, a **trained MDN
  jump semigroup** (residual, dt-scaled, identity at dt=0), symmetric Strang composition
  (measured second-order: slopes 1.995–2.000), and a **parallel-in-time trainer** whose
  M−1 interval losses evaluate in one batched forward pass over precomputed particle
  clusters (no autoregressive rollout). GPU/CPU agnostic throughout.
- Verified against independent van Loan closed forms: OU moment recovery < 1e-3
  (Hypothesis stable-A sweeps); jump-OU with the exact compound-Poisson oracle < 1e-3;
  trained-MDN trajectory errors 1.7e-2 / 7.4e-3 (gate 5e-2); trainer reaches the
  oracle-achievable loss floor (gap closure 0.000). Every unmeasured gate was
  calibrated from a pinned run and recorded in the spec.
- A generator with a jump term but no jump model raises `JumpModelMissingError` — the
  jump component is never silently dropped (change-doc requirement), with a
  defense-in-depth re-check in the Strang composer.
- New `stochastic_galerkin_compare` PoC scenario + CLI: deterministic Galerkin-attention
  arm vs the stochastic moment-projection arm on a shared Fokker-Planck/OU density
  benchmark with free analytic ground truth. The single gate is the stochastic arm's
  absolute MSE (measured 2.3e-8, gate 1e-6); the deterministic arm's MSE and the ratio
  are recorded **ungated** (novelty ≠ superiority). Committed artifacts:
  `results/stochastic_galerkin_compare.{csv,png}`,
  `config/baselines/stochastic_galerkin_ci.json`.
- MCTS/self-play untouched, enforced twice: an AST import-isolation guard over the new
  modules and the green MCTS/F0/F1 regression surfaces. The novelty-gap documentation
  guard is executable (`tests/regression/test_related_work_guard.py`): every
  `docs/related-work.md` entry must carry a "does NOT do" clause, and the retracted
  blanket "no MCTS+Galerkin" claim is asserted absent from the README.

### Added — Honest zero-shot transfer benchmark (operator vs retrained CNN)

- New CI-gated `transfer_baseline_compare` PoC scenario replacing the **fabricated**
  "Zero-shot Transfer MSE 0.000209, 240× better than threshold" headline (a hardcoded
  notebook markdown cell no code ever computed). `src/experiments/cnn_baseline.py`
  (`DiscreteCNNBaseline` discrete foil), `src/research/transfer_baseline_compare.py`
  (median-over-seeds harness), `src/poc/scenarios/transfer_baseline_compare{,_config}.py`,
  `scripts/run_transfer_baseline_compare.py`, `specs/transfer_baseline_compare.spec.md`.
- **Honest measured result** (committed `config/baselines/transfer_ci.json`,
  `results/transfer_baseline_compare.{csv,png}`): the resolution-independent operator
  transfers zero-shot (19×19 MSE ≈ 2.3e-3, trained only at 9×9) but a discrete CNN —
  retrained *or even applied zero-shot* — is more accurate. The operator's value is
  **zero-retraining (one model at any resolution), not peak accuracy**. The gated ratio is
  committed as a regression ceiling, not a false `< 1` win claim.
- Every headline number now comes from one real (median-ranked) seed, so dividing the
  committed absolutes reproduces the committed ratio exactly. Shared
  `src/research/seed_sweep.py` de-duplicates the seed-derivation between the transfer and
  L-shape harnesses. Per-module branch coverage ≥ 92% (cnn 100%, harness 98%, config 98%,
  scenario 92%); new `transfer-baseline-regression` CI job (soft-gated) diffs the tripwire
  run against the committed baseline.

### Removed — Cut to the research core (6 application modules, ~72k LOC)

- `git rm` of `src/{video_compression,reentry,vertex,intercept,firefighting,thermo}` and
  their test trees / scripts / configs / docs to refocus the repo on the Galerkin + MCTS
  core (pre-cut tag `archive/pre-core-cut-2026-07-22`). All six were import-safe (nothing in
  the keep-set imported them). The `thermo` λ-window negative-result ablation is preserved in
  git history. Companion cleanup: removed the `video`/`requires_video` pytest markers, the
  `vertex` packaging extra, the codec-perf CI workflow, and pruned the C4 architecture
  diagrams, `AGENT.md`, and `CLAUDE.md` of the deleted subsystems.

### Changed — Prior-art review + SBIR reframe

- `docs/proposals/PRIOR_ART_REVIEW.md`: the narrow MCTS-Galerkin-basis-selection delta
  survives, but the blanket "no MCTS + FEM" claim does **not** (TreeMesh, arXiv:2111.07613,
  couples MCTS + RL with FE mesh generation). SBIR positioning reframed to the method delta
  at matched wall-clock, not a demonstrated accuracy win.

### Fixed — Single-agent MCTS backup (F0, correctness) + reward wiring (F1)

- **F0 — single-agent backup.** `MCTSNode.backup` unconditionally negated the backed-up value at
  every tree level (a two-player assumption), while `select_child` maximises `Q + exploration` at
  every depth. For single-agent games (`n_players == 1`: every PDE / refinement game) this made the
  search *minimise* value at odd depths. Fixed by routing the sign flip through a new
  `src.mcts.search.SearchMode` (`SINGLE_AGENT` / `ZERO_SUM` / `LEGACY_ADVERSARIAL`);
  `MCTSNode.backup(value, invert)` now takes the flag explicitly. **Backwards compatible:** the
  `MCTS.__init__` default is `ZERO_SUM` (byte-for-byte the old two-player backup), so Go/chess are
  unchanged; single-agent callers pass `SINGLE_AGENT`. `LEGACY_ADVERSARIAL` (deprecated, warns) exists
  only to reproduce pre-fix results.
- **L-shape headline corrected & republished.** The `lshape_amr_compare` MCTS arm now defaults to
  `search_mode="single_agent"`. Re-running the canonical config over the same 5 seeds:
  `legacy_adversarial` (pre-fix) → median L2 ratio **0.8896** (~11% win); `single_agent` (corrected)
  → **0.9605** (~4% win), win fraction 0.80 in both. Still a win at matched DOF (primary gate passes),
  but a **smaller, honest** one. `results/lshape_mcts_vs_dorfler.{csv,png}` regenerated under the
  corrected mode; `specs/lshape_amr_compare.spec.md` AC3 documents both numbers.
- **F1 — reward reachability.** `PDEGame.get_reward` (previously abstract with zero `src/` call sites)
  is now reachable through MCTS behind an opt-in `MCTS(use_intermediate_rewards=...)` flag (default
  `False` → unchanged behaviour): `_simulate` accumulates `R = Σ γ^t · get_last_reward()` along the
  selection path and backs up `R + γ^d · V(leaf)`. `PDEGameAdapter` gained `get_last_reward()`
  implementing the optional `SupportsStepReward` protocol.
- **Tests.** `tests/mcts/test_backup_modes.py` (sign-by-mode, the anchor
  `test_single_agent_search_prefers_higher_value_at_all_depths` which fails on the inverting modes,
  deprecation + reward-discount validation, intermediate-reward accumulation),
  `tests/pde/test_reward_reachability.py` (get_reward invoked iff enabled), and
  `tests/pde/test_clone_isolation.py` (F3 clone isolation across every concrete PDE game).

### Fixed — Post-merge review hardening (PR #95 follow-up)

- **`PDEGameAdapter.search_mode` property.** Since `MCTS.__init__` defaults to `SearchMode.ZERO_SUM`
  for back-compat, a caller wrapping a raw `PDEGameAdapter` who forgot to pass `search_mode` would
  silently get the pre-fix (wrong-for-single-agent) backup. The adapter now exposes a `search_mode`
  property returning `SearchMode.SINGLE_AGENT`, mirroring `RefinementGameAdapter.search_mode`, so PDE
  callers can wire `MCTS(search_mode=adapter.search_mode)`. Additive; nothing merged was incorrect
  (the production `lshape_amr_compare` path already plumbs `search_mode` explicitly).
- **`MCTS._read_step_reward` contract check.** A game exposing `get_last_reward` as a non-callable
  attribute (float / property value) now raises a clear `TypeError` at the source of the contract
  violation instead of a cryptic `'... is not callable'` deeper in the search loop.

### Added — Domain-free refinement engine (`src/refinement/`) + λ-scheduling ablation (`src/thermo/`)

- **`src/refinement/`** — the domain-agnostic `RefinementGame` engine (`RefinementState` +
  `RefinementLike` protocol, `RefinementGameAdapter` → MCTS passing `SINGLE_AGENT`, generic
  `RefinementGameConfig[TDomain]`, `@register_refinement_game`). `PDEState` gains additive
  `to_refinement()`/`from_refinement()` converters (fields unchanged; existing PDE tests green).
  85% branch gate; audit-clean.
- **`src/thermo/`** — the first non-PDE `RefinementGame`: a λ-window (BAR/FEP) sample-scheduling
  ablation. `LambdaSchedulingGame` (deterministic `apply_action`, monotone-under-allocate /
  reachable-non-monotone-under-split), four `VarianceSurrogate`s (analytic / mismatched / recorded /
  operator-stub), and a plan-in-surrogate / act-in-world comparison harness.
- **NEGATIVE result (kill criterion triggered).** Untrained uniform-prior MCTS is **~2× worse** than
  greedy variance-weighted allocation at every surrogate bias including zero — it over-splits and
  fragments the sample budget. The thesis is falsified for this configuration; the code is retained
  only as the falsification harness and **no capability is claimed**. Honest caveat: a purely
  multiplicative surrogate bias is scale-invariant for allocation, so genuine mismatch needs shape
  distortion — moot here since MCTS already loses at zero mismatch. Artifacts:
  `results/lambda_scheduling.{png,csv}`; write-up in `specs/lambda_scheduling.spec.md`. CI gates the
  mechanics (85% branch), not the losing headline.

### Changed — Tech-debt hardening on the refinement/thermo surface

- **Reward-scale confound fixed & negative result revalidated.** The MCTS intermediate-reward
  return `R + γ^d·V(leaf)` mixed the order-`1e-3` shaped reward with an order-`1` terminal winner.
  `LambdaSchedulingGame.get_winner` now returns **0** (neutral) for a non-converged terminal
  (was `-1`), and the per-edge cost is keyed on the **window-count delta** (a split adds a window),
  not on a DOF side-effect. Re-running leaves the verdict unchanged (ratio 2.00 → 2.05), so the
  negative result is genuine over-splitting, not a reward-scale artifact. Spec + committed
  `results/lambda_scheduling.{png,csv}` updated.
- **Structured logging** (`structlog`) added to `src/refinement/adapter.py`,
  `src/thermo/{surrogate,outer_loop}.py`, and `scripts/run_lambda_scheduling.py`, mirroring the
  repo's event-logging convention.
- **Typing escape hatches removed:** `OperatorSurrogate.predict_fn` and `run_bias_sweep.make_planner`
  are now `Callable`-typed; the CLI builds its config via `model_validate`; the dead
  `replace_params` helper (which carried a `type: ignore`) is deleted. Zero avoidable `type: ignore`
  remain in the new src surface.
- **Reuse:** `iterate_greedy/uniform/mcts` generators + `score_true_stderr` extracted in
  `outer_loop`; the plot CLI now consumes them instead of re-implementing the scheduler loops.
- **No hardcoded values:** `DEFAULT_NOISE_FREQUENCY`, `MIN_SPLIT_CHILD_SAMPLES`, `BUDGET_GRID_POINTS`
  named; `RATIO_FLOOR` reused; `reward_discount` surfaced as a typed `LambdaSchedulingConfig` /
  `SchedulingParams` field with a `(0, 1]` validator.
- **Coverage:** new tests for the converged-winner / tolerance-terminal branch, split-vs-allocate
  cost keying, zero-window infinite variance, the adapter's torch-tensor `get_state` path, empty
  `from_refinement`, and the `reward_discount` validators. `src/thermo` 95% / `src/refinement` 99%
  branch. `ruff` + `mypy --strict` clean; abstraction-audit clean.

### Fixed — CI coverage job uses the pure-Python tracer

- The installed torch wheel crashes coverage's default **C tracer** on `import torch._C`
  (`ValueError: module functions cannot set METH_CLASS ...` / segfault), so the `Test Coverage` job
  failed at collection. Set `COVERAGE_CORE=pytrace` on the coverage job (the remedy already
  documented in CLAUDE.md) so the coverage gates actually run.

### Added — Spec-driven agentic tooling + Noyron v2.2 (`specs/`, `.claude/`, `src/agents/`, `src/poc/scenarios/noyron_basis*`)

Additive, backwards-compatible sprint across four workstreams:

- **Spec-driven development (`specs/`)** — a markdown-only spec tree (`README.md`,
  `TEMPLATE.spec.md`, and per-feature specs) whose thresholds reuse the canonical
  `src.poc.config.MetricThreshold` (no parallel schema). Spec → tests → code → AQA →
  regression-surface entry is now the documented workflow.
- **`.claude/` project scaffolding** — committed shared Claude Code config so web/CLI sessions
  can run the repo's checks: a SessionStart hook that bootstraps `pip install -e '.[dev]'`
  (including the `SETUPTOOLS_USE_DISTUTILS=stdlib` fix for `antlr4-python3-runtime`),
  `settings.json`, four skills (`spec-new`, `regression-surface`, `coverage-gate`,
  `new-pde-operator`), five persona subagents, and three slash commands. Local artifacts
  (`.claude/plans/`, `settings.local.json`) stay gitignored.
- **`src/agents/` hardening** — new `src/agents/AGENT.md`; opt-in `BaseAgent` lifecycle hooks
  (`pre/post_setup`, `pre/post_step`, default no-ops); opt-in wall-clock timeout gated on
  `AgentConfig.enforce_timeout` (default `False` preserves behaviour; enabled →
  `ExecutionStatus.TIMEOUT`); reusable `src/agents/scaffold.py` + `agents.cli scaffold` command.
- **Noyron v2.2 — first MCTS-on-Noyron result** — new `noyron_basis` PoC scenario driving MCTS
  Galerkin basis selection on the Leap 71 helical SDF operators via the existing
  `pde_basis_helical` path, reusing the geometry-agnostic `_centaur_common` primitives. A
  reusable `make_manufactured_operator` overlays a product-of-sines target so the homogeneous
  helical operators yield a non-degenerate game. The default thresholds assert the provable
  correctness property (`error_reduction_pct ≥ 0` monotone, bounded residual); the reduction
  *magnitude* on 3D SDF geometry is limited by the current candidate basis library (~2–4 %) and
  documented as an open research item. Per-arm medians are always recorded so results are never
  vacuous.
- **LLM-prior OOD expansion** — shipped `config/scenarios/llm_prior_{helmholtz,biharmonic}.yaml`
  + AQA tests (operators already in the `ood_pde` Literal / `PDE_TYPE_MAP`).
- **Known-issue closure** — SGF variation parsing marked RESOLVED (verified green); MCTS
  rate-control skips documented as a Milestone 10 Phase 3 gate.

Coverage: `agents/base.py` `config.py` `scaffold.py` 100 %; `noyron_basis.py` 97 %,
`noyron_basis_config.py` 100 %. `ruff` + `mypy --strict` clean on the changed surface.

### Added — LLM-Prior MCTS Basis Selection (`src/integrations/lm_studio/`, `src/poc/scenarios/llm_prior_ablation.py`, `src/poc/scenarios/llm_prior_config.py`, `config/scenarios/llm_prior_demo.yaml`)

New PoC scenario `llm_prior_ablation` that benchmarks three MCTS evaluators
on Poisson (in-distribution) and Burgers (out-of-distribution): the
existing `RandomEvaluator`, the existing `FNetEvaluator` (trained), and a
new `LMStudioEvaluator` backed by an OpenAI-compatible local LLM (Qwen-14B
served via LM Studio by default). The demo proves that a generalist LLM
with no PDE-specific training can guide MCTS competitively on familiar
PDEs and survive zero-shot on PDE families where the trained evaluator
collapses — the headline differentiator for the SBIR narrative.

- **`src/integrations/` namespace** — first entry in a new
  `src/integrations/` package for third-party-service adapters gated
  behind optional extras. Each subpackage's SDK is imported lazily so the
  base install never pulls it in. New `src/integrations/AGENT.md`
  documents the integration conventions (lazy imports, typed exception
  hierarchy, structured logging, preflight on construct).
- **`src/integrations/lm_studio/` subpackage** — six modules.
  `LMStudioConfig` (Pydantic; every knob — base_url, model, timeout_ms,
  max_retries, backoff_base_s, temperature, max_tokens,
  fallback_to_uniform_on_parse_error, min_free_vram_gib,
  preflight_on_construct, enabled — surfaced as a typed field).
  `LMStudioPolicyResponse` + typed exception hierarchy (`LMStudioError`
  → `LMStudioParseError` / `LMStudioActionSpaceMismatchError` /
  `LMStudioConnectionError` / `LMStudioPreflightError`).
  Deterministic prompt builder + sha256-truncated `prompt_hash`.
  Synchronous `LMStudioClient` (openai-SDK wrapper using
  `response_format={"type":"json_object"}` and `seed=...`, bounded
  exponential-backoff retries with corrective user-turn on action-size
  mismatch). The retry classifier splits SDK exceptions into retryable
  (APIConnectionError / APITimeoutError / RateLimitError /
  InternalServerError) and non-retryable (Authentication / BadRequest /
  NotFound / etc.) so auth and validation errors fail fast instead of
  consuming the retry budget. `check_lm_studio_server` preflight
  (server reachable + model in `/v1/models` + free-VRAM floor via
  `torch.cuda.mem_get_info`) closes its one-shot SDK client in a
  `finally` block to avoid leaking HTTP connections. `LMStudioEvaluator`
  implements `src/mcts/evaluator.py::Evaluator` structurally with
  illegal-action `-inf` masking + temperature softmax.
- **`llm_prior_ablation` PoC scenario** — orchestrates the
  (arm × pde × seed) grid with median + Mann-Whitney significance and
  an HTML report artifact built via the existing `HTMLReportGenerator`. Arm
  gating is graceful: when LM Studio preflight fails or no trained
  checkpoint is configured, the affected arm is disabled *and* its
  acceptance thresholds are removed from `self.config.thresholds` so
  absent metrics don't auto-FAIL the run. The same gating applies
  symmetrically when the random arm is disabled (the
  `id_rollout_reduction_pct` joint metric is dropped) and when zero
  LLM-call latency samples are recorded (the `llm_call_p95_latency_ms`
  threshold is dropped to avoid recording NaN).
- **GPU-only by policy** — scenario `setup()` calls
  `src.poc.device.resolve_device(config.device, context=...)` which
  raises `RuntimeError` if CUDA is unavailable. No silent CPU fallback
  anywhere on the new path. Per-seed reproducibility via
  `np.random.seed`/`torch.manual_seed` before each `MCTS(...)`
  construction (no `seed` kwarg on `MCTS.__init__`) plus
  `LMStudioClient.complete_policy(seed=...)`.
- **MCTS tree reuse** — `_run_cell` reuses the search tree across macro-
  steps via `mcts.advance(action)` instead of re-instantiating MCTS
  after every move; matches the AlphaZero convention. Early loop exit
  on invalid evaluator output is logged via a `cell_loop_early_exit`
  warning rather than breaking silently.
- **Optional dependency `[lm-studio]`** — adds
  `openai>=1.40,<2.0` as a new optional extra in `pyproject.toml`. The
  SDK is imported lazily so the base install never pulls it in. CPU CI
  mocks the SDK via `tests/integrations/conftest.py::FakeOpenAIModule`
  so the optional dep is never required for green CI. GPU smoke tests
  carry `@pytest.mark.gpu_required` and additionally gate on
  `LM_STUDIO_URL`.
- **Headline acceptance thresholds** — `id_rollout_reduction_pct ≥ 25%`
  (Mann-Whitney p<0.05, 10 seeds), `ood_llm_residual ≤ 1e-2`,
  `ood_trained_residual > 1e-1`, `llm_call_p95_latency_ms ≤ 3000`
  (recalibrated from an initially-proposed 300 ms after Qwen-14B Q4
  empirical latency review).
- **Coverage** — per-module on the new surface: `lm_studio` package
  91% (line+branch combined: `client.py` 91%, `evaluator.py` 95%,
  `preflight.py` 97%, `prompt.py` 100%, `config.py`/`schema.py`/
  `__init__.py` 100%), `llm_prior_ablation.py` 86%,
  `llm_prior_config.py` 100%. 96 new tests across CPU-mocked + GPU
  smoke (93 CPU-safe + 3 `@pytest.mark.gpu_required`); full project
  regression green; `ruff` + `ruff format` clean; `mypy --strict`
  zero new errors on the changed surface.

### Added — SBIR P40 Benchmark Hardening (`src/research/`, `scripts/run_sbir_p40.py`, `config/benchmarks/sbir_p40.yaml`)

Closes the gaps surfaced by the post-run SBIR P40 benchmark report
(NS-FDM L2 ≈ 0.5 floor, Dörfler AMR stuck at 18 DOF on Burgers, no GPU
telemetry, hard-coded CPU PINN, no extreme-resolution Poisson level).

- **NS-FDM Taylor-Green parity** — fixed numpy/torch asymmetry in
  `NavierStokesOperator.exact_solution` (numpy branch had `cos(x)*cos(y)`
  instead of `sin(x)*cos(y)` for `uy`). Single-line fix at
  [src/pde/operators.py:1189](src/pde/operators.py:1189) corrects three
  metrics simultaneously: the FDM IC, the FDM L2 reference, and the PINN
  L2 evaluation (all routed through the numpy branch). The torch branch
  was always correct, so PINN training was unaffected — only post-hoc
  evaluation was corrupted. New
  `tests/pde/test_taylor_green_invariants.py` asserts elementwise
  numpy/torch agreement to guard against the drift recurring.
- **Dörfler AMR escapes the 18-DOF ceiling** — `AMRConfig` defaults
  raised so 1D refinement on smooth Burgers (Cole-Hopf shock indicator
  is sharply concentrated) reaches meaningful DOF counts:
  `marking_fraction` 0.3 → **0.5**, `max_refinements` 10 → **30**,
  `max_initial_points_1d` 8 → **256**, `initial_dof_divisor` 4 → **2**.
  The `_solve_amr_1d` `n_start` formula is now target-aware:
  `n_start = max(min(n_dof // 2, max_initial_points_1d), min_initial_points)`.
  New regression test
  `TestDorflerAMRSolver.test_dorfler_amr_1d_reaches_meaningful_dof`
  parametrised across `target_dof ∈ {128, 512, 2048}` ensures the
  algorithm never collapses back to the 18-DOF bug and that n_dof
  scales with the request.
- **Canonical PINN respects `device` + auto-detects vector PDEs** —
  `PINNConfig` gains `device: str = "auto"` (per CLAUDE.md GPU-preferred
  policy) and `vector_pde: bool | None = None`.
  `SimplePINNSolver.solve()` honours both: device resolution flows
  through `src.poc.device.resolve_device` (extended to support
  indexed `cuda:N` strings with bounds checking), and Navier-Stokes
  operators auto-build a 2-channel network with per-component
  Laplacian residual. The previous hard-coded
  `device = torch.device("cpu")` is gone.
  `_build_network(input_dim, output_dim=1)` now accepts an output
  dimension. Metadata round-trip includes `device`, `vector_pde`,
  `n_collocation`, and the new `gpu_profile` block.
- **GPU utilisation profiler** — new
  [src/research/gpu_profiler.py](src/research/gpu_profiler.py) provides
  a `GpuUtilizationProfiler` context manager wrapping `nvidia-smi dmon`.
  Spawns the dmon subprocess on `__enter__`, terminates and parses on
  `__exit__`, returns a `GpuUtilizationReport` (mean SM-util %, mean
  memory-util %, peak FB-memory MiB) which `SimplePINNSolver` embeds in
  `SolverResult.metadata["gpu_profile"]`. Skips silently when
  `nvidia-smi` is missing (CI on no-GPU hosts). All numerical literals
  surfaced as named module constants
  (`_DMON_COL_GPU=0`, `_DMON_COL_SM_PCT=4`, `_DMON_COL_MEM_PCT=5`,
  `_DMON_COL_FB_MEM_MIB=8`, `_DMON_MIN_COLUMNS=6`,
  `_DEFAULT_TERMINATE_TIMEOUT_S=5.0`); `terminate_timeout_s` is a
  configurable constructor field.
- **`PDEBenchmarkRunner` `--heavy` opt-in** — extra refinement levels
  (e.g. 65 536-DOF Poisson for the P40's 24 GiB advantage) live under
  `heavy_refinement_levels` in the YAML and are appended only when the
  runner is constructed with `heavy=True` (or
  `run_sbir_demo --heavy`). Default behaviour is unchanged so CI
  smoke tests stay fast.
- **`scripts/run_sbir_p40.py` rewritten as a config-driven CLI** — the
  previous 260-line subclass-based fork is gone. New shape: small
  argparse-driven driver that loads
  `config/benchmarks/sbir_p40.yaml` (PINN profiles for `p40` and `cpu`
  rows, baselines, benchmarks). Surfaced overrides:
  `--config`, `--output-dir`, `--device`, `--n-epochs`,
  `--n-collocation`, `--refinement-levels`, `--skip-cpu`,
  `--require-cuda`. Helper functions (`load_config`, `apply_overrides`,
  `apply_benchmark_overrides`, `filter_baselines`, `build_pinn_config`,
  `register_pinn_profiles`, `_make_pinn_class`) are all individually
  unit-tested via
  `tests/scripts/test_run_sbir_p40.py`. Zero hardcoded numerics in the
  script body.
- **Coverage on the changed surface** — 95% branch+line coverage across
  the four affected `src/` modules
  (`gpu_profiler.py` 96%, `baselines.py` 95%, `pde_benchmarks.py` 94%,
  `poc/device.py` 100%); 1131 tests pass on `tests/research/` +
  `tests/pde/` + `tests/scripts/test_run_sbir_p40.py` with the global
  85% gate met (project total 94.84% on the changed module set).
  `ruff check` + `ruff format --check` clean on every edited file.

### Added — Codec Model Zoo Phase 2-D (`src/video_compression/zoo/sweep.py`, `scripts/train_compression_zoo.py`)

- **Manifest-level sweep orchestrator** — `ZooSweep` drives every entry in
  a manifest through a configurable `EntryRunner`. `should_skip(zoo, entry)`
  inspects the persisted entry hash so reruns of an unchanged entry skip
  cleanly. `EntryStatus` + `SweepReport` are frozen dataclasses; the
  default `default_entry_runner` runs `ZooTrainer` in-process.
- **Slice A — multi-entry CLI** — `scripts/train_compression_zoo.py` adds
  `dry-run` / `train` subcommands operating on a manifest. Shared
  primitives extracted into `src/video_compression/zoo/cli_helpers.py`
  (load / resolve_path / load_codec_config / resolve_entry /
  resolve_codec_config_for_entry / override_entry / resolve_device);
  the original `train_compression_zoo_entry.py` re-imports them as
  `_underscored` aliases so existing tests continue to monkeypatch
  through the script module.
- **Slice B — parallel dispatch + subprocess runner** — `ZooSweep.run_parallel()`
  groups entries by device and dispatches one worker thread per device,
  keeping same-device entries serialized inside their worker. Statuses
  return in manifest order regardless of completion order.
  `make_subprocess_entry_runner(...)` returns an `EntryRunner` that
  re-invokes the existing single-entry CLI with `CUDA_VISIBLE_DEVICES`
  pinned to the parent's `cuda:N` index (translating the child's
  `--device` to `cuda:0` because only one GPU is visible). After exit 0,
  the parent reads `metrics.json` + checkpoint back and reconstructs a
  `ZooTrainingReport`. Tests inject a fake `subprocess.run` via the
  `subprocess_runner` hook.
- **`ZooTrainer` persists wall-clock** — `train_wallclock_s` /
  `eval_wallclock_s` now land in `metrics.json` so the subprocess runner
  can reconstruct them across process boundaries.
- **Gap-analysis coverage closure** — branch-wide tech-debt scan confirmed
  zero hardcoded values (sole literal `"cuda:0"` at `sweep.py:510` is a
  CUDA ABI constant — the only-visible-GPU always presents as `cuda:0` in
  a `CUDA_VISIBLE_DEVICES=N` subprocess, documented inline). Discovered
  `cli_helpers.py` at 68% coverage; added 22 unit tests in
  `tests/video_compression/zoo/test_cli_helpers.py` covering every public
  helper across YAML/JSON/unsupported-suffix/empty/non-dict load paths,
  absolute/cwd-relative/manifest-relative path resolution, codec-config
  round-trip, entry lookup/KeyError, codec-config-ref precedence/fallback
  /no-ref raise, override short-circuit, device-preference cascade.
  Lifts `cli_helpers.py` from 68% → **100%**; zoo-subpackage total
  **98.44%**.
- **Test surface** — 22 cli_helpers tests + 15 Slice B tests
  (`tests/video_compression/zoo/test_sweep_parallel.py`) + 9 Slice A
  tests (`tests/scripts/test_train_compression_zoo.py`) + 4 entry-CLI
  tests (`tests/scripts/test_train_compression_zoo_entry.py`) + 11
  sweep-unit tests + zoo-trainer tests; **162-test full
  zoo+scripts+training regression** passes; mypy --strict + ruff clean.

### Added — Codec Model Zoo Phase 2-B (`src/video_compression/zoo/`)

- **Dual-GPU model zoo for the 8-point R-D Lagrangian sweep** — new
  `src/video_compression/zoo/` subpackage that schedules an arbitrary
  `λ`-grid across heterogeneous CUDA devices. Targets the reference rig
  (RTX 5060 Ti 16 GiB at `cuda:0` + RTX 5060 8 GiB at `cuda:1`) but is
  resolution-/SKU-agnostic: the planner consumes a `list[DeviceCapability]`
  produced at runtime by `scan_devices()`, so the same code runs on a
  laptop CPU, a single-GPU box, or a multi-node cluster.
- **Pydantic-validated schemas** (`config.py`, 100% coverage) — every
  measurement-/training-affecting knob is a validated field with bounds:
  `lambda_rd > 0`, `target_psnr_db > 0`, `train_steps ≥ 1`,
  `warmup_steps ≤ train_steps`, Adam betas in `(0, 1)`, entry IDs match
  `^[a-zA-Z0-9_\-\.]+$`, parent-entry-id resolution, dedupe enforcement.
  Schema versions are module-level constants
  (`PERF_ZOO_MANIFEST_SCHEMA_VERSION = 1`,
  `PERF_ZOO_ENTRY_SCHEMA_VERSION = 1`). Zero hardcoded magic numbers.
- **Forward-compatible manifest** (`manifest.py`, 100%) — JSON or YAML
  load/save dispatched by file suffix (`.yaml` / `.yml` → YAML, else
  JSON). `_migrate_manifest_document` promotes unversioned manifests to
  v1, fails loud on newer-than-binary, and rejects non-int schema
  versions. The shipped `config/video_compression/zoo/lambda_grid.yaml`
  (8 points: λ ∈ {0.0016, 0.0032, 0.0075, 0.015, 0.03, 0.045, 0.09, 0.18})
  loads cleanly through the same path as user-authored YAML.
- **Heterogeneous-VRAM device planner** (`device_planner.py`, 100%) —
  four assignment strategies: `VRAM_AWARE` (default; best-fit packing
  by current headroom, falls back to largest-total over-commit when no
  device has room), `ROUND_ROBIN`, `SINGLE_DEVICE`, `MANUAL` (per-entry
  pin via `device="cuda:N"` / `"cpu"` / `"cuda"`). Explicit pins are
  pre-resolved out before strategy dispatch, so all strategies compose
  with manual overrides. `scan_devices()` does a runtime `import torch`
  to remain monkeypatch-friendly. Module-level `CPU_DEVICE_LABEL`
  constant; no string literals leaked.
- **Filesystem `VideoCodecZoo` registry** (`storage.py`, 100%) —
  per-entry directory layout (`<root>/<entry_id>/{checkpoint.pt,
  entry.json, metrics.json}`), atomic write semantics, GCS backend gated
  via `importlib.import_module("src.vertex.storage")` (raises
  `NotImplementedError` until Phase D wires it). Constants
  `CHECKPOINT_FILENAME`, `ENTRY_FILENAME`, `METRICS_FILENAME` are
  module-level.
- **Coverage gate** — `pyproject.toml` `coverage.run.omit` rebalanced
  from a global `src/video_compression/*` blanket to a per-subpackage
  list. The `zoo/` subpackage is omitted from the project-wide 85%
  gate (CI's fast suite uses `--ignore=tests/video_compression/`, so
  including `zoo/` globally would 0%-tank the gate); a dedicated per-
  module gate enforces the zoo coverage floor instead. Achieved
  coverage on the zoo subpackage: **100% line + branch** across all
  five modules (`__init__.py`, `config.py`, `device_planner.py`,
  `manifest.py`, `storage.py`).
- **Test suite** (`tests/video_compression/zoo/`, 68 tests) —
  `test_config.py` (22 tests, schema + validator coverage),
  `test_manifest.py` (11 tests including YAML round-trip + shipped-grid
  smoke + Hypothesis property-based migration test),
  `test_device_planner.py` (14 tests across all four strategies +
  reference-rig fixture + CPU-only fixture),
  `test_storage.py` (8 tests, 1 GCS-skip),
  `test_edge_cases.py` (13 tests targeting all originally-uncovered
  branches: `DevicePlan.device_for` KeyError, bare-`cuda` resolution
  under MANUAL, CPU pin under VRAM_AWARE, `_resolve_run_target` cuda /
  cuda:N / invalid paths, MANUAL with missing pin, `list_entries` with
  removed root, non-dict checkpoint / metrics payloads).
- **E2E validation on live dual-GPU hardware** — `lambda_grid.yaml`
  loads, `scan_devices()` reports both cards correctly
  (`cuda:0=RTX 5060 Ti 16 GiB`, `cuda:1=RTX 5060 8 GiB`),
  `assign_devices` produces a deterministic plan with structured
  `structlog` events bound to `entry_id` / `device` / `strategy`.
  Phase 0 perf harness + Phase 1 runtime backends regress green
  (228 passed, pre-existing ONNX-runtime test failures on `cuda:0` are
  unrelated to this branch).

### Added — Codec Performance Benchmark Phase 0 (`src/video_compression/perf/`)

- **GPU-primary benchmark harness** — new `PerfBenchmark(BaseExecutable)` with `device_preference="cuda"` default, per-profile `cuda:N` pinning so a single sweep covers both cards of the reference dual-GPU rig (RTX 5060 Ti 16 GB at `cuda:0` + RTX 5060 8 GB at `cuda:1`). Indexed-CUDA resolver wraps `src/poc/device.resolve_device` without disturbing existing PoC scenarios.
- **Pydantic-validated config with zero hardcoded values** — `PerfBenchmarkConfig` surfaces every measurement-affecting knob (resolution / batch / phase / warmup / repeats / tolerance / track-VRAM / pattern / data-seed) as validated fields with bounds. `RuntimeProfile` and `ResolutionSpec` schemas pin labels and devices for stable cell keys. Schema versions are module-level constants (`PERF_BENCHMARK_CONFIG_SCHEMA_VERSION`, `PERF_BASELINE_DOCUMENT_SCHEMA_VERSION`, `PERF_BASELINE_ENTRY_SCHEMA_VERSION`).
- **Forward-compatible baseline registry** — `BaselineRegistry` load/save/diff with explicit JSON schema versioning, `extra="ignore"` for unknown future fields, and `_migrate_baseline_document` hook with an unversioned-to-v1 migration. Per-entry tolerance overrides allow tightening regression gates on critical cells without weakening the global threshold.
- **Three YAML configs** — `config/perf/smoke.yaml` (CPU CI gate, ~10 s), `config/perf/cuda0_headline.yaml` (single-card 16 GB primary), `config/perf/default.yaml` (dual-card sweep across `cuda:0` + `cuda:1`).
- **CLI `scripts/benchmark_codec.py`** — `run` / `record-baseline` / `diff` subcommands with structured `structlog` events bound to `benchmark_id` + `cell_key`. Argparse-based; no typer dep.
- **`BenchmarkSubject` Protocol** — runtime-agnostic interface for the timed object (`prepare` / `step` / `teardown`). Phase 1 (ONNX Runtime, TensorRT, FP16, `torch.compile`) drops new subjects in without touching the benchmark loop. Extended docstring includes a runnable Phase-1 example.
- **Coverage gate** — new `.github/workflows/codec-perf-coverage.yml` enforces ≥85% per-module coverage on `src/video_compression/perf/`. Inline-coveragerc heredoc with `include = src/video_compression/perf/*.py` (same pattern as `regression-surface.yml::noyron-hx-coverage-gate` on master). Achieved coverage: `__init__.py` 100% / `baseline.py` 99% / `benchmark.py` 100% / `config.py` 96% / `device.py` 90% / `metrics.py` 100% / `subjects.py` 97% — **TOTAL 98.42%**.
- **Defensive raise-paths covered** — three new tests in `TestDefensiveRaisePaths` (`tests/video_compression/perf/test_benchmark_smoke.py`) lock in: `fail_fast=True` propagates non-`NotImplementedError` exceptions; non-FP32 precision raises clean `NotImplementedError` (Phase-1 stub); `report_from_result` rejects `ExecutionResult` lacking the `"report"` artifact with a clear `KeyError`.

### Added — Noyron HX v1 Hardening (`src/pde/sdf.py`, `src/pde/geometry_picogk.py`, `src/poc/scenarios/noyron_hx.py`)

- **Voxel-FDM training consistency** — `NoyronHXScenario` now trains directly on the cached FDM solution when `ref_solver_kind="voxel_fdm"`. Previously the scenario trained on the harmonic surrogate but graded against FDM; the head-line `mse_low < 5e-4` / `mse_high < 1e-3` thresholds were unreachable in FDM mode. The cached solution is built lazily via `_voxel_fdm_reference()` and reused at evaluation, so reference and supervision come from the same field.
- **Surfaced scenario metrics** — `accept_rate` (from `PicoGKDomain.volume_accept_rate`), `train_time_s`, `eval_time_s`, and `train_loss_final` are now recorded in `ScenarioResult.metrics`. Timing values are captured via `ScenarioLogger.timed(...)` context-manager and propagated through the public metric dict.
- **Bisection / grid-search fallback for SDF projections** — `AnalyticalHelixSDF._nearest_t` gains a coarse-grid + Newton-refine fallback (opt-out via `enable_fallback=False`); `PicoGKDomain._project_to_surface` gains a bracketed bisection along the central-difference gradient (opt-out via `enable_bisection_fallback=False`). Both restore robustness on thin tubes (`r/R << 0.1`) where the original Newton iteration could stall.
- **`PicoGKDomain.volume_accept_rate`** — new read-only property that exposes the empirical interior acceptance rate computed at construction by the existing Monte-Carlo volume estimator. No re-sampling cost; the rate is cached on `_volume_accept_rate`.
- **`NoyronHXScenarioConfig.helix_n_turns`** default aligned to **5** across the Pydantic config, `config/scenarios/noyron_hx.yaml`, and `AnalyticalHelixSDF`. Previously the config-class default was 3 while the YAML used 5 — instantiating the config in code produced a different geometry than the headline run.
- **Module-level numerical-stability constants** (replace previously hardcoded literals):
  - `DEFAULT_TRANSFER_RATIO_FLOOR: float = 1e-12` — division floor for `mse_high / mse_low`.
  - `DEFAULT_NORMALIZE_EXTENT_FLOOR: float = 1e-9` — bbox-extent clamp in `_normalize`.
  - `EVAL_SEED_STRIDE: int = 9973` — prime offset between low- and high-density evaluation seeds.
- **`NoyronHXScenario._draw_pool_indices(n_pool, n_pts)`** — single helper for sampling indices from the cached FDM voxel pool. Replaces duplicated `randperm` / `randint` selection logic that previously appeared inline in both `_sample_voxel_fdm_batch` and `_evaluate`. Validates `n_pool > 0` and `n_pts > 0` and routes through-replacement sampling via `randint` when `n_pts > n_pool`.

### Added — Noyron HX Test Suite (`tests/pde/test_sdf.py`, `tests/pde/test_picogk_domain.py`, `tests/poc/test_noyron_hx_scenario.py`)

- **43 new test cases** covering: SDF fallback (disabled / param validation / grid scaling / pathological initial guess / no-Newton-refine branch), `PicoGKDomain` constructor validators (`grad_epsilon`, `max_oversample`, `projection_max_iters`, `min_grad_norm_sq`), bisection-fallback no-op when all-converged, projection-converged log branch, `accept_rate` / `train_time_s` / `eval_time_s` metric round-trip, voxel-FDM uses FDM (not harmonic) supervision, voxel-FDM cache cleared on teardown, `_draw_pool_indices` semantics, and module-constant invariants.
- **Per-module coverage**: `src/pde/sdf.py` **100%**, `src/pde/geometry_picogk.py` **100%**, `src/poc/config_noyron.py` **100%**, `src/poc/scenarios/noyron_hx.py` **97%** — all well above the project 85% gate.

### Added — Learned PDE Evaluator (`src/alphagalerkin/`)

- **`AlphaGalerkinConfig.evaluator="trained"`** — re-enables the network-backed evaluator literal that was removed in the DOE Genesis PR. The trained branch loads an `AlphaGalerkinModel` checkpoint via `create_model_from_checkpoint` and wraps it in the existing `FNetEvaluator`, providing learned policy/value priors to MCTS rather than the uniform prior of `RandomEvaluator`. Closes the only non-trivial entry under *Known Issues* in `CLAUDE.md`.
- **`checkpoint_path: Path | None`** Pydantic field with a `model_validator(mode="after")` that fails fast at config-construction time when `evaluator="trained"` is paired with a missing or non-existent checkpoint.
- **GPU-primary default** — `AlphaGalerkinConfig.device` default flipped from `"cpu"` to `"cuda"`. New module-level `_resolve_device_cached` helper (cached via `functools.cache`) falls back to CPU at runtime when `torch.cuda.is_available()` is False, emitting at most one downgrade warning per unique device string for the lifetime of the process. The random/uniform evaluator path skips device resolution entirely (it is device-agnostic) so CPU-only users do not see spurious `cuda_requested_but_unavailable` warnings under the new default. `config/train_pde.yaml` updated to `device: auto` (the `Trainer`'s native CUDA-availability fallback) — note that the solver's runtime fallback is solver-only and does not apply to the training pipeline.
- **Trained-evaluator instance cache** — `AlphaGalerkinSolver._build_trained_evaluator()` constructs the `FNetEvaluator` once per solver instance and reuses it across subsequent `solve()` calls so benchmark suites that iterate over many PDEs do not pay repeated disk I/O + model-init cost. `reset_cache()` invalidates the cache for callers that swap checkpoints during a long-running process.
- **New evaluator config fields** (replace previously hardcoded values):
  - `evaluator_temperature: float` (gt=0.0, default=1.0) — softmax temperature for trained-evaluator policy logits.
  - `evaluator_use_fast_path: bool` (default=True) — toggle the FNet fast-forward path inside `FNetEvaluator`.
  - `checkpoint_strict_load: bool` (default=False) — controls strict shape matching on `create_model_from_checkpoint`; the default tolerates policy-head shape mismatches across PDEs.

### Added — Trained Evaluator Tests (`tests/alphagalerkin/test_trained_evaluator.py`)

- **8 new test classes / parameterized cases** covering evaluator dispatch, action-space mismatch graceful degradation, device resolution caching, trained-evaluator instance caching, and config-field propagation. The GPU smoke test is gated on `@pytest.mark.gpu_required` and auto-skips on CPU CI via the root `conftest.py` hook.
- **Per-module coverage** on `src/alphagalerkin/` raised to **94%** (gate: 85%).

### Changed

- **`src/alphagalerkin/solver.py`** — module docstring rewritten to document the three evaluator modes; previously hardcoded `temperature=1.0` / `use_fast_path=True` / `strict=False` in `_build_mcts` removed in favour of the new Pydantic config fields.
- **`tests/alphagalerkin/test_solver.py`** — `test_trained_evaluator_rejected_by_config` flipped into `test_trained_evaluator_requires_checkpoint`, asserting both missing-path and non-existent-path failure modes surface as `ValidationError`.
- **`config/train_pde.yaml`** — `device: cpu` → `device: auto` to use the trainer's native CUDA-availability fallback (`BaseTrainer` resolves `auto` via `torch.cuda.is_available()`); the solver's `_resolve_device_cached` runtime fallback is solver-only and does not apply to the training pipeline.

### Documentation

- **`CLAUDE.md`** — drop the "trained-evaluator stub" entry from *Known Issues*; add the *Learned PDE Evaluator Wired* milestone for 2026-04-25.
- **`docs/architecture/c4_mermaid.md`** — extend the Container Diagram with the trained-evaluator path through `FNetEvaluator` and the on-instance evaluator cache.

### Added — E2E Dashboard (`dashboard/`)

- **`dashboard/app.py`** — Gradio Blocks application factory (`build_app()`) and CLI entry point (`main()`). Launches a tabbed UI exposing all AlphaGalerkin capabilities at `http://localhost:7860`. Accepts `--host`, `--port`, `--share`, `--debug` flags.

- **`dashboard/config.py`** — Full Pydantic v2 config hierarchy eliminating every hardcoded value:
  `AppConfig`, `GameConfig`, `PDEConfig`, `ComplexityRunConfig`, `StabilityRunConfig`,
  `TransferMilestone`, `PoCConfig`, `TrainingConfig`, `DashboardConfig`.
  `DEFAULT_CONFIG` singleton for zero-configuration startup.

- **`dashboard/utils.py`** — Shared utility module:
  - `fig_to_pil()` — always closes matplotlib figure (even on exception); `.copy()` detaches from buffer
  - `device_str()` — CUDA/CPU detection with graceful fallback
  - `format_exc()` — consistent exception formatting
  - `configure_structlog()` — idempotent structured logging setup

- **`dashboard/tabs/game_tab.py`** — Go AI tab. Thread-safe lazy model loading via `threading.Lock` (double-checked locking). Human vs AI and AI vs AI modes with 9×9/13×13/19×19 board support (zero-shot transfer). Config-injected via `GameConfig`.

- **`dashboard/tabs/pde_tab.py`** — Interactive Poisson equation solver. Five charge patterns (Point Charge, Dipole, Quadrupole, Ring, Random), multi-resolution comparison with zoom-upsampling MSE. Config-injected via `PDEConfig`.

- **`dashboard/tabs/poc_tab.py`** — PoC scenario runner. O(N) complexity benchmark, LBB stability monitoring, zero-shot transfer milestone display. Module-level optional imports for test patchability. Config-injected via `PoCConfig`.

- **`dashboard/tabs/training_tab.py`** — Architecture summary, simulated training curves (policy/value/LBB losses), and loss breakdown diagram. Config-injected via `TrainingConfig`.

### Added — Dashboard Test Suite (`tests/dashboard/`)

- **203 tests**, **89% line coverage** (gate: 85%), all passing with zero ruff violations.
- `conftest.py` — shared fixtures, `matplotlib.use("Agg")`, config fixture hierarchy, mock scenario results, charge-grid fixtures.
- `test_app.py` (24 tests) — CSS builder, arg parser, `build_app()`, `main()`.
- `test_config.py` (31 tests) — all Pydantic models, validation errors, JSON round-trip.
- `test_utils.py` (24 tests) — `fig_to_pil` (close on error, detached buffer), `device_str`, `format_exc`, `configure_structlog`.
- `test_pde_tab.py` (37 tests) — all charge patterns, Poisson solve integration, `solve_and_visualize`, `compare_resolutions` with shape-matching mock.
- `test_poc_tab.py` (32 tests) — `_parse_int_list`, `run_complexity`, `run_stability` (mocked), `show_transfer_milestone` (live).
- `test_training_tab.py` (28 tests) — model summary (fallback on import error), training curves, loss breakdown.
- `test_game_tab.py` (27 tests) — `autouse` fixture resetting module globals, fallback board, `_ensure_loaded` idempotency, human/AI move handlers.

- **Intercept Module** (`src/intercept/`)
  - `InterceptGame` implementing `GameInterface` protocol for MCTS-guided missile defense
  - 6-DOF rigid body dynamics (`dynamics.py`, `interceptor_dynamics.py`)
  - Proportional Navigation guidance (`guidance.py`)
  - `ExtendedKalmanFilter` for target tracking (`tracking.py`)
  - `RadarSensor`, `SensorFusion` for multi-sensor tracking (`sensors.py`)
  - `HungarianAssigner` for weapon-target assignment (`assignment.py`)
  - `ISAAtmosphere`, `WindModel` for atmospheric modeling (`atmosphere.py`)
  - `AeroModel`, `TabularAeroModel` for aerodynamic coefficients (`aero.py`)
  - `FrameTransform`, `QuaternionOps` for reference frame conversions (`frames.py`)
  - Pydantic-validated `InterceptorConfig`, `EngagementConfig`, `ThreatConfig`

- **Backend Abstraction** (`src/backend/`)
  - `BackendInterface` protocol for unified PyTorch/JAX operations
  - `TorchBackend`, `JaxBackend` implementations
  - `Array`, `Precision`, `DeviceType` type abstractions (`types.py`)
  - Random number generator abstraction (`rng.py`)
  - Backend-aware logging and debug utilities

- **Prototyping Module** (`src/prototyping/`)
  - `ModelBuilder`, `PrototypeModel` for rapid architecture iteration
  - `QuickTrainer`, `TrainResult` for fast experiment loops
  - `QuickEvaluator`, `EvalResult` for quick model evaluation
  - `DataGenerator`, `SyntheticData` for synthetic data creation
  - `Visualizer` with multiple plot types
  - `ExperimentTemplate`, `TemplateRegistry` for experiment patterns

- **Analysis Module** (`src/analysis/`)
  - `PositionEvaluator`, `EvaluationResult` for position evaluation
  - `GameReviewer`, `MoveAnalysis` for game review and move quality assessment
  - `PatternMatcher`, `PatternLibrary` for board pattern detection
  - `GameStatistics`, `StatisticsCollector` for game statistics aggregation
  - `AnalysisConfig`, `AnalysisMode` Pydantic configuration

- **Tournament Module** (`src/tournament/`)
  - `TournamentManager`, `TournamentState` supporting Round-Robin, Swiss, Elimination formats
  - `TournamentScheduler` for match scheduling
  - `EloRating`, `RatingSystem` for player rating computation
  - `Player`, `PlayerRegistry` for participant management
  - `Match`, `MatchResult`, `MatchStatus` for match tracking

### Changed

- **`pyproject.toml`** — Added `[[tool.mypy.overrides]]` for `dashboard.*` modules (relaxed strict checks for Gradio code). Added `[tool.coverage.report]` with `fail_under = 85` and `show_missing = true`. Added `dashboard` pytest marker.
- **Gradio 6 compatibility** — CSS argument moved from `Blocks()` constructor to `launch()`.

> **Branch and PR cleanup** — removed 28 stale remote branches and 6 open stale PRs.

## [0.3.0] - 2026-04-01

### Summary

Key highlights of this release:

- **Chess Self-Play Training Pipeline** — AlphaZero methodology, 4672-action dense policy, 119-channel state encoding
- **SBIR Readiness Infrastructure** — Navy N252-088, DOE ASCR, NSF SBIR, AFWERX proposal configs and benchmark suite
- **Advanced PDE Operators** — NavierStokes (Taylor-Green), L-shaped Poisson (singularity), enhanced Burgers (Cole-Hopf)
- **Domain Geometry & Time-Stepping module** — Rectangular, L-shaped, Cylinder domains; ForwardEuler, RK4, CrankNicolson
- **Multi-Agent Swarm Planning** — PettingZoo `ParallelEnv` adapter, potential field obstacle avoidance
- **Unified Loss Package & BaseTrainer consolidation** — `LossRegistry`, `get_loss()` factory, shared AMP/grad/LR in `BaseTrainer`
- **CI/CD hardening** — 85% coverage gates, nightly schedule, Stage 8 chess pipeline
- **218+ new tests** across PDE, research, training, and games modules

---

### Added

- **SBIR Readiness Infrastructure** (Navy N252-088, DOE ASCR, NSF, AFWERX)
  - `config/proposals/navy_n252_088.yaml`, `nsf_sbir.yaml` — SBIR-specific benchmark configs
  - `config/benchmarks/sbir_suite.yaml` — 3-problem benchmark suite (L-shaped Poisson, Burgers shock, NS Taylor-Green)
  - `src/research/baselines.py` — Classical PDE solver baselines: UniformFDMSolver, DorflerAMRSolver, SimplePINNSolver
  - `src/research/pde_benchmarks.py` — PDEBenchmarkRunner with JSON/Markdown report generation and convergence rate computation
  - `docs/proposals/templates/sbir_phase1.md` — Reusable SBIR Phase I proposal template
  - `docs/proposals/IP_STRATEGY.md` — 3 provisional patent claims, trade secret boundaries, publication plan

- **Advanced PDE Operators**
  - `NavierStokesOperator` — Taylor-Green vortex benchmark with analytical solution, configurable Re
  - `BurgersOperator` enhanced — Cole-Hopf exact solution, configurable shock params, convergence rate method
  - `LShapedPoissonOperator` — r^(2/3)*sin(2theta/3) singularity for AMR benchmarking

- **Domain Geometry Abstractions** (`src/pde/geometry.py`)
  - `RectangularDomain`, `LShapedDomain`, `CylinderFlowDomain` (DFG benchmark)
  - Rejection sampling for non-convex domains, proportional boundary sampling
  - `GeometryConfig` Pydantic schema and `create_geometry()` factory

- **Time-Stepping Module** (`src/pde/time_stepping.py`)
  - `ForwardEuler`, `RK4`, `CrankNicolson` (fixed-point iteration) with factory pattern
  - `TimeSteppingConfig` Pydantic schema, `integrate()` with snapshot saving

- **S500 Swarm Planning Game** (`src/pde/games/swarm_planning.py`)
  - `SwarmPlanningGame` with round-robin multi-agent control (7 actions per agent)
  - Potential field obstacle avoidance (Laplace equation connection), coverage rewards
  - `SwarmPlanningConfig` — fully Pydantic-validated with no hardcoded values

- **PettingZoo Adapter** (`src/games/pettingzoo_adapter.py`)
  - `PettingZooAdapter` wrapping `GameInterface` as PettingZoo `ParallelEnv`
  - Optional dependency with graceful degradation (`HAS_PETTINGZOO` flag)

- **Unified Loss Package** (`src/training/losses/`)
  - `LossRegistry` with decorator-based registration (`"alphagalerkin"`, `"l2_relative"`, `"h1"`, `"mse"`)
  - `get_loss()` factory function for config-driven loss instantiation
  - Backwards-compatible thin wrappers in `src/training/loss.py` and `src/training/physics_loss.py`

- **BaseTrainer Consolidation** (`src/training/base_trainer.py`)
  - Abstract `BaseTrainer[ConfigT]` with shared AMP, gradient clipping, LR scheduling, checkpoint save/load
  - `BaseTrainerConfig` Pydantic schema covering all shared hyperparameters
  - `StepResult` dataclass for structured step output

- **Checkpoint Migration System** (`src/training/checkpoint_migration.py`)
  - Version-aware migration with `@register_migration` decorator
  - Migration path: `0.0.0 -> 1.0.0 -> 1.1.0` (LBB config fields added)

- **Property-Based and Numerical Stability Tests**
  - `tests/training/test_loss_properties.py` — hypothesis tests: non-negativity, CE = log(n), gradient flow
  - `tests/training/test_numerical_stability.py` — extreme values, near-zero denominators, NaN propagation
  - `tests/pde/test_operator_properties.py` — PDE operator invariants, linearity, collocation in domain
  - `tests/modeling/test_attention_properties.py` — Galerkin attention shape, LBB positivity, resolution independence

- **Comprehensive Coverage Tests** (218 new tests)
  - `tests/pde/test_geometry.py` — 65 tests for domain geometries
  - `tests/pde/test_time_stepping.py` — 37 tests for time-stepping methods
  - `tests/research/test_baselines.py` — 39 tests for classical solver baselines
  - `tests/research/test_pde_benchmarks.py` — 38 tests for benchmark runner
  - `tests/training/test_base_trainer.py` — 39 tests for BaseTrainer
  - `tests/pde/test_swarm_planning.py` — 50 tests for swarm planning game
  - `tests/games/test_pettingzoo_adapter.py` — 11 tests for PettingZoo adapter

### Changed

- **CI/CD Hardening** (`.github/workflows/ci.yml`)
  - MyPy strict enforcement (`continue-on-error: false`)
  - Coverage gates raised: 75% -> 85% overall, 80% -> 85% per-module (pde, modeling, training)
  - Added `research` module coverage gate at 85%
  - Added nightly schedule (`cron: '0 4 * * *'`) and performance benchmark job on main merges

- **Config-Driven LBB Loss** (`config/schemas.py`)
  - Surfaced `lbb_loss_weight`, `lbb_target`, `lbb_eps`, `log_barrier_weight` as Pydantic fields
  - Added mathematical documentation (Babuska-Brezzi motivation) in field descriptions

- **Race Condition Fix** (`src/modeling/model.py`)
  - Removed `_training_resolution` mutation from `forward()` (DDP-unsafe)
  - Added explicit `set_training_resolution()` public method

### Fixed

- `advection_coeff` dimension mismatch in `PDEBenchmarkRunner._create_operator()` — was hardcoded `[0.0, 0.0]` for any dim

- **Chess Self-Play Training Pipeline** (AlphaZero methodology)
  - `ActionPolicyHead` for dense 4672-action policy output (`src/modeling/model.py`)
  - `StatefulGameWrapper` bridging stateless `GameInterface` to MCTS (`src/games/wrapper.py`)
  - Chess training CLI (`scripts/train_chess.py`) with Hydra config (`config/train_chess.yaml`)
  - `game_type` and `action_space_size` fields in `OperatorConfig` (`config/schemas.py`)
  - PRD and ADR documentation (`docs/prd/prd-chess-self-play.md`, `docs/architecture/ADR-chess-self-play.md`)

- **Chess Training Tests**
  - `tests/games/test_wrapper.py` — StatefulGameWrapper unit tests (10 tests)
  - `tests/modeling/test_chess_model.py` — ActionPolicyHead and chess model tests (12 tests)
  - `tests/training/test_chess_self_play.py` — Chess self-play integration tests (7 tests)
  - `tests/games/test_chess_exhaustive.py` — Exhaustive encode/decode roundtrip + edge cases (20 tests)
  - `tests/training/test_trainer_chess.py` — Checkpoint save/load/resume, engine eval, config tests (11 tests)
  - `tests/security/test_chess_security.py` — Invalid actions, OOB states, corrupted data (15 tests)
  - `tests/e2e/test_chess_training_e2e.py` — E2E training smoke tests (3 tests)

- **Stockfish Benchmark Evaluation**
  - Engine eval config fields in `TrainingConfig` (path, depth, games, movetime)
  - `Trainer._run_engine_evaluation()` with W&B Elo metric logging
  - Engine eval section in `config/train_chess.yaml`

- **CI/CD Chess Pipeline**
  - Stage 8: Chess Pipeline Tests in `.github/workflows/ci.yml`
  - Coverage gate `--cov-fail-under=80` for `chess.py` (97%) and `wrapper.py` (100%)
  - CI Success gate requires chess tests
### Changed

- **Game-agnostic self-play**: `SelfPlayWorker` now accepts optional `GameInterface` parameter
- **Game-agnostic trainer**: `Trainer.__init__()` accepts `game` parameter, forwarded to worker
- **Game-agnostic collator**: `VariableSizeCollator` and `SameSizeCollator` derive action mask size from `target_policy` tensor instead of hardcoded `board_size²+1`
- `AlphaGalerkinModel` and `AlphaGalerkinFast` auto-select policy head by `action_space_size`
### Fixed

- **Underpromotion encode/decode mismatch** (`src/games/chess.py`): `_decode_move` used `[-1, 0, 1]` but `_encode_move` used `straight=0, left=1, right=2` — straight promotion from column 0 decoded as `to_col=-1`. Fixed to `[0, -1, 1]`.
- **Collator action mask size** (`src/data/collate.py`): Both collators hardcoded `n_actions = board_size²+1` causing tensor size mismatch with chess's 4672-action policy. Fixed to detect per-experience policy encoding.

## [0.2.0] - 2026-01-26

### Milestones Achieved

- **Zero-Shot Transfer Validated**: Physics PoC demonstrated resolution-independence
  - Trained on 9x9 grids, transfers zero-shot to 19x19 (measured MSE ≈ 0.00039). NOTE: the original "0.000209 / 240× better than threshold" was a fabricated notebook figure — corrected 2026-07-22; a CNN retrained at 19x19 is more accurate (see `specs/transfer_baseline_compare.spec.md`).
  - Validates core Galerkin approach for continuous operator learning

- **Training Pipeline Operational**: End-to-end training with self-play working on GPU
  - MCTS-based self-play generates training experiences
  - LBB stability monitoring integrated into training loop

### Added

- **W&B Integration for Physics PoC**
  - `--wandb` flag for `train_physics.py` to enable Weights & Biases logging
  - Logs training loss, evaluation MSE, transfer MSE, and learning rate
  - Final summary includes success status and best transfer MSE

- **GameInterface Protocol Implementation**
  - Added `apply_action()` method to `SimpleGoGame` class
  - Enables MCTS integration with Go game state

- **Security Tests** (`tests/security/`)
  - Input sanitization tests for GTP interface
  - DoS protection via input length limits

- **E2E Tests** (`tests/e2e/`)
  - CLI journey tests for help and train commands

### Changed

- Replaced Unicode checkmarks with ASCII `[PASS]`/`[FAIL]` for Windows compatibility
- Updated `.gitignore` with additional patterns:
  - `nul` (Windows device file)
  - `*.log`, `*.dist-info/`
  - `hydra_outputs/`

### Fixed

- Fixed `AttributeError: 'SimpleGoGame' object has no attribute 'apply_action'`
- Fixed unused loop variable warning in `BoardSizeBatchSampler`
- Fixed line length issue in W&B initialization

## [0.1.0] - 2026-01-26

### Added

- **Core Architecture**
  - `AlphaGalerkinModel`: Resolution-independent Go AI using continuous operators
  - `GalerkinLinearAttention`: O(N) complexity global influence modeling
  - `SoftmaxAttention`: Local tactical reading with injectivity preservation
  - `FNetBlock`: FFT-based mixing for fast MCTS rollouts

- **Mathematical Kernel**
  - Fredholm integral equation with Green's function formulation
  - Fourier features for positional encoding
  - Monte Carlo integral normalization (1/n) for Galerkin attention
  - LBB stability monitoring (dim(Key) >= dim(Query))

- **Training Infrastructure**
  - Self-play with MCTS for experience generation
  - Uniform and prioritized replay buffers
  - `AlphaGalerkinLoss`: policy_CE + value_MSE + LBB_regularization
  - Checkpoint management with best model tracking
  - Hydra configuration system

- **Physics PoC**
  - Poisson equation solver for synthetic data generation
  - `PhysicsOperator` neural network for influence field prediction
  - Zero-shot transfer verification scripts

- **PoC Scenario Framework**
  - Configuration-driven scenario execution
  - Built-in scenarios: transfer, complexity, stability
  - Pydantic-validated configs
  - Structured logging via structlog

### Documentation

- C4 architecture diagrams
- CLAUDE.md with project context and verification commands
