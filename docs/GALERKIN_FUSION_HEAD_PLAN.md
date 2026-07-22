# Plan — Galerkin Fusion Head: 2-Week Cross-Repo Execution

## Context

**Why this work:** Mouse-Droid-AGI currently fuses heterogeneous sensors (IMX500 camera features, HC-SR04 ultrasonic, ESP32 encoder ticks) through a hand-written adapter inside `sensing/`. The hypothesis is that a Galerkin-attention block — built from the resolution-independent operators already validated in AlphaGalerkin (zero-shot transfer, measured MSE ≈ 4e-4) — can replace that adapter and either (a) free RAM headroom on the Orin Nano's 8GB budget, (b) shorten frame latency tails, or (c) improve downstream RSSM/MCTS quality. Any one of those is a win; all three flat is a publishable negative result.

**Why a cross-repo plan:** Mouse-Droid-AGI will consume AlphaGalerkin via **git submodule** (decided up front). That means this repo (AlphaGalerkin) needs a stable, importable surface for `GalerkinAttention`, `FNetBlock`, `MultiScaleFourierFeatures`, and `StabilityGuard`. Mouse-Droid-AGI's `sensing/galerkin_fusion.py` **must import only from the top-level package** (`from src.modeling import GalerkinAttention, FNetBlock, ...`) — not from the underlying `src.modeling.*` submodules. The re-export surface declared in `docs/architecture/ADR-mouse-droid-fusion-integration.md` is the stability contract; deep-submodule imports bypass it.

**Outcome the plan must produce:** A merged feature-flagged PR in Mouse-Droid-AGI, four benchmark plots on Orin Nano, a 1000–3000 word technical note, and an ADR recorded in **both** repos by end of Day 10 — recording either "continue to Week 3 (ICM refinement + HJB)" or "redirect to SEAL SQE / AlphaGalerkin proper."

---

## Repos & Branches

| Repo | Branch | Role |
|------|--------|------|
| `ianshank/AlphaGalerkin` | `claude/plan-galerkin-fusion-WpojY` | Stabilize public API; ADR; benchmark mirror |
| `ianshank/Mouse-Droid-AGI` | `feat/galerkin-fusion-head` (to be created) | Submodule wiring; `sensing/galerkin_fusion.py`; benchmarks; technical note |

---

## Reused AlphaGalerkin Operators (no reimplementation)

All resolution-independent. Import paths verified by Phase 1 exploration.

| Operator | Path | Signature highlights |
|----------|------|----------------------|
| `GalerkinAttention` | `src/modeling/attention.py:26` | `(d_model, n_heads, d_key=None, d_value=None, dropout=0.0, normalize_features=True)`; forward `[B, n, d] → [B, n, d]`; optional `return_lbb=True`. **Requires float32 for SVD.** |
| `FNetBlock` | `src/modeling/fnet.py:132` | `(d_model, d_ffn=None, dropout=0.1, use_2d_fft=True)`; forward `(x, board_size=None) → [B, n, d]`. 2D mode requires `board_size² == seq_len`. |
| `MultiScaleFourierFeatures` | `src/modeling/multiscale_fourier.py:93` | Use to embed heterogeneous coordinate inputs (camera pixel grid, scalar ultrasonic, encoder tick coords) into a shared feature space before attention. Configured via `FourierFeaturesConfig`. |
| `StabilityGuard` | `src/modeling/stability.py:25` | `(beta_threshold=1e-6, regularization_strength=0.01, log_interval=100, margin_multiplier=10.0)`; methods `compute_lbb_constant`, `check_stability`, `regularization_loss`. Standalone-importable. |
| `BaseModuleConfig` | `src/templates/config.py:120` | Pydantic base for the new `SensingConfig.fusion_backend` schema. `extra="forbid"`. |
| `create_registry` | `src/templates/registry.py:240` | Decorator pattern for `make_fusion_backend()`. |

**Exposed via:** `src/modeling/__init__.py` already re-exports `GalerkinAttention`, `FNetBlock`, `StabilityGuard`. `src/__init__.py` is currently version-only — see AlphaGalerkin Day 1 task.

---

## AlphaGalerkin-Side Work (this repo)

Small surface — just enough to make the submodule import path stable for two weeks.

### Day 1 (parallel with Mouse-Droid-AGI Day 1)

- **Edit** `src/modeling/__init__.py`: add `MultiScaleFourierFeatures` to the explicit re-exports (it's currently importable but not in `__all__`). Verify `GalerkinAttention`, `FNetBlock`, `StabilityGuard` are all listed.
- **Add** `docs/architecture/ADR-mouse-droid-fusion-integration.md` documenting that Mouse-Droid-AGI consumes `src.modeling.*` as a submodule and that those four classes' constructor signatures are now considered a stable interface for the duration of this experiment.
- **Additive hardening permitted** on the four target modules: new constructor parameters with defaults that preserve prior behaviour, structured logging, input validation, and new test coverage. Any such change must stay backwards compatible per the rules in the integration ADR. **Non-additive changes** (renaming params, changing defaults, altering forward signatures) are *not* allowed without bumping the ADR.
- **No packaging changes** (git submodule does not need pyproject changes).

### Day 9–10 (parallel with Mouse-Droid-AGI publish/decision)

- **Mirror benchmark artifacts** into `docs/benchmarks/galerkin_fusion/` (PNGs + `results_summary.md`). Mouse-Droid-AGI is the source of truth; this is for SBIR-narrative continuity in AlphaGalerkin's own publication trail.
- **Write** `docs/architecture/ADR-post-fusion-direction.md` mirroring the Mouse-Droid-AGI ADR. This protects future-you from re-litigating the decision in the AlphaGalerkin context (especially if "redirect to AlphaGalerkin proper" wins).

### Critical AlphaGalerkin files

- `src/modeling/__init__.py` — edit to add `MultiScaleFourierFeatures` export
- `docs/architecture/ADR-mouse-droid-fusion-integration.md` — new
- `docs/architecture/ADR-post-fusion-direction.md` — new on Day 10
- `docs/benchmarks/galerkin_fusion/` — new directory with mirrored plots

**Out of scope for this repo:** No *non-additive* changes to attention/fnet/multiscale_fourier internals (no renaming params, changing defaults, or altering forward signatures). No new operators. No packaging. Constructor signatures are frozen for two weeks (any breaking change requires updating the integration ADR).

---

## Mouse-Droid-AGI Work (separate repo, summarized; see source plan for full daily detail)

### Week 1 — Implementation

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Add AlphaGalerkin as submodule at `external/AlphaGalerkin`. Read `sensing/`, run baseline 752 tests, capture baseline memory/latency/norm telemetry. Define `GalerkinFusionProtocol` (no-op impl + tests). | `sensing/fusion_protocol.py` + no-op passes baseline tests |
| 2 | Implement `GalerkinFusion` (~200 lines) using `MultiScaleFourierFeatures` for heterogeneous-input embedding → `GalerkinAttention` over shared Q-tokens → projection to `latent_shape`. Shape-only unit tests. | `sensing/galerkin_fusion.py` shape tests green |
| 3 | Wire `make_fusion_backend(cfg)` factory behind `config.sensing.fusion_backend ∈ {adapter, galerkin}`. Full existing test suite still green. | Feature flag switches at runtime |
| 4 | Correctness vs. baseline on recorded sensor data: assert outputs differ from adapter (not a no-op), but preserve invariants (bounded norms, no NaN/Inf, consistent dims). | `tests/sensing/test_galerkin_fusion_invariants.py` ≥10 assertions |
| 5 | On-Jetson shakedown: 5 min stationary + 5 min walking. `tegrastats` for memory/thermal. **If OOM:** scope down Q-tokens / projection dim per Day 2 fallback notes. | `benchmarks/galerkin_fusion/shakedown/` telemetry committed |

### Week 2 — Benchmark & Publish

| Day | Task | Deliverable |
|-----|------|-------------|
| 6 | Build `scripts/bench_fusion.py` — runs both backends on same recorded episode, ≥5 trials, deterministic given seed. CSV per-frame metrics. | Raw CSV committed |
| 7 | Four matplotlib plots: peak memory boxplot, frame-latency ECDF, latent-entropy line plot, rollouts/sec bar chart. Tables with mean/std/95% CI. Honest negatives reported. | `docs/benchmarks/galerkin_fusion/plots/` + `results_summary.md` |
| 8 | Technical note draft: Problem / Approach / Results / Limitations, ≤3000 words. Specific limitations (no vague hand-waving). | `docs/notes/galerkin-fusion-head.md` |
| 9 | Reproduce benchmark from clean clone (fix harness if it doesn't reproduce, never the numbers). Self-review note. Update `CLAUDE.md` / `AGENTS.md`. Open PR against main. | PR opened |
| 10 | Run decision gate. Record ADR in **both repos**. If continue → draft Week 3. If redirect → handoff note + Monday redirected to SEAL SQE / AlphaGalerkin proper. | `docs/adr/NNNN-post-fusion-direction.md` (Mouse-Droid) + mirror in AlphaGalerkin |

---

## Module Structure (Mouse-Droid-AGI)

```
external/
└── AlphaGalerkin/                  # NEW: git submodule
sensing/
├── __init__.py
├── factory.py                      # EDIT: add make_fusion_backend()
├── observation_bundle.py           # unchanged
├── adapter_backend.py              # rename of existing adapter
├── fusion_protocol.py              # NEW: Protocol + no-op
├── galerkin_fusion.py              # NEW: ~200 lines
└── tests/
    ├── test_factory.py             # existing
    ├── test_galerkin_fusion.py     # NEW: shape + behavior tests, ≥20 cases
    └── test_galerkin_fusion_invariants.py  # NEW: ≥10 invariant assertions
config/
└── sensing.yaml                    # EDIT: add fusion_backend field
scripts/
└── bench_fusion.py                 # NEW: deterministic A/B harness
docs/
├── adr/NNNN-post-fusion-direction.md   # NEW: Day 10
├── benchmarks/galerkin_fusion/         # NEW: plots + CSV
└── notes/galerkin-fusion-head.md       # NEW: technical note
```

**Conventions to mirror from AlphaGalerkin** (for Mouse-Droid-AGI consistency):
- Pydantic configs inherit `BaseModuleConfig` pattern (`extra="forbid"`, `Field(...)` constraints, `@model_validator(mode="after")` for cross-field checks)
- Registry via `create_registry("FusionBackend", GalerkinFusionProtocol)` — decorator-based registration
- structlog for logging: `logger.info("fusion_step", backend="galerkin", latency_ms=lat)`
- pytest + hypothesis; mark on-hardware tests with `@pytest.mark.gpu_required` so CI auto-skips
- mypy `--strict`, ruff with same rule set (E, F, W, I, N, D, UP, ANN, B, C4, SIM)

---

## Success Criteria (from source plan, mechanically verifiable)

1. `mypy sensing/galerkin_fusion.py --strict` passes
2. All existing 752 `sensing/` tests pass unchanged with `fusion_backend=adapter`
3. `tests/sensing/test_galerkin_fusion.py` ≥20 tests, ≥95% coverage on the new module
4. `config.sensing.fusion_backend ∈ {adapter, galerkin}` switches at runtime, no rebuild
5. ≥1 plot in `docs/benchmarks/galerkin_fusion/` with reproducer script
6. Technical note 1000–3000 words, four sections: Problem / Approach / Results / Limitations
7. ADR committed in **both** repos by end of Day 10

---

## "What Counts as a Win" (decision-gate inputs)

Any one is enough to publish:
- Memory reduction ≥15% with no regression on the other three metrics
- Latency mean flat-or-lower **and** p99 ≥10% lower
- Latent entropy meaningfully different (Cohen's d ≥ 0.3) — either direction, with explanation
- Rollouts/sec up ≥10% at fixed 100ms decision budget

If none hold → publish as honest negative result; the decision gate uses that signal directly.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Galerkin block doesn't fit in 8GB shared RAM | Day 5 shakedown; documented scope-down path on Day 2 (fewer Q-tokens, lower projection dim) |
| 752 existing tests break in non-obvious ways | Feature flag keeps adapter path alive; CI matrices both backends |
| AlphaGalerkin operator signature changes mid-experiment | Day 1 ADR freezes signatures for 2 weeks; submodule pinned to commit SHA |
| Benchmark shows no delta | Already allowed-for in success criteria; informs decision gate cleanly |
| Two weeks slips to three | Hard stop at Day 10 for decision gate; finish note in Week 3 if rough |
| Scope creep toward ICM / HJB / patent drafting | Out-of-scope list explicit; capture impulses in `docs/followups.md` without acting |
| On-Jetson thermal/power flakiness | Shakedown on Day 5 (not Day 10) leaves buffer; workstation fallback with explicit caveat |

---

## Verification

**End-to-end sanity check (run on Day 9 from clean clone):**

```bash
# Mouse-Droid-AGI clean-clone reproduction
git clone --recurse-submodules <mouse-droid-url> /tmp/mdagi-verify
cd /tmp/mdagi-verify
pip install -e .
pip install -e external/AlphaGalerkin

# Both backends still type-check
mypy sensing/ --strict

# Both backends pass invariant tests
pytest tests/sensing/ -v

# Benchmark reproduces from scratch
python scripts/bench_fusion.py --episodes 10 --seeds 5 --output /tmp/bench
diff /tmp/bench/results_summary.md docs/benchmarks/galerkin_fusion/results_summary.md
```

**AlphaGalerkin-side verification (this repo):**

```bash
# Re-exports stable
python -c "from src.modeling import GalerkinAttention, FNetBlock, MultiScaleFourierFeatures, StabilityGuard"

# Existing test suite unchanged
pytest tests/modeling/ -v

# Day-1 ADR committed
ls docs/architecture/ADR-mouse-droid-fusion-integration.md

# Day-10 ADR (created after the decision gate — see §8; not expected to exist
# on day 1 of the sprint, only after the benchmark results are in)
# ls docs/architecture/ADR-post-fusion-direction.md
```

---

## One thing to remember

The benchmark plot is the artifact; the **ADR is the commitment**. If Day 10 ends without an ADR in both repos, the two weeks did not fully succeed regardless of code quality. The point of doing the fusion head first is that it is cheap enough to be a *decision tool* — treat it that way.
