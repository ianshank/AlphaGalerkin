# `dashboard/` — Gradio E2E dashboard (developer notes)

The local, full-featured Gradio app: `python dashboard/app.py`. Eight tabs exposing the Go AI,
the physics/benchmark/architecture demos, an interactive PDE solver, PoC scenarios, a training
summary, and an About page. Distinct from [`hf_space/`](../hf_space/AGENT.md), which is the
public HuggingFace Space bundle — see *Two UIs, two Gradio majors* below.

## Layout

| File | Contents |
|---|---|
| `app.py` | `build_app()` factory → `gr.Blocks`; CLI entry (`--host/--port/--share/--debug`); About markdown |
| `config.py` | All Pydantic config. Pure `typing` + `pydantic` — **no gradio import** (load-bearing, see below) |
| `utils.py` | `fig_to_pil`, `device_str`, `format_exc`, `configure_structlog` |
| `tabs/*.py` | One `create_<name>_tab(cfg)` factory per tab, called inside `build_app()`'s `gr.Blocks` context |

Every tunable is a typed Pydantic `Field` on a model in `config.py`, composed into
`DashboardConfig` / `DEFAULT_CONFIG`. No magic numbers in tab code.

## Claim fidelity is enforced here

Any figure a tab renders is a **claim** under the charter's *UI Claim Fidelity* Requirement
([`openspec/specs/project-charter/spec.md`](../openspec/specs/project-charter/spec.md)), guarded by
`tests/docs/test_charter_alignment.py::test_ui_claims_match_committed_artifacts`. Concretely:

- Numbers must trace to a committed artifact — `config/baselines/transfer_ci.json` and
  `results/transfer_baseline_compare.csv`. `TransferMilestone` in `config.py` carries them as
  defaults and is AQA-tested against the baseline in `tests/dashboard/test_config.py`.
- The retracted `0.000209` figure and the retracted blanket novelty claim are **banned outright**
  in `dashboard/**/*.py` (no retraction-marker excuse — a UI has no reason to *discuss* a
  retracted number).
- No self-comparison framing: a ratio against an arbitrary pass threshold is not a result. Compare
  against the committed baseline, in whichever direction the artifacts support.
- Synthetic data must be labelled. `training_tab.py` says "(simulated)" on its curves; do the same
  for anything not measured.

The guard loads `config.py` **standalone** via `importlib`, never `import dashboard.config` —
`__init__.py` imports `app.py`, which pulls in gradio, and `tests/docs/` deliberately stays
stdlib-only. **Keep `config.py` free of heavy imports** or that guard breaks.

## The `sys.path` shadowing hazard

`app.py` inserts `hf_space/` into `sys.path` **before** the repo root, so `import src.X` inside a
dashboard process resolves to `hf_space/src/` — the hand-maintained, knowingly-diverged mirror —
not the maintained tree. It does this because four modules exist *only* there:
`config.board`, `src.endgame`, `src.game_manager`, `src.rendering.board_renderer` (all imported by
`tabs/game_tab.py`). `tabs/training_tab.py::get_model_summary` re-inserts the same path locally.

Two things follow, and both matter:

1. **`tests/dashboard/conftest.py` uses the opposite order** (root first) *and* mocks
   `_ensure_loaded`, so the app and its tests import different code for the same module names.
   The real import path is not exercised by any test.
2. **Reordering `sys.path` is not a fix.** Root `src/` and `config/` both have `__init__.py`, so
   they are *regular* packages: whichever path entry wins claims the whole namespace, with no
   fall-through. Putting root first makes the Go tab raise `ModuleNotFoundError` immediately. The
   fix is to relocate those four modules into the maintained tree — tracked as WS3 in
   [`openspec/changes/dashboard-uplift/`](../openspec/changes/dashboard-uplift/design.md) (AD4).

## Two UIs, two Gradio majors

`dashboard/` requires **gradio ≥6.0** (a `[dev]` dependency — the dashboard is not a runtime
dependency of the package). `hf_space/` pins **gradio 4.44.1** to match its Space SDK version.
Code shared between them must be valid on both majors, which is why edits to `hf_space/app.py`
from dashboard work are kept text-only.

## Quality gates

| Gate | Command | Notes |
|---|---|---|
| Lint | `ruff check src/ tests/ dashboard/` | Hard gate in CI's `lint` job |
| Format | `ruff format src/ tests/ dashboard/ --check` | Hard gate; matches pre-commit, which has no `files:` filter |
| Coverage | `pytest tests/dashboard/ --cov=dashboard --cov-branch --cov-fail-under=84` | Recorded in the charter's gates register |
| Claim fidelity | `pytest tests/docs/test_charter_alignment.py -k ui_claims` | See above |

**Coverage is gated at 84, not 85**, against a measured 84.85%. The whole deficit is
`tabs/game_tab.py` (~53%): `_ensure_loaded` and the AI-move paths are untestable while the
`hf_space` shadowing forces `conftest.py` to mock them. Raising this to 85 is a WS3 task —
relocating those modules is what makes that code reachable.

**mypy does not run on `dashboard/`.** The CI step is `mypy src/ --strict` and is
`continue-on-error`; the `pyproject.toml` override for `dashboard.*` disables 13 error codes
because gradio's stubs are incomplete. Extending the step to cover `dashboard/` would add the
appearance of type-checking without the substance, so it is deliberately not done. The override is
**wildcarded** (`"dashboard"` + `"dashboard.*"`) so a new module cannot silently land under full
strict — the previous hand-enumeration had exactly that trap.

## Gotchas

- `build_app()` is really called in `tests/dashboard/test_app.py`, inside a real `gr.Blocks`
  context — a broken tab factory fails there immediately. Only the domain layer (scenarios,
  models, game manager, solver) is mocked; gradio is not stubbed.
- The demo tabs are constructed with `model=None`, so `PhysicsDemo.predict()` returns zeros. That
  path is labelled a placeholder rather than reporting `mean(ground_truth²)` as model error.
  Wiring the real checkpoint through is WS4.
- Tab callbacks must bind the injected `cfg`. Passing a bare function reference to `.click()`
  silently renders `DEFAULT_CONFIG` instead, so a custom config can describe one thing and display
  another.
- `gr.Image` used as an output should set `interactive=False`, otherwise users can upload over it.
