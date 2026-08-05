# Tasks: `dashboard-uplift`

WS1, WS2, and WS6 have landed. WS3–WS5 are designed in `design.md` and delivered separately;
their tasks are recorded here so the sequencing and its charter consequences stay visible.

WS6 was taken ahead of WS3–WS5 deliberately: it is the cheapest workstream and it protects
everything the others will touch. One dependency runs the other way — the dashboard coverage gate
is pinned at 84 rather than 85 because `tabs/game_tab.py` cannot be covered until WS3 removes the
`hf_space` shadowing (see task 3.6).

## 1. WS1 — Claim fidelity (P0)

- [x] 1.1 `dashboard/config.py` — replace `TransferMilestone.achieved_mse` spike defaults
      (`{9: 2.5e-6, 13: 2.04e-4, 19: 3.93e-4}`) with the committed 3-seed-median operator
      figures from `results/transfer_baseline_compare.csv`
- [x] 1.2 `dashboard/config.py` — add `cnn_retrained_mse_19x19`, `cnn_zeroshot_mse_19x19`, and
      `transfer_ratio_19x19`, each citing `config/baselines/transfer_ci.json`
- [x] 1.3 `dashboard/config.py` — remove the `0.000209` literal from the `achieved_mse`
      description; cite `specs/transfer_baseline_compare.spec.md` instead
- [x] 1.4 `dashboard/tabs/poc_tab.py` — rewrite `show_transfer_milestone` as a three-arm
      comparison (operator zero-shot / CNN retrained / CNN zero-shot at 19×19) plus the
      operator's real 9→13→19 curve; signature unchanged
- [x] 1.5 `dashboard/tabs/poc_tab.py` — delete the synthetic `default_rng(7)` curve and every
      `ratio = threshold / mse` / `N× better` string
- [x] 1.6 `dashboard/tabs/poc_tab.py` — replace the tab blurb's `min(achieved_mse.values())`
      (the 9×9 in-distribution number) with the 19×19 figure and the zero-retraining framing
- [x] 1.7 `dashboard/app.py` — correct the About-table transfer row
- [x] 1.8 `hf_space/app.py` — add the zero-retraining framing beside the board-size table
      (text-only; the Space is pinned to Gradio 4.44.1)
- [x] 1.9 `src/demos/physics_demo.py` + `hf_space/src/demos/physics_demo.py` — label the
      `model is None` output a placeholder instead of reporting it as an MSE

## 2. WS2 — Guards

- [x] 2.1 `openspec/specs/project-charter/spec.md` — add `### Requirement: UI Claim Fidelity`
      (prose + `SHALL` + Scenarios; no delimited register, per `design.md` AD2)
- [x] 2.2 `tests/docs/test_charter_alignment.py` — register the Requirement in `_GUARDED`
- [x] 2.3 `tests/docs/test_charter_alignment.py` — implement
      `test_ui_claims_match_committed_artifacts`: bare ban on `FABRICATED_FIGURE` and
      `RETRACTED_BLANKET_CLAIM` across `dashboard/**/*.py`, plus agreement between the
      dashboard's transfer figures and `config/baselines/transfer_ci.json`
- [x] 2.4 `tests/dashboard/test_config.py` — AQA test binding `TransferMilestone` to the
      committed baseline; assert the new fields
- [x] 2.5 `tests/dashboard/test_poc_tab.py` — assert the honest framing is rendered and the
      retracted strings are absent
- [x] 2.6 Mutation-test both new guards: reintroduce a spike figure and the retracted literal,
      confirm each fails, revert
- [x] 2.7 `CLAUDE.md` — Regression Surface row; `CHANGELOG.md` — `## [Unreleased]` entry

## 3. WS3 — Un-shadow the mirror (deferred)

- [ ] 3.1 Relocate `hf_space/src/{game_manager,endgame}.py`, `hf_space/src/rendering/`, and
      `hf_space/config/board.py` into the maintained tree. **Not** a `sys.path` reorder — root
      `src/` and `config/` are regular packages, so reordering alone breaks the Go tab
      (`design.md` AD4)
- [ ] 3.2 Charter delta for *Scope Integrity* + `ARCHITECTURE.md` package map, since 3.1 adds a
      `src/` package
- [ ] 3.3 `dashboard/app.py` — put `ROOT` before `HF_SPACE` once 3.1 lands
- [ ] 3.4 `dashboard/tabs/training_tab.py` — remove the second shadowing site (the local
      `sys.path.insert` inside `get_model_summary`)
- [ ] 3.5 Add a test that exercises the real import path, so the divergence between
      `dashboard/app.py` and `tests/dashboard/conftest.py` cannot recur
- [ ] 3.6 Once 3.1–3.5 land, cover `tabs/game_tab.py::_ensure_loaded` and the AI-move paths
      (~53% today, the entire dashboard coverage deficit) and raise the WS6 gate from 84 to 85
      in both `.github/workflows/ci.yml` and the charter's gates register

## 4. WS4 — Registry-driven scenarios + Results tab (deferred)

- [ ] 4.1 Drive the PoC tab from `ScenarioRegistry().get_all()` after importing
      `src.poc.scenarios` (all 10 register), replacing the hardcoded 3
- [ ] 4.2 Gate each scenario on `requires_gpu` / CUDA availability / LM Studio preflight —
      3 are CPU-safe outright, 3 more with `device="cpu"`, 4 need GPU and/or LM Studio
- [ ] 4.3 Fix `src/poc/cli.py::register_builtin_scenarios`, which registers only 3 scenarios and
      makes `python -m src.poc.cli list` under-report
- [ ] 4.4 New Results tab rendering the committed `results/*.png` and CSVs, reusing
      `src/poc/visualization/reports.py::HTMLReportGenerator` into `gr.HTML`; degrade gracefully
      when `results/` is absent (it is not packaged). Excludes `lambda_scheduling` (AD5)
- [ ] 4.5 Wire the loaded checkpoint into the physics and architecture demo tabs, closing AD6

## 5. WS5 — Clickable Go board (deferred)

- [ ] 5.1 Have `BoardRenderer` expose its intersection→pixel mapping, or render square —
      `figsize` is `(6.5, 6.0)` with `set_aspect("equal")`, so the axes letterbox and pixel
      inversion is not a clean affine map today
- [ ] 5.2 Wire `gr.Image.select` → `(row, col)` (supported in both 4.44.1 and ≥6.0); keep the
      Textbox as the accessible path
- [ ] 5.3 Set `interactive=False` on both board images in `dashboard/tabs/game_tab.py` — they
      currently accept uploads over the board

## 6. WS6 — Quality gates + docs

- [x] 6.1 Add `dashboard/` to CI's `ruff check` and `ruff format --check` — pre-commit already
      lints it, so CI-green code can fail a contributor's commit hook today
- [x] 6.2 Add a `dashboard` coverage gate; record it in the charter's gates register only once
      CI enforces it (the charter⊆CI direction is what the gate guard checks)
- [x] 6.3 Author `dashboard/AGENT.md`, modelled on `hf_space/AGENT.md`
- [x] 6.4 Decide `mypy` posture for `dashboard/` — `pyproject.toml` currently relaxes strictness
      per-module for gradio's incomplete stubs

## 7. Verification

- [x] 7.1 Charter regression surface: `pytest tests/docs/ tests/regression/test_related_work_guard.py tests/hf_space/ -v`
- [x] 7.2 Dashboard surface: `pytest tests/dashboard/ -v`
- [x] 7.3 `python scripts/check_doc_links.py` (CI runs it on `openspec/**` changes)
- [x] 7.4 `ruff check` + `ruff format --check` on every touched file
- [x] 7.5 CI's blanket selection: `pytest tests/ -m "not slow and not e2e and not gpu_required" -q`
- [~] 7.6 Manual smoke: `python dashboard/app.py` → PoC Scenarios → Zero-Shot Transfer.
      Done headlessly — `build_app()` constructs the full Blocks tree and
      `show_transfer_milestone()` renders the three-arm chart and summary (asserted in
      `tests/dashboard/`). A human browser pass is still worth doing before release; this
      environment has no display.
