# Contributing to AlphaGalerkin

Thanks for your interest in contributing! This project favors **small, verified,
spec-driven changes** over large speculative ones. This guide points you at the
conventions already used in the repo rather than inventing new process.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## 1. Development setup

```bash
git clone https://github.com/ianshank/AlphaGalerkin.git
cd AlphaGalerkin

python -m venv venv && source venv/bin/activate   # Linux/Mac
pip install -e ".[dev]"

# Install the git hooks so lint/format/type checks run before each commit
pre-commit install
```

See [`docs/getting-started.md`](docs/getting-started.md) for a clone-to-first-run
walkthrough (optional extras: `dev`, `viz`, `test-extras`, `fem`, `jax` /
`jax-gpu`, `picogk`, `lm-studio`, `docs`; no `dashboard` extra) and
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the repository map.

## 2. Scope and spec-driven development

Before proposing a feature, check it is in scope: the project charter
([`openspec/specs/project-charter/spec.md`](openspec/specs/project-charter/spec.md)) is the
supreme authority on mission, non-goals, and the evidence standard every numeric claim must meet.

Non-trivial features start as a **markdown spec** before any code:

1. Copy [`specs/TEMPLATE.spec.md`](specs/TEMPLATE.spec.md) to `specs/<feature>.spec.md`.
2. Fill in the **Data Contract** (Pydantic fields), **Acceptance Criteria**
   (Given/When/Then, each mapped to a test), and **Thresholds** (reusing the
   canonical `src.poc.config.MetricThreshold` — do **not** invent a parallel schema).
3. Write the tests, then the code, then an AQA test asserting spec ↔ config agreement.
4. Register the guarding test command in the **Regression Surface** table in
   [`CLAUDE.md`](CLAUDE.md).

The full workflow is documented in [`specs/README.md`](specs/README.md). The
`spec-new` skill scaffolds this for you.

## 3. Quality gates (run locally before pushing)

These mirror what CI enforces (`.github/workflows/ci.yml`). Run them before opening a PR:

```bash
# Lint + format, exactly as CI's ruff-only `lint` job runs them (hard gates)
ruff check src/ tests/ dashboard/ scripts/ config/ conftest.py deploy_space.py
ruff format --check src/ tests/ dashboard/ scripts/ config/ conftest.py deploy_space.py

# Type check + abstraction audit: CI's `typecheck` job (parallel to `lint`).
# The audit is a hard gate; mypy is informational (continue-on-error).
python -m scripts.audit_abstractions src/mcts src/refinement src/pde src/research --fail-on-missing
mypy src/ --strict --ignore-missing-imports

# The hermetic fast lane, exactly as CI's `test-fast` job runs it (see below)
make test-fast

# The regression surface for the code path you touched — see the table in CLAUDE.md.
# Example (PDE / scenario changes):
pytest tests/pde/ tests/poc/ -m "not gpu_required"

# Or run all the hooks at once
pre-commit run --all-files
```

### The hermetic fast lane and the `network` marker

`make test-fast` (and `make coverage`) run under
[`pytest-socket`](https://pypi.org/project/pytest-socket/) with
`--disable-socket --allow-unix-socket --allow-hosts=127.0.0.1,localhost`, the same
`HERMETIC_PYTEST_FLAGS` CI's `test-fast` and `coverage` jobs use. Every socket except
loopback and unix sockets is refused, so a test that reaches the network fails on your
machine the same way it fails in CI instead of passing on whatever egress you happen
to have. Loopback stays open because `torch.multiprocessing`, `DataLoader` workers and
gloo on localhost need it.

- A test that **genuinely needs egress** (a model download, a live server) must be marked
  `@pytest.mark.network`. The marker is registered in `pyproject.toml`; the fast lane
  deselects it with `-m "... and not network"`, so the test still runs in any lane that does
  not carry that expression. Prefer mocking or a fixture-provided fake first — `network`
  is for tests whose *purpose* is the real call.
- `HERMETIC_PYTEST_FLAGS= make test-fast` runs the lane unsandboxed if you need to.
- `tests/docs/test_fast_lane_is_hermetic.py` keeps the three flag copies (`ci.yml`,
  `Makefile`, this convention) equal and drives a planted outbound connect to prove the
  block is real; do not edit one copy without the others.

### Optional-extra markers

Tests that need an optional extra carry a registered marker and are gated by the root
`conftest.py` through one shared `_gate_optional_extra` body:

| Marker | Extra | Where it runs in CI |
|---|---|---|
| `@pytest.mark.fem_required` | `[fem]` (scikit-fem) | `test-extras` |
| `@pytest.mark.eval_harness_required` | `[eval-harness]` (git extra) | `test-extras` (`-m "eval_harness_required"`, with its own coverage gate) |
| `@pytest.mark.gpu_required` | CUDA | auto-skipped on every CPU runner |

On a base install a marked test **skips with a count** in the terminal summary. With
`ALPHAGALERKIN_REQUIRE_EXTRAS=1` set, a missing extra is a `pytest.UsageError` naming the
`pip install -e '.[...]'` to run — the form CI's `test-extras` job uses, so a
half-installed extra errors instead of silently skipping. Do **not** use a module-level
`pytest.importorskip(...)` for an extra: it collapses the file to zero items, which no
hook can count and no `-m` expression can select (that is how
`tests/integrations/eval_harness/` went unmeasured — hygiene B37). Import the extra inside
a fixture instead and mark the module.

### Coverage

CI enforces **85% branch coverage** globally, plus **per-module gates** (e.g.
`mcts ≥ 90`, `refinement ≥ 85`, `pde ≥ 85`, `distributed ≥ 60`, games (Go + Chess) `≥ 80`). New
code needs tests that keep the changed module above its gate. The
[`coverage-gate`](.claude/) skill runs the exact per-module command for you.

> **Note (changed 2026-09-02):** this section used to require `COVERAGE_CORE=pytrace`
> because a torch wheel was believed to crash coverage's default C tracer. That claim
> was re-verified and does not reproduce; the pin is retired repo-wide and the default
> tracer is ~3× faster. Do not set `COVERAGE_CORE`.

The per-module gates run in the `coverage-gates` job, which is **sharded four ways** by
a `matrix.shard == N` condition on each gate step. Assignment is data, not policy: to
move a gate, change its `N`. `tests/docs/test_coverage_gate_shards.py` fails a step with
no shard condition (it would run on every shard), one naming a shard outside the matrix
(it would run on none) and an empty shard.

### Merge gates

`ci-success` names every non-nightly job in `needs` and exits 1 on any result other than
`success`. `tests/docs/test_ci_success_hard_gates.py` requires each job in `ci.yml` to be
either a hard gate or a disclosed, self-expiring exemption — so when you add a CI job, add
it to both `needs` and the `exit 1` block, or record why not (the `wire-a-ci-job` skill
walks through it). `lint` (ruff), `typecheck` (abstraction audit + mypy), `focus`
(scope containment), `secrets` (gitleaks) and every test job are hard gates.

### Module-size budget and shape baseline

Two ratchets stop the codebase's *shape* drifting while nobody is looking:

- **Module-size budget** (`tests/docs/test_module_size_budget.py`): every `src/` module
  over 600 lines and every test module over 1000 lines is frozen at its recorded line
  count. An unlisted module may not cross the ceiling; a listed one may not grow past its
  row; one that drops below the ceiling must have its row removed. Splitting a listed
  module (the `god-file-split` skill) is the intended way to shrink a row.
- **Shape baseline** (`config/shape_baseline.yaml`, `tests/docs/test_shape_baseline.py`):
  nine code-shape metrics over `src/` — ruff `C901` complexity findings, `PLR2004` magic
  values, `T201` library prints, lazy first-party imports, ad-hoc device-resolution sites,
  orphan modules, dead `ImportError` guards, deprecation-less shim files, `hf_space`
  mirror-diverged files — may not **grow**. Check locally with
  `python -m scripts.measure_shape check`. After a genuine reduction, regenerate the
  baseline with `python -m scripts.measure_shape write` and commit
  `config/shape_baseline.yaml`; do not hand-edit a count (the file carries a hash over
  `src/**/*.py`, and the guard fails when a recorded count disagrees with a live measure
  under a matching hash).

### Scope freeze (`focus`) and the `focus-override` label

Two tracks are frozen this cycle ([`docs/FOCUS.md`](docs/FOCUS.md), enforced from
[`config/focus.yaml`](config/focus.yaml)). The rule is not "do not touch frozen code" but
"do not make a *substantive* change to a frozen track in the same pull request as core
work" — substantive being the per-track `incidental_line_budget`. The `focus` job runs on
every pull request and is a hard merge gate; check locally with
`python -m scripts.check_focus --base origin/<base> --fail-on-violation`. If the coupling
is genuinely necessary, add the **`focus-override`** label to the pull request: the job
skips visibly, `ci-success` accepts that skip only when the label is present, and the
override is on the record.

## 4. Coding conventions

- **Style:** enforced by Ruff (line length 100; pydocstyle `D`, annotations `ANN`,
  bugbear `B`, and more — see `[tool.ruff.lint]` in `pyproject.toml`). Let
  `ruff format` do the formatting.
- **Types:** strict typing; `mypy --strict` with narrow, documented per-module
  carve-outs in `pyproject.toml`. New code should not add carve-outs without reason.
- **No hardcoded values:** configuration is Pydantic-validated. Surface magic
  numbers as typed config fields with docstrings.
- **Structured logging** via `structlog`, not `print`.
- **Tests:** property-based tests (Hypothesis) for mathematical operators;
  `@pytest.mark.gpu_required` for anything needing CUDA (auto-skipped on CPU CI);
  `@pytest.mark.network` for anything that must reach the real network;
  `@pytest.mark.fem_required` / `@pytest.mark.eval_harness_required` for the optional
  extras (see §3). Every marker must be registered in `pyproject.toml` —
  `--strict-markers` rejects an unknown marker on a test, and
  `tests/docs/test_marker_vocabulary.py` rejects an unknown identifier inside a `-m`
  expression, which `--strict-markers` does not.

## 5. Commits & pull requests

- **Commit messages:** [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`…), validated by Commitizen in
  the pre-commit hooks.
- **Changelog:** add a bullet under `[Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md).
  The file's *structure* is guarded by `tests/docs/test_changelog_headers.py`
  (Keep a Changelog 1.0.0 + PEP 440): exactly **one** `## [Unreleased]` header, and it is
  the first `##`; one preamble; under `[Unreleased]` each `###` group is one of
  `Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` / `Security` and appears at
  most **once** — a titled sub-section inside a group is a `####` heading, never a second
  `### Added`; release headers carry an ISO date and descend in version and date. A merge
  that duplicates a `### Added` block fails the guard rather than silently producing two
  changelogs.
- **Branch:** work on a feature branch; open a PR against the repository's default branch.
- **PR description:** fill in the [pull request template](.github/PULL_REQUEST_TEMPLATE.md).
  Keep PRs focused and reviewable.
- **Releases:** see [`RELEASING.md`](RELEASING.md) for the versioning/release process.

## 6. Concurrent agents and worktrees

If you dispatch more than one coding agent (or work alongside one) in the same
checkout, follow the rule in the root [`AGENT.md`](AGENT.md#concurrent-subagents):
concurrent subagents work in their own `git worktree` and never run `git stash`,
`git reset`, `git checkout -- <path>` or `git clean` in a shared working tree. Those
commands act on the whole tree, not the caller's file scope, so in a shared tree each
one reaches every other agent's uncommitted edits — non-overlapping *file* scopes do not
isolate agents; separate *worktrees* do (hygiene B22 is the incident). Create the
worktree as a sibling directory outside the repository, commit on a dedicated branch,
run `python -m …` from the worktree root, and hand the branch back to the orchestrator.
`tests/claude/test_worktree_rule.py` keeps the rule's anchor sentence identical in
`AGENT.md` and every Bash-granted `.claude/agents/*.md`.

## 7. Where things live

The repository map, the two-domain (game-AI vs. PDE) split, and the
core-vs-experimental maturity labels are in [`ARCHITECTURE.md`](ARCHITECTURE.md).
Per-module developer guides are the `AGENT.md` files inside each `src/` package.

Questions? See [SUPPORT.md](SUPPORT.md).
