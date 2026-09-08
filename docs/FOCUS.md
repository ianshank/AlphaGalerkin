# Focus — what this cycle is working on, and what it is not

This repository has 28 `src/` packages and a backlog longer than any one cycle
can absorb. That is not a problem by itself; it becomes one when attention is
spread across all of it at once. This document records an owner decision about
where attention goes, and — more usefully — the mechanism that keeps the
decision honest after everyone has stopped remembering it.

The machine-readable half is [`config/focus.yaml`](../config/focus.yaml), read by
`scripts/check_focus.py` in CI. This page and that file are kept in step by
`tests/scripts/test_check_focus.py`: a track named in one and missing from the
other fails the build. Prose that CI cannot read is a suggestion.

## The current focus

The cycle's thesis — **multi-step tree search beats greedy marking for
adaptive mesh refinement** — now has an interpretable answer on the element-local
substrate. Untrained MCTS vs Dörfler on `SkfemTriSubstrate` at θ=0.5, policy
`max_dof=600` (matched DOF 287): median `l2_error_ratio_at_matched_dof` **0.9532**
(`results/mcts_classical_amr_arena.csv`). Look-ahead does **not** win at matched
solves (ratio 9.23, ungated). Adequacy rates remain gate evidence, not this
result. Legacy `results/lshape_mcts_vs_dorfler.csv` is **non-informative for
element-local policy**.

The freeze **lifts** on that signed result. `config/focus.yaml` still lists
`codec` and `interactive-surfaces` so the split-attention gate keeps working
until a follow-up re-scopes tracks (empty `frozen_tracks` is rejected; promoting
the `focus` job into `ci-success` is a separate PR).

Active surfaces: `src/refinement/`, `src/pde/`, `src/mcts/`, `src/research/`,
and the governance layer (`openspec/`, `specs/`, `tests/docs/`,
`tests/regression/`) that keeps the resulting numbers auditable.

## Frozen tracks

Frozen means **paused, not abandoned**. Frozen code stays in the tree, stays
green in CI, and keeps its coverage gate. Nothing here is deleted, deprecated,
or scheduled for removal — the 2026-07-22 "cut to the core" already demonstrated
what deletion costs when the call is made too early, and `video_compression` was
reinstated the following day.

| Track | Paths | Why |
| --- | --- | --- |
| `codec` | `src/video_compression/`, `config/video_compression/`, `tests/video_compression/`, the `scripts/*compression*` / `*_video.py` / `benchmark_codec.py` entry points, `.github/workflows/phase2-zoo-validation.yml` | The largest non-core surface in the tree, and complete enough to sit still. It competes for exactly the reviewer attention the refinement work needs. |
| `interactive-surfaces` | `dashboard/`, `hf_space/`, `tests/dashboard/`, `tests/hf_space/`, `deploy_space.py` | Demo surfaces, not evidence. `hf_space/src/` is additionally a ~55k-LOC near-duplicate of `src/` (hygiene backlog B14, and a disclosed charter deviation), so substantive work there costs roughly double. |

## Explicitly *not* frozen

Naming these matters as much as naming the frozen ones, because "we're focusing"
is otherwise read as "everything else is dead":

- **Games** (`src/games/`, Go and Chess) — the original domain, and the
  back-compat anchor for every `SearchMode.ZERO_SUM` change.
- **Noyron / PicoGK / Leap 71** (`src/pde/operators_picogk.py`, `src/pde/sdf.py`,
  `src/poc/scenarios/noyron_*`) — an external-collaboration surface with its own
  cadence.
- **SBIR** (`src/research/pde_benchmarks.py`, `docs/business/proposals/`) — the
  proposal work has external deadlines this cycle does not control.

## How the gate reads a diff

A changeset may touch a frozen track. It may touch the core surface. What it may
not do is make a *substantive* change to both at once — that is what a split
attention span looks like in a diff.

"Substantive" is a line budget (`incidental_line_budget`, currently 20 changed
lines per track), not a file count. The distinction is doing real work: this very
cycle's PR edits `hf_space/src/__init__.py` to single-source a version string
alongside a new `src/research/` module, and that is a seven-line shim, not codec
work. A budget expresses the actual intent — *feature work is never seven lines*
— in one auditable number.

The alternative, an exemption list, only ever grows, and every entry silently
narrows the gate until it reports nothing.

```bash
python -m scripts.check_focus --base origin/<base-branch> --fail-on-violation
```

### The override

If the coupling is genuinely real, add the `focus-override` label to the pull
request. CI skips the check and the label is the record of the decision. That is
deliberately a *visible* escape hatch rather than a silent one — an override
nobody can see is the same as no gate.

## When the freeze lifts

The freeze lifted on the committed Phase 2 artifacts
(`results/mcts_classical_amr_arena.{csv,run.json}`): a matched-DOF win (median
ratio 0.9532 at θ=0.5, policy `max_dof=600`) that evaporates at matched compute.
That is an interpretable answer either way — not a smoke pass. Parked work
(certificates, Noyron geometry, dashboard WS3–5, codec B35, LLM GPU smokes)
may resume in **separate** changesets; do not mix it with a new solver number
in the same PR. The `codec` / `interactive-surfaces` rows below remain the
machine-readable freeze until a follow-up edits `config/focus.yaml`.
