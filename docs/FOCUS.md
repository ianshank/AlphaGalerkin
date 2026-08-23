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

The cycle's thesis is that **multi-step tree search beats greedy marking for
adaptive mesh refinement** — and that thesis is, as of today, unfalsified. Not
disproven: *untested*, because the substrate the experiment ran on refines by
tensor-product grid lines, so a marking-policy comparison on it measures the
substrate rather than the policy. Everything in the active set exists to get
that experiment to the point where its result means something.

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

The freeze is tied to a result, not a date. It lifts when the refinement
experiment has an interpretable answer — whichever way it goes. A negative
result ends the cycle just as legitimately as a positive one, and the charter
already requires reporting it either way.

The physical repository split that a frozen track invites stays deferred until
then: restructuring around an unfalsified thesis is the expensive version of
this mistake.
