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
adaptive mesh refinement** — is **not yet decided**: it has been tested only on
elliptic controls, where it failed, and the testbed that can decide it is not
built. The committed arena
(`results/mcts_classical_amr_arena.{csv,run.json}`: untrained MCTS vs Dörfler on
`SkfemTriSubstrate` at θ=0.5, matched DOF 287; the binding limit is
`max_steps=12`, not `max_dof=600`) reports a median
`l2_error_ratio_at_matched_dof` of **0.9532**, but that is single-element greedy
marking against Dörfler bulk marking: the artifact's greedy control makes the
same 12 decisions as the MCTS arm on every seed (MCTS/greedy 1.0,
`decisions_diverging_from_greedy_max` 0), so **search contributed no
decisions**. Reading 0.9532 as an MCTS win was corrected on 2026-10-08. At
matched solves MCTS is at 9.23 (ungated). Gate 1 of
`docs/business/COMMERCIALIZATION_PEER_REVIEW.md`, pre-registered in
`specs/lookahead_vs_greedy.spec.md`, asked whether look-ahead beats greedy on two
elliptic testbeds (the L-shape and a two-corner Z-tetromino) and returned
**NO-GO** on both: the deterministic search made greedy's 30 decisions on each, and
the best classical arm (Dörfler θ=0.3 / θ=0.5) beat it at matched DOF
(`results/lookahead_vs_greedy_lshape.{csv,run.json}`,
`results/lookahead_vs_greedy_zshape.{csv,run.json}`). That is the expected
control result; the thesis stays open until the deferred moving-front testbed
(T3) runs. Adequacy rates remain gate evidence, not this result. Legacy
`results/lshape_mcts_vs_dorfler.csv` is **non-informative for element-local
policy**.

The freeze **lifted** on that signed result's pre-registered verdict
(`l2_error_ratio_at_matched_dof < 1`), which still holds: the correction changes
what the verdict is evidence of, not the verdict, and re-freezing is an owner
decision it does not make. `config/focus.yaml` still lists
`codec` and `interactive-surfaces` so the split-attention gate keeps working
until a follow-up re-scopes tracks (empty `frozen_tracks` is rejected). The
`focus` job is a **hard merge gate** in `ci-success` since 2026-09-11 (plan
R-09): it runs on pull requests only, so `ci-success` accepts a `skipped`
result exactly when the event is not a pull request or the visible
`focus-override` label is present, and fails the build on a skip it cannot
explain. `tests/docs/test_ci_success_hard_gates.py` keeps both clauses in
place. Nothing in the YAML changed for that promotion (decision D9 of
`docs/ENGINEERING_REFLECTION_2026-09-11.md`).

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
(`results/mcts_classical_amr_arena.{csv,run.json}`): a pre-registered matched-DOF
verdict (median ratio 0.9532 at θ=0.5; the binding limit is `max_steps=12`, not
`max_dof=600`) that evaporates at matched compute — a signed result, not a smoke
pass. Since the 2026-10-08 re-record with a greedy control, that ratio is known
to be single-element greedy marking (**search contributed no decisions**), so the
lift rests on the verdict, not on an answer to the look-ahead question. Gate 1
returned NO-GO on its two elliptic testbeds (search again contributed no
decisions); the question stays open until the moving-front testbed runs. Parked work
(certificates, Noyron geometry, dashboard WS3–5, codec B35, LLM GPU smokes)
may resume in **separate** changesets; do not mix it with a new solver number
in the same PR. The `codec` / `interactive-surfaces` rows below remain the
machine-readable freeze until a follow-up edits `config/focus.yaml`.
