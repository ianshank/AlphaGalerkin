# Commercialization peer review — adjudicating the three-model meta-analysis

**Reviewed at:** `6052281` (2026-09-25) · **Method:** every load-bearing premise checked
against code on disk, not against the narrative.

A prior meta-analysis fused three frontier-model reviews of this repository and concluded
that the project is "a domain-agnostic constrained-allocation engine bundled with
enterprise-grade DevOps tooling", then proposed a four-phase extract-and-pivot plan. The
strategic reading is largely right. Roughly half of the *factual* premises under it are
not.

This document does three things: it scores each premise against the code, it identifies
the four findings that change the plan, and it replaces the plan with one that can
actually merge through this repository's own gates.

---

## 1. Verification scorecard

| # | Premise from the meta-analysis | Verdict | Evidence |
|---|---|---|---|
| 1 | `picogk` is an unstated generative-design prototype | **False** | `src/pde/sdf.py:437` raises `NotImplementedError` unconditionally. The `[picogk]` extra ships only `pythonnet>=3.0` — not PicoGK. Everything runs on `AnalyticalHelixSDF`. |
| 2 | "MCTS loses by 9.23× at matched wall-clock compute" | **Mislabelled** | `9.23` is `l2_error_ratio_at_matched_solves`. The wall-clock-proxy metric is `error_per_dof_ratio_mcts_over_dorfler` = **30.84**. The review understated the loss on the axis it named. |
| 3 | `video_compression` is decoupled from `pde` | **True** | One import in either direction: `src/video_compression/perf/device.py:16` → `src.poc.device`. No `src/pde`, `src/mcts`, `src/refinement` or `src/research` import at all. |
| 4 | `tests/claude/` is extractable today | **True** | 3 files, 1,014 LOC, **zero** `src.*` imports. |
| 5 | `tests/docs/` is extractable today | **Partial** | 22 files, 8,104 LOC, imports `src.poc`, `src.templates`, `src.tools`. Most of it is charter/CI-alignment logic that is meaningless outside this repo. |
| 6 | `tests/security/` is extractable today | **False as stated** | Imports six `src` packages including `src.video_compression`. The *payload corpus* is portable; the tests around it are not. |
| 7 | `device_planner.py` exists | **True** | `src/video_compression/zoo/device_planner.py` — `scan_devices()`, `assign_devices()`, `VRAM_AWARE` best-fit. Named correctly. |
| 8 | Dual-budget accounting is a new contribution | **Already implemented** | `results/mcts_classical_amr_arena.csv` already carries `wall_time_seconds`, `n_apply_actions`, `n_cache_misses`, `n_cache_hits`; the sidecar already reports three separate ratios. |
| 9 | Phase 2 must "sever MCTS from the Go/PDE legacy" | **Already done and CI-gated** | `tests/regression/test_import_contracts.py` contract `search-engine-does-not-know-its-domains` forbids `src.mcts` importing `src.pde`, `src.refinement`, `src.research`, `src.poc`, `src.games`. |
| 10 | `GameInterface` is vestigial | **False — it is the boundary** | Two deliberate definitions: `src/mcts/search.py:85` (`Protocol`, structural, domain-free) and `src/games/interface.py:56` (`ABC`, game-AI base). The Protocol exists *precisely so* `src/mcts` need not import `src/games`. Deleting it breaks contract #9. |
| 11 | Hardware provenance is an open honour-system loophole | **Partial** | `collect_hardware_tag()` already exists (`src/research/run_manifest.py:313`) and calls `torch.cuda.get_device_name()`. The real gap is narrower and worse — see finding C. |
| 12 | Pivot A (video rate control) must be built | **Already partly built** | `src/video_compression/mcts/rate_control.py` — 500 LOC, `MCTSRateController` + `GOPPlanner`, already wired into `codec.py:170`. |
| 13 | Agent-count independence for swarms | **Architecturally plausible** | `GalerkinAttention.forward` operates on `b n (h d)` (`src/modeling/attention.py:108`) with no grid assumption, and normalises by `1/n`. The claim is untested, but nothing in the architecture forbids it. |

**Score: 4 true, 4 false, 3 partial, 1 mislabelled, 1 plausible-but-untested.**

---

## 2. The four findings that change the plan

### A. The isomorphism is not a discovery — it is already in the tree, as a copy-paste

The meta-analysis credits Opus with recognising that adaptive mesh refinement and video
rate control are the same MDP. That recognition already happened, and somebody already
acted on it — by duplicating the search engine rather than sharing it.

```
src/mcts/node.py:84                          UCB(s,a) = Q + c_puct · P · √N_parent / (1 + N)
src/video_compression/mcts/rate_control.py:84  exploration = c_puct * prior * sqrt(parent_visits) / (1 + visit_count)
```

Two `MCTSNode` classes. Two `ucb_score` methods. Identical formula. Zero shared code.
`src/video_compression/mcts/` (900 LOC) imports nothing from `src/mcts/` (2,057 LOC).

This inverts the plan's premise. The work is not *"abstract MCTS into a planning SDK, then
port it to video."* The work is *"two implementations of the same algorithm already exist
and are free to drift; unify them."* That is a smaller, better-defined, and far more
defensible change — and it converts Pivot A from a port into a configuration change.

It is also a live correctness risk today: a fix to selection, backup or exploration
semantics in one implementation silently does not reach the other. This repository already
paid for exactly that class of defect once — the F0 two-player backup applied to a
single-agent game, which produced a retracted headline.

### B. The plan's headline sequencing is illegal under this repo's own merge gate

`config/focus.yaml` freezes two tracks. `src/video_compression/` is one of them
(track `codec`), with an `incidental_line_budget` of **20 changed lines**. `core_paths`
are `src/mcts/`, `src/pde/`, `src/refinement/`, `src/research/`. The `focus` job has been
a **hard merge gate** in `ci-success` since 2026-09-11 (`docs/FOCUS.md`).

The rule: a changeset may touch a frozen track, or a core path, but a *substantive* change
to both at once fails the build.

The proposed Phase 3 Pivot A — "wire the `PlanningEnvironment` to the video codec lab" — is
precisely the diff shape that gate exists to reject. So is any single PR that both extracts
a planner from `src/mcts/` and adopts it in `src/video_compression/`.

This is not an obstacle to route around. It is a sequencing instruction, and honouring it
produces a better plan (see §4, Phase 1): one core-only PR, then one codec-only PR.

### C. The provenance gap is real, but it is not the one identified

The review proposed a CI guard that parses Markdown performance tables and asserts the GPU
name matches. The capture machinery for that already exists. The actual gap is cruder:

**Four of six committed result CSVs have no provenance sidecar at all.**

```
results/lambda_scheduling.csv              — no .run.json
results/lshape_mcts_vs_dorfler.csv         — no .run.json
results/stochastic_galerkin_compare.csv    — no .run.json
results/transfer_baseline_compare.csv      — no .run.json
```

And of the two that do, `lshape_adaptive_vs_uniform.run.json` records
`hardware_tag: "unknown"`. So the field exists, is schema-versioned, is checked by
`assert_proposal_grade` — and is empty on the artifact that has it.

A guard that cross-checks a *claimed* GPU against a *measured* one is the right idea aimed
at the wrong layer. The cheaper, higher-value fix is to require a sidecar with a non-`unknown`
hardware tag for every committed artifact the charter's evidence register cites. The
`run-provenance` and `claims-ledger` skills already encode the ritual; nothing enforces
completeness across the set.

### D. The strategic verdict survives, but the evidence is worse than stated

On the arena's own artifact (`results/mcts_classical_amr_arena.run.json`, `dirty: false`,
SHA `19609d4`):

| Metric | Value | Reading |
|---|---|---|
| `l2_error_ratio_at_matched_dof` | **0.9532** | Below 1.0 — favours look-ahead when the budget is degrees of freedom |
| `l2_error_ratio_at_matched_solves` | **9.23** | Look-ahead loses heavily when the budget is solver calls |
| `error_per_dof_ratio_mcts_over_dorfler` | **30.84** | Look-ahead loses by ~31× end-to-end on wall-clock |
| `adequacy_error_ratio` | 0.0946 | Substrate gate passed; not a policy result |

The conclusion — *this is not a faster solver* — holds, and holds harder than the review
argued. But note what the same artifact says: the sign of the answer **depends on which
budget you charge for**. That is not a consolation prize; it is the actual product thesis,
and it is the one thing all three models circled without naming. The technology is not
"search finds better refinements". It is **"search converts a budget you have into a budget
you don't"** — spend CPU cycles offline to buy degrees of freedom, bits, or spatial
footprint at deployment. Every credible pivot below is an instance of that trade, and it is
only credible where the deployment budget is genuinely scarcer than the planning budget.

---

## 3. Adjudicating the three models

The meta-analysis's adjudication is fair in outline and wrong in attribution.

**Opus — product visionary.** Correctly credited for the AMR↔rate-control isomorphism and
the agent-count-independence reframing, which is the one speculative claim that survives
contact with the code (finding #13). But the forensic call it was most praised for — the
`picogk` prototype — is its worst error: the class raises `NotImplementedError` on the
first line of its constructor. Treating a stub as hidden capability is the exact failure
mode this repository has a retraction ledger for.

**Sol — pragmatic architect.** Credited with dual-budget accounting as the fix that
"neutralises the scientific critique". The repo has had it since the arena landed:
three ratios in the sidecar, per-solve counters in the CSV. Sol's contribution is real but
different from the one claimed — it is the observation that the accounting should be a
*contract* rather than a convention. That reframing is worth keeping and is cheap to
enforce (Phase 2 below).

**Nemotron — meta-observer.** Correct and under-credited. `.claude/` and `openspec/` are a
tested agent harness: `tests/claude/` is 1,014 LOC with zero `src` imports — the single
cleanest extraction in the tree, and the only Phase 1 candidate that survives inspection
intact.

**On the monorepo-vs-split question**, the meta-analysis's verdict (reject the split, adopt
the SDK refactor) is right, for a reason none of the three models gave: the split is not
blocked by cohesion, it is blocked by *governance surface*. Charter alignment, the
architecture map, the shape baseline, the artifact manifest and the `hf_space` mirror all
bind `src/` package identity to tests that fail on drift. Splitting the repo means
rewriting that machinery; keeping it means the machinery keeps working for free.

**On the rename** (`src/mcts/` → `src/planning/`): **do not do it.** It touches 54 importing
files across 13 packages, plus the `hf_space` mirror, plus `ARCHITECTURE.md`, the charter
scope register, `config/shape_baseline.yaml` hashes and the import contracts. It buys
nothing that contract #9 does not already guarantee, and it would burn the exact review
attention the commercial work needs. Terminology in docstrings is free; package identity is
not.

---

## 4. The revised plan

Each phase names the PR shape, the gate it must clear, and a falsifiable exit criterion.
Phases 0–2 are sequenced so that no single PR is both core-touching and codec-touching.

### Phase 0 — Delete the false premises (days 1–3)

| Task | Action | Exit criterion |
|---|---|---|
| **0.1 `picogk` decision** | It is a stub behind an extra that installs only `pythonnet`. Either retitle the Noyron surface honestly as an *analytical-surrogate* benchmark, or drop the extra. Do **not** market a 3D-printing lane. | `docs/` contains no claim implying live PicoGK geometry ingestion. |
| **0.2 Provenance completeness** | Regenerate the four missing `.run.json` sidecars; fix `hardware_tag: unknown`. | Every `results/*.csv` cited by the evidence register has a sidecar with a non-`unknown` hardware tag. |
| **0.3 Provenance guard** | Extend the existing manifest guard: a cited artifact without a complete sidecar fails CI. Mutation-kill it per `harden-a-guard`. | Deleting any sidecar turns a *named* test red. |

Phase 0 touches `docs/`, `results/` and `tests/` only — no frozen track, no core path.

### Phase 1 — Unify the two search engines (weeks 1–3) — *the highest-value work in this plan*

Two PRs, in this order, because the `focus` gate requires it:

**PR-1 (core only — zero codec lines).** Inside `src/mcts/`, extract the selection/backup/
expansion core that both implementations share, behind the existing domain-free `Protocol`.
Do not rename the package. Do not import anything new. Contract #9 must stay green
unmodified.
*Exit:* `src/mcts` tests green at its existing 90 branch gate; `audit_abstractions` clean;
import contracts unchanged.

**PR-2 (codec only — zero core lines).** Delete `MCTSNode`/`ucb_score` from
`rate_control.py`; adopt the core planner. `MCTSRateController` and `GOPPlanner` keep their
public signatures.
*Exit:* codec coverage gate (83) holds; a planted change to the core UCB formula visibly
changes rate-control behaviour — proving the duplication is gone rather than merely hidden.

**Why this is worth doing first:** it is the only item in the entire plan that is
simultaneously a correctness fix, a maintenance win, and a commercial enabler. It converts
Pivot A from a port into a config change, and it removes a silent-divergence risk of the
same class as the F0 defect.

*Contingency:* if PR-1 cannot be built without a core-path change that the `focus` gate
reads as substantive alongside anything else in flight, land it alone on a quiet branch
rather than reaching for the `focus-override` label. The label exists; using it to dodge a
gate that is correctly firing would forfeit the thing that makes this repo's numbers worth
anything.

### Phase 2 — Promote dual-budget from convention to contract (week 3, parallel)

The data already exists. Make it a rule: any comparative claim must report **both** the
domain budget (DOF / bits / footprint) and the planning budget (solves / wall-clock), and a
claim that reports only the favourable one fails CI. This is one guard over the evidence
register, in the idiom `test_amr_policy_ratios_cite_a_manifest` already uses.

*Exit:* a synthetic single-budget claim planted in the register turns a named test red.

**This is the highest strategic-value-per-line item in the plan.** It converts the 0.9532 /
9.23 / 30.84 spread from an embarrassment into the product's defining specification.

### Phase 3 — Extraction, honestly scoped (weeks 2–5, parallel, low risk)

Ranked by *verified* extractability, not by narrative appeal:

1. **`tests/claude/` → agent-harness conformance suite.** 1,014 LOC, zero `src` imports.
   Ships today. This is the cleanest asset in the repository and the meta-analysis
   under-rated it.
2. **`device_planner.py` → VRAM-aware scheduler.** Self-contained, real, correctly named.
   Note it lives in a frozen track: extract by *copy* to a new package, leave the original
   untouched, and the gate never fires.
3. **`tests/security/` payload corpus → reusable fixture.** Extract the *corpus*, not the
   tests. The tests import six `src` packages.
4. **`tests/docs/` → CI-integrity linter.** The generalisable core is the coverage-gate
   integrity check and the "can this job actually fail?" parser. Budget for real work: 8,104
   LOC, most of it charter-specific.
5. **SBIR scaffold fork.** `docs/business/` + `config/proposals/` both exist as described.
   Lowest engineering cost, genuine developer-relations value.

### Phase 4 — Pivots, ranked by distance-to-evidence (months 2–4)

| Pivot | Distance | Honest status |
|---|---|---|
| **A. Predictive transcoding** | **Closest** | `MCTSRateController` + `GOPPlanner` exist and are wired. After Phase 1 PR-2 this is a measurement task, not a build. First deliverable is a BD-rate number against `h265_baseline.py` with both budgets reported. |
| **B. Content-aware bitrate for detection** | Near | The genuinely novel framing: optimise QP against downstream detector confidence, not PSNR. Needs a detector in the loop; nothing in-repo blocks it. Strongest fit to stated domain expertise. |
| **C. Agent-count-independent swarms** | Far but real | Architecturally plausible (finding #13) and completely unbuilt. The one-week falsification test: train on N=4, evaluate at N=16, report whether the attention's `1/n` normalisation holds. Cheap to disprove — run it before committing a roadmap to it. |
| **D. Adaptive CI test selection** | Furthest | Greenfield, and competes with mature commercial tools. The repo's CI-integrity assets (Phase 3.4) are the more defensible play in the same market. Deprioritise. |

---

## 5. What to drop, and why

| Dropped | Reason |
|---|---|
| `src/mcts/` → `src/planning/` rename | 54 files + mirror + charter + shape baseline, for zero guarantee that contract #9 does not already provide. |
| "Delete the vestigial `GameInterface`" | It is not vestigial. It is the `Protocol` that keeps the search engine domain-free; deleting it breaks the contract the same plan wants to establish. |
| "Build dual-budget accounting" | Already exists at the data layer. Re-scoped to *enforcing* it (Phase 2). |
| GPU-name-vs-Markdown-table CI guard | Aimed at the wrong layer. Replaced by sidecar completeness (Phase 0.3). |
| Physically splitting the repository | Blocked by governance surface, not by cohesion. The monorepo keeps the guard machinery working for free. |
| 3D-printing / generative-design lane | The dependency is a stub that raises on construction. There is no lane. |

---

## 6. Open questions for the owner

1. **Does the `codec` freeze lift?** `docs/FOCUS.md` says the freeze lifted on the signed
   arena result, but `config/focus.yaml` still lists `codec`. Phase 1 PR-2 and all of Pivot A
   depend on the answer. Re-scoping the YAML is a one-PR `openspec` change and should happen
   before Phase 1 starts, not during it.
2. **Is Pivot B (detector-aware bitrate) in scope?** It is the strongest commercial fit and
   the only pivot that is both novel and close to existing code, but it needs a detection
   model in-repo, which is a new dependency and a charter scope change.
3. **Who owns the extracted packages?** Phase 3 produces four shippable artifacts with no
   home. Extraction without a maintenance owner produces four more unmaintained repositories.

---

## 7. Bottom line

The meta-analysis is right that this is not a faster PDE solver, right that the enterprise
tooling is the most immediately liquid asset, and right to reject the repo split. It is
wrong about `picogk`, wrong about which budget the 9.23× figure describes, wrong that
`GameInterface` is vestigial, wrong that dual-budget accounting and MCTS/domain decoupling
still need building, and wrong about three of its five extraction candidates.

The single largest finding it missed is sitting in the tree: **the isomorphism it proposes
to discover has already been acted on, by copying the search engine instead of sharing it.**
Unifying those two implementations is a correctness fix, a maintenance win and the enabler
for the most valuable pivot — and it is, conveniently, the one piece of work that this
repository's own merge gates are already shaped to accept.
