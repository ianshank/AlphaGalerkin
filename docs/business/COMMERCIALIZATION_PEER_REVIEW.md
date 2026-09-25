# Commercialization peer review — adjudicating the three-model meta-analysis

**Revision 2** · reviewed at `6052281` (2026-09-25) · **Method:** every premise checked against
the code on disk, and every empirical premise re-measured. Revision 1 is in git history; §2 lists
what it got wrong.

> **Summary.** The meta-analysis — and revision 1 of this review — built a commercialization
> strategy on a premise nobody had tested: that the tree search does something. On the only
> committed result, it does not. The published MCTS trajectory is bit-for-bit identical to greedy
> single-element marking, and stays identical at every simulation budget tried (1 to 128) under the
> committed action filter. That reorders the plan. Before any pivot, spin-out or SDK, the project
> needs the one experiment it has never run — a problem where greedy is myopic — and a go/no-go
> decision on its outcome.

---

## 1. The finding that reorders everything

### The committed "MCTS wins at matched DOF" result contains no look-ahead

`results/mcts_classical_amr_arena.csv` is the artifact that the charter's evidence register, its
Novelty requirement, `docs/FOCUS.md` and `README.md` all cite as the answer to the cycle thesis:
median `l2_error_ratio_at_matched_dof` **0.9532**, "MCTS ~4.7% better at matched DOF".

Re-running the committed configuration with only the search budget changed:

| Search configuration | Legal actions per step | MCTS trajectory vs greedy | Matched-DOF ratio vs Dörfler | Real solves |
|---|---|---|---|---|
| `n_simulations=1` (cannot compare alternatives) | 8 | — (this *is* greedy) | 0.9532 | 13 |
| `n_simulations=2` | 8 | identical, bit for bit | 0.9532 | 13 |
| `n_simulations=4` | 8 | identical | 0.9532 | 14 |
| **`n_simulations=8` (committed)** | 8 | **identical** | **0.9532** | **43** |
| `n_simulations=16` | 8 | identical | 0.9532 | 121 |
| `n_simulations=32` | 8 | identical | 0.9532 | 290 |
| `top_k_actions=4`, `n_simulations=64` (deep tree) | 4 | identical | 0.9532 | 521 |
| `top_k_actions=8`, `n_simulations=128` (deep tree) | 8 | identical | 0.9532 | 559 |
| `top_k_actions=2`, `n_simulations=64` (deepest tree) | 2 | diverges at step 3 | **1.0289** (gate fails) | 433 |

**Why.** Three settings make multi-step look-ahead structurally impossible at the committed budget:
`get_legal_actions` pre-filters to the top 8 elements by residual indicator, sorted descending
(`src/pde/games/substrate_refinement.py:175-187`); the prior is a softmax over those same indicators
(`src/research/substrates/residual_evaluator.py`); and 8 simulations over 8 children is a one-ply
sweep. The search can at most confirm the indicator ranking one step deep, and on this problem it
always does. Given room for a real tree (4 children and 64 simulations; 8 and 128), it *still*
always agrees. In the one configuration where deep search overrode the indicator, it finished on
the wrong side of the Dörfler reference curve: **1.0289**, where greedy sits at 0.9532.

**What 0.9532 actually measures.** Single-element maximum marking (refine the largest-indicator
element, one per step) against Dörfler bulk marking at θ=0.5 — the expected trade of finer marking
in adaptive FEM (more solve iterations for more DOF-efficient meshes), not a search effect. At equal
accuracy the greedy mesh needs 2.6% fewer DOF (287 vs ≈294.7) for about twice the solves (13 vs
≈6.2): a trade that pays back after roughly 180–240 downstream reuses of the mesh, if downstream
solve cost scales between DOF¹ and DOF^1.5. The search layer on top produces the *identical* mesh
with 30 more solves, so it never pays back.

**What it means.**

- The charter evidence row (`openspec/specs/project-charter/spec.md:129`), its Novelty text
  (`:203-207`), its frozen-tracks deviation row (`:343`), `docs/FOCUS.md:19`, `README.md:58-59`
  and the 2026-09-08 `CLAUDE.md` milestone all attribute to MCTS a result produced by greedy marking.
- The cycle thesis — *"MCTS multi-step look-ahead beats classical greedy marking"*
  (`specs/mcts_classical_amr_arena.spec.md:11`) — is **not answered** by this artifact. Both arms
  it compares are greedy: Dörfler bulk marking, and an MCTS arm that reduces to single-element greedy
  marking. The control that separates look-ahead from marking granularity — a single-element greedy
  arm — is absent.
- The meta-analysis read the 9.23× matched-solves loss as the price of a quality edge; revision 1
  called it "converting a budget you have into a budget you don't". There is no trade. Greedy reaches
  the same mesh at 13 solves, so the search's extra cost buys nothing.

**What it does not mean.** This is one problem (L-shape Poisson, P1, `SkfemTriSubstrate`), one
Dörfler θ, an untrained evaluator, and a 12-step cap. It does not show that look-ahead can never
help. It shows that the committed evidence contains none, and — consistent with the repo's own
2026-07-05 note that greedy residual marking is near-optimal for local elliptic singularities — that
this testbed cannot show it.

**Status of these numbers.** Exploratory and uncommitted — not headline claims. The full table
reproduces in a few minutes on CPU. The one claim checkable against committed data takes a single
run of about ten seconds:

```bash
pip install -e '.[fem]'
python -m scripts.run_mcts_classical_amr_arena --n-simulations 1 --output-dir /tmp/arena_greedy
python - <<'EOF'
import csv
mcts = lambda p: [(r["seed"], r["n_dof"], r["l2_error"])
                  for r in csv.DictReader(open(p)) if r["method"] == "mcts"]
print(mcts("results/mcts_classical_amr_arena.csv")
      == mcts("/tmp/arena_greedy/mcts_classical_amr_arena.csv"))  # True
EOF
```

The other rows vary `--n-simulations`; the `top_k_actions` rows copy
`config/scenarios/mcts_classical_amr_arena.yaml` with that one field changed and pass `--config`.

---

## 2. Corrections to revision 1

Revision 1 scored the meta-analysis and was itself wrong on these points:

| Revision 1 said | What the code says |
|---|---|
| The codec's MCTS copies `src/mcts` ("identical formula … unify them") | A different algorithm. The codec controller is MuZero-style — learned `representation`/`dynamics`/`prediction` networks (`src/video_compression/mcts/rate_control.py:100-129`). `src/mcts` is AlphaZero-style over a real environment. They share only the PUCT selection formula. |
| The duplication is "a silent-divergence risk of the F0 class" | The codec backup is correct single-agent code, with no sign inversion (`rate_control.py:333-336`). Its real defects are different ones — §4. |
| Pivot A is "closest … after unification, a measurement task" | Search-based rate control is unbuilt: default-off, never trained, budget-blind (§4). The only H.265 reference is labelled `hardware_tag: placeholder` by its own file. |
| `video_compression` has "one import in either direction" | It imports `src.templates` (15×), `src.training` (2×), `src.poc` and `src.constants`. What is true: nothing from `src/pde`, `src/mcts`, `src/refinement` or `src/research`, in either direction. |
| 30.84 is "the wall-clock-proxy metric" | Precisely `error_per_dof_ratio_at_matched_wall_clock`. It survives; re-measured at 31.32 (wall-clock noise). |
| "Regenerate the four missing sidecars" | One is impossible (`lambda_scheduling.csv`'s producer was cut; the charter pins the file). One is the charter-exempt golden. Two fall outside the charter's sidecar rule, which covers only AMR policy ratios. The real provenance defects are in §5. |
| The hardware-provenance guard is "aimed at the wrong layer" | Wrong. The meta-analysis was pointing at a live claim (`README.md:352-360`). |
| `docs/FOCUS.md` and `config/focus.yaml` contradict each other | They do not. The charter's deviation register (`spec.md:343`) records `codec` as "remains paused" pending a follow-up edit. The `focus` gate admits a frozen-track-only PR of any size, so un-pausing is a policy decision, not a gate. |
| `tests/claude/` "ships today" | It has zero `src` imports, but `test_harness_validation.py` has about twenty lines bound to this repo's own files and inventory. The pattern ships; the file needs parameterising. |
| `device_planner.py` extracts "by copy" | `assign_devices` takes a `ModelZooManifestConfig`, so extraction needs a generic entry type. |
| The swarm pivot is "completely unbuilt" | `src/pde/games/swarm_planning.py` (644 LOC), 474 LOC of tests and `src/games/pettingzoo_adapter.py` exist. They are dormant: no consumer, scenario or result. |
| `sdf.py:437` | The raise is at `sdf.py:435`. |

---

## 3. The meta-analysis, rescored

| # | Premise | Verdict | Evidence |
|---|---|---|---|
| 1 | The repo is secretly a domain-agnostic allocation engine | **Stated, not hidden** | The charter's Purpose (`spec.md:8-16`): "`src/mcts/` is the domain-agnostic engine, and each domain adapts into it." |
| 2 | Look-ahead buys a matched-DOF edge at a compute cost | **False** | §1 — the edge comes from greedy marking; search adds only cost. |
| 3 | "MCTS loses 9.23× at matched wall-clock" | **Mislabelled** | 9.23 is matched *solves*; the matched-wall-clock figure is error-per-DOF 30.84. Neither measures look-ahead. |
| 4 | `picogk` hides a generative-design prototype | **False** | `PicoGKSDFEvaluator.__init__` raises `NotImplementedError` (`src/pde/sdf.py:435`); the extra installs only `pythonnet`. |
| 5 | Hardware provenance is an honour-system loophole | **True** | `README.md:352-360`: three latency and simulations-per-second rows on an "NVIDIA RTX 3090" — no artifact, no guard, and none of the rigs the repo documents (RTX 5060 Ti / 5060, GTX 1660 Ti, Tesla P40). |
| 6 | Video rate control can reuse the search | **True in principle, false in the tree** | §4. |
| 7 | Swarms are "agent-count independent" | **Untested; the nearest precedent is costly** | The host game exists (§2). The committed transfer benchmark goes from 81 to 361 tokens and loses ≈14× to a retrained specialist (`spec.md:124-126`). A 4→16-agent transfer is a comparable scale-up. |
| 8 | MCTS must be severed from its domains | **Done, CI-gated** | `tests/regression/test_import_contracts.py`, contract `search-engine-does-not-know-its-domains`. |
| 9 | `GameInterface` is vestigial | **False** | The `Protocol` at `src/mcts/search.py:85` is the boundary that contract depends on. |
| 10 | Dual-budget accounting is new | **Exists as data** | The arena CSV carries solves, cache hits and wall time; the sidecar reports three ratios. What is missing is a rule that claims report both. |
| 11 | `tests/claude`, `tests/docs`, `tests/security` and `device_planner` extract cleanly | **Pattern yes, code no** | §2; `tests/security` imports six `src` packages. |
| 12 | The tooling can fix a "2-star adoption problem" | **Premise true** | 2 stars, 0 forks. |
| 13 | Swarms are a new defence-grade capability | **Re-treads a cut domain** | `intercept` (MCTS missile defence, 6-DOF dynamics) was built, then removed on 2026-07-22 as "Domain PoC; not on the core solver path" (`spec.md:94-97`). |

---

## 4. What is real, and what is scaffold

The meta-analysis inventoried mechanisms and read them as capabilities. Separating the two:

| Component | Status | Evidence |
|---|---|---|
| Element-local AMR substrate and adequacy gate | **Real, measured** | The arena's policy metrics reproduce bit for bit; adaptive-vs-uniform separation is gated in CI. |
| Search engine (`src/mcts`) | **Real, correct, domain-free** — but contributes no decisions to the committed result | §1; single-agent backup, 90% coverage gate, import contract. |
| Operator transfer | **Real, honest** — zero retraining at ≈14× accuracy cost | Charter evidence rows at `spec.md:124-126`. |
| Neural codec, zoo, runtimes, BD-rate pipeline | **Real code, no committed result** | Coverage gate at 83; no `results/` artifact; the H.265 anchors are placeholders by their own label. |
| Codec MCTS rate control | **Scaffold with defects** | See below. |
| Swarm planning game | **Dormant code** | 644 LOC and 50 tests; no consumer, scenario or result; round-robin single-agent control; no learned evaluator. |
| PicoGK geometry | **Stub** | Raises on construction. |
| README performance table | **Unbacked claim** | No artifact; the hardware matches no documented rig. |
| Governance and evidence tooling | **Real, mutation-tested** | Dozens of guards with recorded kills, and eight recorded incidents of checks that ran green while measuring nothing. |

**The codec rate controller, specifically:**

- It is off by default (`use_mcts_rate_control=False`), and no training loop exists anywhere for its
  three networks — the only `backward()` calls on them are single-pass gradient-flow unit tests.
  Enabled, it searches randomly initialised models.
- `bits_used` and `target_bits_per_frame` are assigned (`rate_control.py:132-133`) and never read,
  so the rate budget is absent from the search state.
- `GOPPlanner.plan_gop` computes `_frame_target_bits` and discards it (`:472`); the underscore
  silences ruff's unused-variable rule. The planner is a per-frame loop.
- A first-visited leaf takes its value from the most recent prediction of a *different* node
  (`:226-228`), and children are created with their parent's state (`:264-270`).
- Q-values are not min-max normalised, while the value support spans ±25.
- The one test named for its purpose, `test_mcts_adapts_to_content`, is
  `pytest.skip("Requires trained MCTS model for meaningful results")`.

**The pattern.** Mechanisms keep shipping ahead of the evidence that they work: a controller with no
training loop, a game with no consumer, a stub behind an extra, a benchmark table with no run — and
now an arena whose search layer never changes a decision. The charter was written because *numbers*
outran evidence. The same failure recurs one level up, as *mechanisms* outrunning evidence, and a
strategy built by inventorying the tree — as both the meta-analysis and revision 1 were — inherits
it. The remedy is not more building. It is converting one mechanism into evidence, end to end.

---

## 5. Provenance defects (replacing revision 1's Finding C)

1. **A charter-cited sidecar fails the repo's own proposal-grade check.**
   `results/lshape_adaptive_vs_uniform.run.json`, cited at `spec.md:131`, records
   `git.dirty: true` and `config_hash: "unknown"`, and `assert_proposal_grade` rejects it. It
   predates that check, but it is neither re-recorded nor disclosed as a deviation.
2. **An unbacked, unguarded performance table.** `README.md:352-360`. The README evidence guard
   covers only AMR policy ratios. `tests/benchmarks/test_mcts_perf.py` already measures simulations
   per second, so the table can be backed rather than deleted.
3. **Seeds that cannot differ.** The arena's three seeds are identical by construction
   (`add_noise=False`, `temperature=0`; the sidecar records `l2_ratio_seed_std = 0.0`). A "median
   over 3 seeds" is one measurement.
4. **The binding limit is misreported.** Every citation says "policy `max_dof=600`", but the MCTS
   arm stops at `max_steps=12`, which is what sets matched DOF at 287.

---

## 6. Rewritten plan

### Constraints the plan is sized to

- **One maintainer.** `CODEOWNERS` routes every path to a single owner. The repo's own norm is one
  concern per PR and at most 300 diff lines where possible
  (`docs/ENGINEERING_REFLECTION_2026-09-11.md`); review time, not compute, is the bottleneck.
- **"Prefer deletion to abstraction"** is the repo's stated principle. The meta-analysis's SDK layer
  runs the other way.
- **The charter is supreme.** Scope and claim changes go through `openspec-change` and `claims-ledger`.
- **The `focus` gate** blocks core and frozen-track work in the same PR; either alone is legal.

The meta-analysis proposed three packages, an SDK refactor and three pivots in four months. At this
repo's review bandwidth that is likely closer to a year — spent before learning whether the search
does anything.

### Gate 0 — Correct the record (week 1 · five small PRs)

| # | PR | Why now | Exit criterion |
|---|---|---|---|
| 0.1 | Add a **single-element greedy arm** (`n_simulations=1`) and a `decisions_diverging_from_greedy` metric to the arena. Correct the claim — the charter's evidence row, Novelty text and frozen-tracks deviation row, `docs/FOCUS.md`, `README.md`, `CLAUDE.md` and the arena spec's "Win" row — via `openspec-change` + `claims-ledger`. Add a guard: a register row that attributes a result to look-ahead must cite a run with divergence above zero. | The cycle thesis is recorded as answered by a result that does not test it. | A planted row attributing 0.9532 to look-ahead turns a named test red. |
| 0.2 | Re-record `lshape_adaptive_vs_uniform` from a clean tree (hash-pin protocol), or disclose it as a deviation. | A charter-cited artifact fails the repo's own standard. | `assert_proposal_grade` passes on every register-cited sidecar, or a deviation row names the exception. |
| 0.3 | Back the README performance table with `test_mcts_perf.py` output plus a sidecar recorded on real hardware, or remove it. Extend the README guard to unit-bearing numbers (`ms`, `/sec`). | A live unbacked claim. | A planted unbacked `ms` figure turns a named test red. |
| 0.4 | Deprecate `use_mcts_rate_control` with a warning that says why (untrained, budget-blind), and remove it after the deprecation window. A codec-only PR is gate-legal, but the track is paused, so this is the owner's call. | A public API advertises a controller that does not control. | The warning is emitted and the removal is dated. |
| 0.5 | Retitle Noyron as an analytical-surrogate benchmark, with no geometry-ingestion claims. | A stub behind an extra. | No document implies live PicoGK ingestion. |

### Gate 1 — The go/no-go (from week 2; the decisive testbed is the long pole): can look-ahead beat greedy anywhere?

Pre-register it with `spec-new` on problems where greedy has a structural reason to fail. For
elliptic problems with a reliable estimator, adaptive-FEM optimality theory already shows that
estimator-driven marking with a small enough marking fraction reaches the optimal convergence rate —
consistent with what §1 observed — so the candidates are ranked by how much room they leave for
look-ahead:

| Testbed | Why greedy could be myopic | Cost on existing code | Role |
|---|---|---|---|
| A moving front (advection, or Burgers before shock formation) | The current residual is blind to where error *will* appear. This is the regime of VDGN's anticipatory refinement, which uses RL without an explicit tree — the sharpest statement of the novelty boundary. | High — needs a time-dependent substrate | **The real test** |
| Goal-oriented refinement for a quantity of interest | The primal residual is the wrong signal (the fair baseline is a dual-weighted greedy) | Medium | Secondary |
| Two reentrant corners of unequal strength under a hard DOF budget | Budget allocation across competing singularities | Low — a new geometry predicate on `SkfemTriSubstrate` | A two-day check that §1's null is not L-shape-specific |

Every testbed needs the same arms: greedy (single-element maximum marking); Dörfler at
θ ∈ {0.1, 0.3, 0.5}; and MCTS with `n_simulations` well above `top_k_actions`, so the tree can reach
three or more plies. Seeds must actually vary (root noise on, or perturbed initial meshes), or be
reported as n = 1. Every row reports both budgets.

**Pre-registered exit — written before any run:**

- **GO** if MCTS beats the *best* classical arm (not Dörfler at a single θ) at matched DOF on a
  majority of seeds, with `decisions_diverging_from_greedy > 0`, and a finite, stated break-even
  reuse count against greedy.
- **NO-GO** otherwise. The AMR look-ahead thesis closes honestly, with the result committed.

### Gate 2 — Conditional on Gate 1

- **If GO:** the project has a demonstrated method delta. The commercial framing becomes "planning
  compute for mesh efficiency", and the break-even reuse count is the buyer's test — the value holds
  wherever one mesh is reused more often than that count (parametric sweeps, design loops, digital
  twins).
  Only then is a second domain worth building: a *budgeted* rate-control game on `src/mcts`, with the
  budget in the state and real bit counts from the entropy model, preceded by a measured H.265
  reference from `FFmpegBaselineRunner`. Not the MuZero scaffold.
- **If NO-GO:** lead with what is verified — the evidence-governance tooling, and the operator's
  zero-retraining transfer at a stated accuracy cost. Both are true today.

Either way: **no** `src/planning` SDK or `PlanningEnvironment` protocol until two domains work. An
abstraction extracted from one working example and one scaffold encodes the scaffold's mistakes.

### Track T — one extraction, not four (parallel, from week 2)

With 2 stars, each spin-out is another repository for one maintainer to keep alive. Ship **one**,
chosen for generality: the **CI-gate integrity checker** (`tests/docs/test_coverage_gate_integrity.py`,
`tests/support/workflows.py`, `tests/support/marker_expr.py`). It answers "does this gate measure
anything, and can this job fail?" — a problem every CI user has — and the repo's recorded incidents
are ready-made case studies. Its repo-specific exemption tables need lifting into configuration.

Deferred until a user asks for them: the harness validator (`tests/claude`), `device_planner` (which
needs a generic entry type), the pickle-payload corpus, and the SBIR scaffold fork.

### Dropped

| Item | Reason |
|---|---|
| `src/mcts` → `src/planning` rename | 54 importing files, plus the mirror, charter and shape baseline, for nothing the import contract does not already give. |
| `PlanningEnvironment` / state-encoder abstraction | One working domain; the abstraction would precede the evidence. |
| "Unifying" the codec MCTS into `src/mcts` | It is a scaffold, not a duplicate. Deprecate it (0.4); build properly if Gate 2 calls for it. |
| Adaptive CI test selection | No code, and mature competitors. |
| The swarm pivot, now | It waits on Gate 1 and on decision 5. If un-deferred, the one-week falsifier is an operator evaluator trained at N=4 and evaluated at N=16, reporting the accuracy penalty the transfer benchmark predicts. |

---

## 7. Owner decisions

Only these need you; everything else is sequenced.

1. **Accept the arena-claim correction (0.1).** The most important decision here, because it changes
   what the charter says the project has shown.
2. **Choose the Gate 1 testbed and pre-register its thresholds** before any run.
3. **Codec MCTS:** deprecate then remove (recommended), or keep it with a disclosed-defects deviation row.
4. **Which single extraction**, if any, before Gate 1 closes.
5. **Is defence robotics in scope?** `intercept` was cut once for being off the core path. A swarm
   product is a charter Purpose amendment, not an engineering task.

---

## 8. Bottom line

The meta-analysis is right that this is not a faster PDE solver, right that the governance tooling is
the most liquid asset, right about hardware provenance and adoption, and right to keep the monorepo.
Revision 1 of this review corrected several of its facts — and got its own central finding wrong.

What neither saw is that **the result every strategy was built on does not contain the search it is
attributed to.** The published MCTS trajectory is greedy marking, decision for decision. That does not
end the project. It means the thesis has not been tested yet. The cheapest test takes days on
existing code; the decisive one — a moving front — needs a time-dependent substrate, and is still
smaller than any pivot on the table. Every pivot should wait for that answer.
