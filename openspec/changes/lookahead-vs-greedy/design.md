# Design: `lookahead-vs-greedy`

Only the decisions a reviewer could reasonably disagree with.

## D1 — The reference is the best classical arm, read at one shared matched DOF

**Decision.** Per MCTS run, `N*` is the largest DOF *every* arm reached (the run plus the five
classical arms), and the reference is the minimum of the five classical L2 values at `N*`.

**Reason.** A pairwise matched DOF per classical arm would compare MCTS with different arms at
different DOF, and the minimum over those ratios has no single meaning. One `N*` makes "beats every
classical policy at this mesh size" literal. The classical arms run to a DOF budget well above what
30 single-element steps reach, and the harness *checks* that they span the game-driven arms (an
error otherwise), so `N*` is set by the game-driven arms and not silently shrunk by a classical
budget.

## D2 — Root noise lives in the decision rule

**Decision.** Both MCTS arms play the arena's game, built from an `MCTSClassicalAMRArenaConfig`
that satisfies its scored locks (`add_noise=False`, `temperature=0`). The robustness arm passes
`add_noise=True` to `MCTS.get_action` inside its `DecisionRule`.

**Reason.** The game, the stopping rules and the point recording must be byte-for-byte the ones
greedy uses (`build_arena_episode`). Relaxing the arena config's lock to admit noise would weaken
the lock that protects the committed arena result.

## D3 — Realized depth counts visited nodes only

**Decision.** `subtree_depth(node)` is the largest number of edges from `node` to a descendant with
`visit_count > 0`.

**Reason.** `expand` creates every child of a leaf it evaluates, with only a prior; no state was
solved at those children. Counting them would report one ply more than the search evaluated — with
one simulation per step it would report depth 2 for a search that looked one action ahead.

## D4 — Break-even at equal accuracy, by first passage

**Decision.** `ε* = max(L_mcts(N*), L_greedy(N*))`; each arm's DOF and wall-clock at `ε*` are read
at the first segment of its trajectory that reaches `ε*`, log-log in error and DOF, linear in
wall-clock, with one shared crossing fraction.

**Reason.** Taking the *worse* of the two errors guarantees both arms reach `ε*` inside their
recorded trajectories, so the DOF saving is always finite and never extrapolated. First passage (not
`np.interp` on error) is correct when an L2 trajectory is not monotone. The downstream cost model
`t_α(N) = κ_α N^α` is calibrated from greedy's measured per-step time, as the peer review asks; the
GO criterion is gated on the DOF saving, which is equivalent to a finite `K*` for every α > 0 and
does not depend on wall-clock noise.

## D5 — The verdict is the thresholds, evaluated once

**Decision.** The GO/NO-GO verdict is `all(threshold.evaluate(metrics[name]))` over
`LookaheadVsGreedyConfig.get_default_thresholds()` — the same `MetricThreshold`s the PoC framework
evaluates for pass/fail.

**Reason.** Two implementations of the criteria would be two places a threshold could drift. The
thresholds are typed fields whose defaults are the pre-registered values, so a change is a visible
config diff and moves the config hash.

## D6 — Exit code is not the verdict

**Decision.** `scripts/run_lookahead_vs_greedy.py` exits 0 on GO and on NO-GO; 3 on an adequacy
abort, 2 when a run cannot be proposal-grade, 1 on any other error.

**Reason.** A NO-GO is a research outcome, not a process failure. The arena CLI's
"exit 1 iff the gate fails" convention would make an honest negative look like a crash.

## D7 — A module-level `poc → research` import in the new scenario

**Decision.** `src/poc/scenarios/lookahead_vs_greedy.py` imports the harness at module level.

**Reason.** The shape ratchet forbids new lazy first-party imports. The poc↔research cycle the
older compare scenarios avoid with lazy imports needs a *runtime* `research → poc` edge to close,
and the harness has none: its poc imports are `TYPE_CHECKING`-only. A test pins that importing the
harness loads no `src.poc` module, so the module-level import cannot become a cycle unnoticed.

## D8 — Size budgets are kept, not raised

**Decision.** `src/mcts/search.py` (716/716) gains its `root` property by condensing the adjacent
`get_root_value` docstring; `src/research/mcts_classical_amr_arena.py` (602/628) does not grow.

**Reason.** The budgets are frozen for this cycle; raising one is a reviewer-visible decision this
change does not need.
