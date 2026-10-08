# Proposal: `lookahead-vs-greedy`

## Why

The cycle thesis — **MCTS multi-step look-ahead beats classical greedy marking** — has never been
tested, and the committed evidence cannot test it.

`results/mcts_classical_amr_arena.csv` is the only scored MCTS-vs-classical artifact. Gate 0.1
(`src/research/greedy_control.py`) added the single-element greedy control and the
`decisions_diverging_from_greedy` counter, and the merged arena reads **divergence 0, MCTS/greedy
1.0, greedy/Dörfler 0.9532**: the published MCTS trajectory is the greedy trajectory, decision for
decision. Its ratio against Dörfler measures marking granularity, not search
(`docs/business/COMMERCIALIZATION_PEER_REVIEW.md` §1). Two settings leave the search no room: a
legal set pre-ranked by the same indicator the prior is built from, and too few simulations per
candidate for PUCT to revisit.

The peer review's Gate 1 (§6) asks for the one experiment the thesis needs: problems where greedy
has a structural reason to fail, a search with room to grow several plies, the *best* classical
policy as the reference rather than one Dörfler θ, seeds that actually vary, and a go/no-go written
down before any run. The second testbed it needs already exists: the Z-tetromino with two reentrant
corners of unequal strength (`zshape_poisson`, merged in C1).

## What Changes

1. **Pre-registration** — `specs/lookahead_vs_greedy.spec.md`, committed before any comparison run:
   question, testbeds (T1 L-shape, T2 Z-tetromino `c1 = 1.0`, `c2 = 0.25`, T3 moving front
   deferred), arms, budgets, the primary metric, the break-even formula, the depth definition and
   the four GO criteria.
2. **Realized tree depth from `src/mcts`** — a read-only `MCTS.root` property and a pure
   `subtree_depth(node)` in `src/mcts/node.py`. Search behaviour is unchanged.
3. **Arena config accepts the Z testbed** — `operator_name` gains `zshape_poisson`; the adequacy
   validator accepts it with its own gate, and `adequacy_gate()` becomes operator-aware through
   `adequacy_gate_for_operator`. L-shape behaviour and the committed arena artifact are unchanged.
4. **Harness** — `src/research/lookahead_vs_greedy.py` (arms) and
   `src/research/lookahead_vs_greedy_metrics.py` (matched-DOF reading against the best classical arm,
   first-passage break-even, verdict), composing the Phase A primitives: `build_arena_episode` /
   `run_arena_episode`, `greedy_action` / `GreedyDivergence`, `run_classical_arm`,
   `compare_trajectories`, `measure_adequacy` / `gate_violations`, `RunRecorder`. The MCTS arms are
   a `DecisionRule` closure over `MCTS.get_action`; root noise lives in the rule, not the game.
5. **Scenario + CLI** — `@scenario("lookahead_vs_greedy")`, `LookaheadVsGreedyConfig` (pre-registered
   values as defaults; validators for the invariants), two YAMLs, and
   `scripts/run_lookahead_vs_greedy.py`, whose exit code reports aborts and errors but never the
   verdict.
6. **Capability register** — one row, `lookahead_vs_greedy`, in the charter's capability region.
7. **Artifacts** — proposal-grade `results/lookahead_vs_greedy_{lshape,zshape}.{csv,png,run.json}`
   and a regenerated `results/MANIFEST.sha256`.
8. **Reporting, after the runs (task 4)** — two evidence-register rows (NO-GO on T1 and T2), the
   Novelty paragraph and the frozen-tracks deviation row state the result; `README.md`,
   `docs/FOCUS.md`, the arena spec and the peer review's status table follow; and
   `tests/docs/test_lookahead_attribution.py` reads Gate 1 sidecars through their own divergence
   metric (`primary_decisions_diverging_from_greedy`), so a Gate 1 claim must carry the
   "search contributed no decisions" label exactly as an arena claim must.

## Impact

- Charter: the *Capability Register Accuracy* Requirement gains one register row, landed with the
  code. After the runs, *Evidence-Backed Claims* gains two rows, and *Novelty Claim Discipline*
  and *Accepted Deviation Disclosure* state the result (see the delta). The result's wording was
  written after the runs, from the sidecars; no threshold or pre-registered value changed.
- `src/mcts/search.py` stays at its 716-line size budget: the `root` property's lines are offset by
  condensing `get_root_value`'s docstring.
- The shape ratchet's lazy-import count is unchanged: the config dispatch for the new scenario adds
  one lazy import to `load_config_from_dict` (its established pattern), and a redundant lazy import
  in `src/research/experiment.py` (of a module that file already imports at module level) is
  hoisted.

## What this change does NOT do

- It does not build T3 (the moving front), the testbed the plan calls decisive.
- It does not train an evaluator, tune any search knob, or change a threshold after a result.
- It did not write the charter evidence, novelty or deviations rows, `README.md`, `docs/FOCUS.md`,
  `CLAUDE.md` or `CHANGELOG.md` before the runs. The result's wording was proposed in the run
  report and landed afterwards through `claims-ledger` (task 4).
- It does not change `MCTS`'s default `search_mode`, the arena's scored locks, or the committed
  arena artifact.
- A NO-GO on T1/T2 does not close the thesis; a GO needs replication before any claim.
