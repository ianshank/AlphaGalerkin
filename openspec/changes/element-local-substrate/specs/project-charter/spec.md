# Delta: `project-charter` — element-local substrate

This change modifies the **Scope Integrity** and **Accepted Deviation Disclosure**
Requirements. The other six are untouched.

## MODIFIED Requirements

### Requirement: Scope Integrity

The scope register gains the modules this change adds. No package is removed, and no non-goal
is re-entered.

Added to the scope register (domain `pde`):

| Package | Domain | Note |
| --- | --- | --- |
| `src/research/substrates/` | pde | Concrete refinement substrates. `skfem_tri.py` is in the coverage `omit`, matching `fem_baseline.py`, because its tests require the optional `[fem]` extra. |

`src/refinement/` is **already** in the register and gains modules rather than entering it.

#### Scenario: The substrate package is added without a charter row
- GIVEN `src/research/substrates/__init__.py` exists on disk
- WHEN the scope guard runs
- AND the scope register does not list it
- THEN the guard SHALL fail naming the package

### Requirement: Accepted Deviation Disclosure

One deviation is **retired** and one is **added**, the latter explicitly time-boxed.

**Retired.** *"`RefinementGameRegistry` has zero runtime registrants"* — this change delivers
its first, so the deviation no longer describes reality. Its removal is asserted by a test that
the registry contains the new game after its `register_games` module is imported: the deviation
may only be removed once something actually registers.

**Added, time-boxed.**

| Deviation | Reason |
| --- | --- |
| Two refinement game/adapter paths coexist (`src/pde/games/lshape_amr.py` + `PDEGameAdapter`, and the new substrate path on `src/refinement/`) | The legacy pair is **frozen as the back-compat golden reference** — its bitwise reproduction of `results/lshape_mcts_vs_dorfler.csv` is what proves the substrate abstraction changed no behaviour. Deleting it would remove that proof; extending it would fork development. **Retirement condition:** when the golden test is its only remaining consumer, the legacy path is deleted and this row is removed. |

The retirement condition is stated because a deviation without one is a permanent exemption
wearing a temporary label — the failure mode the *Non-Goal Exclusion* Requirement exists to
prevent at package scope, applied here at module scope.

#### Scenario: A deviation is added without a retirement condition
- GIVEN a deviation row describing a temporary state
- WHEN the deviation guard runs
- AND the reason cell states no condition under which the row is removed
- THEN this is a disclosure failure, and the row SHALL be rewritten or the state made permanent

> **Note on guard coverage.** The existing guard checks that every deviation states a *reason*
> of at least 20 characters. It does not — and this change does not make it — check for a
> retirement condition, because "states a condition" is not mechanically decidable from prose.
> The scenario above is a **review** obligation, recorded honestly as such rather than claimed
> as enforced. Claiming otherwise would be exactly the kind of unguarded assertion this charter
> was written to stop.
