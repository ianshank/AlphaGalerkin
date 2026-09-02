# Delta: `project-charter` — element-local substrate

This change modifies the **Scope Integrity** and **Accepted Deviation Disclosure**
Requirements. The other six are untouched.

## MODIFIED Requirements

### Requirement: Scope Integrity

`src/refinement/` is **already** in the scope register and gains modules rather than entering it.
No package is removed, and no non-goal is re-entered.

**Amended 2026-09-02**: this Requirement originally instructed adding a
`src/research/substrates/` row to the scope register. That is now known to be impossible without
breaking the guard it was meant to satisfy, and the row is dropped rather than added.

`tests/docs/test_charter_alignment.py`'s scope register is **top-level-package granular**:
`_ARCH_ROW_PACKAGE = re.compile(r"\|\s*`src/([a-z0-9_]+)/`\s*\|")` — the character class excludes
`/`, so a nested path cannot match the pattern the guard parses — and its on-disk comparison set
is built from `SRC.glob("*/__init__.py")`, one level deep only. A `src/research/substrates/` row
fails `assert not extra` (the register would name a package the glob never finds) and
`assert not phantom` (the register would claim a package with no `src/<name>/__init__.py`) in
both directions, on a guard that is otherwise green. Widening that guard to nested paths is a
larger, separate change to a load-bearing scope/claims-drift check, not a one-line row addition.

The concern the row was meant to address is already covered: `src/research/substrates/` is a
subpackage of `src/research/`, which already has its own row in the register. Nothing added by
this change sits outside the scope that row already claims.

#### Scenario: The substrate package is a subpackage of an already-scoped package
- GIVEN `src/research/substrates/__init__.py` exists on disk
- AND the scope register lists `src/research/` (domain `pde`)
- WHEN the scope guard runs
- THEN the guard SHALL pass without a dedicated `src/research/substrates/` row

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
