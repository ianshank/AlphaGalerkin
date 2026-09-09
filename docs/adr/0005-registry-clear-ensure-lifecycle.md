# 0005. Process-global registry `clear` / `ensure` lifecycle

- **Status:** Accepted
- **Date:** 2026-09-09
- **Deciders:** @ianshank

## Context

`src.templates.registry.create_registry` builds a process-global singleton.
`register()` raises `ValueError` on a duplicate name. `clear()` empties the
map and is documented as a test-isolation tool.

Importing a registrant module runs `@register_*` **once**. A later
`clear()` leaves `Available: []`. A second `import` of the same module is a
no-op, so an `ensure_*` helper that only imports cannot recover.

That combination failed default-tip CI on `a85d265` ([run
34292047225](https://github.com/ianshank/AlphaGalerkin/actions/runs/34292047225)):
`tests/refinement/test_substrate.py` cleared `RefinementSubstrateRegistry`
without restoring; later fast-lane tests called
`ensure_substrate_registrants()` and still saw an empty map
(`KeyError: 'tensor_grid' not registered`). The same process-global shape
already bit `ScenarioRegistry` (documented in `tests/poc/conftest.py` and
hygiene B16): several modules `clear()` and never restore, and a
package-level snapshot/restore fixture made the end state *worse* because
those suites also purge `sys.modules`.

This ADR records the contract for **new and production `ensure_*` helpers**
and for tests that `clear()`. It does **not** rewrite every registry in
the tree.

## Decision

We will treat `clear()` on a process-global registry as a poison unless a
later `ensure_*` can rebuild the production map.

1. **`ensure_*` is a production path, not a test helper.** It must be
   idempotent: calling it when the kinds are already present is a no-op.
   Calling it after `clear()` must re-register every production kind the
   helper owns. Import-only ensure is not sufficient once the registrant
   modules are in `sys.modules`.
2. **Re-register only missing names.** `register()` raises on duplicates, so
   the helper checks `registry.get(kind) is not None` before decorating.
   Unconditional re-register is a defect: it crashes the first `ensure`
   after the decorator has already fired.
3. **Poisoner tests restore.** A suite that `clear()`s a production
   registry restores in teardown by calling the same `ensure_*` (not by
   importing). Restore-in-teardown is the neighborly half; `ensure_*` remains
   the production recovery so a missed teardown cannot empty the rest of
   the process.
4. **Do not write a package-level snapshot/restore for a registry whose
   tests also purge `sys.modules`.** That is the B16 lesson
   (`ScenarioRegistry`). Per-test restore of a partial snapshot removes
   registrations the test legitimately created. Subprocess reads stay the
   correct workaround until those local fixtures are reworked.
5. **Do not rewrite all registries in the change that learns this.** New
   `ensure_*` helpers follow this contract. Existing `ScenarioRegistry` /
   `AgentRegistry` / `BACKEND_REGISTRY` fixtures stay until their own
   suites are the subject of a change.

Enforced for substrates by
`tests/research/substrates/test_factory_registry_lookup.py::test_ensure_registrants_re_registers_after_clear`
(primes, then `clear()`, then `ensure`; the first `ensure` is load-bearing
so the test cannot pass on an import-only body when it runs first).

## Consequences

- Production lookups (`build_substrate_from_config`,
  `src.pde.register_refinement_games`) survive a prior test's `clear()`.
- Adding a new substrate kind means adding it to
  `ensure_substrate_registrants` (and the `SUBSTRATE_KIND_*` constants),
  not relying on a decorator that already ran.
- A test that `clear()`s and does not restore is now a documented defect
  class, not an isolation style.
- Generalizing this into `BaseRegistry.ensure` on `src.templates.registry`
  is a follow-up, not this decision. A shared helper that does not know
  its registrants cannot re-register them.

## Alternatives considered

- **Import-only ensure.** Cheaper, and what shipped. A second import is a
  no-op; this is the failure that emptied `RefinementSubstrateRegistry`.
- **Ban `clear()` in tests.** Would force every isolation test to use a
  private registry instance. `create_registry` is a singleton by design;
  the poison is real and the tests that exercise `clear()` are the ones
  that prove it. Restore + ensure is the smaller contract.
- **Package-level snapshot/restore autouse.** Tried for
  `ScenarioRegistry` and reverted (B16). Not repeated here.
- **Rewrite every registry now.** Out of scope. The substrate lookup is
  the first non-test reader of `RefinementSubstrateRegistry`; that is why
  it had to be production-correct. Other registries still have only
  test-side readers and keep their existing fixtures.
