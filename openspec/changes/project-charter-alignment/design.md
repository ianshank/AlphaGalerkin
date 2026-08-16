# Design: `project-charter-alignment`

## Technical Approach

The charter is only worth writing if it is enforceable. The design principle is therefore:
**every Requirement carries a delimited machine-readable register and exactly one guard test.**
OpenSpec's `#### Scenario:` GIVEN/WHEN/THEN blocks map 1:1 onto the repository's existing AQA test
convention (`specs/TEMPLATE.spec.md` already uses Given/When/Then), so the mapping is native, not
bolted on.

## Architecture Decisions

### AD1 — The charter is thin and referential, not a container of copies

**Decision.** The charter asserts *equality with an existing owner* rather than restating it.

**Rationale.** `ARCHITECTURE.md:30-39` already ships a documentation-hierarchy table naming five
owners. A charter that re-listed the package map, the coverage gates, and the prior-art analysis
would create a fifth source of truth for each — three-way sync cost for zero new detection, and a
new drift surface. The charter's scope register is asserted equal to `ARCHITECTURE.md`'s package
map; the map remains bound to disk by the existing `tests/docs/test_architecture_map.py`. One root
cause, one failure.

**Consequence.** "Supreme" means *wins on conflict*, not *contains everything*.

### AD2 — Registry truth is read at runtime, in a subprocess

**Decision.** The capability guard shells out to read `ScenarioRegistry().list_scenarios()`.

**Rationale.** Two independent hazards.

*Grep is wrong.* Scenarios register as `@scenario(SCALING_SCENARIO_NAME)` — a module constant. A
string-literal grep finds 4 of the 10 registered scenarios. Static AST analysis would need
cross-module constant resolution (the constant lives in the sibling `*_config.py`) and would still
only approximate runtime truth.

*In-process reads are order-dependent.* `ScenarioRegistry` is a process-wide singleton, and nine
`tests/poc/*` modules call `ScenarioRegistry().clear()` in autouse fixtures **with no teardown**;
two of them also purge `src.poc.scenarios*` from `sys.modules`, so a re-import cannot restore the
registrations. Measured with a probe test, not assumed:

| Invocation | What an in-process read sees |
| --- | --- |
| `pytest tests/poc tests/docs` | 10 — recovers, because a later `tests/poc` module re-imports |
| `pytest tests/poc/test_complexity_scenario.py tests/poc/test_registry.py <probe>` | **0** |

The failure is therefore *latent and selection-dependent*: green for the full suite, red for a
contributor running a subset, and unpredictable under `pytest-randomly` or xdist. That is worse
than a consistent failure, because nothing surfaces it until it misleads someone. Relocating the
test does not help — `tests/regression/` sorts *after* `tests/poc/`.

**Consequence.** ~3.9 s cost, hermetic regardless of ordering, and `tests/docs/` stays
stdlib-pure at collection time.

### AD3 — Non-goal enforcement checks directory existence only

**Decision.** The non-goal guard asserts `not (SRC / name).is_dir()`. It does **not** grep source
for the cut module names.

**Rationale.** The names collide with live vocabulary: `vertex` has 12 legitimate hits in
`src/pde/games/mesh_refinement.py`, `intercept` has 10 in `src/research/`. Even a
word-boundaried import-form regex self-hits `tests/hf_space/test_mirror_guard.py`, whose own
comment documents the `"src.thermo" in "src.thermodynamics"` collision. A guard that cries wolf
gets suppressed, and a suppressed guard protects nothing.

### AD4 — No guard may depend on git history

**Decision.** Provenance for the 2026-07-22 cut is `CHANGELOG.md` and the `CLAUDE.md` milestone,
never the `archive/pre-core-cut-2026-07-22` tag.

**Rationale.** CI checks out shallow with no tags (`ci.yml` uses bare `actions/checkout@v4`);
`git tag -l` is empty in a fresh clone. A guard resolving that tag would fail everywhere except a
full local clone.

**Consequence.** `results/lambda_scheduling.{csv,png}` are **kept**, not deleted — with the tag
unresolvable they are the only in-tree evidence of that negative result, and `ARCHITECTURE.md`
already declares changelog-referenced artifacts deliberate.

### AD5 — Guards are pruned to those that always mean something

Rejected, with reasons: a charter↔disk scope check (duplicates `test_architecture_map.py`); any
date/expiry column on deviations (turns red on a calendar day with no code change); the CI ⊆
charter direction on gates (adding a CI gate would nag a charter edit for no protection); an
unbounded "all backticked paths must exist" (belongs in the repo-wide link checker); mission
keyword assertions and GIVEN/WHEN/THEN structural assertions (ceremony).

Four guards that always mean something beat eight that get `# noqa`'d.

## Data Flow

```
ARCHITECTURE.md package-map region ─┐
src/*/__init__.py ──────────────────┼─► R1 scope guard
charter charter:scope region ───────┘

charter charter:non-goals region ───► R2 ─► (SRC/name).is_dir() must be False

charter charter:evidence region ────► R3 ─► brace-expand ─► Path.exists()

charter text ───────────────────────► R4 ─► retracted-string absence

subprocess: import src.poc.scenarios
            ScenarioRegistry().list_scenarios() ─┐
charter charter:capabilities region ─────────────┴─► R5 set equality

.github/workflows/ci.yml steps ─────┐
charter charter:gates region ───────┴─► R6 charter ⊆ CI

charter charter:deviations region ──► R7 ─► Reason cell non-trivial
```

## Parsing Strategy

Markdown tables inside `<!-- charter:<name>:start -->` / `:end` regions. `### Requirement:`
headings are parsed only by the meta-guard. GIVEN/WHEN/THEN is never machine-read — it is prose
for humans, and each Scenario's *enforcement* is the corresponding guard function.

Pitfalls handled explicitly:

- **Strip fenced code blocks first.** Otherwise an example table inside a fence parses as data.
- **Anchor on the literal `^### Requirement:`** — `^###` also matches `####`.
- **Do not require a backticked token in the first cell.** An earlier draft did this (mirroring
  `_ROW_PACKAGE` in `test_architecture_map.py`, which can afford to because every real row there
  starts with a backticked package path) — but two of the six registers here (evidence,
  deviations) key on a *prose* claim/deviation label with no backtick, and a parser that
  silently skipped non-backticked rows would make exactly those two guards vacuous. Instead,
  the header/separator row is dropped **structurally**: find the `| --- |` separator and take
  everything after it; if no separator is present, the first row is a header and everything
  after it is data — and if there is no "everything after it" (a table of only a header, every
  real row deleted), that must return `[]`, not the header cells reinterpreted as data. An
  earlier version of this fallback returned the header itself in that single-line case, which
  would have let a fully-emptied `deviations` table pass R7 silently (that register has no
  external source of truth to cross-check against, unlike scope/non-goals/capabilities/gates —
  each of those would catch a phantom row as an unexpected "extra"). Fixed before merge; the
  shipped `_row_lines` in `tests/docs/test_charter_alignment.py` implements this correctly.
- **Brace expansion is single-level**, then assert no `{` remains — fail loudly rather than
  silently skipping. Order-agnostic: `CHANGELOG.md` writes `{png,csv}`, specs write `{csv,png}`.
- **Reject absolute and `..` paths** before `exists()`.
- **Never `parametrize` over parsed rows.** An empty parse would collect zero tests and report
  green. Aggregate into a sorted failure list instead, and add a meta-guard asserting every region
  parses non-empty with its markers appearing exactly once.

## File Changes

| File | Change |
| --- | --- |
| `openspec/project.md` | New — conventions + document precedence |
| `openspec/specs/project-charter/spec.md` | New — the charter, R1–R7 |
| `openspec/changes/project-charter-alignment/*` | New — this change package |
| `tests/docs/test_charter_alignment.py` | New — one guard per Requirement + 2 meta-guards |
| `tests/support/cut_modules.py` | New — `CUT_MODULES` promoted to a shared constant |
| `tests/hf_space/test_mirror_guard.py` | Import the shared constant instead of redeclaring |
| `scripts/check_doc_links.py` | Resolve repo-path-shaped inline code spans (glob + allowlist) |
| `.github/workflows/docs.yml` | Widen `paths:` so docs-only PRs run the link checker |
| `ARCHITECTURE.md` | Add the charter to the documentation-hierarchy table |
| `CLAUDE.md`, `README.md`, `CONTRIBUTING.md`, `specs/README.md`, `docs/README.md` | Point at the charter |
| P0/P1/P2 content fixes | 15 documents — see `tasks.md` §1–§2 |

## Alternatives Considered

**A root `CHARTER.md` instead of `openspec/`.** Rejected — the user selected canonical OpenSpec
adoption, and the `openspec/` tree keeps the project-level charter cleanly separated from the
per-feature `specs/` system rather than competing with it in the same directory.

**An ADR for the scope decision.** `docs/adr/` is a genuinely good home for the cut-to-the-core
decision, and an ADR would be immutable and supersedable. Deferred rather than rejected: the
charter's non-goal register already carries the decision and its reasons, and splitting it across
two documents now would reintroduce the multi-owner problem this change exists to solve.

**Deleting `results/lambda_scheduling.*`.** Rejected — see AD4.
