# AlphaGalerkin — OpenSpec Project Context

This directory is AlphaGalerkin's [OpenSpec](https://github.com/Fission-AI/OpenSpec) tree:
`specs/` holds the source of truth for current governing behaviour, `changes/` holds proposed
modifications as self-contained change packages.

## Document precedence

AlphaGalerkin carries several documents that each claim authority over some slice of the
project. When two disagree, resolve in this order:

| Rank | Document | Owns |
| --- | --- | --- |
| 1 | [`openspec/specs/project-charter/spec.md`](specs/project-charter/spec.md) | **The charter.** Mission, scope, non-goals, the novelty claim, the evidence standard, accepted deviations. Supreme — on any conflict, the charter wins. |
| 2 | [`ARCHITECTURE.md`](../ARCHITECTURE.md) | Repository *layout* — the package map, layering, naming gotchas. The charter delegates layout detail here and asserts equality with it. |
| 3 | [`CLAUDE.md`](../CLAUDE.md) | Agent operational context: the append-only milestone log and the Regression Surface table. |
| 4 | [`specs/`](../specs/README.md) | Per-feature contracts — data contract, acceptance criteria, and `MetricThreshold` pass/fail values. |

The charter is deliberately **thin and referential**. It does not restate the package map, the
coverage gates, or the prior-art analysis — it names the owner and asserts agreement, so there is
exactly one place to edit when reality changes. A charter that copied those registers would rot
faster than the documents it governs.

## Two spec systems, on purpose

`specs/*.spec.md` (the repo's own format, described in [`specs/README.md`](../specs/README.md))
remains the home of **per-feature** contracts. Its thresholds reuse the canonical
`src.poc.config.MetricThreshold`, and `CONTRIBUTING.md`, the `spec-new` skill, and CI all
reference it. That system is unchanged.

`openspec/` governs **project-level** questions: what is in scope, what is explicitly not, and
what makes a claim admissible. Migrating the per-feature specs into `openspec/` is deliberately
out of scope.

## Conventions inherited from the repo

- **Pydantic configs, no hardcoded values.** Every tunable is a typed `Field(default=…, <bounds>,
  description=…)`; numerical-stability literals become named module constants.
- **Registries over imports.** Scenarios, PDE operators, games, and losses register through
  thread-safe singleton registries. Note that decorators take module *constants*
  (`@scenario(SCALING_SCENARIO_NAME)`), so registry contents must be **enumerated at runtime**,
  not grepped — a string-literal grep finds 4 of the 10 scenarios.
- **structlog everywhere**, never `print()`.
- **Executable documentation.** Where a document makes a checkable assertion, it carries a
  delimited region and a guard test. Precedents: `tests/docs/test_architecture_map.py`,
  `tests/regression/test_related_work_guard.py`, `tests/hf_space/test_mirror_guard.py`.

## Layout

```
openspec/
├── project.md                          # this file
├── specs/
│   └── project-charter/spec.md         # the charter (current truth)
└── changes/
    └── project-charter-alignment/      # the change that introduced it
        ├── proposal.md
        ├── design.md
        ├── tasks.md
        └── specs/project-charter/spec.md   # delta
```

## Validation

```bash
npx openspec validate --strict          # optional; network-gated
pytest tests/docs/test_charter_alignment.py -v   # the charter's guards
```
