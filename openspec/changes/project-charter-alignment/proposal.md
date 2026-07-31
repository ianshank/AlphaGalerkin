# Proposal: `project-charter-alignment`

## Why

AlphaGalerkin had no charter. Mission, scope, non-goals, and the standard for an admissible claim
were spread across `README.md`, `ARCHITECTURE.md`, `CLAUDE.md`, `AGENT.md`, `specs/`, and
`docs/business/` — six documents with no reconciliation, drifting independently.

The drift was not cosmetic. An audit (three independent reviewers: adversarial plan review,
exhaustive drift sweep, guard-test design) found four P0 defects — **claims the repository's own
committed artifacts contradict**:

1. `CLAUDE.md` carried a **retracted** AMR headline (~11–14% win, ~15–55× wall-clock) that
   `specs/lshape_amr_compare.spec.md` had already corrected to ~4% / ~350×, produced by the F0
   two-player-backup defect.
2. `README.md` advertised a transfer MSE of ≈4e-4 while citing a spec whose committed artifacts
   say ≈2.3e-3 — the favourable number came from an uncommitted spike, and had propagated into
   nine outward-facing SBIR documents.
3. `docs/business/proposal/concept_note.md` asserted a headline Pareto plot was *"archived at"* a
   path that does not exist, in the same sentence as *"no numerical performance claim … is not
   traceable to that artifact."*
4. `CLAUDE.md` documented the `video_compression` subsystem — deleted 2026-07-22 — as live, across
   five milestones naming eight non-existent paths.

This is the same failure mode as the fabricated `0.000209 / 240×` figure retracted on 2026-07-22.
That correction was right but reactive, and the pattern recurred within months. Prose will not
catch the next one.

## What Changes

**Content first.** Every P0 and P1 defect is corrected before the charter lands, so the charter
never certifies broken documentation.

**Then a charter that is executable.** A new `openspec/` tree with
`openspec/specs/project-charter/spec.md` as the supreme governing document — seven Requirements in
OpenSpec format, each carrying a delimited machine-readable register and a guard test in
`tests/docs/test_charter_alignment.py`.

The charter is deliberately **thin and referential**. It asserts equality with existing owners
(`ARCHITECTURE.md` for layout, `ci.yml` for gates, the scenario registry for capabilities) rather
than copying them. A charter that duplicated those registers would rot faster than what it governs
— and would recreate exactly the drift it exists to prevent.

**Then the guards get armed.** `scripts/check_doc_links.py` runs today only when `docs/**` or
`src/**` changes, so a PR touching only `CLAUDE.md`, `README.md`, or `specs/**` never triggers it.
That is fixed.

## In Scope

- Correcting the four P0 and the P1/P2 documentation defects the audit surfaced.
- The `openspec/` tree: `project.md`, the charter, and this change package.
- `tests/docs/test_charter_alignment.py` — one guard per Requirement.
- Widening the existing cut-module guard from `hf_space/` to the whole repository.
- Arming `check_doc_links.py` on docs-only PRs.
- Authority wiring: adding the charter to `ARCHITECTURE.md`'s documentation-hierarchy table and
  pointing the other governing documents at it.

## Out of Scope

- Rewriting `CLAUDE.md`'s milestone history. Entries receive correction and removal banners; the
  append-only record stands.
- Migrating the six existing `specs/*.spec.md` into `openspec/`. They keep owning per-feature
  `MetricThreshold` contracts.
- `hf_space/` single-sourcing — an explicit charter non-goal.
- Writing the missing `src/refinement/AGENT.md` and `src/alphagalerkin/AGENT.md`.
- Resolving the empty `RefinementGameRegistry` dead abstraction — disclosed in the charter's
  deviation register, fixed separately.
- Writing `scripts/train_distributed.py`. The never-runnable commands are removed and the gap is
  stated plainly; building the launcher is separate work.
- Any change to `src/` behaviour.

## Impact

- **Affected docs:** `README.md`, `CLAUDE.md`, `AGENT.md`, `CONTRIBUTING.md`, `ARCHITECTURE.md`,
  `specs/README.md`, and nine files under `docs/`.
- **Affected tests:** one new file under `tests/docs/`; one shared constant promoted out of
  `tests/hf_space/test_mirror_guard.py`.
- **Affected CI:** `.github/workflows/docs.yml` path filter widened. No new job; the fast test job
  already collects `tests/docs/`.
- **Risk:** low. No `src/` behaviour changes. The one item with real blast radius — resolving
  repo-path-shaped inline code spans in the link checker — is gated behind a glob/allowlist design
  and can be dropped without losing the rest.
