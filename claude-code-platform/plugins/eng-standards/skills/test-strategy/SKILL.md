---
name: test-strategy
description: House testing standards covering the test pyramid, per-module coverage gates, Hypothesis property-based testing, subprocess contract tests for CLIs, fixtures, deterministic tests, and marker-gated GPU/external tests. Use when writing tests, adding a module, reviewing test coverage, or deciding what kind of test a change needs — trigger phrases include "add tests", "write a test for", "coverage gate", "is this tested", "test this CLI", or "flaky test".
---

# Test Strategy

## Test Pyramid

- **Many unit tests**: pure logic, config validation, single classes with injected fakes. Fast (<100ms each), no I/O, no network.
- **Fewer integration tests**: real interfaces wired together (module → adapter → engine micro-runs), tiny inputs, still CPU-only and seconds-fast.
- **Few e2e tests**: full CLI/config journeys against shipped configs. One per headline user path.
- Push assertions down: if a unit test can catch it, do not write an integration test for it.

## Non-Negotiable Rules

1. **Tests ship with the code.** Every new module or changed public behavior includes new/updated tests in the same PR. No follow-up-PR promises.
2. **Coverage is gated per-module**, not just globally. Target ≥85% branch coverage on each changed package (`--cov=<package> --cov-branch --cov-fail-under=85`). A global number that hides an untested module does not pass.
3. **Deterministic always.** Seed every randomness source (`random`, numpy, framework RNGs) at test start. No wall-clock dependence (inject clocks or freeze time), no live network — fake or mock at the boundary.
4. **Fixtures over setup duplication.** Shared construction goes in `conftest.py` fixtures; parametrize instead of copy-pasting near-identical tests. A fixture used once belongs inline.
5. **Gated tests are marked, never silently skipped.** GPU, live-server, or credentialed tests carry a marker (e.g. `@pytest.mark.gpu_required`) and an auto-skip hook keyed on the actual capability check — never `skipif(True)` or commented-out tests.

## Technique Selection

| Code under test | Required technique |
|---|---|
| Pure functions / math / parsers / migrations | Hypothesis property-based test (invariants, round-trips, idempotence) plus example tests for known edge cases |
| Config schemas | Validation tests: defaults, boundary values, invalid input raises with useful message |
| Classes with dependencies | Unit test with hand-written fakes injected via constructor (Protocol makes this trivial) |
| CLI / scripts | Subprocess contract test: invoke as a real subprocess, assert stdout/stderr content, exit codes, and produced files — not internals |
| Cross-module wiring | Integration micro-run with real components and minimal inputs |
| Anything hardware/external | Marked + auto-skipped smoke test, with a mocked CPU-path twin that always runs in CI |

## Writing Order

1. Name the behaviors the change adds or alters (public surface, not private helpers).
2. For each behavior, pick the pyramid level and technique from the table.
3. Write the failure-mode tests first (invalid input, missing resource, boundary), then the happy path.
4. Run the changed-module suite with the branch-coverage gate; close gaps in branches, not lines.

## Definition of Done

- New/changed behavior has a test that fails if the behavior regresses.
- Per-module branch coverage ≥85% on the changed surface.
- Full relevant suite green locally; no unexplained skips.
- No test depends on execution order, timing, or ambient state.

See references/test-pyramid.md for worked examples: a unit test with fixtures, a Hypothesis property test, a subprocess contract test, and the marker-gating pattern.
