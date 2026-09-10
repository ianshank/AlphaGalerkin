# CI `--ignore` / `--deselect` ledger

The fast lane (`test-fast` / "Run fast unit tests") and the coverage job
("Run tests with coverage") apply **the same** `--ignore` and `--deselect`
list. The Makefile variable `CI_TEST_EXCLUDES` is a third copy, so
`make test-fast` / `make coverage` measure what CI measures.

This page is the **owner / reason / reopen** register for that list. It is
not a fourth copy of the flags: the YAML block below is the machine-readable
ledger, and `tests/docs/test_ci_exclusion_ledger.py` asserts it matches all
three invocation sites.

**DRY note.** Unit Tests (Fast) and Test Coverage duplicate the list in
`.github/workflows/ci.yml`. Collapsing them onto one shared args file is
hygiene backlog B7. That extraction is **deferred**: rewriting CI's test
invocation is a high-risk cosmetic win (a red PR for a list that already
matches). Until B7 lands, update the YAML block, both `ci.yml` steps, and
`Makefile` `CI_TEST_EXCLUDES` together. The guard fails if any copy drifts.

`covered_by: ~` means no other CI job runs the excluded tests. That is a
disclosed gap, not an accident of an `--ignore` whose covering job was
deleted. **None remain** after 2026-09-10: the cabb5ff/17612ee
`covered_by: ~` clusters (multiprocess file ignore, DataLoader trio,
autodiff Laplacian, two MCTS nodeids) were isolation-green and green in a
combined process of `tests/{data,mcts,experiments,distributed,training,pde}`
(2825 passed) and were dropped from the three invocation copies.

<!-- ci-exclusion-ledger:begin -->
```yaml
version: 1
ignores:
  - path: tests/e2e/
    owner: build-engineer
    covered_by: test-e2e
    reason: >
      Own blocking E2E job. The fast lane also passes -m "not e2e".
      Ignoring the directory keeps the unit-test signal fast; the
      invisibility defect was the directory being ignored *and* named
      in no other step.
    reopen: >
      Delete this ignore only if the dedicated test-e2e job is retired
      and the tier is merged into the fast lane. Never delete while
      test-e2e exists — that is the correct half of a split.

  - path: tests/integration/
    owner: build-engineer
    covered_by: test-integration
    reason: >
      Own Integration Tests job (needs test-fast). Too heavy for the
      unit-test signal.
    reopen: >
      Delete only if test-integration is retired and the directory is
      merged into the fast lane.

  - path: tests/demos/
    owner: build-engineer
    covered_by: "test-fast step: Run demo and notebook suites"
    reason: >
      Slower than a unit test. The same test-fast job runs tests/demos/
      in a named step before the broad pytest tests/ invocation.
    reopen: >
      Delete from the broad run only if that dedicated step is removed
      and demos are merged into pytest tests/. Never delete both.

  - path: tests/training/test_extended_config.py
    owner: src/training
    covered_by: test-integration
    reason: >
      Added to the ignore list in 17612ee as a collection-error risk
      (optional OmegaConf / Hydra surface). test-integration runs
      tests/training/ without this ignore, so the file is not invisible.
    reopen: >
      Drop from the fast-lane ignore after proving the file collects
      under the fast-lane extra set. Keep ignored here while
      test-integration covers it if the file is still slower than a
      unit test.

  - path: tests/notebooks/
    owner: build-engineer
    covered_by: "test-fast step: Run demo and notebook suites"
    reason: >
      Same split as tests/demos/: slower than a unit test, given its
      own step in the test-fast job (18f533d closed the 226-test gap).
    reopen: >
      Delete from the broad run only if the dedicated step is removed
      and notebooks are merged into pytest tests/. Never delete both.

deselects:
  - nodeid: tests/games/test_chess.py::TestChessEdgeCases::test_invalid_move_notation
    owner: src/games
    covered_by: test-chess
    reason: >
      Point-deselected in cabb5ff as a pre-existing chess edge-case
      failure. test-chess runs tests/games/test_chess.py without
      this deselect.
    reopen: >
      Drop from the fast-lane deselect after proving green in the
      combined fast lane. test-chess covering the file is not
      proof the combined process is clean.

  - nodeid: tests/games/test_chess.py::TestChessEdgeCases::test_illegal_move_notation
    owner: src/games
    covered_by: test-chess
    reason: >
      Point-deselected in cabb5ff as a pre-existing chess edge-case
      failure. test-chess runs the same file without this deselect.
    reopen: >
      Drop from the fast-lane deselect after proving green in the
      combined fast lane.

  - nodeid: tests/training/test_self_play.py::TestParallelSelfPlayWorker::test_generate_games_sequential_fallback_on_error
    owner: src/training
    covered_by: test-integration
    reason: >
      Point-deselected in cabb5ff (originally with two CUDA-device
      fallback tests that have since been removed). test-integration
      runs tests/training/ without this deselect.
    reopen: >
      Drop from the fast-lane deselect after proving green in the
      combined fast lane. Integration covering the file is not
      proof the combined process is clean.
```
<!-- ci-exclusion-ledger:end -->
