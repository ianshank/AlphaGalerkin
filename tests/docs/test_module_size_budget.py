"""Module-size budget: a ceiling on every module, and an allowlist that expires.

**Defect class (one sentence):** a module grows past the size one reader can
hold in their head, and nothing notices, because size is not a lint rule and
the diff that crosses the line reads like any other diff.

This repo's hygiene audit (``docs/CODE_HYGIENE_AUDIT.md`` B4) named the god
files, split five of seven, and left the rest as prose. Prose does not ratchet:
between the audit and ``docs/ENGINEERING_REFLECTION_2026-09-11.md`` the count of
``src/`` modules over 600 lines went from 29 to **44** with no check firing,
because there was no check. This file is that check (ticket R-10).

The budget is deliberately a *ratchet*, not a standard:

* An **unlisted** file must be at or under its tier's ceiling
  (``CEILING_SRC`` for ``src/**/*.py``, ``CEILING_TESTS`` for ``tests/**/*.py``).
* A **listed** file may be over the ceiling, but may not grow past the line
  count recorded in its entry. Raising the recorded count is a one-line diff
  that names the file and states a reason -- visible in review, which is the
  whole point.
* The allowlist is **self-expiring in both directions** (harden-a-guard step
  8): an entry whose file is now at or under the ceiling fails (a stale
  exemption silently shrinks the guard), and an entry whose file no longer
  exists fails (a rule about nothing passes forever).

Line count is ``len(path.read_text(encoding="utf-8").splitlines())``. Under the
repo's ``end-of-file-fixer`` pre-commit hook every file ends with exactly one
newline, so this equals ``wc -l``. Scope is ``src/`` and ``tests/`` only:
``hf_space/src/`` is a frozen mirror (charter deviation, hygiene B14) and
``dashboard/`` / ``scripts/`` are not in the ticket.

Recording a frozen codec file here is data, not an edit -- no frozen path is
touched, so ``scripts/check_focus.py`` is unaffected.

Hermetic: reads the tree, runs nothing, ~0.2 s.

Mutation-kill record, 2026-09-11 (5 planted defects, 226 tests -- the two
numbers are kept apart on purpose; each defect is killed by a NAMED test, and
each mutation was confirmed applied by re-measuring the file before running):

1. ``src/poc/cli.py`` (listed at 601) padded with 50 lines to 651 ->
   ``test_listed_modules_have_not_grown_past_their_recorded_count[src/poc/cli.py]``
   FAILED, 225 passed -- exactly one failure, so it is attributable.
2. A new unlisted ``src/zz_r10_mutation.py`` of exactly 601 lines ->
   ``test_unlisted_modules_are_within_the_ceiling[src]`` FAILED, 225 passed.
   Control: the same file at exactly 600 lines -> 226 passed. The boundary is
   ``>``, as ``TestBudgetPredicates::test_boundary_is_strictly_greater_than``
   also pins on synthetic input.
3. ``src/poc/cli.py`` truncated to 550 lines with its entry kept ->
   ``test_listed_modules_are_still_over_the_ceiling[src/poc/cli.py]`` FAILED,
   225 passed.

Adjacent weaker defects (harden-a-guard step 5), also planted and killed:

4. ``src/pde/game.py`` deleted with its entry kept ->
   ``test_every_allowlist_key_exists[src/pde/game.py]`` FAILED (the two sibling
   per-entry tests for that path also fail on the missing read; 223 passed).
5. The ``src`` tier pointed at a directory that does not exist ->
   ``test_the_scan_is_not_vacuous[src_missing]`` FAILED -- and
   ``test_unlisted_modules_are_within_the_ceiling[src_missing]`` stayed GREEN
   on the empty scan, which is the vacuous pass the floor exists to catch.

Not planted, structurally rejected: an entry keyed under the other tree
(``tests/...`` in ``ALLOWLIST_SRC``) is scanned by nothing and is refused by
``test_allowlist_keys_live_under_their_tier``; a repeated key, which a dict
literal silently collapses, is refused by ``test_allowlist_keys_are_unique``
reading this file's AST (its reader is unit-tested on a planted duplicate).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Ceiling for ``src/**/*.py``. Files over this need an ``ALLOWLIST_SRC`` entry.
CEILING_SRC = 600

#: Ceiling for ``tests/**/*.py``. Test modules are legitimately longer (one
#: file per subject, parametrised tables), so the bar is higher, not absent.
CEILING_TESTS = 1000

#: Vacuity floors. A scan that matches fewer files than this is a scan of the
#: wrong directory, and every parametrised assertion below would pass on it.
#: Measured 2026-09-11: 410 ``src`` files, 558 ``tests`` files.
MIN_SRC_FILES = 300
MIN_TEST_FILES = 400

#: ``src`` modules over ``CEILING_SRC``, as ``path -> (recorded lines, reason)``.
#: 44 rows recorded 2026-09-11. Reasons are short and honest: "split scheduled"
#: cites the reflection plan's WS1 row; "frozen codec track" cites
#: ``config/focus.yaml``; the rest name why the file is cohesive as it stands.
#: To raise a count: edit the number here, in the same PR, with a reason.
ALLOWLIST_SRC: dict[str, tuple[int, str]] = {
    "src/training/trainer.py": (1246, "trainer facade; split touch-triggered (plan 1.2)"),
    "src/games/chess.py": (1242, "rules class; move-encoding seam split deferred (plan 1.4)"),
    "src/training/checkpoint.py": (994, "split scheduled (plan 1.3)"),
    "src/video_compression/codec/codec.py": (976, "frozen codec track"),
    "src/research/lshape_amr_compare.py": (942, "comparison harness; split cut (plan 1.8)"),
    "src/training/base_trainer.py": (900, "shared trainer base; split deferred (plan 1.6)"),
    "src/demos/visualizations.py": (810, "demo module"),
    "src/pde/games/basis_selection.py": (800, "split scheduled (plan 1.5, basis_library)"),
    "src/modeling/model.py": (781, "heads/blocks split deferred (plan 1.6)"),
    "src/research/fem_baseline.py": (776, "reference baseline behind the [fem] extra"),
    "src/agents/config.py": (750, "schema module"),
    "src/modeling/multiscale_fourier.py": (742, "one family of Fourier-feature layers"),
    "src/pde/games/mesh_refinement/game.py": (731, "MCTS game half of the B4 split"),
    "src/training/loss_balancing.py": (731, "one family of loss balancers"),
    "src/research/transfer_baseline_compare.py": (722, "comparison harness"),
    "src/mcts/search.py": (716, "core search engine"),
    "src/backend/torch_backend.py": (713, "backend adapter"),
    "src/mcts/gumbel.py": (712, "Gumbel MCTS variant"),
    "src/alphagalerkin/solver.py": (706, "solver facade"),
    "src/poc/scenarios/llm_prior_ablation.py": (699, "scenario"),
    "src/training/self_play.py": (697, "self-play loop"),
    "src/tools/gtp.py": (696, "GTP protocol handler"),
    "src/poc/statistics/significance.py": (694, "statistics module"),
    "src/training/evaluation.py": (692, "evaluation module"),
    "src/math_kernel/integral.py": (684, "math kernel"),
    "src/video_compression/config.py": (683, "frozen codec track"),
    "src/video_compression/perf/benchmark.py": (666, "frozen codec track"),
    "src/math_kernel/spectral.py": (665, "math kernel"),
    "src/tournament/manager.py": (657, "tournament manager"),
    "src/pde/geometry.py": (656, "one family of domain geometries"),
    "src/video_compression/demo/runner.py": (652, "frozen codec track"),
    "src/demos/benchmark_demo.py": (649, "demo module"),
    "src/distributed/config.py": (646, "schema module"),
    "src/pde/games/swarm_planning.py": (644, "game"),
    "src/distributed/trainer.py": (631, "DDP trainer"),
    "src/demos/architecture_demo.py": (630, "demo module"),
    "src/analysis/reviewer.py": (629, "analysis module"),
    "src/research/mcts_classical_amr_arena.py": (628, "arena harness"),
    "src/games/go.py": (624, "rules class"),
    "src/demos/sbir_demo.py": (608, "demo module"),
    "src/prototyping/templates.py": (606, "prototyping templates"),
    "src/analysis/patterns.py": (605, "pattern catalogue"),
    "src/pde/game.py": (605, "PDEGame base class"),
    "src/poc/cli.py": (601, "CLI dispatcher; one line over"),
}

#: ``tests`` modules over ``CEILING_TESTS``. 8 rows recorded 2026-09-11.
#: ``tests/docs/test_e2e_visibility.py`` is itself a guard: each new clause
#: there bumps its own recorded count, visibly, in the PR that adds the clause.
ALLOWLIST_TESTS: dict[str, tuple[int, str]] = {
    "tests/pde/test_operators.py": (1717, "one suite per operator family"),
    "tests/pde/test_mesh_refinement.py": (1507, "mesh + game suite"),
    "tests/training/test_trainer_coverage.py": (1377, "trainer coverage sweep"),
    "tests/poc/test_complexity_scenario.py": (1148, "scenario suite"),
    "tests/training/test_checkpoint.py": (1102, "checkpoint suite; follows plan 1.3"),
    "tests/docs/test_e2e_visibility.py": (1074, "itself a guard; grows one clause at a time"),
    "tests/research/test_baselines.py": (1069, "baseline suite incl. shared primitives"),
    "tests/training/test_trainer_physics.py": (1001, "physics trainer suite; one line over"),
}


@dataclass(frozen=True)
class Tier:
    """One scanned tree with its ceiling and allowlist."""

    name: str
    ceiling: int
    allowlist: dict[str, tuple[int, str]]
    min_files: int


TIERS: tuple[Tier, ...] = (
    Tier("src", CEILING_SRC, ALLOWLIST_SRC, MIN_SRC_FILES),
    Tier("tests", CEILING_TESTS, ALLOWLIST_TESTS, MIN_TEST_FILES),
)

_ENTRIES: list[tuple[Tier, str]] = [(tier, path) for tier in TIERS for path in tier.allowlist]
_ENTRY_IDS = [path for _tier, path in _ENTRIES]
_TIER_IDS = [tier.name for tier in TIERS]


# ---------------------------------------------------------------------------
# Primitives -- pure functions, so ``TestBudgetPredicates`` can drive them on
# synthetic input and pin the boundary independently of the live tree.
# ---------------------------------------------------------------------------


def line_count(path: Path) -> int:
    """``wc -l`` under end-of-file-fixer (see the module docstring)."""
    return len(path.read_text(encoding="utf-8").splitlines())


def scan_tree(root: Path, top: str) -> dict[str, int]:
    """``repo-relative posix path -> line count`` for every ``top/**/*.py``.

    ``__pycache__`` is skipped by path component. It normally holds only
    ``.pyc``, but a stray ``.py`` there (a copied fixture, an editor artefact)
    must not be counted as a module.
    """
    found: dict[str, int] = {}
    for path in sorted((root / top).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        found[path.relative_to(root).as_posix()] = line_count(path)
    return found


def unlisted_over_ceiling(
    counts: dict[str, int], ceiling: int, allowlist: dict[str, tuple[int, str]]
) -> dict[str, int]:
    """Files strictly over ``ceiling`` with no allowlist entry."""
    return {p: n for p, n in counts.items() if n > ceiling and p not in allowlist}


def grown_past_record(
    counts: dict[str, int], allowlist: dict[str, tuple[int, str]]
) -> dict[str, tuple[int, int]]:
    """``path -> (recorded, actual)`` for listed files that grew past their record."""
    return {
        p: (recorded, counts[p])
        for p, (recorded, _reason) in allowlist.items()
        if p in counts and counts[p] > recorded
    }


def no_longer_over_ceiling(
    counts: dict[str, int], ceiling: int, allowlist: dict[str, tuple[int, str]]
) -> dict[str, int]:
    """Listed files now at or under the ceiling -- their entry has expired."""
    return {p: counts[p] for p in allowlist if p in counts and counts[p] <= ceiling}


def _allowlist_literal_keys(source: Path, name: str) -> list[str]:
    """String keys of the dict *literal* assigned to ``name`` in ``source``.

    A Python ``dict`` literal with a repeated key silently keeps the last value,
    so ``len(ALLOWLIST)`` cannot see a duplicate; only the AST can.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        if not (isinstance(target, ast.Name) and target.id == name):
            continue
        assert isinstance(value, ast.Dict), f"{name} must be a dict literal"
        return [
            k.value for k in value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)
        ]
    raise AssertionError(f"no module-level assignment to {name} in {source.name}")


# ---------------------------------------------------------------------------
# Vacuity first (harden-a-guard step 1).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS, ids=_TIER_IDS)
def test_the_scan_is_not_vacuous(tier: Tier) -> None:
    """Every assertion below iterates this scan; an empty one passes them all."""
    counts = scan_tree(REPO_ROOT, tier.name)
    assert len(counts) >= tier.min_files, (
        f"scan of {tier.name}/ found only {len(counts)} .py files (floor {tier.min_files}); "
        "either the tree moved or the scanner is pointed at the wrong root"
    )


# ---------------------------------------------------------------------------
# The budget.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS, ids=_TIER_IDS)
def test_unlisted_modules_are_within_the_ceiling(tier: Tier) -> None:
    """A file over the ceiling with no entry is the defect this guard exists for."""
    offenders = unlisted_over_ceiling(scan_tree(REPO_ROOT, tier.name), tier.ceiling, tier.allowlist)
    by_size = sorted(offenders.items(), key=lambda kv: -kv[1])
    listing = "\n".join(f"  {n:5d}  {p}" for p, n in by_size)
    assert not offenders, (
        f"{len(offenders)} {tier.name}/ module(s) exceed the {tier.ceiling}-line ceiling and are "
        f"not in ALLOWLIST_{tier.name.upper()}:\n{listing}\n"
        "Split it, or if it is cohesive add its allowlist entry with a reason."
    )


@pytest.mark.parametrize(("tier", "path"), _ENTRIES, ids=_ENTRY_IDS)
def test_listed_modules_have_not_grown_past_their_recorded_count(tier: Tier, path: str) -> None:
    """The ratchet: an allowlisted file may stay large, but may not get larger silently."""
    recorded, _reason = tier.allowlist[path]
    actual = line_count(REPO_ROOT / path)
    assert actual <= recorded, (
        f"{path} grew from its recorded {recorded} lines to {actual}. "
        "Split it, or if it is cohesive raise its allowlist entry with a reason."
    )


@pytest.mark.parametrize(("tier", "path"), _ENTRIES, ids=_ENTRY_IDS)
def test_listed_modules_are_still_over_the_ceiling(tier: Tier, path: str) -> None:
    """Self-expiring: an entry for a file now under the ceiling is a stale exemption.

    Left in place it would let the file grow straight back to its recorded
    count without the "unlisted" check ever seeing it.
    """
    actual = line_count(REPO_ROOT / path)
    assert actual > tier.ceiling, (
        f"{path} is {actual} lines, at or under the {tier.ceiling}-line ceiling, "
        f"but still carries an ALLOWLIST_{tier.name.upper()} entry. Delete the entry -- "
        "a stale exemption silently shrinks the guard's scope."
    )


# ---------------------------------------------------------------------------
# Allowlist integrity -- the exemption itself must be well-formed and live.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("tier", "path"), _ENTRIES, ids=_ENTRY_IDS)
def test_every_allowlist_key_exists(tier: Tier, path: str) -> None:
    """An entry for a deleted or renamed file is a rule about nothing."""
    assert (REPO_ROOT / path).is_file(), (
        f"ALLOWLIST_{tier.name.upper()} names {path}, which does not exist. "
        "Delete the entry, or re-key it to the file's new path."
    )


@pytest.mark.parametrize(("tier", "path"), _ENTRIES, ids=_ENTRY_IDS)
def test_every_allowlist_entry_states_a_reason(tier: Tier, path: str) -> None:
    """A bare number is a suppression; a reason is a decision someone can revisit."""
    recorded, reason = tier.allowlist[path]
    assert reason.strip(), f"ALLOWLIST_{tier.name.upper()}[{path!r}] has an empty reason"
    assert recorded > tier.ceiling, (
        f"ALLOWLIST_{tier.name.upper()}[{path!r}] records {recorded} lines, which is not over "
        f"the {tier.ceiling} ceiling -- the entry is malformed"
    )


@pytest.mark.parametrize("tier", TIERS, ids=_TIER_IDS)
def test_allowlist_keys_live_under_their_tier(tier: Tier) -> None:
    """An entry keyed under the other tree is scanned by neither and silently inert."""
    prefix = f"{tier.name}/"
    strays = [p for p in tier.allowlist if not p.startswith(prefix)]
    assert not strays, f"ALLOWLIST_{tier.name.upper()} keys must start with {prefix!r}: {strays}"


@pytest.mark.parametrize("tier", TIERS, ids=_TIER_IDS)
def test_allowlist_keys_are_unique(tier: Tier) -> None:
    """A dict literal keeps the *last* duplicate silently; read the AST, not the dict."""
    keys = _allowlist_literal_keys(Path(__file__), f"ALLOWLIST_{tier.name.upper()}")
    assert len(keys) == len(tier.allowlist), "AST key count disagrees with the live dict"
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    assert not duplicates, f"ALLOWLIST_{tier.name.upper()} repeats keys: {duplicates}"


def test_the_ceilings_are_ordered() -> None:
    """Tests may be longer than modules, never the reverse -- a swapped pair is a typo."""
    assert 0 < CEILING_SRC < CEILING_TESTS


# ---------------------------------------------------------------------------
# The predicates on synthetic input -- pins the boundary and the exclusions
# without depending on what the live tree happens to contain today.
# ---------------------------------------------------------------------------


def _write(path: Path, n_lines: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"# line {i}\n" for i in range(n_lines)), encoding="utf-8")


class TestBudgetPredicates:
    """Drive the scanner and predicates on a fabricated tree."""

    def test_line_count_matches_wc_l_convention(self, tmp_path: Path) -> None:
        target = tmp_path / "m.py"
        _write(target, 7)
        assert line_count(target) == 7
        # A file missing its trailing newline still counts its last line, which
        # is where `splitlines()` and a naive `count("\n")` would disagree.
        target.write_text("a\nb\nc", encoding="utf-8")
        assert line_count(target) == 3

    def test_scan_excludes_pycache_and_non_python(self, tmp_path: Path) -> None:
        _write(tmp_path / "src" / "ok.py", 3)
        _write(tmp_path / "src" / "__pycache__" / "stray.py", 3)
        (tmp_path / "src" / "notes.txt").write_text("x\n", encoding="utf-8")
        assert scan_tree(tmp_path, "src") == {"src/ok.py": 3}

    def test_scan_of_a_missing_tree_is_empty_not_an_error(self, tmp_path: Path) -> None:
        # Which is exactly why ``test_the_scan_is_not_vacuous`` exists.
        assert scan_tree(tmp_path, "src") == {}

    def test_boundary_is_strictly_greater_than(self) -> None:
        counts = {"src/at.py": 600, "src/over.py": 601}
        assert unlisted_over_ceiling(counts, 600, {}) == {"src/over.py": 601}

    def test_listed_file_is_not_reported_as_unlisted(self) -> None:
        counts = {"src/big.py": 900}
        assert unlisted_over_ceiling(counts, 600, {"src/big.py": (900, "r")}) == {}

    def test_growth_past_record_is_reported_with_both_numbers(self) -> None:
        counts = {"src/big.py": 950, "src/same.py": 700}
        allow = {"src/big.py": (900, "r"), "src/same.py": (700, "r")}
        assert grown_past_record(counts, allow) == {"src/big.py": (900, 950)}

    def test_expired_entry_is_reported_at_and_below_the_ceiling(self) -> None:
        counts = {"src/at.py": 600, "src/under.py": 10, "src/live.py": 601}
        allow = dict.fromkeys(counts, (700, "r"))
        assert no_longer_over_ceiling(counts, 600, allow) == {"src/at.py": 600, "src/under.py": 10}

    def test_ast_key_reader_sees_a_duplicate_the_dict_hides(self, tmp_path: Path) -> None:
        source = tmp_path / "mod.py"
        source.write_text(
            'X: dict[str, tuple[int, str]] = {\n    "a": (1, "r"),\n    "a": (2, "r"),\n}\n',
            encoding="utf-8",
        )
        assert _allowlist_literal_keys(source, "X") == ["a", "a"]

    def test_ast_key_reader_fails_loudly_on_a_missing_name(self, tmp_path: Path) -> None:
        source = tmp_path / "mod.py"
        source.write_text("Y = 1\n", encoding="utf-8")
        with pytest.raises(AssertionError, match="no module-level assignment to X"):
            _allowlist_literal_keys(source, "X")
