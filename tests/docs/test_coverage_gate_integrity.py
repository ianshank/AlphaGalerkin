"""Guards coverage gates against the two ways they silently measure nothing.

This repo has now hit the same false pass **three** times — `video_compression`,
`demos`, and `src/research/substrates/skfem_tri.py` — and each time it was found
by a human reading the workflow, not by a check. The shape is always identical:
a CI step passes ``--cov=<target>`` and ``--cov-fail-under=<N>``, the step goes
green, and the gate is enforcing nothing at all. A gate that cannot fail is
worse than no gate, because it is *reported* as coverage.

Two mechanisms, both mechanical and therefore both checkable:

1. **The `omit` collision.** A target listed in pyproject.toml's global
   ``[tool.coverage.run] omit`` is never traced, so a bare ``--cov`` of it
   reports ``0.00%`` with ``CoverageWarning: No data was collected``. The
   established escape is an inline-generated ``.coveragerc`` that drops the
   omit for that step only, selected with ``--cov-config=``. This test requires
   that escape wherever the collision exists.

2. **The file-path spec.** ``coverage`` 7.x silently ignores a ``--cov=<...>.py``
   argument — no error, no warning, just no measurement. Directory targets and
   the native ``coverage run --include=`` runner are the two working forms.

Hermetic: parses YAML and TOML, runs nothing. The meta-guard at the end is what
stops the whole file becoming a no-op if the parser stops matching anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
PYPROJECT = REPO_ROOT / "pyproject.toml"

_COV_TARGET = re.compile(r"--cov=(?P<target>[^\s\\'\"]+)")
_COV_CONFIG = re.compile(r"--cov-config=")

#: ``--cov`` accepts a module *name* as well as a path. Those are not what this
#: file is about, and a dotted spec has its own documented hazard (it crashes on
#: this repo's torch build), so they are skipped rather than mis-analysed.
_NON_PATH_TARGET = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$")


@dataclass(frozen=True)
class CovInvocation:
    """One ``--cov=<target>`` occurrence, with the step it lives in."""

    workflow: str
    job: str
    step: str
    target: str
    has_cov_config: bool

    def __str__(self) -> str:  # pragma: no cover - failure-message sugar only
        return f"{self.workflow}::{self.job}::{self.step} (--cov={self.target})"


def _iter_run_scripts() -> list[tuple[str, str, str, str]]:
    """Every ``run:`` script body in every workflow, tagged with where it lives."""
    found: list[tuple[str, str, str, str]] = []
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_name, job in (document.get("jobs") or {}).items():
            for index, step in enumerate(job.get("steps") or []):
                script = step.get("run")
                if not isinstance(script, str):
                    continue
                label = step.get("name") or f"step[{index}]"
                found.append((path.name, str(job_name), str(label), script))
    return found


def _cov_invocations() -> list[CovInvocation]:
    invocations: list[CovInvocation] = []
    for workflow, job, step, script in _iter_run_scripts():
        has_config = bool(_COV_CONFIG.search(script))
        for match in _COV_TARGET.finditer(script):
            target = match.group("target")
            if _NON_PATH_TARGET.match(target):
                continue
            invocations.append(
                CovInvocation(
                    workflow=workflow,
                    job=job,
                    step=step,
                    target=target,
                    has_cov_config=has_config,
                )
            )
    return invocations


def _ancestors(path: str) -> list[str]:
    """Every directory prefix of ``path``, longest first (``a/b/c.py`` -> a/b, a)."""
    parts = path.split("/")[:-1]
    return ["/".join(parts[: i + 1]) for i in reversed(range(len(parts)))]


_OMIT_OVERRIDE = re.compile(r"--(cov-config|rcfile)=")


def _step_really_gates(script: str, path: str) -> bool:
    """Report whether this run-script actually measures ``path``, omit and all.

    Two conditions, and the second is the one that makes this a test rather
    than a formality:

    1. It **selects** ``path`` — by ``--include=`` naming it, or by a ``--cov``
       of the file or a directory containing it.
    2. It **overrides** the global omit, with ``--cov-config=`` (pytest-cov) or
       ``--rcfile=`` (native runner). Without this the step inherits
       pyproject.toml's ``omit`` and measures nothing of ``path``, however
       precisely it selects it — verified empirically: ``coverage run
       --include='*/src/research/fem_baseline.py'`` alone reports "No data was
       collected", because ``coverage run`` reads pyproject.toml too.
    """
    if not _OMIT_OVERRIDE.search(script):
        return False
    if re.search(rf"--include=['\"]\*/{re.escape(path)}['\"]", script):
        return True
    # Word-boundary match, not a substring one: `--cov=src/research` occurs
    # inside `--cov=src/research/substrates`, so a naive `in` test let the
    # substrates gate stand in as proof for a *sibling* file. A mutation check
    # (deleting the fem_baseline gate entirely) passed against that version.
    for candidate in [path, *_ancestors(path)]:
        if re.search(rf"--cov={re.escape(candidate)}(?=[\s\\'\"]|$)", script):
            return True
    return False


_OMIT_BLOCK = re.compile(r"^omit\s*=\s*\[(?P<body>.*?)^\]", re.M | re.S)


def _omit_patterns() -> list[str]:
    """Read ``[tool.coverage.run] omit`` from pyproject.toml.

    Anchored regex rather than ``tomllib``, following the sibling
    ``tests/docs/test_version_consistency.py``: ``tomllib`` is stdlib only from
    3.11 and this repo's declared floor is 3.10, so importing it would be a
    *collection* error on the oldest supported interpreter — taking the whole
    run down, not just this file. ``tests/docs/test_python_floor_compatibility.py``
    catches exactly that, and did catch it here.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    match = _OMIT_BLOCK.search(text)
    assert match, "could not locate the coverage `omit` list in pyproject.toml"
    # Strip `#` comments first: the omit block carries prose explaining each
    # entry, and prose containing an apostrophe or a quoted name would
    # otherwise be harvested as if it were a pattern (it was -- "Eval-harness
    # adapter" showed up in the first version's output).
    body = "\n".join(line.split("#", 1)[0] for line in match.group("body").splitlines())
    return re.findall(r"[\"']([^\"']+)[\"']", body)


#: Targets for which the global ``omit`` is the *intended* configuration rather
#: than a defeat. Only the repo-wide root qualifies: the omit list exists
#: precisely to shape that gate (excluding optional-extra and frozen packages),
#: so requiring it to override its own configuration would be nonsense. Every
#: narrower target is a different matter — see ``_is_omitted``. Each entry needs
#: a reason, and the meta-guard below asserts the exemption is still load-bearing.
_OMIT_IS_INTENDED_FOR: dict[str, str] = {
    "src": (
        "the repo-wide gate; the global omit list is authored FOR it, to exclude "
        "optional-extra and frozen packages from the whole-tree percentage"
    ),
}

#: ``(target, omit-pattern)`` pairs where the omitted module genuinely cannot be
#: traced on that job (it needs an optional extra the job does not install) AND
#: is gated somewhere else instead. This is the *only* acceptable reason for a
#: package gate to under-measure part of itself, and it is not taken on trust:
#: ``test_every_elsewhere_gated_exemption_names_a_real_gate`` below requires the
#: named module to actually appear in some workflow's coverage invocation, so an
#: exemption whose "gated elsewhere" gate is deleted fails rather than rots.
_OMITTED_BUT_GATED_ELSEWHERE: dict[tuple[str, str], str] = {
    ("src/research", "src/research/fem_baseline.py"): (
        "needs the optional [fem] extra, which the `coverage` job does not "
        "install; gated at 83 in `test-extras`, which does"
    ),
    ("src/research", "src/research/substrates/skfem_tri.py"): (
        "same optional [fem] extra; gated at 95 as part of "
        "--cov=src/research/substrates in `test-extras`"
    ),
}


def _is_omitted(target: str, patterns: list[str]) -> str | None:
    """Return an omit pattern that defeats a ``--cov`` of ``target``, or ``None``.

    Collides in **both** directions, which is the correction that matters:

    * *Ancestor* — the omit covers the target or a parent of it, so the target
      is traced not at all and coverage reports ``0.00%``.
    * *Descendant* — the omit covers something **inside** the target. This looks
      harmless (the package total is still a real number) and is exactly how the
      third instance of this bug got written: a gate added specifically to
      measure ``skfem_tri.py`` would have reported ~99% from the *other* four
      files in the package while its actual subject contributed nothing. A
      mutation check caught the first version of this function passing that case.

    ``_OMIT_IS_INTENDED_FOR`` carves out the repo-wide root, where the omit is
    the point rather than the problem.
    """
    normalised = target.rstrip("/")
    if normalised in _OMIT_IS_INTENDED_FOR:
        return None
    for pattern in patterns:
        if (normalised, pattern) in _OMITTED_BUT_GATED_ELSEWHERE:
            continue
        prefix = pattern[:-2] if pattern.endswith("/*") else pattern
        if normalised == prefix:
            return pattern
        if normalised.startswith(prefix + "/"):  # omit is an ancestor
            return pattern
        if prefix.startswith(normalised + "/"):  # omit is a descendant
            return pattern
    return None


#: Omit patterns that no CI gate covers, each with the reason it is acceptable.
#:
#: This dict exists because the checks above answer only half the question. They
#: verify that a gate a workflow *declares* is not secretly neutered by the
#: omit. They say nothing about a package that is omitted **and never gated at
#: all** -- which is not a fake gate, it is no gate, and it is invisible for
#: exactly the same reason. ``src/backend`` sat in that blind spot: 2873 LOC and
#: 213 passing tests measuring nothing anywhere, because ``src/backend/*`` was
#: omitted from the whole-tree gate (legitimately -- ``jax_backend.py`` needs the
#: optional [jax] extra) at *package* granularity, which also silently excused
#: ``logging.py`` and ``rng.py`` at literally 0%. Found by hand on 2026-09-02,
#: not by this file, which is the finding.
#:
#: An entry here is a claim that no gate is warranted, not a note that none
#: exists -- and ``test_every_orphaned_omit_exemption_is_still_orphaned``
#: fails it the moment a gate appears, so it cannot outlive its reason.
#: CORRECTED 2026-09-02. The original reason for the entry below was false in
#: both of its checkable clauses, and it was written by the same change that
#: added this guard -- the guard fired, and it was silenced with an untruth:
#:
#:   * "which no CI job installs" -- ``.github/workflows/ci.yml`` installs
#:     ``.[dev,test-extras,eval-harness]`` in the ``test-extras`` job, with a
#:     comment saying it does so *specifically* to un-skip these modules.
#:   * "enforced instead by the `Eval-harness adapter` Regression Surface
#:     command" -- no such row exists in ``CLAUDE.md``. It never did.
#:
#: ``test_every_orphaned_omit_exemption_states_a_reason`` only checks a reason is
#: *present*, and ``..._is_still_orphaned`` only that no gate has appeared. A
#: false reason satisfied both forever. ``test_regression_surface_rows_cited_by_
#: exemptions_exist`` below closes the half of that hole which is mechanically
#: decidable.
_OMIT_WITHOUT_A_CI_GATE: dict[str, str] = {
    "src/integrations/eval_harness/*": (
        "needs the optional [eval-harness] git dependency (langfuse-eval-harness, "
        "a git URL, not on PyPI). The test-extras CI job DOES install it, but runs "
        "no test under tests/integrations/eval_harness/, so 8 of those 11 modules "
        "skip at import and 914 LOC is measured by nothing anywhere. This is a "
        "DISCLOSED GAP, not a justified exemption: the fix is a per-module gate in "
        "test-extras, which already has the extra installed. Tracked as B37 in "
        "docs/CODE_HYGIENE_AUDIT.md."
    ),
}

#: Marker an exemption reason uses when it claims a ``CLAUDE.md`` Regression
#: Surface row enforces the module instead. Any such claim is checkable, so it
#: is checked.
_REGRESSION_SURFACE_CLAIM = "Regression Surface"

#: The citation shape: a backticked row name immediately followed by the marker.
_REGRESSION_SURFACE_CITATION = re.compile(r"`([^`]+)`\s+" + _REGRESSION_SURFACE_CLAIM)


def _rows_cited(reason: str) -> list[str]:
    """Every Regression Surface row an exemption reason names in backticks.

    Extracted so ``TestRowsCited`` can drive it on synthetic input. The first
    version of the guard below inlined this regex in a comprehension and was
    **vacuous on live data**: the sole exemption cites no row, ``findall``
    returned ``[]``, the inner loop never entered, and coverage still reported
    the function 100% covered -- a comprehension collapses to one line with no
    branch arc. It failed under mutation (restoring the false citation) but had
    never once demonstrated the regex *accepts* a valid citation, so a regex
    typo matching nothing would have left it green forever. Same shape as the
    ``describe()`` tautology caught one commit earlier.
    """
    return _REGRESSION_SURFACE_CITATION.findall(reason)


def _rows_missing_from_surface(exemptions: dict[str, str], surface: str) -> list[tuple[str, str]]:
    """``(pattern, row)`` for every cited row absent from ``surface``."""
    return [
        (pattern, row)
        for pattern, reason in exemptions.items()
        for row in _rows_cited(reason)
        if row not in surface
    ]


def _omit_pattern_is_gated(pattern: str) -> bool:
    """Whether some workflow step both selects ``pattern``'s module and beats the omit.

    Reuses ``_step_really_gates`` rather than re-deriving the two conditions:
    the whole point of the ``src/backend`` miss was two checks that should have
    agreed but never met.
    """
    path = pattern[:-2] if pattern.endswith("/*") else pattern
    return any(_step_really_gates(script, path) for _wf, _job, _step, script in _iter_run_scripts())


def test_every_omit_entry_is_gated_somewhere() -> None:
    """An omitted package must be measured by *some* gate, or be a stated exemption.

    The omit list is not a list of code that does not matter -- it is a list of
    code the whole-tree percentage cannot honestly include. Each entry therefore
    owes a per-module gate that overrides it. Without this test, adding a package
    to ``omit`` silently removes it from every gate in the repo, and the diff that
    does it is one line that reads like configuration.
    """
    orphaned = [
        pattern
        for pattern in _omit_patterns()
        if pattern not in _OMIT_WITHOUT_A_CI_GATE and not _omit_pattern_is_gated(pattern)
    ]
    assert not orphaned, (
        "these `[tool.coverage.run] omit` entries are measured by no coverage gate "
        f"anywhere, so their tests report nothing: {orphaned}. Add a per-module gate "
        "using the inline-coveragerc technique (see the `demos`/`backend` steps in "
        "ci.yml), or record the entry in `_OMIT_WITHOUT_A_CI_GATE` with a reason."
    )


def test_every_orphaned_omit_exemption_is_still_orphaned() -> None:
    """A stale exemption is worse than none: it hides a gate that now exists.

    Mirrors ``scripts/audit_abstractions.py``'s ``_STAGED_FOR_UPCOMING_TASK``
    staleness check and ``tests/regression/test_import_contracts.py``'s
    ``test_every_exemption_is_still_needed``. If a gate is later added for an
    exempted pattern, the exemption must be *deleted*, not left to imply the
    package is still ungatable.
    """
    now_gated = [p for p in _OMIT_WITHOUT_A_CI_GATE if _omit_pattern_is_gated(p)]
    assert not now_gated, (
        f"these patterns are exempted as ungatable but now have a real gate: {now_gated}. "
        "Delete their `_OMIT_WITHOUT_A_CI_GATE` entries."
    )


def test_every_orphaned_omit_exemption_states_a_reason() -> None:
    """A reason is what distinguishes an exemption from a hole someone tolerated."""
    for pattern, reason in _OMIT_WITHOUT_A_CI_GATE.items():
        assert reason.strip(), f"{pattern} is exempted with no reason"
        assert len(reason.split()) >= 8, f"{pattern}'s reason is too thin to review: {reason!r}"


def test_every_orphaned_omit_exemption_is_actually_omitted() -> None:
    """An exemption for a pattern no longer in `omit` guards nothing.

    Same vacuity failure the import contracts guard against: a renamed or
    removed omit entry turns its exemption into a rule about nothing, and a
    rule about nothing passes.
    """
    patterns = set(_omit_patterns())
    stale = [p for p in _OMIT_WITHOUT_A_CI_GATE if p not in patterns]
    assert not stale, (
        f"these `_OMIT_WITHOUT_A_CI_GATE` keys are not in pyproject.toml's omit list: {stale}"
    )


INVOCATIONS = _cov_invocations()


def test_no_cov_target_is_silently_swallowed_by_the_global_omit() -> None:
    """The `omit` collision: `--cov` of an omitted path measures 0.00%.

    Escape hatch, and the only accepted one: the same step passes
    ``--cov-config=`` pointing at an inline-generated rcfile that drops the
    omit for that run. That is the technique `video_compression`, `demos` and
    `src/research/substrates` all use.
    """
    patterns = _omit_patterns()
    offenders = [
        (invocation, matched)
        for invocation in INVOCATIONS
        if (matched := _is_omitted(invocation.target, patterns)) is not None
        and not invocation.has_cov_config
    ]
    assert not offenders, (
        "coverage gate(s) measuring nothing -- the target is in pyproject.toml's "
        "[tool.coverage.run] omit, so coverage reports 0.00% with "
        "'No data was collected' and --cov-fail-under cannot fail:\n"
        + "\n".join(f"  {inv} collides with omit {pat!r}" for inv, pat in offenders)
        + "\nFix: generate an inline .coveragerc without the omit and pass "
        "--cov-config= in the same step."
    )


def test_no_cov_target_is_a_file_path_spec() -> None:
    """Coverage 7.x silently ignores ``--cov=<...>.py`` -- no error, no measurement."""
    offenders = [inv for inv in INVOCATIONS if inv.target.endswith(".py")]
    assert not offenders, (
        "file-path --cov spec(s), which coverage 7.x silently drops:\n"
        + "\n".join(f"  {inv}" for inv in offenders)
        + "\nUse the directory form, or the native runner "
        "(python -m coverage run --include='*/path.py')."
    )


def test_every_cov_target_exists_on_disk() -> None:
    """A gate pointed at a renamed package measures nothing and still goes green."""
    missing = [inv for inv in INVOCATIONS if not (REPO_ROOT / inv.target).exists()]
    assert not missing, "coverage target(s) that do not exist:\n" + "\n".join(
        f"  {inv}" for inv in missing
    )


class TestTheGuardItself:
    """Meta-guards. A parser that matches nothing passes every test above."""

    def test_the_omit_parser_reads_exactly_the_real_patterns(self) -> None:
        """Pins the hand-rolled TOML parse. Every entry must be a plausible path.

        The regex form is forced by the 3.10 floor (no ``tomllib``), which makes
        it the weak link in this file: a parse that silently returns fewer
        patterns makes every collision test above vacuous, and one that returns
        *extra* junk makes them noisy. The first version harvested a comment
        string as if it were a pattern.
        """
        patterns = _omit_patterns()
        assert patterns, "the omit list parsed as empty -- every collision test is now vacuous"
        for pattern in patterns:
            assert pattern.startswith("src/"), f"{pattern!r} is not a source path"
            assert " " not in pattern, f"{pattern!r} looks like prose, not a path"
        assert "src/research/substrates/skfem_tri.py" in patterns
        assert "src/demos/*" in patterns

    def test_the_parser_finds_the_known_gates(self) -> None:
        targets = {inv.target for inv in INVOCATIONS}
        # Three gates that must exist for this repo to be gated at all.
        assert "src" in targets, "the repo-wide --cov=src gate was not found"
        assert "src/research/substrates" in targets
        assert len(INVOCATIONS) > 20, f"only {len(INVOCATIONS)} --cov invocations parsed"

    def test_the_known_collisions_are_recognised_as_collisions(self) -> None:
        """The three real cases must be *detected*, not merely absent.

        Without this, deleting the omit list would make the collision test pass
        vacuously. Each of these is a real target that is really omitted; the
        steps that use them are green only because they pass --cov-config.
        """
        patterns = _omit_patterns()
        for target in (
            "src/demos",
            "src/video_compression",
            "src/research/substrates/skfem_tri.py",
        ):
            assert _is_omitted(target, patterns) is not None, (
                f"{target} is in the omit list but _is_omitted did not flag it -- "
                "the collision detector is broken, so the gate above is vacuous"
            )

    def test_an_unrelated_target_is_not_flagged(self) -> None:
        """Scope: a rule that flags everything is as useless as one that flags nothing."""
        patterns = _omit_patterns()
        assert _is_omitted("src/mcts", patterns) is None
        assert _is_omitted("src/refinement", patterns) is None

    def test_a_target_containing_an_omitted_file_is_flagged(self) -> None:
        """The direction the first version of this guard missed.

        ``--cov=src/research/substrates`` with the global omit in force reports
        ~99% from four files while the fifth -- ``skfem_tri.py``, the module the
        gate was added for -- contributes nothing. A mutation check (dropping
        ``--cov-config`` from that CI step) passed against the ancestor-only
        rule, which is what turned this into a real test rather than a comment.
        """
        assert (
            _is_omitted("src/research/substrates", ["src/research/substrates/skfem_tri.py"])
            == "src/research/substrates/skfem_tri.py"
        )

    def test_every_elsewhere_gated_exemption_names_a_real_gate(self) -> None:
        """An exemption's "gated elsewhere" claim must be a fact, not a comment.

        Each exemption claims the omitted module is measured by some *other*
        coverage invocation. Verified by requiring some step to both *select*
        it and *override* the global omit. Delete that gate and this fails,
        instead of the exemption quietly becoming a hole.
        """
        blobs = [script for _, _, _, script in _iter_run_scripts()]
        for (target, omitted), reason in _OMITTED_BUT_GATED_ELSEWHERE.items():
            assert reason, f"exemption ({target}, {omitted}) has no reason"
            gated = any(_step_really_gates(blob, omitted) for blob in blobs)
            assert gated, (
                f"{omitted} is exempted from the {target} gate on the promise that it is "
                f"'gated elsewhere' ({reason}) -- but no workflow step both selects it "
                f"AND overrides the global omit, so nothing measures it anywhere. "
                f"Either restore that gate or drop the exemption."
            )

    def test_an_ancestor_gate_is_not_accepted_as_proof(self) -> None:
        """The subtlety a mutation check caught in the first version of this rule.

        An ancestor ``--cov`` (``--cov=src``) inherits the *same* global omit, so
        it cannot possibly be the "gated elsewhere" gate — accepting it made the
        exemption self-satisfying, and deleting the real gate left this file
        green. Proof now requires an omit **override** in the same step.
        """
        assert not _step_really_gates("pytest --cov=src --cov-fail-under=85", "src/demos/x.py")
        assert not _step_really_gates(
            "coverage run --include='*/src/demos/x.py' -m pytest", "src/demos/x.py"
        ), "an --include without an rcfile does not override pyproject's omit"
        assert _step_really_gates(
            "coverage run --rcfile=.rc --include='*/src/demos/x.py' -m pytest", "src/demos/x.py"
        )
        assert _step_really_gates(
            "pytest --cov=src/demos --cov-config=.rc --cov-fail-under=81", "src/demos/x.py"
        )

    def test_every_exemption_is_still_needed(self) -> None:
        """A stale exemption silently shrinks the rule's scope.

        If an omit pattern is removed from pyproject.toml, its exemption stops
        exempting anything and should be deleted rather than left behind.
        """
        patterns = set(_omit_patterns())
        stale = [
            (target, omitted)
            for (target, omitted) in _OMITTED_BUT_GATED_ELSEWHERE
            if omitted not in patterns
        ]
        assert not stale, (
            f"exemption(s) for omit pattern(s) that no longer exist: {stale} -- delete them"
        )

    def test_the_repo_wide_root_is_exempt_and_the_exemption_is_load_bearing(self) -> None:
        """`--cov=src` must stay exempt, and the exemption must still be needed.

        A stale exemption silently shrinks a rule's scope. If the omit list ever
        stopped overlapping ``src`` — i.e. the carve-out became unnecessary —
        this fails and the entry should be deleted rather than left to rot.
        """
        patterns = _omit_patterns()
        assert _is_omitted("src", patterns) is None, "the repo-wide gate must not be flagged"
        assert _OMIT_IS_INTENDED_FOR["src"], "an exemption without a reason is not an exemption"
        without_exemption = [
            pattern
            for pattern in patterns
            if pattern.rstrip("/*").startswith("src/") or pattern.rstrip("/*") == "src"
        ]
        assert without_exemption, (
            "no omit pattern overlaps 'src' any more, so the _OMIT_IS_INTENDED_FOR "
            "entry is doing nothing -- delete it"
        )

    @pytest.mark.parametrize(
        ("target", "pattern"),
        [
            ("src/demos", "src/demos/*"),
            ("src/demos/foo.py", "src/demos/*"),
            ("src/research/fem_baseline.py", "src/research/fem_baseline.py"),
        ],
    )
    def test_collision_matching_covers_both_pattern_shapes(self, target: str, pattern: str) -> None:
        assert _is_omitted(target, [pattern]) == pattern

    def test_a_prefix_that_is_not_a_path_boundary_is_not_a_match(self) -> None:
        """``src/demos/*`` must not swallow ``src/demos_extra``."""
        assert _is_omitted("src/demos_extra", ["src/demos/*"]) is None


def test_regression_surface_rows_cited_by_exemptions_exist() -> None:
    """An exemption may not point at a ``CLAUDE.md`` row that does not exist.

    The specific untruth this file shipped with: an exemption claiming coverage
    was "enforced instead by the `Eval-harness adapter` Regression Surface
    command", when ``grep -n "Eval-harness" CLAUDE.md`` returns nothing. Both
    pre-existing exemption meta-guards passed -- one checks a reason is present,
    the other that no gate has appeared -- because neither reads the reason.

    Only the mechanically decidable half is enforced here: a reason that names a
    Regression Surface row in backticks must name one that exists. "This module
    is fine, trust me" remains unfalsifiable, and is supposed to be rare.
    """
    surface = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    missing = _rows_missing_from_surface(_OMIT_WITHOUT_A_CI_GATE, surface)
    assert not missing, (
        "these exemptions justify themselves with a CLAUDE.md Regression Surface "
        f"row that does not exist: {missing}. Either add the row, or state the "
        "real reason -- an exemption backed by a citation that does not resolve is "
        "worse than no exemption, because it reads as enforced."
    )


class TestRowsCited:
    """Unit-tests the citation predicate on synthetic input, so it cannot be vacuous.

    Per the ``TestGatePredicate`` precedent in
    ``tests/research/test_amr_arena_interpretability.py``: a predicate that only
    ever runs behind live data whose current shape never enters its body is a
    predicate nothing checks. These cases do not depend on what
    ``_OMIT_WITHOUT_A_CI_GATE`` happens to contain today.
    """

    def test_a_valid_citation_is_extracted(self) -> None:
        reason = "enforced by the `Solver wiring` Regression Surface row instead"
        assert _rows_cited(reason) == ["Solver wiring"]

    def test_a_reason_with_no_citation_yields_nothing(self) -> None:
        assert _rows_cited("needs the optional [fem] extra; disclosed gap, see B37") == []

    def test_backticks_without_the_marker_are_not_a_citation(self) -> None:
        """``src/foo/*`` in backticks is a path, not a row claim."""
        assert _rows_cited("gated by `src/research/substrates` in test-extras") == []

    def test_missing_rows_are_separated_from_present_ones(self) -> None:
        """The property the live guard asserts, on data where both cases exist."""
        exemptions = {
            "src/a/*": "covered by the `Solver wiring` Regression Surface row",
            "src/b/*": "covered by the `Eval-harness adapter` Regression Surface row",
        }
        surface = "| Solver wiring | pytest tests/... |"
        assert _rows_missing_from_surface(exemptions, surface) == [
            ("src/b/*", "Eval-harness adapter")
        ]
