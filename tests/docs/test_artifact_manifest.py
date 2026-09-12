"""Hermetic guard: the committed benchmark artifacts match ``results/MANIFEST.sha256``.

Defect class, in one sentence: **a committed artifact that a headline number is
quoted from can be edited, replaced or deleted without anything in the tree
noticing.** Three headline claims have been retracted here because the number
was written down before, or instead of, the artifact behind it; a manifest makes
the artifact itself the thing that cannot drift silently. The manifest is
regenerated only by the ``claims-ledger`` / ``run-provenance`` skills, i.e. when
an artifact is *deliberately* produced or replaced -- any other byte change is a
defect, and one of the named tests below goes red on it.

Everything here is independent of the script's own ``check`` subcommand where it
matters: the tracked-file listing is a direct ``git ls-files``, and the hashes are
recomputed with :mod:`hashlib`. The script's config class *is* imported, because
the patterns must be one source of truth -- a guard whose vocabulary drifts from
the CLI's would be guarding a different set of files than the one being written.

The second half guards the *wiring*: a manifest nothing runs is a file. CI's
``lint`` job must run ``check`` as a hard step, the ``Makefile`` must mirror it,
the script's module-scope imports must all be installed by that job (it installs
a minimal set, so an innocent ``import yaml`` at module scope would break the
build), the ``transfer-baseline-regression`` upload must be the run's own output
rather than the committed files, and no workflow may write into ``results/``.

Mutation kills (``harden-a-guard``; planted on the real tree 2026-09-11 and
restored, each asserted to have applied before the run was trusted, none of
the killers carrying ``gpu_required``/``fem_required``):

1. **Flip one byte** of ``results/transfer_baseline_compare.csv``
   -> ``test_every_hash_matches_the_tree[results/transfer_baseline_compare.csv]``.
2. **Delete the manifest line** for that CSV
   -> ``test_every_tracked_artifact_has_an_entry[results/transfer_baseline_compare.csv]``
   (and ``test_manifest_is_not_vacuous``, since 9 < 10).
3. **Add an unlisted CSV** (``results/planted.csv``, ``git add``-ed so it is tracked)
   -> ``test_every_tracked_artifact_has_an_entry[results/planted.csv]``.
4. **Demote a hashed CSV to a presence-only comment line** -- the adjacent weaker
   defect. Confirmed during the kill: plain ``sha256sum -c`` exits **0** on the
   demoted manifest, because the line is now a comment
   -> ``test_every_tracked_artifact_has_an_entry[results/transfer_baseline_compare.csv]``
   (kind mismatch).
5. **Hash the excluded template** (``config/baselines/poc_headline.example.json``)
   -> ``test_every_entry_is_an_expected_artifact[config/baselines/poc_headline.example.json]``
   and ``test_the_manifest_and_the_template_are_never_hashed``.
6. **Revert the upload path** to ``results/transfer_baseline_compare.*``
   -> ``test_the_transfer_upload_is_the_run_not_the_committed_files``.
7. **Delete the lint step** ``python -m scripts.artifact_manifest check``
   -> ``test_the_lint_job_runs_the_check_as_a_hard_step``.
8. **Soften the lint step** with ``continue-on-error: true``
   -> ``test_the_lint_job_runs_the_check_as_a_hard_step``.
9. **Delete the Makefile target** ``artifact-manifest``
   -> ``test_the_makefile_target_mirrors_ci`` and ``test_pre_pr_chains_the_target``.
10. **Hoist ``import yaml`` to module scope** in the script
    -> ``test_the_lint_job_installs_every_module_scope_import``.
11. **Write into the frozen directory from CI** (``--output-dir results``)
    -> ``test_no_workflow_writes_into_the_frozen_directory`` and
    ``test_the_transfer_upload_is_the_run_not_the_committed_files``.

``test_regenerating_the_manifest_reproduces_it_byte_for_byte`` and
``test_the_ci_command_reaches_the_same_verdict`` failed on 1-5 as well (and
``test_plain_sha256sum_verifies_the_manifest`` on 1); the named per-entry
tests are listed because they say *which* file moved. Mutation 12 -- the
logging defect the partial branch shipped with -- is recorded in
``tests/scripts/test_artifact_manifest.py``, whose named test kills it.

11 planted defects here (12 with the unit file's), each killed by a *named*
test; the test count is larger and is not the number being claimed.
"""

from __future__ import annotations

import ast
import hashlib
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

import pytest

from scripts.artifact_manifest import (
    EXIT_OK,
    ArtifactManifestConfig,
    EntryKind,
    ManifestEntry,
    ManifestFormatError,
    _glob_match,
    parse_manifest,
    render_manifest,
)
from tests.support.workflows import (
    CI_SUCCESS_JOB,
    CI_WORKFLOW,
    CI_WORKFLOW_FILENAME,
    MAKEFILE,
    hard_gate_jobs,
    iter_commands,
    iter_run_scripts,
    job_needs,
    load_workflow,
    makefile_target_recipe,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: The shipped defaults are the contract; the CLI and this guard read the same class.
CONFIG: Final[ArtifactManifestConfig] = ArtifactManifestConfig()
MANIFEST: Final[Path] = REPO_ROOT / CONFIG.manifest_path
SCRIPT: Final[Path] = REPO_ROOT / "scripts" / "artifact_manifest.py"

#: Vacuity floor, deliberately a literal here and *not* read from the config: a
#: guard that asserts ``>= config.min_hashed_entries`` goes vacuous the moment
#: someone sets that field to 0. Ten is the count at first landing (2026-09-11);
#: the artifacts it freezes are the charter's evidence register, so this number
#: should only ever go up.
MIN_HASHED_ENTRIES: Final[int] = 10
MIN_PRESENCE_ENTRIES: Final[int] = 1

#: The one command CI, the Makefile and the subprocess test below all run.
MANIFEST_CHECK_COMMAND: Final[str] = "python -m scripts.artifact_manifest check"
LINT_JOB: Final[str] = "lint"
MAKE_TARGET: Final[str] = "artifact-manifest"
TRANSFER_JOB: Final[str] = "transfer-baseline-regression"

#: The script's documented module-scope dependency footprint. Anything else it
#: imports at module scope must be added here *and* to the lint job's install
#: step, or the ``lint`` job -- which installs a minimal set -- fails on import.
EXPECTED_MODULE_SCOPE_THIRD_PARTY: Final[frozenset[str]] = frozenset({"structlog", "pydantic"})

#: Import name -> distribution name, for the few that differ. Only consulted
#: when an import name is not itself found in the install step.
_IMPORT_TO_DIST: Final[dict[str, str]] = {"yaml": "pyyaml"}

#: The directory the manifest freezes; CI must never write into it.
FROZEN_DIR: Final[str] = "results"

_OUTPUT_DIR_FLAG: Final[re.Pattern[str]] = re.compile(r"(?:--output-dir|--out)(?:=|\s+)(\S+)")


def _git_tracked(scan_roots: tuple[str, ...]) -> list[str]:
    """Direct ``git ls-files``, independent of the script's lister."""
    out = subprocess.check_output(
        ["git", "ls-files", "-z", "--", *scan_roots], cwd=REPO_ROOT, timeout=60
    )
    return sorted(p for p in out.decode("utf-8").split("\0") if p)


def _expected_kinds() -> dict[str, EntryKind]:
    """Every tracked file the config says is frozen, with its required kind."""
    kinds: dict[str, EntryKind] = {}
    for path in _git_tracked(CONFIG.scan_roots):
        kind = CONFIG.classify(path)
        if kind is not None:
            kinds[path] = kind
    return kinds


def _listed() -> dict[str, ManifestEntry]:
    """The manifest's entries, by path. Empty (not an error) if absent or unreadable.

    Feeds ``parametrize`` at collection time, so a missing or malformed manifest
    must yield zero cases rather than a collection crash -- ``test_manifest_exists``
    and ``test_manifest_parses`` are the tests that report those states clearly,
    and ``test_manifest_is_not_vacuous`` goes red on the empty result either way.
    """
    if not MANIFEST.is_file():
        return {}
    try:
        entries = parse_manifest(MANIFEST.read_text(encoding="utf-8"), CONFIG)
    except ManifestFormatError:
        return {}
    return {entry.path: entry for entry in entries}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matches(path: str, pattern: str) -> bool:
    return _glob_match(path, pattern)


def _ci_job(name: str) -> dict[str, Any]:
    jobs = load_workflow(CI_WORKFLOW).get("jobs", {})
    assert name in jobs, f"{CI_WORKFLOW_FILENAME} has no job {name!r}"
    job = jobs[name]
    assert isinstance(job, dict)
    return job


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    steps = job.get("steps")
    assert isinstance(steps, list) and steps, "job has no steps"
    return [step for step in steps if isinstance(step, dict)]


# --------------------------------------------------------------------------- #
# Existence and vacuity -- before anything else, because everything below      #
# parametrises over these sets and passes on an empty one.                     #
# --------------------------------------------------------------------------- #


def test_manifest_exists() -> None:
    """Without the file, every parametrised test below iterates nothing and passes."""
    assert MANIFEST.is_file(), (
        f"{CONFIG.manifest_path} is missing; run `python -m scripts.artifact_manifest write`"
    )


def test_manifest_parses() -> None:
    """A malformed manifest is reported here, not as a collection crash or an empty set."""
    parse_manifest(MANIFEST.read_text(encoding="utf-8"), CONFIG)


def test_manifest_is_not_vacuous() -> None:
    """A manifest freezing (almost) nothing is the false-pass shape this repo keeps hitting."""
    listed = _listed()
    hashed = [e for e in listed.values() if e.kind is EntryKind.HASHED]
    presence = [e for e in listed.values() if e.kind is EntryKind.PRESENCE]
    assert len(hashed) >= MIN_HASHED_ENTRIES, f"only {len(hashed)} hashed entries"
    assert len(presence) >= MIN_PRESENCE_ENTRIES, f"only {len(presence)} presence entries"


def test_the_scanner_reaches_real_subjects() -> None:
    """Each pattern in the shipped config must match at least one tracked file.

    One arm still matching can mask another going unexamined: a pattern that
    matches nothing is a *vocabulary* defect (the ``Dörfler``-vs-``dorfler``
    class), and this is the test that would have caught it.
    """
    tracked = _git_tracked(CONFIG.scan_roots)
    assert tracked, "git ls-files returned nothing under the scan roots"
    for pattern in (*CONFIG.hash_patterns, *CONFIG.presence_patterns):
        hits = [p for p in tracked if CONFIG.classify(p) is not None and _matches(p, pattern)]
        assert hits, f"pattern {pattern!r} matches no tracked file -- it guards nothing"


def test_excluded_paths_are_still_real_and_still_need_excluding() -> None:
    """An exemption must expire in both directions.

    If the excluded file is gone, the entry is dead and should be deleted; if it
    no longer matches any pattern, the exclusion is doing nothing and should be
    deleted. Either way a stale exclusion silently shrinks what is auditable.
    """
    tracked = set(_git_tracked(CONFIG.scan_roots))
    for excluded in CONFIG.exclude:
        assert excluded in tracked, f"exclude entry {excluded!r} is not a tracked file"
        would_match = any(
            _matches(excluded, pattern)
            for pattern in (*CONFIG.hash_patterns, *CONFIG.presence_patterns)
        )
        assert would_match, f"exclude entry {excluded!r} matches no pattern; it is inert"


# --------------------------------------------------------------------------- #
# Tree -> manifest: every frozen artifact is listed, with the right kind.       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", sorted(_expected_kinds()))
def test_every_tracked_artifact_has_an_entry(path: str) -> None:
    """A tracked artifact with no entry landed without the skill.

    Wrong kind is the adjacent weaker defect: a hashed file demoted to a
    presence-only comment line still passes ``sha256sum -c``.
    """
    listed = _listed()
    assert path in listed, (
        f"{path} is a tracked artifact but has no entry in {CONFIG.manifest_path}; "
        f"if it was deliberately added, regenerate the manifest via the claims-ledger skill"
    )
    expected_kind = _expected_kinds()[path]
    assert listed[path].kind is expected_kind, (
        f"{path} is listed as {listed[path].kind.value} but the config requires "
        f"{expected_kind.value}"
    )


# --------------------------------------------------------------------------- #
# Manifest -> tree: every entry names a real, tracked, frozen file.            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", sorted(_listed()))
def test_every_entry_is_an_expected_artifact(path: str) -> None:
    """Every entry must name a tracked, frozen, present file.

    Catches a deleted artifact, an entry for an untracked file, the manifest
    hashing itself, and the excluded template being hashed.
    """
    expected = _expected_kinds()
    assert path in expected, (
        f"{CONFIG.manifest_path} lists {path}, which is not a tracked frozen artifact "
        f"(deleted, untracked, excluded, or the manifest itself)"
    )
    assert (REPO_ROOT / path).is_file(), f"{path} is listed but absent from the working tree"


@pytest.mark.parametrize(
    "path",
    sorted(p for p, e in _listed().items() if e.kind is EntryKind.HASHED),
)
def test_every_hash_matches_the_tree(path: str) -> None:
    """The load-bearing assertion: the bytes a number was quoted from have not moved."""
    entry = _listed()[path]
    file_path = REPO_ROOT / path
    if not file_path.is_file():
        pytest.fail(f"{path} is listed but absent -- see test_every_entry_is_an_expected_artifact")
    actual = _sha256(file_path)
    assert entry.digest is not None
    assert actual == entry.digest, (
        f"{path} has changed since the manifest was written "
        f"(manifest {entry.digest[:12]}..., tree {actual[:12]}...). If this is a deliberate "
        f"regeneration, follow the claims-ledger / run-provenance skill and rerun "
        f"`python -m scripts.artifact_manifest write`; otherwise revert the artifact."
    )


def test_the_manifest_and_the_template_are_never_hashed() -> None:
    """The two exclusions a future config edit is most likely to get wrong.

    Spelled out even though ``test_every_entry_is_an_expected_artifact``
    covers both, so the failure names the rule rather than the symptom.
    """
    listed = _listed()
    assert CONFIG.manifest_path not in listed, "the manifest hashes itself"
    for excluded in CONFIG.exclude:
        assert excluded not in listed, f"excluded template {excluded} has an entry"


# --------------------------------------------------------------------------- #
# Whole-file agreement and format compatibility.                               #
# --------------------------------------------------------------------------- #


def test_regenerating_the_manifest_reproduces_it_byte_for_byte() -> None:
    """The committed manifest is exactly what ``write`` would produce today.

    Stronger than the per-entry tests together: it also pins ordering, header
    and line endings, so a hand-edited manifest that happens to list the right
    hashes still fails, which is what "regenerated only by the skill" means.
    """
    entries = [
        ManifestEntry(
            path=path,
            kind=kind,
            digest=_sha256(REPO_ROOT / path) if kind is EntryKind.HASHED else None,
        )
        for path, kind in sorted(_expected_kinds().items())
    ]
    assert MANIFEST.read_bytes() == render_manifest(entries, CONFIG).encode("utf-8")


def test_the_ci_command_reaches_the_same_verdict() -> None:
    """The exact command CI runs, as a subprocess from the repo root, must exit 0.

    A subprocess rather than an in-process ``main()``: that is what the ``lint``
    job does, and it keeps the CLI's global ``structlog.configure`` out of this
    test session.
    """
    argv = shlex.split(MANIFEST_CHECK_COMMAND)
    assert argv[0] == "python"
    completed = subprocess.run(
        [sys.executable, *argv[1:]],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == EXIT_OK, completed.stdout + completed.stderr
    assert "OK: every frozen artifact matches" in completed.stdout


def test_plain_sha256sum_verifies_the_manifest() -> None:
    """``sha256sum -c`` compatibility is a documented property, so it is tested.

    Presence-only lines are ``#`` comments, which GNU coreutils skips; a marker
    that was *not* a comment would make this exit non-zero with
    "improperly formatted lines".
    """
    tool = shutil.which("sha256sum")
    if tool is None:
        pytest.skip("GNU sha256sum not on PATH (macOS ships shasum); format checked in-process")
    completed = subprocess.run(
        [tool, "-c", "--strict", CONFIG.manifest_path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_the_manifest_uses_lf_line_endings() -> None:
    """``sha256sum -c`` treats a trailing carriage return as part of the filename."""
    assert b"\r" not in MANIFEST.read_bytes()


# --------------------------------------------------------------------------- #
# Wiring: a manifest nothing runs is a file.                                   #
# --------------------------------------------------------------------------- #


def _lint_step_running_the_check() -> dict[str, Any] | None:
    for step in _steps(_ci_job(LINT_JOB)):
        script = step.get("run")
        if isinstance(script, str) and MANIFEST_CHECK_COMMAND in iter_commands(script):
            return step
    return None


def test_the_lint_job_runs_the_check_as_a_hard_step() -> None:
    """Defect class: a check that exists but runs nowhere, or runs and cannot fail.

    The step must be in the ``lint`` job (the smallest install, so the
    dependency-footprint guard below has teeth), must not be ``continue-on-error``,
    and ``lint`` itself must be a hard ``ci-success`` gate -- a job in ``needs``
    whose result is only echoed blocks nothing.
    """
    lint_scripts = [
        rs
        for rs in iter_run_scripts()
        if rs.workflow == CI_WORKFLOW_FILENAME and rs.job == LINT_JOB
    ]
    assert lint_scripts, f"no run: steps found in the {LINT_JOB!r} job"
    hits = [rs for rs in lint_scripts if MANIFEST_CHECK_COMMAND in iter_commands(rs.script)]
    assert len(hits) == 1, (
        f"expected exactly one {LINT_JOB!r} step running {MANIFEST_CHECK_COMMAND!r}, "
        f"found {[str(rs) for rs in hits]}"
    )
    step = _lint_step_running_the_check()
    assert step is not None
    assert not step.get("continue-on-error"), f"{step.get('name')} is soft; it gates nothing"

    document = load_workflow(CI_WORKFLOW)
    assert LINT_JOB in job_needs(document, CI_SUCCESS_JOB)
    success_script = next(
        rs.script
        for rs in iter_run_scripts()
        if rs.workflow == CI_WORKFLOW_FILENAME and rs.job == CI_SUCCESS_JOB
    )
    assert LINT_JOB in hard_gate_jobs(success_script), f"{LINT_JOB!r} is not a hard gate"


def test_the_makefile_target_mirrors_ci() -> None:
    """``make artifact-manifest`` must run the same command as CI, via ``$(PYTHON)``."""
    recipe = makefile_target_recipe(MAKE_TARGET)
    assert recipe, f"Makefile has no {MAKE_TARGET!r} target"
    normalised = [command.replace("$(PYTHON)", "python") for command in recipe]
    assert normalised == [MANIFEST_CHECK_COMMAND], normalised
    phony = re.search(r"^\.PHONY:((?:.*\\\n)*.*)$", MAKEFILE.read_text(encoding="utf-8"), re.M)
    assert phony is not None, "no .PHONY declaration"
    assert MAKE_TARGET in phony.group(1).replace("\\\n", " ").split(), f"{MAKE_TARGET} not .PHONY"


def test_pre_pr_chains_the_target() -> None:
    """``make pre-pr`` is the documented pre-PR mirror of CI; it must include this gate."""
    text = MAKEFILE.read_text(encoding="utf-8")
    match = re.search(r"^pre-pr:\s*(?P<deps>.*)$", text, re.M)
    assert match is not None, "no pre-pr target"
    assert MAKE_TARGET in match.group("deps").split()


def _module_scope_third_party_imports(script: Path) -> set[str]:
    """Top-level names imported at module scope that are not in the stdlib."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    return {name for name in names if name not in stdlib}


def _lint_job_installed_distributions() -> set[str]:
    """Every distribution name handed to ``pip install`` in the lint job, lowercased."""
    installed: set[str] = set()
    for step in _steps(_ci_job(LINT_JOB)):
        script = step.get("run")
        if not isinstance(script, str):
            continue
        for command in iter_commands(script):
            tokens = shlex.split(command)
            if "install" not in tokens or "pip" not in tokens:
                continue
            for token in tokens[tokens.index("install") + 1 :]:
                if token.startswith("-"):
                    continue
                installed.add(re.split(r"[<>=!~;\[]", token, maxsplit=1)[0].lower())
    return installed


def test_the_lint_job_installs_every_module_scope_import() -> None:
    """Defect class: the check imports something the job that runs it never installed.

    The ``lint`` job installs a minimal, explicitly-named set. The script's
    module-scope third-party imports are pinned here as the documented
    footprint (stdlib + structlog + pydantic) *and* each must be named in that
    install step -- so hoisting ``import yaml`` to module scope, or dropping
    ``structlog`` from the install line, fails this test rather than the build.
    """
    third_party = _module_scope_third_party_imports(SCRIPT)
    assert third_party == EXPECTED_MODULE_SCOPE_THIRD_PARTY, (
        f"module-scope footprint changed: {sorted(third_party)}; update the docstring, "
        f"EXPECTED_MODULE_SCOPE_THIRD_PARTY and the lint job's install step together"
    )
    installed = _lint_job_installed_distributions()
    assert installed, f"no `pip install` found in the {LINT_JOB!r} job -- the guard reached nothing"
    for name in sorted(third_party):
        candidates = {name.lower(), _IMPORT_TO_DIST.get(name, name).lower()}
        assert candidates & installed, (
            f"{name!r} is imported at module scope by {SCRIPT.name} but the {LINT_JOB!r} job "
            f"does not install it (installs: {sorted(installed)})"
        )


def _output_dirs(script: str) -> list[str]:
    return [
        match.group(1).strip("'\"")
        for command in iter_commands(script)
        for match in _OUTPUT_DIR_FLAG.finditer(command)
    ]


def test_the_transfer_upload_is_the_run_not_the_committed_files() -> None:
    """Defect class: the uploaded "artifact" is the repository, not the run.

    The run step writes ``--output-dir <X>``; the upload step's ``path`` must
    sit under ``<X>`` and must not point into the frozen directory. Until
    2026-09-11 it uploaded ``results/transfer_baseline_compare.*`` -- files the
    run never touches -- so the artifact could not tell a reader anything the
    checkout did not already say.
    """
    steps = _steps(_ci_job(TRANSFER_JOB))
    output_dirs = [
        d for step in steps if isinstance(step.get("run"), str) for d in _output_dirs(step["run"])
    ]
    assert len(output_dirs) == 1, f"expected one --output-dir in {TRANSFER_JOB}, got {output_dirs}"
    output_dir = output_dirs[0].rstrip("/")
    assert not (output_dir == FROZEN_DIR or output_dir.startswith(FROZEN_DIR + "/"))

    uploads = [s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    assert len(uploads) == 1, f"expected one upload-artifact step in {TRANSFER_JOB}"
    with_block = uploads[0].get("with")
    assert isinstance(with_block, dict)
    path = str(with_block.get("path", "")).strip()
    assert path, "upload step has no path"
    assert path == output_dir or path.startswith(output_dir + "/"), (
        f"upload path {path!r} is not under the run's --output-dir {output_dir!r}"
    )
    assert not path.startswith(FROZEN_DIR), f"upload path {path!r} is the committed tree"


def test_no_workflow_writes_into_the_frozen_directory() -> None:
    """Defect class: a CI run overwrites a committed artifact and uploads the result.

    Every ``--output-dir`` / ``--out`` in every workflow must point outside
    ``results/``. The shipped scenario YAMLs default to ``output_dir: results``,
    so a step that omits the flag inherits that default -- which is why the
    ``claims-ledger`` skill says to prefer ``--output-dir outputs/...``.
    """
    seen: list[tuple[str, str]] = []
    for rs in iter_run_scripts():
        for target in _output_dirs(rs.script):
            seen.append((str(rs), target))
    assert seen, "no --output-dir found in any workflow; the guard examined nothing"
    offending = [
        (where, target)
        for where, target in seen
        if target.rstrip("/") == FROZEN_DIR or target.startswith(FROZEN_DIR + "/")
    ]
    assert not offending, f"workflow steps writing into {FROZEN_DIR}/: {offending}"
