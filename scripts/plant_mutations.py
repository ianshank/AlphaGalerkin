r"""Plant each defect in a spec, run the narrowest selection, record the NAMED kills, restore.

Usage (from the repo root):

    python -m scripts.plant_mutations path/to/mutations.py [--only LABEL ...]

The spec is a Python file defining ``MUTATIONS``, a list of dicts:

    MUTATIONS = [
        {
            "label": "N1 multi-input spec left flat",  # printed; --only matches its prefix
            "path": "src/deployment/export_onnx.py",   # repo-relative
            "old": "shapes = TupleOutputWrapper.dynamic_shapes_for(shapes)\n",  # exactly once
            "new": "pass\n",
            "select": ["tests/deployment/test_export_adapters.py::TestExportDispatch"],
        },
    ]

For each entry the file is changed, pytest runs on ``select`` with CI's marker filter, and the
file is restored byte-for-byte (then re-read to prove it). Steps 2-4 of the skill, mechanised:

- an anchor that is missing or not unique is reported as ``ANCHOR``, never run: a mutation that
  silently no-ops produces a green run that looks like a survival, or worse, a kill elsewhere;
- each selection first runs **unmutated**; a test already failing there is not counted as a
  kill, and a selection that is red before any mutation is reported as ``BASELINE-RED``;
- a kill is a ``FAILED <node id>`` line that was green in the baseline. A run that only
  ``ERROR``s (collection or fixture errors, an ImportError taking the module down) is
  ``BROKEN``, not a kill -- it names no test that understood the defect;
- every verdict prints the run's first ``E`` line as its ``reason``. A test that failed because a
  dependency raised is not a kill (B52's first two plants made onnxruntime raise and never
  reached the assertion), so a mutation may set ``expect``, a regex the reason must match; a
  kill for any other reason is then ``WRONG-REASON``. Without ``expect``, read the reason;
- a run longer than ``--timeout`` seconds is ``TIMEOUT``;
- the marker filter defaults to CI's ``not gpu_required and not fem_required``, so a killer that
  skips on CI cannot be recorded as one.

Restoration survives a hard kill: before a file is mutated its original bytes are journalled to
``<repo>/.plant-journal.json`` (gitignored), and the next run restores any file a killed run left
mutated before doing anything else -- but only if the file still holds exactly the mutated
bytes (the journal records their SHA-256), never over work done since. The journal lives in
the repository, not the shared temp
directory, and every path it or a spec names must resolve inside the repository: a journal
planted by another user cannot become an arbitrary file write. Writes are bytes, so line
endings survive (``Path.write_text`` writes CRLF on Windows). Exits non-zero unless at least
one mutation ran and every one was killed.

``main`` takes the repository root and the pytest runner as keyword arguments, so
``tests/scripts/test_plant_mutations.py`` drives every verdict without a subprocess. The
``harden-a-guard`` skill (``.claude/skills/harden-a-guard/SKILL.md``) is its procedure.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import runpy
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_MARKER_EXPR = "not gpu_required and not fem_required"
DEFAULT_TIMEOUT_S = 600
JOURNAL_NAME = ".plant-journal.json"
FAILED_LINE = re.compile(r"^FAILED (\S+)")
ERROR_LINE = re.compile(r"^ERROR (\S+)")
REASON_LINE = re.compile(r"^E\s")
SUMMARY_LINE = re.compile(r"=+ .*(passed|failed|error|no tests ran).* =+$")


class RunResult(NamedTuple):
    """What one pytest run reported: failed and errored node ids, summary, first ``E`` line."""

    failed: tuple[str, ...]
    errored: tuple[str, ...]
    summary: str
    reason: str


Runner = Callable[[list[str], argparse.Namespace, Path], RunResult]


def parse_pytest_output(stdout: str) -> RunResult:
    """Read the node ids, the summary line and the first ``E`` line out of pytest's output."""
    lines = stdout.splitlines()
    failed = tuple(m.group(1) for line in lines if (m := FAILED_LINE.match(line)))
    errored = tuple(m.group(1) for line in lines if (m := ERROR_LINE.match(line)))
    summary = next((line for line in reversed(lines) if SUMMARY_LINE.search(line)), "")
    reason = next((line for line in lines if REASON_LINE.match(line)), "")
    return RunResult(failed, errored, summary.strip("= "), reason)


def run_pytest(select: list[str], args: argparse.Namespace, repo_root: Path) -> RunResult:
    """The default runner: pytest on ``select`` in a subprocess rooted at ``repo_root``."""
    # -B: a same-size mutation written within the second a baseline compiled could reuse its .pyc.
    command = [sys.executable, "-B", "-m", "pytest", *select, "-q", "-p", "no:randomly"]
    command += ["-p", "no:cacheprovider", "--no-header", "-m", args.marker_expr, *args.pytest_arg]
    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        timeout=args.timeout,
        stdin=subprocess.DEVNULL,
    )
    return parse_pytest_output(result.stdout)


def journal_path(repo_root: Path) -> Path:
    """Where a run journals the file it is about to mutate: inside the repository itself."""
    return repo_root / JOURNAL_NAME


def contained_path(repo_root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``repo_root``; raise ``ValueError`` if it lands outside."""
    root = repo_root.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"{relative!r} resolves outside the repository {root}")
    return target


def encode_original(original: bytes) -> str:
    """The journal's encoding of a file's original bytes."""
    return base64.b64encode(original).decode()


def journal_entry(path: str, original: bytes, mutated: bytes) -> dict[str, str]:
    """The journal record for one mutation: what to restore, and what it must replace."""
    return {
        "path": path,
        "original": encode_original(original),
        "mutated_sha256": hashlib.sha256(mutated).hexdigest(),
    }


def recover(repo_root: Path) -> str | None:
    """Restore a file a killed run left mutated; return its path, or None if none was.

    Raises:
        ValueError: The journal names a path outside the repository.
        RuntimeError: The file holds neither the original nor the journalled mutation,
            so it was edited since the killed run; nothing is written.

    """
    journal = journal_path(repo_root)
    if not journal.exists():
        return None
    entry = json.loads(journal.read_text(encoding="utf-8"))
    target = contained_path(repo_root, entry["path"])  # raises before anything is written
    original = base64.b64decode(entry["original"])
    current = target.read_bytes() if target.exists() else None
    if current == original:
        journal.unlink()
        return None
    mutated_digest = entry.get("mutated_sha256")
    if current is None or hashlib.sha256(current).hexdigest() != mutated_digest:
        raise RuntimeError(
            f"refusing to recover {entry['path']}: it changed since the killed run (it holds "
            f"neither the original nor the mutation). Inspect it, then delete {journal}."
        )
    target.write_bytes(original)
    journal.unlink()
    return str(entry["path"])


class Verdict(NamedTuple):
    """One mutation's outcome, and the node ids or messages that support it."""

    kind: str
    names: tuple[str, ...]
    summary: str
    reason: str


def run_mutation(
    mutation: dict[str, Any],
    baseline: frozenset[str],
    args: argparse.Namespace,
    repo_root: Path,
    runner: Runner,
) -> Verdict:
    """Plant ``mutation``, run its selection, restore the file, and judge the run."""
    path = contained_path(repo_root, mutation["path"])
    original = path.read_bytes()
    text = original.decode("utf-8")
    count = text.count(mutation["old"])
    if count != 1:
        return Verdict("ANCHOR", (f"anchor occurs {count} times in {mutation['path']}",), "", "")

    mutated = text.replace(mutation["old"], mutation["new"], 1).encode("utf-8")
    journal = journal_path(repo_root)
    journal.write_text(
        json.dumps(journal_entry(mutation["path"], original, mutated)), encoding="utf-8"
    )
    path.write_bytes(mutated)
    try:
        result = runner(list(mutation["select"]), args, repo_root)
    except subprocess.TimeoutExpired:
        return Verdict("TIMEOUT", (f"no result within {args.timeout} s",), "", "")
    finally:
        path.write_bytes(original)
        journal.unlink(missing_ok=True)
    if path.read_bytes() != original:
        raise RuntimeError(f"{mutation['path']} was not restored byte-for-byte -- stop and check")

    killers = tuple(name for name in result.failed if name not in baseline)
    expect = mutation.get("expect")
    if killers and expect and not re.search(expect, result.reason):
        return Verdict("WRONG-REASON", killers, result.summary, result.reason)
    if killers:
        also = tuple(f"(also errored: {name})" for name in result.errored)
        return Verdict("KILLED", killers + also, result.summary, result.reason)
    if result.errored:
        return Verdict("BROKEN", result.errored, result.summary, result.reason)
    return Verdict("SURVIVED", (), result.summary, result.reason)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("spec", type=Path, help="Python file defining MUTATIONS")
    parser.add_argument("--only", nargs="*", default=[], help="run labels with these prefixes")
    parser.add_argument("--marker-expr", default=CI_MARKER_EXPR, help="pytest -m expression")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help="seconds per run")
    parser.add_argument(
        "--pytest-arg", action="append", default=[], help="extra pytest argument (repeatable)"
    )
    return parser.parse_args(argv)


def _print_verdict(label: str, verdict: Verdict) -> None:
    print(f"\n== {label}: {verdict.kind}  {verdict.summary}")
    for name in verdict.names:
        print(f"    {name}")
    if verdict.reason:
        print(f"    reason: {verdict.reason.removeprefix('E').strip()}")


def main(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path | None = None,
    runner: Runner | None = None,
) -> int:
    """Run every selected mutation; return 1 unless each was killed by a named test."""
    args = _parse_args(argv)
    root = repo_root or REPO_ROOT
    run = runner or run_pytest
    recovered = recover(root)
    if recovered is not None:
        print(f"recovered {recovered} from a previous run that was killed mid-mutation")

    mutations: list[dict[str, Any]] = runpy.run_path(str(args.spec))["MUTATIONS"]
    selected = [
        m for m in mutations if not args.only or any(m["label"].startswith(o) for o in args.only)
    ]
    if not selected:
        print(f"no mutation matches {args.only or 'the spec'} -- nothing would be checked")
        return 1

    baselines: dict[tuple[str, ...], frozenset[str]] = {}
    bad = 0
    for mutation in selected:
        key = tuple(mutation["select"])
        if key not in baselines:
            clean = run(list(key), args, root)
            baselines[key] = frozenset(clean.failed)
            if clean.failed or clean.errored:
                print(f"\n== BASELINE-RED for {list(key)}: {clean.summary}")
                for name in clean.failed + clean.errored:
                    print(f"    {name}")
        verdict = run_mutation(mutation, baselines[key], args, root, run)
        bad += verdict.kind != "KILLED"
        _print_verdict(mutation["label"], verdict)
    print(f"\n{'all planted defects killed' if not bad else f'{bad} not killed'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
