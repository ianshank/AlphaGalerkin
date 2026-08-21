"""The `--allow-unsafe-pickle` opt-in must exist and reach the loader.

Five CLI entry points take a checkpoint path straight from an argument. Each was
changed to deserialize safely by default and to expose the unsafe path only
behind an explicit flag. Two things can silently undo that, and neither shows up
in a normal test run:

1. The flag is declared but never threaded to the loader, so passing it does
   nothing and an operator with a genuinely legacy file has no way through
   except editing source -- which is how these call sites acquired
   ``weights_only=False`` in the first place.
2. The flag is threaded but the *default* drifts to ``True``, re-opening the
   hole while every functional test still passes.

These are argument-plumbing assertions, deliberately: the deserialization
behaviour itself is covered by ``tests/security/test_checkpoint_safety.py`` and
``test_codec_checkpoint_safety.py``. Running the scripts end-to-end would need
real checkpoints and, for the video pair, real bitstreams.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

pytestmark = pytest.mark.security

REPO_ROOT = Path(__file__).resolve().parents[2]

# (module path, the argparse dest the flag must produce)
CLI_ENTRY_POINTS = [
    "src/experiments/verify_transfer.py",
    "scripts/play_engine.py",
    "scripts/encode_video.py",
    "scripts/decode_video.py",
    "scripts/inspect_checkpoint.py",
]

FLAG = "--allow-unsafe-pickle"
DEST = "allow_unsafe_pickle"


@pytest.mark.parametrize("rel_path", CLI_ENTRY_POINTS)
def test_declares_the_flag(rel_path: str) -> None:
    """Every checkpoint-taking entry point exposes the opt-in."""
    source = (REPO_ROOT / rel_path).read_text()
    assert FLAG in source, f"{rel_path} does not declare {FLAG}"


@pytest.mark.parametrize("rel_path", CLI_ENTRY_POINTS)
def test_flag_defaults_to_off(rel_path: str) -> None:
    """`store_true` is what makes the safe path the default.

    Parsed from the AST rather than grepped, so a ``default=True`` added
    alongside it cannot slip through on a substring match.
    """
    tree = ast.parse((REPO_ROOT / rel_path).read_text())
    found = False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"):
            continue
        if not any(isinstance(a, ast.Constant) and a.value == FLAG for a in node.args):
            continue
        found = True
        kwargs = {k.arg: k.value for k in node.keywords}
        assert isinstance(kwargs.get("action"), ast.Constant), f"{rel_path}: no action="
        assert kwargs["action"].value == "store_true", (
            f"{rel_path}: {FLAG} must be store_true so the default is safe"
        )
        if "default" in kwargs:
            assert isinstance(kwargs["default"], ast.Constant)
            assert kwargs["default"].value is False, (
                f"{rel_path}: {FLAG} has an explicit truthy default"
            )
    assert found, f"{rel_path}: {FLAG} is not registered via add_argument"


@pytest.mark.parametrize("rel_path", CLI_ENTRY_POINTS)
def test_flag_is_threaded_to_a_loader(rel_path: str) -> None:
    """Declared is not enough — it must reach a call.

    The flag has to appear as a keyword argument somewhere, otherwise it parses
    into a namespace nothing reads.
    """
    tree = ast.parse((REPO_ROOT / rel_path).read_text())
    threaded = any(
        isinstance(node, ast.Call) and any(kw.arg == DEST for kw in node.keywords)
        for node in ast.walk(tree)
    )
    assert threaded, f"{rel_path}: {FLAG} is parsed but never passed to any call"


def test_verify_transfer_threads_through_every_level() -> None:
    """The one entry point where the flag crosses three functions.

    `main` -> `run_verification` -> `load_model`. A default that stops being
    keyword-with-default at any level would break existing callers, and a level
    that drops the argument would silently ignore the flag.
    """
    from src.experiments import verify_transfer as vt

    for fn in (vt.load_model, vt.run_verification):
        params = inspect.signature(fn).parameters
        assert DEST in params, f"{fn.__name__} does not accept {DEST}"
        assert params[DEST].default is False, f"{fn.__name__} does not default {DEST} to False"
        assert params[DEST].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{fn.__name__}: {DEST} should be keyword-only so positional callers are unaffected"
        )


def test_load_codec_exposes_the_opt_in_on_the_primary_path() -> None:
    """Guards the gap where the flag reached only `decode_video`'s fallback.

    If `load_codec` cannot accept the opt-in, an operator passing the flag for a
    legacy file gets the fallback's reduced load (`strict=False`, possibly a
    default config) instead of the real one.
    """
    from src.video_compression.codec.codec import load_codec

    params = inspect.signature(load_codec).parameters
    assert DEST in params
    assert params[DEST].default is False
    assert params[DEST].kind is inspect.Parameter.KEYWORD_ONLY


def test_inspect_checkpoint_parses_and_defaults_safely() -> None:
    """Exercises the script's OWN parser, via `build_parser`.

    This previously constructed a local `ArgumentParser` with the same two
    arguments and asserted against that -- a tautology that stayed green
    independent of the script, and would have passed with the flag removed from
    it entirely. `build_parser` exists so the assertion has a real subject.
    """
    from scripts import inspect_checkpoint

    parser = inspect_checkpoint.build_parser()

    assert parser.parse_args(["some.pt"]).allow_unsafe_pickle is False
    assert parser.parse_args(["some.pt", FLAG]).allow_unsafe_pickle is True

    params = inspect.signature(inspect_checkpoint.inspect).parameters
    assert params[DEST].default is False


def _flag_argument_nodes(rel_path: str) -> list[ast.expr]:
    """Every value node passed as the ``allow_unsafe_pickle=`` keyword in a file."""
    tree = ast.parse((REPO_ROOT / rel_path).read_text())
    return [
        kw.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == DEST
    ]


@pytest.mark.parametrize("rel_path", CLI_ENTRY_POINTS)
def test_the_flag_value_flows_from_the_parsed_arguments(rel_path: str) -> None:
    """Closes failure mode 2 in this module's docstring, which was NOT covered.

    `test_flag_is_threaded_to_a_loader` only asserts that the identifier appears
    as *some* keyword in *some* call, so `f(allow_unsafe_pickle=True)` satisfies
    it; `test_inspect_checkpoint_parses_and_defaults_safely` asserts on the
    parser, not on what `main` does with the parsed value. Mutation-verified
    before this test existed: hardcoding `allow_unsafe_pickle=True` at every one
    of these call sites left the whole suite green -- 94 passed -- while the
    safe default was gone. `verify_transfer.py` was at 99% line coverage and the
    mutation still survived, which is what coverage without a value assertion
    buys you.

    So assert the *shape* of the value: it must flow from a variable
    (`args.allow_unsafe_pickle`, or a parameter threaded down from it), never a
    literal. A constant here means the operator's choice has been overridden in
    source -- in the unsafe direction if `True`, and if `False`, the flag is
    inert and the operator has no way through at all.
    """
    values = _flag_argument_nodes(rel_path)
    assert values, f"{rel_path} passes {DEST} to nothing"

    literals = [v for v in values if isinstance(v, ast.Constant)]
    assert not literals, (
        f"{rel_path} hardcodes {DEST}="
        f"{[v.value for v in literals]} instead of passing the parsed value; "
        "the operator's choice is overridden in source"
    )
    for value in values:
        assert isinstance(value, ast.Name | ast.Attribute), (
            f"{rel_path} passes {DEST} as {type(value).__name__}, which this guard "
            "cannot prove flows from the CLI; use the parsed value directly"
        )


def test_no_source_file_hardcodes_the_opt_in_on(tmp_path: Path) -> None:
    """Repo-wide, not just the five entry points.

    The same drift can land on any of the loaders that now take the keyword
    (`OperatorTrainer`, `VideoCompressionTrainer`, `DistributedTrainer`,
    `load_codec`), none of which are CLI entry points and so none of which the
    parametrized guard above reaches.
    """
    offenders: list[str] = []
    for root in ("src", "scripts", "dashboard"):
        for py in (REPO_ROOT / root).rglob("*.py"):
            for value in _flag_argument_nodes(str(py.relative_to(REPO_ROOT))):
                if isinstance(value, ast.Constant) and value.value is True:
                    offenders.append(str(py.relative_to(REPO_ROOT)))

    assert not offenders, f"{DEST}=True is hardcoded in: {sorted(set(offenders))}"
