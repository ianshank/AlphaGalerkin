"""Floor drift-guard for the ``hf_space/`` HuggingFace Space deploy mirror.

``hf_space/src/`` and ``hf_space/config/`` are a **manual, partial,
independently-formatted** copy of the repository's ``src/`` / ``config/`` trees
(see ``hf_space/AGENT.md``). The mirror has already diverged from ``src/`` and is
**not** kept byte-identical, so this guard deliberately does **not** assert
parity — a parity check would fail today and effectively force the full
single-sourcing refactor that is intentionally out of scope.

Instead this is a *floor*: it catches the regressions that would actually break
the deployed Space or resurface removed/retracted content, without pretending
the mirror is in sync. Full single-sourcing is a tracked follow-up.

Guards:
    1. Every top-level ``src.*`` / ``config.*`` module imported by
       ``hf_space/app.py`` resolves to a file inside ``hf_space/`` — so the
       Space cannot ``ImportError`` on launch because the mirror lost a module.
    2. Every ``.py`` under ``hf_space/`` parses — catches a truncated/corrupt
       copy.
    3. The mirror stays scrubbed of the 2026-07-22 "cut to the core" modules and
       the retracted fabricated zero-shot-transfer figure.

Pure stdlib (``ast``/``pathlib``) — no torch/gradio import — so it runs on the
CPU CI surface.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_SPACE = REPO_ROOT / "hf_space"

# Roots whose imports the Space resolves against its own tree (app.py inserts
# hf_space/ onto sys.path, so ``src.`` / ``config.`` mean ``hf_space/src`` /
# ``hf_space/config``).
_MIRROR_IMPORT_ROOTS = frozenset({"src", "config"})

# Packages removed in the 2026-07-22 "cut to the core"; must not resurface here.
_CUT_MODULES = ("video_compression", "reentry", "vertex", "intercept", "firefighting", "thermo")

# The retracted, fabricated zero-shot-transfer figure (corrected to ~4e-4; see
# specs/transfer_baseline_compare.spec.md and the WS3 review banners).
_FABRICATED_FIGURE = "0.000209"


def _py_files() -> list[Path]:
    return sorted(HF_SPACE.rglob("*.py"))


def _app_module_imports() -> list[str]:
    """Top-level ``src.*`` / ``config.*`` modules imported by ``hf_space/app.py``.

    Runs at collection time (it feeds ``parametrize``), so it is defensive:
    returns ``[]`` if ``app.py`` is missing or unparseable rather than raising
    and erroring collection. Those conditions are reported as clean failures by
    ``test_hf_space_mirror_present`` and ``test_all_mirror_files_parse`` instead.
    """
    try:
        tree = ast.parse((HF_SPACE / "app.py").read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in _MIRROR_IMPORT_ROOTS:
                modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _MIRROR_IMPORT_ROOTS:
                    modules.add(alias.name)
    return sorted(modules)


def _module_exists_in_mirror(dotted: str) -> bool:
    parts = dotted.split(".")
    return (
        HF_SPACE.joinpath(*parts).with_suffix(".py").exists()
        or (HF_SPACE.joinpath(*parts) / "__init__.py").exists()
    )


def _references_cut_module(text: str, name: str) -> bool:
    # Word-boundaried so a cut name that is a prefix of a legitimate identifier
    # (e.g. "thermo" in "thermodynamics", "src.thermo" in "src.thermodynamics")
    # does not false-positive; matches ``src.<name>`` and ``import/from <name>``.
    escaped = re.escape(name)
    return re.search(rf"\bsrc\.{escaped}\b|\b(?:import|from)\s+{escaped}\b", text) is not None


def test_hf_space_mirror_present() -> None:
    assert HF_SPACE.is_dir(), "hf_space/ deploy mirror is missing"
    assert (HF_SPACE / "app.py").is_file(), "hf_space/app.py is missing"


@pytest.mark.parametrize("module", _app_module_imports())
def test_app_imports_resolve_in_mirror(module: str) -> None:
    """``app.py`` must not import a module the mirror does not ship."""
    assert _module_exists_in_mirror(module), (
        f"hf_space/app.py imports {module!r} but no matching file exists under "
        f"hf_space/ — the Space would ImportError on launch. Add the module to "
        f"the mirror (see hf_space/AGENT.md)."
    )


def test_all_mirror_files_parse() -> None:
    """Every mirror ``.py`` must parse — catches a truncated/corrupt copy."""
    unparseable: list[str] = []
    for path in _py_files():
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as exc:  # pragma: no cover - failure branch
            unparseable.append(f"{path.relative_to(REPO_ROOT)}: {exc}")
    assert not unparseable, "unparseable mirror file(s):\n" + "\n".join(unparseable)


def test_mirror_stays_scrubbed() -> None:
    """The mirror must not reintroduce cut modules or the retracted figure."""
    offenders: list[str] = []
    for path in _py_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(REPO_ROOT)
        offenders += [
            f"{rel}: references cut module {name!r}"
            for name in _CUT_MODULES
            if _references_cut_module(text, name)
        ]
        if _FABRICATED_FIGURE in text:
            offenders.append(f"{rel}: contains retracted figure {_FABRICATED_FIGURE!r}")
    assert not offenders, "mirror scrub regression:\n" + "\n".join(offenders)
