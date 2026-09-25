"""The ``picogk`` extra must be described as what it is while PicoGK ingestion is a stub.

Defect class, in one sentence: a front-door description of the ``[picogk]`` extra that implies
live PicoGK geometry ingestion while ``PicoGKSDFEvaluator`` is still a stub.

It shipped: ``docs/getting-started.md`` described the extra as "Leap 71 PicoGK voxel/SDF kernel
(Noyron HX)", but the extra installs only ``pythonnet`` and ``PicoGKSDFEvaluator.__init__``
raises ``NotImplementedError`` -- every Noyron scenario runs on ``AnalyticalHelixSDF``
(``docs/business/COMMERCIALIZATION_PEER_REVIEW.md`` §3 row 4, Gate 0 item 0.5).

The guard is a coupling, so it fails for the right reason in both directions:

* while the stub premise holds (clauses a and b), every front-door block that mentions the
  extra must disclose it -- name the ``pythonnet`` bridge and say ingestion is "not
  implemented" (clause c);
* once ingestion is implemented, or the extra gains a PicoGK dependency, clause a or b fails
  and says to rewrite the disclosures -- a disclosure that became false is the same defect
  pointing the other way.

It reads only the front doors (``tests/support/docs_site.py``: ``README.md`` and the docs-site
nav). History, the ledgers, and roadmap rows saying ingestion is future work are legitimate
mentions and are never scanned, so none can false-positive here. ``CONTRIBUTING.md`` only
enumerates extra names and defers to ``docs/getting-started.md`` for what they do.

Mutation kills -- 6/6 planted defects, each failing a NAMED test, none ``gpu_required`` or
``fem_required``:

1. ``docs/getting-started.md``'s row restored to "Leap 71 PicoGK voxel/SDF kernel (Noyron HX)."
   (the literal historical defect) ->
   ``test_front_door_mentions_of_the_extra_disclose_the_stub``.
2. That row restored with the disclosure moved into the neighbouring ``jax`` row -- the
   adjacent weaker defect, green under a whole-table check ->
   ``test_front_door_mentions_of_the_extra_disclose_the_stub``.
3. README's disclosure sentence deleted (its extras list still names ``picogk``) ->
   ``test_front_door_mentions_of_the_extra_disclose_the_stub``.
4. ``raise NotImplementedError`` in ``PicoGKSDFEvaluator.__init__`` replaced by ``return`` (the
   stub "implemented") -> ``test_picogk_ingestion_is_still_a_stub``.
5. ``"PicoGK>=1.0"`` added to the extra in ``pyproject.toml`` ->
   ``test_the_picogk_extra_installs_only_the_bridge``.
6. The mention pattern drifted so it matches nothing, which passes the disclosure check on
   an empty set -> ``test_the_extra_is_described_on_the_front_doors``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Final

from tests.support.docs_site import README, front_door_documents
from tests.support.perf_claims import split_blocks

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
SDF_MODULE = REPO_ROOT / "src" / "pde" / "sdf.py"

#: The stub whose constructor must still raise for the disclosures to be true.
STUB_CLASS: Final[str] = "PicoGKSDFEvaluator"
STUB_EXCEPTION: Final[str] = "NotImplementedError"

#: The extra's name in ``[project.optional-dependencies]``.
EXTRA: Final[str] = "picogk"

#: The only distribution the extra may install while ingestion is a stub.
BRIDGE_DISTRIBUTION: Final[str] = "pythonnet"

#: Phrases every front-door description of the extra must carry (case-insensitive).
DISCLOSURE_TERMS: Final[tuple[str, ...]] = (BRIDGE_DISTRIBUTION, "not implemented")

#: The documents that describe the extra today; each must keep at least one mention.
DESCRIBING_DOCUMENTS: Final[tuple[str, ...]] = (README, "docs/getting-started.md")

#: A mention of the extra, as the docs write it: ``picogk`` in backticks or ``[picogk]``.
_EXTRA_MENTION: Final[re.Pattern[str]] = re.compile(rf"`{EXTRA}`|\[{EXTRA}\]")

#: The extra's list in pyproject.toml (anchored regex: ``tomllib`` is 3.11+, the floor 3.10).
_EXTRA_LIST: Final[re.Pattern[str]] = re.compile(
    rf"^{EXTRA}\s*=\s*\[(?P<body>[^\]]*)\]", re.MULTILINE
)
_QUOTED: Final[re.Pattern[str]] = re.compile(r"[\"']([^\"']+)[\"']")
_DISTRIBUTION_NAME: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _mentions() -> list[tuple[str, int, str]]:
    """``(document, line, text)`` for each front-door unit that mentions the extra.

    A table row is its own unit, so a disclosure elsewhere in the extras table cannot cover
    a row that says something else; prose is judged by its whole paragraph or list item.
    """
    found: list[tuple[str, int, str]] = []
    for document in front_door_documents(REPO_ROOT):
        text = (REPO_ROOT / document).read_text(encoding="utf-8")
        for block in split_blocks(text):
            units = block.lines if block.kind == "table" else ((block.lines[0][0], block.text),)
            found += [(document, n, unit) for n, unit in units if _EXTRA_MENTION.search(unit)]
    return found


def test_the_extra_is_described_on_the_front_doors() -> None:
    """Vacuity: with no mention found, the disclosure check below passes on nothing."""
    documents = {document for document, _, _ in _mentions()}
    missing = [document for document in DESCRIBING_DOCUMENTS if document not in documents]
    assert not missing, f"no mention of the `{EXTRA}` extra found in {missing}"


def test_front_door_mentions_of_the_extra_disclose_the_stub() -> None:
    undisclosed = [
        f"{document}:{line}: {text.strip()[:120]}"
        for document, line, text in _mentions()
        if not all(term in text.lower() for term in DISCLOSURE_TERMS)
    ]
    assert not undisclosed, (
        f"front-door descriptions of the `{EXTRA}` extra that do not say it installs only "
        f"the {BRIDGE_DISTRIBUTION} bridge and that PicoGK ingestion is not implemented:\n  "
        + "\n  ".join(undisclosed)
    )


def test_picogk_ingestion_is_still_a_stub() -> None:
    """The premise of every disclosure: the constructor still refuses to construct."""
    tree = ast.parse(SDF_MODULE.read_text(encoding="utf-8"))
    stub = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == STUB_CLASS), None
    )
    assert stub is not None, f"{STUB_CLASS} is gone from {SDF_MODULE.name}; update this guard"
    init = next(
        (n for n in stub.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None
    )
    assert init is not None, f"{STUB_CLASS} has no __init__; update this guard"
    raised = {
        ast.unparse(node.exc).split("(", 1)[0]
        for node in ast.walk(init)
        if isinstance(node, ast.Raise) and node.exc is not None
    }
    assert STUB_EXCEPTION in raised, (
        f"{STUB_CLASS}.__init__ no longer raises {STUB_EXCEPTION}: PicoGK ingestion looks "
        f"implemented, so the disclosures in {', '.join(DESCRIBING_DOCUMENTS)}, "
        "src/pde/sdf.py and the Noyron scenario docstrings are now false. Rewrite them, "
        "then this guard."
    )


def test_the_picogk_extra_installs_only_the_bridge() -> None:
    match = _EXTRA_LIST.search(PYPROJECT.read_text(encoding="utf-8"))
    assert match is not None, f"pyproject.toml declares no `{EXTRA}` extra; update this guard"
    names = set()
    for requirement in _QUOTED.findall(match.group("body")):
        name = _DISTRIBUTION_NAME.match(requirement.strip())
        assert name is not None, f"unparseable requirement {requirement!r} in the extra"
        names.add(name.group(0).lower())
    assert names == {BRIDGE_DISTRIBUTION}, (
        f"the `{EXTRA}` extra installs {sorted(names)}, not only {BRIDGE_DISTRIBUTION}: the "
        f"disclosure that it is only the .NET bridge is now false. Rewrite it, then this guard."
    )
