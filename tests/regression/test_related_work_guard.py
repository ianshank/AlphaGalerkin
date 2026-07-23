"""AC9: the novelty-gap documentation guard is executable, not prose.

Parses the delimited entries region of ``docs/related-work.md`` and asserts
every entry carries a case-insensitive "does not do" clause; the NKE entry's
clause must name MCTS, adaptive basis/mesh selection, and LBB. Also asserts
the retracted blanket novelty claim stays out of the README.

Spec: specs/stochastic_galerkin_nke.spec.md (AC9, change-doc requirement 3).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RELATED_WORK = REPO_ROOT / "docs" / "related-work.md"
README = REPO_ROOT / "README.md"

START_MARKER = "<!-- entries:start -->"
END_MARKER = "<!-- entries:end -->"

RETRACTED_BLANKET_CLAIM = "no published papers combine MCTS with Galerkin"


def _entries_region() -> str:
    text = RELATED_WORK.read_text(encoding="utf-8")
    assert START_MARKER in text, f"missing {START_MARKER} in {RELATED_WORK}"
    assert END_MARKER in text, f"missing {END_MARKER} in {RELATED_WORK}"
    return text.split(START_MARKER, 1)[1].split(END_MARKER, 1)[0]


def _entries() -> list[tuple[str, str]]:
    """(heading, body) pairs for every ``## `` entry inside the region."""
    region = _entries_region()
    parts = re.split(r"^## ", region, flags=re.MULTILINE)
    entries = []
    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        entries.append((heading.strip(), body))
    return entries


class TestRelatedWorkGuard:
    def test_document_exists_with_markers(self):
        assert RELATED_WORK.exists()
        assert _entries(), "the entries region contains no ## entries"

    def test_every_entry_has_does_not_do_clause(self):
        missing = [heading for heading, body in _entries() if "does not do" not in body.lower()]
        assert not missing, (
            "related-work entries missing the mandatory 'does NOT do' clause "
            f"(novelty-boundary rule): {missing}"
        )

    def test_nke_entry_names_the_three_boundaries(self):
        nke = [body for heading, body in _entries() if "Kolmogorov" in heading]
        assert nke, "the NKE entry is missing from docs/related-work.md"
        body = nke[0].lower()
        for required in ("mcts", "adaptive basis", "lbb"):
            assert required in body, f"NKE does-NOT-do clause must mention {required!r}"

    def test_nke_entry_carries_provenance_caveat(self):
        nke = [body for heading, body in _entries() if "Kolmogorov" in heading]
        assert "unreachable" in nke[0].lower(), (
            "the NKE entry must keep the provenance caveat (paper not read at "
            "implementation time; reviewer cross-check open)"
        )


class TestReadmeNoveltyClaim:
    def test_retracted_blanket_claim_absent(self):
        """README must use the narrow defensible form, not the retracted claim."""
        text = README.read_text(encoding="utf-8")
        assert RETRACTED_BLANKET_CLAIM not in text, (
            "README.md still carries the blanket 'no MCTS+Galerkin' claim that "
            "docs/proposals/PRIOR_ART_REVIEW.md retracted (TreeMesh exists); use "
            "the narrow multi-step-look-ahead form"
        )

    def test_readme_references_treemesh_counterexample(self):
        text = README.read_text(encoding="utf-8")
        assert "TreeMesh" in text, (
            "the narrow novelty claim must cite the TreeMesh counterexample "
            "(PRIOR_ART_REVIEW honesty constraint 1)"
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
