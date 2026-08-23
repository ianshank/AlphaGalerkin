"""The 2026-07-22 "cut to the core" module list — one definition, several guards.

Five subsystems were removed from the repository on 2026-07-22 to refocus on the
Galerkin/MCTS core (`video_compression` was later re-scoped and reinstated as part
of the Codec Model-Zoo work). Two independent guards need that list:

* ``tests/hf_space/test_mirror_guard.py`` — the deploy mirror must stay scrubbed of them.
* ``tests/docs/test_charter_alignment.py`` — the charter's non-goal register must match, and
  none of them may reappear as a ``src/`` package.

Declaring the tuple twice would create two sources of truth for a scope decision, so it lives
here. The charter
(``openspec/specs/project-charter/spec.md``, *Non-Goal Exclusion*) is the human-facing record;
this module is its executable form.

**Provenance is in-tree, deliberately.** ``CLAUDE.md``'s 2026-07-22 milestone cites a
``archive/pre-core-cut-2026-07-22`` tag, but CI checks out shallow with no tags and a fresh
clone resolves none — so nothing here may consult git history. ``CHANGELOG.md`` and the charter
are the provenance.
"""

from __future__ import annotations

from typing import Final

#: Packages removed in the 2026-07-22 "cut to the core". Must not resurface.
#: ``video_compression`` was later reinstated (Codec Model-Zoo) and is no longer cut.
CUT_MODULES: Final[tuple[str, ...]] = (
    "reentry",
    "vertex",
    "intercept",
    "firefighting",
    "thermo",
)

#: The retracted, fabricated zero-shot-transfer figure. It was never computed by any code;
#: see ``specs/transfer_baseline_compare.spec.md``. The committed result is ~2.3e-3.
FABRICATED_FIGURE: Final[str] = "0.000209"

#: The blanket novelty claim retracted by ``docs/business/proposals/PRIOR_ART_REVIEW.md``
#: (TreeMesh, arXiv:2111.07613, already couples MCTS+RL with FE mesh *generation*).
RETRACTED_BLANKET_CLAIM: Final[str] = "no published papers combine MCTS with Galerkin"

#: Retracted 2026-08-23. The project asserted the AMR-RL literature was "uniformly
#: single-step"; VDGN (arXiv:2211.00801) refines *anticipatorily* for features that appear at
#: later times, which is multi-step in effect. The surviving delta is the explicit search tree
#: over refinement sequences, not multi-step reasoning as such. See ``docs/related-work.md``
#: entry 2 and ``docs/business/proposals/PRIOR_ART_REVIEW.md`` Finding 2.
RETRACTED_UNIFORMLY_SINGLE_STEP: Final[str] = "uniformly single-step"

#: Retracted 2026-08-16. The committed L-shape result is that MCTS **loses** at matched DOF
#: (median ratio 1.0996, 1 of 5 seeds) and at matched compute (2.04, 0 of 5). Any live
#: statement that MCTS beats Doerfler is the retracted headline. Both spellings are guarded
#: because the repo uses the umlaut inconsistently.
RETRACTED_AMR_WIN_PHRASES: Final[tuple[str, ...]] = (
    "beats dörfler",
    "beats dorfler",
    "outperforms dörfler",
    "outperforms dorfler",
)
