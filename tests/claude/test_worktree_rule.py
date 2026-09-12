"""The concurrent-subagent worktree-isolation rule is stated where it is read.

Defect class, in one sentence: **a subagent with `Bash` can run a tree-wide git
command that reaches every other agent sharing that working tree, and no file the
subagent actually reads told it not to.** During PR #140 one background agent ran
`git stash` / `git reset` on its own initiative and stashed a second agent's
uncommitted, unrelated edits along with its own (`docs/CODE_HYGIENE_AUDIT.md` B22;
the incident is retold in root `AGENT.md`). Non-overlapping *file* scopes did not
isolate the agents; separate *worktrees* would have.

The rule has one anchor sentence. It lives in root `AGENT.md` under
`## Concurrent subagents` and is repeated **verbatim** in every `.claude/agents/*.md`
whose frontmatter grants `Bash` -- those are the files a dispatched subagent is
handed, and a rule stated only in a document the subagent never opens is a rule
stated nowhere. This suite keeps the three copies from drifting apart:

* ``ANCHOR`` below is compared *both* ways -- against the sentence actually
  written in root `AGENT.md` (so editing the doc without the test fails) and
  against each Bash agent (so editing an agent without the doc fails). A constant
  that is never checked against its source is the "test defends the drift"
  pattern `.claude/agents/sqe.md` warns about.
* Discovery is data-driven, reusing `tests/claude/test_harness_validation.py`'s
  ``AGENTS`` glob and ``_frontmatter`` parser rather than a second parser or a
  hardcoded list. A new agent that declares `Bash` is checked the moment it lands.
* Vacuity first: ``test_at_least_one_agent_declares_bash`` fails if the
  parametrised list is empty, because a guard over nothing passes forever.
* `settings.json` must actually grant `Bash(git worktree:*)` -- root `AGENT.md`
  *claims* it does, and a claim about configuration is checked against the
  configuration, not trusted.

Clause 4 of the rule (run `python -m ...` from the worktree root) is asserted
too, in ``test_bash_agent_states_the_run_from_worktree_root_clause``: it is the
trap subagents hit in practice (the editable install points at the primary
checkout), and it retires with the lockfile ticket R-04b in
`docs/ENGINEERING_REFLECTION_2026-09-11.md` -- delete that test with the clause.

Mutations planted per `.claude/skills/harden-a-guard/SKILL.md` (each confirmed to
apply, each restored afterwards):

1. **Anchor deleted from one Bash agent** (`build-engineer.md`) -- killed by
   ``test_bash_agent_carries_the_anchor_sentence[build-engineer]``.
2. **One word altered in root `AGENT.md`'s copy** (`never` -> `rarely`) -- killed
   by ``test_anchor_appears_exactly_once_in_root_agent_md`` and
   ``test_anchor_constant_equals_the_sentence_in_root_agent_md``; every per-agent
   test stays green, which is correct -- the agents still carry the canonical
   sentence, the doc is what drifted.
3. **`Bash` stripped from one agent's tools while its sentence is kept**
   (`reviewer.md`) -- **survives by design**, and the survival is the decision:
   removing `Bash` removes the capability the rule constrains, so the sentence
   becomes surplus prose, not a false statement. A non-Bash agent is neither
   required to carry the rule nor forbidden from carrying it. The guard is
   deliberately one-directional (Bash implies anchor, not the converse); what it
   *does* refuse is the limit of that mutation --
4. **`Bash` stripped from every agent** -- killed by
   ``test_at_least_one_agent_declares_bash``, the vacuity guard, which is the
   only thing standing between "no agent needs the rule" and "the rule is
   checked on nobody".
5. **`Bash(git worktree:*)` removed from `settings.json`** -- killed by
   ``test_settings_permit_git_worktree``.
6. **Clause 4's phrase removed from one Bash agent** (`sqe.md`, "from the
   worktree root" -> "from anywhere") -- killed by
   ``test_bash_agent_states_the_run_from_worktree_root_clause[sqe]``.

**5/5 mutation-killed** of the six planted; mutation 3 is the recorded, reasoned
survivor, not an escape. Observed on 2026-09-11 with each mutation confirmed to
have changed the file (an anchor assertion before the write) and the suite
confirmed green again after restore. Under mutation 4 pytest reports the two
parametrised tests as *skipped* ("empty parameter set") -- precisely the
silent-green that ``test_at_least_one_agent_declares_bash`` turns into a failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.claude.test_harness_validation import AGENTS, CLAUDE_DIR, REPO_ROOT, _frontmatter

#: The rule's anchor sentence. Byte-identical in root ``AGENT.md`` and in every
#: Bash-declaring agent; ``test_anchor_constant_equals_the_sentence_in_root_agent_md``
#: is what stops this constant and the document drifting apart.
ANCHOR: str = (
    "Concurrent subagents work in their own `git worktree` and never run `git stash`, "
    "`git reset`, `git checkout -- <path>` or `git clean` in a shared working tree."
)

ROOT_AGENT_MD: Path = REPO_ROOT / "AGENT.md"
SETTINGS_JSON: Path = CLAUDE_DIR / "settings.json"

#: The section heading under which the rule lives, in root ``AGENT.md`` and in
#: each Bash agent.
SECTION_HEADING: str = "## Concurrent subagents"

#: Clause 4's load-bearing phrase: the editable install points at the primary
#: checkout, so ``python -m ...`` must be run from the worktree root. Retires
#: with reflection ticket R-04b (the lockfile).
WORKTREE_ROOT_CLAUSE: str = "from the worktree root"

#: The permission root ``AGENT.md`` claims ``settings.json`` grants.
WORKTREE_PERMISSION: str = "Bash(git worktree:*)"


def _declared_tools(path: Path) -> set[str]:
    """Tool names an agent's frontmatter grants, via the shared parser."""
    return {t.strip() for t in _frontmatter(path)["tools"].split(",")}


BASH_AGENTS: list[Path] = [p for p in AGENTS if "Bash" in _declared_tools(p)]


def _section_body(text: str, heading: str) -> str:
    """The Markdown between *heading* and the next ``## `` heading (or EOF).

    Asserts the heading occurs exactly once -- two sections with the same name
    would make "the" anchor ambiguous, and zero would make every read vacuous.
    """
    marker = f"\n{heading}\n"
    found = text.count(marker)
    assert found == 1, f"expected exactly one {heading!r} section, found {found}"
    after = text.split(marker, 1)[1]
    return after.split("\n## ", 1)[0]


class TestConcurrencyRule:
    """The anchor sentence is stated once in the doc and verbatim in every Bash agent."""

    def test_at_least_one_agent_declares_bash(self) -> None:
        """Vacuity guard: the parametrised tests below iterate ``BASH_AGENTS``.

        If no agent declared `Bash`, every per-agent assertion would be skipped
        as "no parameters" and the rule would be checked on nobody while this
        file reported green. Today all six agents declare it.
        """
        assert AGENTS, "no agents discovered -- the glob or the directory moved"
        assert BASH_AGENTS, (
            "no .claude/agents/*.md declares Bash; the per-agent worktree-rule tests would "
            "be vacuous. If that is deliberate, delete this suite rather than let it pass."
        )

    def test_anchor_appears_exactly_once_in_root_agent_md(self) -> None:
        """Root ``AGENT.md`` states the anchor sentence exactly once.

        Zero means the rule's home is gone; two means an edit to one copy can
        leave a stale copy behind in the same file.
        """
        assert ROOT_AGENT_MD.is_file(), "root AGENT.md is missing"
        text = ROOT_AGENT_MD.read_text(encoding="utf-8")
        assert text.count(ANCHOR) == 1, (
            f"root AGENT.md contains the anchor sentence {text.count(ANCHOR)} times; "
            f"expected exactly 1 under {SECTION_HEADING!r}"
        )

    def test_anchor_constant_equals_the_sentence_in_root_agent_md(self) -> None:
        """``ANCHOR`` is read back from the document, not merely searched for.

        The first prose line under ``## Concurrent subagents`` in root
        ``AGENT.md`` *is* the anchor. Comparing the constant against that line
        means neither side can be edited alone: change the doc and this fails;
        change the constant and this fails. ``test_anchor_appears_exactly_once``
        alone would not catch a doc edit that left the old sentence somewhere
        else in the file.
        """
        text = ROOT_AGENT_MD.read_text(encoding="utf-8")
        body = _section_body(text, SECTION_HEADING)
        first_prose = next(ln.strip() for ln in body.splitlines() if ln.strip())
        assert first_prose == ANCHOR, (
            "the first sentence under root AGENT.md's `## Concurrent subagents` is not the "
            f"ANCHOR constant in this test:\n  doc : {first_prose}\n  test: {ANCHOR}"
        )

    @pytest.mark.parametrize("path", BASH_AGENTS, ids=lambda p: p.stem)
    def test_bash_agent_carries_the_anchor_sentence(self, path: Path) -> None:
        """Every agent that can run git states the rule, verbatim, in its own file.

        A paraphrase is not accepted: the sentence is the thing the harness
        test can hold constant across seven files, and a paraphrase is the
        first step of a drift.
        """
        text = path.read_text(encoding="utf-8")
        assert SECTION_HEADING in text, (
            f"{path.name} declares Bash but has no {SECTION_HEADING!r} section"
        )
        assert ANCHOR in text, (
            f"{path.name} declares Bash but does not carry the worktree-isolation anchor "
            f"sentence verbatim. Copy it from root AGENT.md's {SECTION_HEADING!r} section."
        )

    @pytest.mark.parametrize("path", BASH_AGENTS, ids=lambda p: p.stem)
    def test_bash_agent_states_the_run_from_worktree_root_clause(self, path: Path) -> None:
        """Clause 4, the trap subagents actually hit, is in every Bash agent.

        Before the lockfile lands, ``pip install -e .`` points at the primary
        checkout, so ``python -m`` from anywhere but the worktree root imports
        the primary tree's code. Retires with R-04b; delete with the clause.
        """
        text = path.read_text(encoding="utf-8")
        assert WORKTREE_ROOT_CLAUSE in text, (
            f"{path.name} does not state the run-`python -m`-{WORKTREE_ROOT_CLAUSE} clause"
        )

    def test_non_bash_agents_are_not_required_to_carry_it(self) -> None:
        """Documents the one-directional design; passes vacuously today.

        Every current agent declares `Bash`, so this asserts nothing about a
        real file yet. It exists so the decision recorded in the module
        docstring (mutation 3) is visible in the test list rather than only in
        prose: an agent without `Bash` cannot run git, so the rule is surplus
        there -- permitted, never required.
        """
        non_bash = [p for p in AGENTS if "Bash" not in _declared_tools(p)]
        for path in non_bash:
            # No assertion on ANCHOR either way; the file may carry it or not.
            assert path.is_file()


class TestWorktreePermission:
    """``settings.json`` grants what root ``AGENT.md`` says it grants."""

    def test_settings_permit_git_worktree(self) -> None:
        """Clause 1 of the rule depends on ``git worktree add`` being pre-approved."""
        data = json.loads(SETTINGS_JSON.read_text(encoding="utf-8"))
        allowed = data["permissions"]["allow"]
        assert WORKTREE_PERMISSION in allowed, (
            f"settings.json does not allow {WORKTREE_PERMISSION!r}; root AGENT.md's "
            "`## Concurrent subagents` clause 1 says it does"
        )

    def test_root_agent_md_makes_that_claim(self) -> None:
        """The permission test above guards a *claim*; the claim must exist to guard.

        If the sentence in root ``AGENT.md`` that names the permission is
        removed, the previous test still passes but no longer corresponds to
        any documented promise -- which is how a guard outlives its reason.
        """
        text = ROOT_AGENT_MD.read_text(encoding="utf-8")
        body = _section_body(text, SECTION_HEADING)
        assert f"`{WORKTREE_PERMISSION}`" in body, (
            f"root AGENT.md's {SECTION_HEADING!r} section no longer names "
            f"{WORKTREE_PERMISSION!r}; either restore the claim or retire "
            "test_settings_permit_git_worktree with it"
        )
