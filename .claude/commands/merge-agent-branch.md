---
description: Merge a finished subagent branch into the current branch the way this repo's orchestration cycle does it — verify first, --no-ff merge, guard sweep, worktree cleanup, ledger rows — instead of re-improvising the ritual.
argument-hint: "<impl/branch> [<worktree path>]"
---

Integrate the subagent branch `$ARGUMENTS` into the current branch. This is the
ritual the 2026-09-11 reflection cycle ran eight times by hand (R-03, R-05, R-08,
R-10, R-11 twice, R-13, R-14, the UCI leak fix, the docs sync); each pass repeated
the same six steps, and the one time a step was skipped (a stale `lint`-job claim
merged unread) the mismatch surfaced only in CI.

## 1. Verify the branch on its own terms before merging

A subagent's report is a claim, not evidence — and the agent may have died before
reporting at all (three did on 2026-09-11 after committing complete work). Run the
branch's own checks **in its worktree** (`cd` there; `python -m …` from that root —
the editable install points at the primary checkout):

```bash
git -C <worktree> log --oneline $(git merge-base HEAD <branch>)..<branch>
git -C <worktree> status --short          # uncommitted remnants are a finding
ruff check <touched files> && ruff format --check <touched files>
python -m mypy --strict --ignore-missing-imports <touched src/ files>
python -m pytest <the branch's own test files> -q -p no:cacheprovider
```

If the branch's commit message quotes a measurement (RSS, coverage, mutation
kills), **re-measure one of them** before believing the rest.

## 2. Merge with a merge commit, never a rebase

```bash
git merge --no-ff <branch> -m "Merge branch '<branch>' into $(git branch --show-current)"
```

Resolve conflicts by hand and re-read the branch's claims against the *merged* tree
— a branch cut before a job was split, a file was renamed or a threshold was raised
still describes the old tree (the R-05 step said its imports were "named in this
job's install step above"; after R-12a that job installed only ruff).

## 3. Run the guards that read what changed

`.claude/hooks/guard_build_config.sh` selects the guard modules that read an edited
file; after a merge, run the union for every build/config/doc path the branch touched,
plus the branch's own tests once more on the merged tree:

```bash
python -m pytest tests/docs tests/claude <branch test files> -q -p no:cacheprovider
python scripts/check_doc_links.py
python -m scripts.artifact_manifest check
```

## 4. Retire the worktree and the branch

```bash
git worktree remove <worktree>        # refuses if the tree is dirty: that is a finding, not an obstacle
git branch -d <branch>                # -d, never -D: an unmerged branch must not be deletable here
```

`tests/claude/test_harness_validation.py` excludes `.claude/worktrees/` from its
scan, but a leftover worktree still costs disk and confuses `git worktree list`.

## 5. Write the three ledger rows in the same commit

- `CHANGELOG.md` `[Unreleased]` → the group the change belongs to (one bullet; the
  `test_changelog_headers.py` guard enforces one `[Unreleased]` and canonical groups).
- `CLAUDE.md` Regression Surface → one row per new guard, its command **copied from
  the CI step it mirrors** (`test_claude_coverage_gates.py` reads coverage rows as data
  and fails on a threshold that CI does not enforce).
- `docs/ENGINEERING_REFLECTION_*.md` implementation ledger, if the cycle has one.

Use the text the subagent's report supplied; if the agent died before reporting,
write it from the commit message and the tests you ran in step 1.

## 6. Commit, then push once the sweep is green

Commit the merge and the ledger rows; run the pre-push sweep (`ruff`, the guard
tiers above); push. CI on the pushed SHA is the final word — a local green from the
wrong subset is how 2026-09-11's `--cov-fail-under=1` row reached CI.
