---
name: close-hygiene-row
description: File or close a row in the hygiene audit's prioritized backlog end-to-end — the next free B-number, the DONE format that keeps the original finding, notes on the rows the change touches, the CHANGELOG bullet, the CLAUDE.md Regression Surface row and dated log line, and the kill records in test docstrings. Use when a fix closes a backlog item, when a review or a scan turns up a finding worth filing, or when a change alters what an existing row says.
---

# close-hygiene-row — the backlog is a record, not a to-do list

`docs/CODE_HYGIENE_AUDIT.md` §5 is the repo's memory of what was wrong and how it was
proved fixed. A closed row that loses its original finding cannot be re-checked. A row
claiming a CI result nobody observed is how "41 passed, green" came to describe a step
that failed 24 of 36 tests (B45). The format is not written down anywhere else, so it
drifts with every closer. This skill is the format.

Related: `harden-a-guard` (the kills a row cites), `claims-ledger` (numeric headline
claims), `regression-surface` (the row's command), `pr-preflight` (before pushing).

## Step 1 — Find the row, or the next free number

```bash
grep -nE "^\| (~~)?B[0-9]+" docs/CODE_HYGIENE_AUDIT.md | tail -5
grep -n "<a distinctive word from the finding>" docs/CODE_HYGIENE_AUDIT.md
```

If a row already describes the finding, update **that** row. Two rows for one defect
split its history. Otherwise take the highest number + 1. Numbers are never reused,
including numbers struck through as DONE.

## Step 2 — Filing a finding

```text
| Bnn | **<the defect in one sentence>** <evidence> | S/M/L | Low/Med/High | <proposed fix> |
```

The evidence is measured, with its coordinates: a CI run id, a commit, dependency
versions, the numbers. Say what was **not** run. B45's row records "Steps 2-3 of the
ONNX conversion were not run (no `onnx` here)". An unrun step stated as unrun is
evidence; an unrun step left implicit becomes a claim.

## Step 3 — Closing it

```text
| ~~Bnn~~ ✅ **DONE (YYYY-MM-DD)** | **Closed: <what is now true>** <how> ... Original finding: **<the original bold sentence>** <the original evidence, verbatim> | <effort> | <risk> | <the original proposed fix> |
```

The "how" names:
- each reproduce-first test, by node id, that failed on the old code;
- the kill count, `N/N planted defects killed`. Planted defects are not tests, so don't
  conflate the two counts (`harden-a-guard` Step 7);
- what was deliberately **not** changed, and why. B50 left `_export_script` alone and
  said why, so B47's owner inherits the reason and not a mystery.

Keep the original finding, and keep the original proposed fix even where the real fix
differed. B45's proposal turned out incomplete twice over, and the row says so; that
is the useful part.

## Step 4 — Rows the change touches

If the change alters what another row says, append a dated note **inside** that row:

```text
**[UPDATED YYYY-MM-DD (Bnn): <what changed, and what still stands>]**
```

Never rewrite another row's history. B10 carries a `CORRECTED` and then an `UPDATED`
note in sequence, and the sequence is the record.

## Step 5 — CHANGELOG

Add the entry under `## [Unreleased]`, in the right group: `### Fixed` for a defect,
and next to related entries rather than at the end. Use the house shape: a bold lead
sentence ending `(hygiene Bnn).`, then what was wrong, the fix, how it was proved, and
what was left. `tests/docs/test_changelog_headers.py` guards the headings, so don't
invent a group.

## Step 6 — CLAUDE.md

- **Regression Surface row.** Copy the command from CI verbatim (`regression-surface`,
  `wire-a-ci-job`). Update its counts, and its kill record (planted defects, not tests).
- **The per-area milestone log is append-only.** Add a dated line. Never edit an old
  one: "41/41" was true on 2026-10-15 and stays true of that date.

## Step 7 — Test docstrings

Each test file whose tests killed a planted defect lists it, with its killers by name,
in the module docstring (`harden-a-guard` Step 7). The audit row cites the count; the
docstrings hold the names.

## Step 8 — Say where it was measured

State the versions a number was measured on. Locally on one stack and in CI on
another is two measurements, not one. Until CI has run, write "measured locally on
torch 2.11.0, …", never "CI passes". The B50 row was written before its PR's first CI
run and says so.

## Step 9 — Verify

```bash
pytest tests/docs/test_changelog_headers.py tests/claude/ -q
```

Then `pr-preflight` before pushing.
