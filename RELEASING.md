# Releasing

AlphaGalerkin follows [Semantic Versioning](https://semver.org/) and
[Keep a Changelog](https://keepachangelog.com/). This document describes how a
release is cut. Releases are cut **by hand**: there is no release automation in
this repository — no on-tag workflow (nothing under `.github/workflows/`
triggers on `tags:` or `release:`), no signed artifacts, no `cz bump`.

## Current state

- `version` in [`pyproject.toml`](pyproject.toml) is `0.4.0-dev`, with the
  classifier `Development Status :: 4 - Beta` (both set by the 2026-08-16 bump;
  the classifier is reviewed at each cut, not assumed to have moved).
- **No tagged release has been cut yet** — `git tag` is empty. Every version
  block in [`CHANGELOG.md`](CHANGELOG.md) so far was a heading rename without a
  tag. `## [0.3.0]` heads two blocks there (`2026-07-22` and `2026-04-01`); it is
  left as append-only history and allowlisted in the changelog guard rather
  than repaired.
- **Cadence: monthly, or at each `openspec/changes/` archive, whichever comes
  first** (decision D8 in
  [`docs/ENGINEERING_REFLECTION_2026-09-11.md`](docs/ENGINEERING_REFLECTION_2026-09-11.md)).

## Versioning policy (SemVer)

Given `MAJOR.MINOR.PATCH`:

- **MAJOR** — incompatible public-API changes (e.g. a break in the `src.modeling`
  stable surface frozen by [ADR 0002](docs/adr/0002-mouse-droid-fusion-integration.md)).
- **MINOR** — backwards-compatible functionality (new scenarios, operators, agents).
- **PATCH** — backwards-compatible bug fixes.

The single source of truth for the version is `version` in
[`pyproject.toml`](pyproject.toml); `src/__init__.py` reads it from the installed
distribution's metadata. Every other place a version is written down is a copy
and is guarded (see step 4 below).

## Commit conventions

Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `build:`, `ci:`…), validated by
the **Commitizen** `commit-msg` hook in `.pre-commit-config.yaml`. That hook
validates messages only: `pyproject.toml` has no `[tool.commitizen]` table, so
`cz bump` is not configured and is not part of the process below — the version
is bumped by hand. The type of a change still maps to the SemVer bump:
`feat:` → MINOR, `fix:` → PATCH, a `!` / `BREAKING CHANGE:` footer → MAJOR.

## Changelog discipline

Every user-facing change adds a bullet under the single `[Unreleased]` heading
in `CHANGELOG.md`, in the appropriate group (`Added` / `Changed` / `Deprecated` /
`Removed` / `Fixed` / `Security`), newest first. A titled sub-section inside a
group uses a `####` heading. Do **not** rewrite historical entries.

The structure is guarded by `tests/docs/test_changelog_headers.py`: exactly one
`[Unreleased]` header and it is first; one preamble; each group name at most once
under `[Unreleased]`; release versions non-increasing top-to-bottom under PEP 440
and unique except for the guard's `DUPLICATE_VERSIONS` allowlist (which is
asserted to still be needed); ISO `YYYY-MM-DD` dates, non-increasing.

## Cutting a release (by hand)

1. Ensure the default branch is green (the `CI Success` aggregate job in
   `.github/workflows/ci.yml`).
2. Decide the new version from the `[Unreleased]` entries (SemVer, above). A
   `-dev` suffix drops at the cut (a dev pre-release becomes its final).
3. In `CHANGELOG.md`, rename `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD` and add a
   fresh empty `[Unreleased]` section on top. Leave older blocks untouched. PEP
   440 ranks a final release above its own `-dev` pre-release, so the new block
   sitting above the existing dev-labelled block satisfies the guard.
4. Bump the version everywhere it is declared. `tests/docs/test_version_consistency.py`
   guards the first four against `pyproject.toml`:
   - `pyproject.toml` — `[project].version`;
   - `README.md` — the `## Project status` paragraph;
   - `hf_space/src/__init__.py` — `__version__` is a literal by necessity (the
     Space installs no distribution to read);
   - this file — the *Current state* section above;
   - `docker/Dockerfile` line 1 — a comment, not guarded; update it by hand.

   Review the `Development Status` classifier in `pyproject.toml` at the same
   time. Once a `uv.lock` exists (introduced by a parallel change), re-lock —
   the lockfile embeds the project version.
5. Run `python -m pytest tests/docs/test_version_consistency.py tests/docs/test_changelog_headers.py -q`.
6. Commit (`chore(release): vX.Y.Z`) and tag: `git tag -a vX.Y.Z -m "vX.Y.Z"`.
7. Push the tag: `git push origin vX.Y.Z`, then create the GitHub Release **by
   hand** from the changelog section. Nothing runs on the tag; the release notes
   are copied, not generated.

## Pre-1.0 note

While on `0.x`, the public API may change between MINOR versions. The
`src.modeling` re-export surface is the one interface held stable ahead of 1.0,
per its ADR.
