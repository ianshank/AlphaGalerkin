# Releasing

AlphaGalerkin follows [Semantic Versioning](https://semver.org/) and
[Keep a Changelog](https://keepachangelog.com/). This document describes how a
release is cut. It is intentionally lightweight — the project has not yet cut a
tagged release (the version is `0.1.0` and everything sits under `[Unreleased]`
in [`CHANGELOG.md`](CHANGELOG.md)).

## Versioning policy (SemVer)

Given `MAJOR.MINOR.PATCH`:

- **MAJOR** — incompatible public-API changes (e.g. a break in the `src.modeling`
  stable surface frozen by [ADR 0002](docs/adr/0002-mouse-droid-fusion-integration.md)).
- **MINOR** — backwards-compatible functionality (new scenarios, operators, agents).
- **PATCH** — backwards-compatible bug fixes.

The single source of truth for the version is `version` in
[`pyproject.toml`](pyproject.toml).

## Commit conventions

Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `build:`, `ci:`…), validated by
**Commitizen** in the pre-commit hooks. The type of a change maps to the SemVer
bump: `feat:` → MINOR, `fix:` → PATCH, a `!` / `BREAKING CHANGE:` footer → MAJOR.

## Changelog discipline

Every user-facing change adds a bullet under the `[Unreleased]` heading in
`CHANGELOG.md`, in the appropriate group (`Added` / `Changed` / `Fixed` /
`Removed`). Do **not** rewrite historical entries.

## Cutting a release

1. Ensure `main` is green (CI: lint, fast tests on 3.10–3.12, 85% branch coverage).
2. Decide the new version from the `[Unreleased]` entries (SemVer, above).
3. In `CHANGELOG.md`, rename `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD` and add a
   fresh empty `[Unreleased]` section on top.
4. Bump `version` in `pyproject.toml` to `X.Y.Z`. Update
   `Development Status` in the classifiers if the maturity changed
   (e.g. `3 - Alpha` → `4 - Beta`).
5. Commit (`chore(release): vX.Y.Z`) and tag: `git tag -a vX.Y.Z -m "vX.Y.Z"`.
6. Push the tag: `git push origin vX.Y.Z`, then create a GitHub Release from the
   changelog section.

## Pre-1.0 note

While on `0.x`, the public API may change between MINOR versions. The
`src.modeling` re-export surface is the one interface held stable ahead of 1.0,
per its ADR.
