# ADR-0003: Release pinning via self-referential sha-pinned catalog sources

**Status:** accepted — 2026-07-03

## Context

Sha pinning lives on **plugin source entries in `marketplace.json`**
(`ref`/`sha`; `sha` wins). With relative in-repo sources
(`"./plugins/eng-standards"`), installed plugins track the marketplace
repo state at install/update time — consumers cannot pin per-plugin, and
the consumer-side `extraKnownMarketplaces` schema has no sha field.
`plugin.json` cannot carry its own release sha (a manifest cannot know
the commit that will contain it).

## Decision

- `release/pins.json` is the pin manifest: `schema_version`, `repo`
  (owner/repo for self-referential sources), and `pins`
  (`plugin-name → {version, ref, sha}`).
- `python -m tools.sync_catalog --write` regenerates the catalog's
  `plugins` array from plugin manifests **plus** pins:
  - manifest version matches its pin → entry source becomes
    `{"source": "github", "repo": <pins.repo>, "ref": <tag>, "sha": <sha>}`;
  - otherwise → relative dev source `./plugins/<name>`.
- Release flow (human-triggered): tag → record the tagged commit's sha in
  `pins.json` → `sync_catalog --write` → commit the catalog. The
  `release-pins` gate fails CI when a pinned version is published with a
  relative or mismatched source.

## Consequences

- Invariant: a consumer installing release X receives bytes identical to
  the tagged release X plugin directory, regardless of marketplace HEAD.
- The catalog is generated from **two** inputs (manifests + pins); both
  are parity-gated so drift is a build failure.
- During development (unpinned versions), relative sources keep the
  `--plugin-dir`-style fast loop intact.
