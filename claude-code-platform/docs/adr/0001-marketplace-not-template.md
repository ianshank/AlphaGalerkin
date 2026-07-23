# ADR-0001: Plugin marketplace repo, not a template repo

**Status:** accepted — 2026-07-03

## Context

Agentic tooling (review agents, test-enforcement skills, quality hooks) is
duplicated and drifting across consumer repos. The distribution mechanism
must give versioned updates, namespacing, and backwards compatibility
without copying files into consumers.

## Decision

This repository is a Claude Code **plugin marketplace**: a git repo with
`.claude-plugin/marketplace.json` cataloging plugins under `plugins/`.
Consumers declare it in `.claude/settings.json` (`extraKnownMarketplaces`)
and enable plugins via `enabledPlugins` (a boolean map,
`"plugin@marketplace": true`). Claude Code resolves, caches, and updates
plugin directories; nothing is copied into consumer repos.

## Consequences

- Versioned, sha-pinnable distribution for free (see ADR-0003).
- Components live under a stable namespace (`plugin-name:component`).
- The catalog and plugin manifests must be kept in parity — enforced by
  `claude plugin validate` (version) and `tools/validate` (description).
