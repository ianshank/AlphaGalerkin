# CLAUDE.md — claude-code-platform

Claude Code plugin marketplace: catalog (`.claude-plugin/marketplace.json`)
plus versioned plugins under `plugins/`, validated by the dev-side harness
in `tools/`.

## Build/Validate Commands

```bash
python -m tools.validate                          # full static validation gate
python -m tools.validate --format json            # machine-readable violations
python -m tools.sync_runtime --write              # re-vendor hook_runtime into plugins
python -m tools.sync_runtime --check              # CI parity mode (exit 1 on drift)
python -m tools.sync_catalog --write              # regenerate catalog from manifests+pins
python -m tools.sync_catalog --check              # CI staleness mode
claude plugin validate .                          # official marketplace check
claude plugin validate ./plugins/<name>           # official manifest check
pytest tests/unit/ -v                             # harness + hook unit tests
pytest tests/e2e -v                               # official-CLI smoke (skips w/o CLI)
ruff check tools/ tests/ plugins/*/hooks/scripts  # lint
# Use `python -m mypy` (a standalone `mypy` binary may run in an isolated
# tool venv without the project's deps and report spurious import errors)
python -m mypy tools/ tests/ plugins/eng-standards/hooks/scripts/quality_scan.py
claude --plugin-dir ./plugins/<name>              # interactive local test
/reload-plugins                                   # pick up edits without restart
```

## Architecture Decisions

- [2026-07-03] ADR-0001 — Marketplace repo, not template repo: versioned
  distribution, sha pinning, no file copying.
- [2026-07-03] ADR-0002 — Hook runtime is **stdlib-only** and **vendored**
  into each plugin (`hooks/scripts/_runtime/`): installed plugins are
  cached per-directory and cannot import outside their root; consumer
  machines have no dependency resolution for hooks. Canonical source:
  `tools/hook_runtime`; vendor with `sync_runtime`; CI parity-gates drift.
- [2026-07-03] ADR-0003 — Release/pinning: catalog entries for released
  versions are self-referential github sources pinned to the release sha;
  `release/pins.json` is the pin manifest; `sync_catalog` stamps entries.
- [2026-07-03] Catalog `plugins` array is GENERATED (manifests + pins);
  marketplace identity fields (name/owner/metadata) are authored.
- [2026-07-03] Hooks fail-safe: warn + exit 0 on unexpected errors; a hook
  marked *gating* fails CLOSED (exit 2) instead. Blocking is opt-in via
  the `gating` tunable, promoted only after measured false-positive rate.
- [2026-07-03] Tunables: `config/defaults.json` per plugin + `CCP_<KEY>`
  env overrides (env wins, type-coerced against the shipped default).
  Plugin-root `settings.json` is NOT a tunables store (docs: it supports
  only `agent`/`subagentStatusLine`).

## Verified Doc Facts (source URLs; re-verify on spec-sensitive work)

Fetched 2026-07-03 from code.claude.com/docs/en/:

- `plugin-marketplaces`: marketplace.json requires `name`, `owner`,
  `plugins[]` (entries: `name`, `source`); source `ref`/`sha` supported,
  sha (40-hex) wins; `extraKnownMarketplaces` consumer schema;
  `enabledPlugins` is a **boolean map** (`"plugin@marketplace": true`);
  `claude plugin validate` checks schemas, frontmatter, hooks.json syntax,
  duplicate names, path traversal, **catalog↔manifest version mismatch**.
- `plugins-reference`: only `plugin.json` inside `.claude-plugin/`;
  components at plugin root; `${CLAUDE_PLUGIN_ROOT}` substitution;
  **installed plugins cannot reference files outside their directory**;
  plugin `settings.json` keys limited to `agent`/`subagentStatusLine`;
  `claude plugin details` reports always-on/on-invoke token cost.
- `discover-plugins`: trusting a repo with declared marketplaces prompts
  install; `/reload-plugins` (with `--force`).
- NOT documented on those pages: hook exit-code semantics for plugin
  hooks (empirical: exit 2 = block; exit 0 = allow — treat as
  best-effort until officially specified).

## Layout Invariants

- `.claude-plugin/` directories contain ONLY the manifest
  (marketplace.json at root scope, plugin.json at plugin scope).
- Hook scripts: Python 3.11+ stdlib-only; every runtime import resolves
  inside the plugin directory; all intra-plugin paths go through
  `${CLAUDE_PLUGIN_ROOT}`.
- No hardcoded absolute paths, usernames, repo names, model strings, or
  secrets anywhere (gated).
- SKILL.md ≤ 150 lines (progressive disclosure; detail in `references/`).

## Known Issues

- Custom JSON handling of marketplace/manifest schemas is best-effort
  (derived from docs); `claude plugin validate` remains authoritative.
- The marketplace *install* flow (trust prompt → enable) is interactive
  and verified manually per release, not in CI (see release checklist in
  README).
