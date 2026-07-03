# Changelog

All notable changes to the marketplace and its plugins.
Format: [Keep a Changelog](https://keepachangelog.com); versions are
per-plugin semver plus a marketplace metadata version.

## [Unreleased]

### Fixed — adversarial-review hardening (pre-release)

- **Gating hooks now fail CLOSED on crashes** (F1): the fail-safe wrapper
  accepts a crash-time gating resolver; `quality_scan` resolves gating
  from env/config safely, so a config typo can no longer silently disable
  an enabled gate. Crash telemetry reports the *effective* gating flag.
- **Gate bypass closures** (F2–F5): stdlib-import gate now scans every
  `*.py` under `hooks/` plus files referenced by hooks.json commands;
  dynamic imports (`importlib`, `__import__`) banned; vendored-runtime
  parity is recursive/content-complete and rejects symlinks; hooks.json
  commands referencing nonexistent files are violations; `sync_runtime`
  mirrors all of the above (recursive vendor/stray removal, symlink
  replacement).
- **Path-literal gate coverage** (F6): trailing-slash requirement dropped
  (`/home/user` now caught), tilde file paths caught, extensionless files
  scanned.
- **Env overrides without defaults file** (F7): `load_tunables` gained
  `fallbacks`; `CCP_*` overrides now work even when `config/defaults.json`
  is absent.
- 24 regression tests pinning every finding (`tests/unit/test_review_fixes.py`).

### Added — marketplace 0.1.0

- Catalog (`.claude-plugin/marketplace.json`) with generated, parity-gated
  plugin entries; release pin manifest (`release/pins.json`, ADR-0003).
- Dev-side validation harness (`tools/validate`): marketplace/manifest/
  hooks schema gates (pydantic, forward-compatible `extra="ignore"`),
  catalog description parity, release-pin consistency, vendored-runtime
  byte parity, machine-path literal gate, hook stdlib-import gate (AST),
  frontmatter lint with 150-line SKILL.md progressive-disclosure limit.
- Canonical stdlib-only hook runtime (`tools/hook_runtime`): typed stdin
  parsing, JSON-lines stderr logging honoring `CCP_LOG_LEVEL`/`CCP_DEBUG`,
  tunables loader (`config/defaults.json` + `CCP_*` env overrides with
  type coercion), fail-safe wrapper (warn+exit 0; gating hooks fail
  closed). Vendored per plugin by `tools/sync_runtime` (ADR-0002).
- Catalog generator `tools/sync_catalog` (manifests + pins → entries).
- CI workflow (ubuntu + macos): lint, types, static gates, sync checks,
  unit + subprocess contract tests, official `claude plugin validate`.

### Added — eng-standards 0.1.0

- Skills: `python-standards` (Protocol DI, pydantic configs,
  no-hardcoded-values, structlog, asyncio discipline), `c4-architecture`
  (C4 + Mermaid authoring), `test-strategy` (pyramid, coverage gates,
  property-based, contract tests) — each with `references/` detail files.
- Agents: `code-reviewer` (read-only adversarial standards review),
  `test-enforcer` (blocks "done" without tests).
- Hook: warn-only PostToolUse `quality_scan` — configurable secret /
  hardcoded-path regex categories; never logs matched text; fail-safe;
  blocking opt-in via the `gating` tunable (`CCP_GATING`).
