# ADR-0002: Hook runtime is stdlib-only and vendored per plugin

**Status:** accepted — 2026-07-03

## Context

Two facts about the runtime environment on consumer machines (verified
against the official plugins-reference docs, 2026-07-03):

1. Installed plugins are cached per-directory
   (`~/.claude/plugins/cache/{marketplace}/{plugin}/{version}/`) and
   **cannot reference files outside their own root** — a shared library
   elsewhere in the marketplace repo is simply not present at runtime.
2. Hook scripts execute with whatever `python3` is on the consumer's
   PATH. There is **no dependency-installation mechanism** for plugin
   hooks, and network calls from hooks are prohibited by policy, which
   also rules out `uv run` with PEP 723 inline metadata.

## Decision

- `tools/hook_runtime/` is the canonical, **stdlib-only** shared library
  (dataclasses for typed input, `logging` with a JSON formatter for
  structured stderr output, env-driven tunables, a fail-safe wrapper).
- `python -m tools.sync_runtime --write` vendors it **verbatim** into
  each hook-shipping plugin as `hooks/scripts/_runtime/`.
- CI parity-gates the vendored copies byte-for-byte
  (`tools/validate` gate `vendored-runtime-parity`; also
  `sync_runtime --check`).
- A `stdlib-imports` gate AST-checks every hook script — every `*.py`
  under `hooks/` plus any file referenced by a hooks.json command,
  wherever it lives in the plugin — and fails on any non-stdlib,
  non-`_runtime` *static* import. Dynamic-import machinery (`importlib`,
  `__import__`) is banned outright because it defeats static analysis.
- The parity gate is recursive and content-complete (all files at any
  depth, bytecode caches excluded) and rejects symlinks anywhere in the
  vendored copy — a symlinked runtime dangles after plugin install.
- Third-party libraries (pydantic, structlog, pyyaml) are permitted only
  in the dev-side harness (`tools/validate`, sync tools, tests).

## Consequences

- Hooks are self-contained and work identically under `--plugin-dir`
  (dev) and marketplace install (consumer) — the failure mode where dev
  tests pass but consumer installs break is structurally eliminated.
- Runtime changes require a re-vendor step; forgetting it is a CI
  failure, not a silent drift.
- The runtime must stay small and dependency-free; anything needing real
  dependencies belongs in an agent or skill, not a hook.

## Alternatives rejected

- **Symlinks** (docs: same-marketplace targets are dereferenced at
  install): subtle, unverified across platforms, breaks on Windows
  checkouts.
- **PyPI package**: adds an install prerequisite on every consumer
  machine and a network dependency the policy forbids.
