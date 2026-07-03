# claude-code-platform

Reusable Claude Code **plugin marketplace**: engineering standards
(skills), review agents, and quality hooks packaged as versioned,
installable plugins — one source of truth consumed by every repo.

> **Note:** currently developed as a subtree of the AlphaGalerkin repo.
> It is fully self-contained (own `pyproject.toml`, tests, CI workflow)
> and designed to be extracted verbatim into `ianshank/claude-code-platform`
> via `git subtree split --prefix=claude-code-platform`.

## Consumer installation

```jsonc
// consumer repo: .claude/settings.json
{
  "extraKnownMarketplaces": {
    "ianshank-platform": {
      "source": { "source": "github", "repo": "ianshank/claude-code-platform" }
    }
  },
  "enabledPlugins": {
    "eng-standards@ianshank-platform": true   // boolean map — no object form
  }
}
```

When a teammate trusts the repo, Claude Code prompts to install the
marketplace and enable the declared plugins. Released plugin versions are
pinned to exact commit shas inside the catalog (ADR-0003), so an install
of release X is byte-identical to tag X.

## Plugins

| Plugin | Version | Components |
|---|---|---|
| `eng-standards` | 0.1.0 | 3 skills (`python-standards`, `c4-architecture`, `test-strategy`), 2 agents (`code-reviewer`, `test-enforcer`), 1 warn-only PostToolUse quality hook |

Each plugin's README documents its components and configuration table.

## Development

```bash
pip install -e '.[dev]'                 # dev-side harness deps (never hook deps)
python -m tools.validate                # static gates (schema/parity/paths/imports)
python -m tools.sync_runtime --write    # re-vendor the stdlib hook runtime
python -m tools.sync_catalog --write    # regenerate catalog from manifests + pins
pytest tests/unit/ -v                   # unit + subprocess contract tests
pytest tests/unit/ --cov --cov-branch   # coverage gate (>=85% branch, pyproject)
pytest tests/e2e -v                     # official `claude plugin validate` smoke
claude --plugin-dir ./plugins/eng-standards   # interactive local test
```

Key invariants (all CI-gated):

1. Every catalog entry mirrors its plugin manifest (description parity
   here; version parity via `claude plugin validate`).
2. Hook scripts are **stdlib-only** and import nothing outside the plugin
   directory; the shared runtime is vendored per plugin and byte-compared
   against `tools/hook_runtime` (ADR-0002).
3. No machine-specific literal paths; hook commands use
   `${CLAUDE_PLUGIN_ROOT}`.
4. Released versions ship as self-referential sha-pinned sources
   (`release/pins.json`, ADR-0003).

## Release checklist (human-triggered)

1. Bump plugin `version` in `.claude-plugin/plugin.json`; update CHANGELOG.
2. Tag the release; record the tagged commit sha in `release/pins.json`.
3. `python -m tools.sync_catalog --write`; commit the regenerated catalog.
4. CI green on both OS targets.
5. **Manual install smoke** (not CI-automatable — the trust prompt is
   interactive): fresh temp dir containing only the consumer
   `.claude/settings.json` above → open Claude Code → confirm the install
   prompt appears and namespaced components are listed.
6. `claude plugin details <plugin>`: always-on token cost within ceiling.

## Architecture

See `docs/architecture/c4.md` (C4 context/container/component + dynamic
view) and `docs/adr/` for the three load-bearing decisions:
marketplace-not-template (0001), vendored stdlib hook runtime (0002),
sha-pinned release model (0003).

## Platform support

Hooks are Python 3.11+ entrypoints tested on Linux and macOS. Windows is
not a supported target; because entrypoints are Python (not shell), they
degrade with a clear interpreter error rather than silently.
