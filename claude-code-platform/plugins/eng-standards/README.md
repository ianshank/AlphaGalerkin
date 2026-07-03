# eng-standards

House engineering standards as a Claude Code plugin: skills that encode
the coding conventions, agents that enforce them in review, and a
warn-only quality hook.

## Install

```jsonc
// .claude/settings.json
{
  "extraKnownMarketplaces": {
    "ianshank-platform": {
      "source": { "source": "github", "repo": "ianshank/claude-code-platform" }
    }
  },
  "enabledPlugins": { "eng-standards@ianshank-platform": true }
}
```

Local development: `claude --plugin-dir ./plugins/eng-standards`.

## Components

| Component | Type | Purpose |
|---|---|---|
| `python-standards` | skill | Protocol DI, pydantic configs, no hardcoded values, structlog, asyncio discipline |
| `c4-architecture` | skill | Authoring C4 context/container/component docs with Mermaid |
| `test-strategy` | skill | Test pyramid, per-module coverage gates, property-based + contract tests |
| `code-reviewer` | agent | Read-only adversarial standards-compliance review (`Read, Grep, Glob`) |
| `test-enforcer` | agent | Blocks "done" without tests; runs the relevant suite (`Read, Grep, Glob, Bash`) |
| `quality_scan` | hook | PostToolUse warn-only secret / hardcoded-path scan on edited files |

## Configuration

Defaults ship in `config/defaults.json`; any key is overridable via a
`CCP_<KEY>` environment variable (env wins; values are type-coerced
against the shipped default). Plugin-root `settings.json` is not used for
tunables.

| Env var | Default | Effect |
|---|---|---|
| `CCP_LOG_LEVEL` | `INFO` | Hook log level (JSON lines on stderr) |
| `CCP_DEBUG` | unset | Any truthy value forces DEBUG logging |
| `CCP_GATING` | `false` | `true`/`1`: findings exit 2 (blocking) instead of warn-only |
| `CCP_SCAN_TOOLS` | `["Write","Edit","MultiEdit","NotebookEdit"]` | JSON list of tool names to scan |
| `CCP_MAX_FILE_BYTES` | `1048576` | Max file size read when content isn't in the payload |
| `CCP_PATTERNS` | see defaults | JSON object: category → regex list |
| `CCP_EXCLUDE_PATH_SUBSTRINGS` | `["/.git/","/_runtime/","/node_modules/"]` | Paths skipped by the scan |

Hook behavior notes:

- **Fail-safe:** unexpected errors are logged (`hook_failsafe_triggered`)
  and exit 0 — a crashing hook never blocks a session.
- **No secret leakage:** findings log category/line/pattern, never the
  matched text.
- **Blocking is opt-in** (`gating`), to be promoted only after the
  false-positive rate is measured in real use.

## Platform

Python 3.11+ on Linux/macOS. Hook scripts are stdlib-only and
self-contained (the shared runtime is vendored under
`hooks/scripts/_runtime/` — do not edit it there; edit
`tools/hook_runtime` and run `python -m tools.sync_runtime --write`).
