# Peer Review: `ianshank/claude-code-platform` Implementation Plan

**Reviewed:** 2026-07-03
**Subject:** Implementation plan for a Claude Code plugin marketplace repo packaging
engineering standards (skills, agents, hooks) for consumption by `Agents`,
`MouseDroid-AGI`, `MangoMAS`, `SEAL`, and future repos.
**Method:** Every load-bearing technical claim in the plan was checked against the
official Claude Code documentation (`code.claude.com/docs/en/{plugins,
plugins-reference, plugin-marketplaces, discover-plugins}`) rather than reviewed
from memory. Verdicts below cite the source page.

---

## Verdict Summary

The plan is **structurally sound and unusually accurate** on Claude Code plugin
mechanics — 11 of 14 verified claims are fully correct, including several details
plans commonly get wrong (catalog location, `${CLAUDE_PLUGIN_ROOT}`, sha-pinning
primitives, `claude plugin details` token costs). However, it contains **three
architecture-breaking defects** that would surface only after consumer rollout
(Phase 3), plus **three major spec/verification errors**. All are fixable without
restructuring the plan; a corrected plan is provided in
[`docs/plans/IMPL_PLAN_20260703_claude_code_platform_marketplace.md`](../plans/IMPL_PLAN_20260703_claude_code_platform_marketplace.md).

**Recommendation: revise before Phase 0.** The critical findings all concern
decisions that Phase 0 (skeleton + harness) hard-codes; catching them after
Phase 1 ships a plugin means reworking every hook script and both C4 levels.

---

## Claim-by-Claim Verification

| # | Plan claim | Verdict | Detail |
|---|---|---|---|
| 1 | Marketplace = git repo with `.claude-plugin/marketplace.json` cataloging plugins | ✅ Confirmed | Required fields: `name`, `owner`, `plugins[]` (each entry needs `name` + `source`). [plugin-marketplaces] |
| 2 | `plugin.json` is the only file in `.claude-plugin/`; `skills/`, `agents/`, `hooks/`, `commands/` live at plugin root | ✅ Confirmed | Docs warn explicitly against nesting component dirs inside `.claude-plugin/`. [plugins-reference] |
| 3 | Consumers declare marketplaces via `extraKnownMarketplaces` with `{"source": {"source": "github", "repo": "owner/repo"}}` | ✅ Confirmed | Schema matches docs exactly. [plugin-marketplaces] |
| 4 | Consumers enable plugins via `enabledPlugins` with `{"enabled": true, "scope": "project"}` objects | ❌ **Incorrect** | Real format is a **boolean**: `{"plugin-name@marketplace": true}`. The `scope` key does not exist. See Finding 4. [plugin-marketplaces] |
| 5 | Marketplace entries can pin plugins to a git sha | ✅ Confirmed | Plugin **source entries** support `ref` + `sha` (40-char); `sha` wins when both set. But see Finding 6 — this does not automatically give *consumers* pinning. [plugin-marketplaces] |
| 6 | `claude plugin validate` exists | ✅ Confirmed | Validates marketplace schema, each `plugin.json`, frontmatter, `hooks/hooks.json` syntax, duplicate names, path traversal, **and marketplace↔manifest version mismatch**. Supports `--strict`. See Finding 7. [plugin-marketplaces] |
| 7 | `claude --plugin-dir` exists for local testing | ✅ Confirmed | Repeatable flag; local copy shadows an installed plugin of the same name. [plugins] |
| 8 | `claude plugin details` reports always-on token cost | ✅ Confirmed | Reports component inventory plus always-on and on-invoke token costs. Machine-readability of the output is unverified — see Finding 9. [plugins-reference] |
| 9 | `/reload-plugins` exists | ✅ Confirmed | Reloads plugins/skills/agents/hooks/MCP without restart; `--force` available. [discover-plugins] |
| 10 | `${CLAUDE_PLUGIN_ROOT}` for intra-plugin paths | ✅ Confirmed | Substituted in hook commands, skill/agent content, MCP/LSP configs. Also available: `${CLAUDE_PLUGIN_DATA}`, `${CLAUDE_PROJECT_DIR}`. [plugins-reference] |
| 11 | Hooks live in `hooks/hooks.json`; `PreToolUse`/`PostToolUse` are valid events | ✅ Confirmed | Also valid inline as a `"hooks"` field in `plugin.json`. Event list is much larger than the plan uses. Exit-code semantics are **not** documented on the plugin pages — see Finding 8. [plugins-reference] |
| 12 | Hook scripts import a shared `tools/hook_runtime` lib from the marketplace repo | ❌ **Broken at runtime** | Installed plugins are cached per-plugin and *cannot reference files outside their directory*. See Finding 1. [plugins-reference] |
| 13 | Opening a consumer repo auto-prompts marketplace install after trust dialog | ✅ Confirmed | "When team members trust the repository folder, Claude Code prompts them to install these marketplaces and plugins." But the named CI test doesn't exercise this path — see Finding 5. [discover-plugins] |
| 14 | Plugins support a root `settings.json` of defaults and `.mcp.json` | ⚠️ **Partially correct** | `.mcp.json` yes. `settings.json` exists but supports **only `agent` and `subagentStatusLine`** — it is not a tunables store for hooks. See Finding 3. [plugins-reference] |

---

## Critical Findings (architecture-breaking)

### Finding 1 — The shared `hook_runtime` library cannot be imported by installed plugins

The plan's Phase 0 item 3, repo layout (`tools/hook_runtime/`), and C4 Level 2
(`Rel(p1, hooklib, "Hook scripts import")`) all assume plugin hook scripts import
a Python library that lives **outside the plugin directory**, elsewhere in the
marketplace repo.

The docs are explicit that this fails after installation:

> "Installed plugins cannot reference files outside their directory. Paths that
> traverse outside the plugin root (such as `../shared-utils`) will not work
> after installation because those external files are not copied to the cache."
> — [plugins-reference], *Plugin caching and file resolution*

Only the plugin directory is copied to
`~/.claude/plugins/cache/{marketplace}/{plugin}/{version}/`. The `--plugin-dir`
dev loop *would* work (the repo is on disk), so this defect passes every test in
Section 4.1 and detonates only on consumer machines in Phase 3 — the worst
possible discovery point.

**Options:**
- **(a) Vendor + sync (recommended).** Keep `tools/hook_runtime/` as the source
  of truth; a `tools/sync_runtime.py` copies it into each plugin
  (`plugins/<name>/hooks/scripts/_runtime/`) and a CI parity gate fails on
  drift — the same generated-artifact pattern the plan already uses for the
  catalog.
- **(b) Symlinks.** Docs state same-marketplace symlink targets are dereferenced
  (copied) at install. Viable but behavior is subtle and needs an empirical test
  before relying on it; symlinks also complicate Windows checkouts.
- **(c) Stdlib-only single-file hooks.** Sidesteps the problem entirely;
  combines with Finding 2's recommendation.

### Finding 2 — Third-party Python dependencies in hook scripts are unresolvable on consumer machines

Soft constraint 2.2 mandates "Pydantic v2 models for all hook I/O, structlog for
logging," and hard constraint 2.1 requires "hook scripts validate inputs with
Pydantic models." Hooks execute with whatever `python` is on the consumer's
PATH; Claude Code provides **no dependency-installation mechanism** for plugin
hook scripts. A consumer machine without `pydantic`/`structlog` in its ambient
interpreter gets an `ImportError` on every hook fire.

The obvious escape hatch — PEP 723 inline metadata + `uv run` — is closed by the
plan's own hard constraint: *"PROHIBITED: Network calls from hook scripts at
runtime"* (uv fetches wheels on first run), and it would add a uv install
prerequisite for every consumer anyway.

**The plan's constraints are internally contradictory.** Recommendation: hook
runtime is **stdlib-only** (`dataclasses` + `json` + `logging` with a JSON
formatter gives typed input parsing and structured stderr logs with zero deps).
Pydantic and structlog stay where they belong per the plan — in the **dev-side
validation harness** (`tools/`), which runs in a controlled CI/dev environment.

### Finding 3 — Plugin `settings.json` is not a tunables store

Hard constraint 2.1 ("all tunables via env vars (CCP_* prefix) **or plugin
settings.json**") and C4 component `cfg` ("Plugin default settings on enable...
hooks read tunables") assume a plugin-root `settings.json` acts as a
configuration store hooks can read.

Per docs, plugin-root `settings.json` currently supports **only the `agent` and
`subagentStatusLine` keys** [plugins-reference]. It is not a general key-value
store, and hooks are not documented to receive anything from it.

**Fix:** tunables come from (1) `CCP_*` env vars and (2) an optional
plugin-owned config file (e.g. `${CLAUDE_PLUGIN_ROOT}/config/defaults.json`)
that scripts read explicitly, with env vars winning. Drop the `cfg` component's
current role from the C4 or re-label it as the defaults file.

---

## Major Findings (spec/verification errors)

### Finding 4 — Appendix B `enabledPlugins` schema is invented

The consumer contract shows:

```jsonc
"enabledPlugins": {
  "eng-standards@ianshank-platform": { "enabled": true, "scope": "project" }
}
```

The documented format is a boolean map [plugin-marketplaces]:

```json
"enabledPlugins": {
  "eng-standards@ianshank-platform": true
}
```

There is no `scope` key. Since Appendix B is the copy-paste integration contract
for four consumer repos, this error propagates on day one of Phase 3.

### Finding 5 — Success criterion 3 is not verifiable by the test it names

Criterion: *"A clean consumer repo with only .claude/settings.json declaring
this marketplace auto-prompts install and exposes every namespaced skill
(verified by scripted `claude --plugin-dir` smoke test in CI)."*

`--plugin-dir` loads a plugin directly from disk and **bypasses the entire
marketplace resolution/install/cache path** — it cannot verify the auto-prompt,
the settings-driven install, or namespacing-as-installed. The trust-dialog flow
is interactive and effectively unscriptable in CI.

**Fix — split it:**
- CI-verifiable: `claude plugin validate --strict` on the marketplace + a
  `--plugin-dir` smoke test asserting components load and skills are listed
  (component correctness, not distribution).
- Release-checklist (manual): fresh temp dir with only the consumer
  `settings.json` → confirm install prompt and namespaced component listing.
  The plan's own Section 4.3 step 2 already describes this manual check; the
  success criterion should reference *that*, not the CI test.

### Finding 6 — Consumer sha-pinning requires a design decision the plan skips

The plan asserts consumers "pin sha" and invariant 4 says "a consumer pinned to
sha X is byte-identical to release X." The mechanics don't compose the way the
plan assumes:

- Sha pinning lives on **plugin source entries inside `marketplace.json`**
  (`ref`/`sha` on `github`/`url`/`git-subdir` sources).
- If plugins use relative in-repo sources (`"./plugins/eng-standards"` — the
  natural choice for a monorepo marketplace), the installed plugin tracks
  whatever the marketplace repo state is at install/update time. There is
  no per-plugin pin.
- The consumer-side `extraKnownMarketplaces` schema documents no sha field for
  pinning the **marketplace itself**.

To make invariant 4 true, the marketplace entries must be **self-referential
github sources with an explicit `sha` per release** (each release tags the repo,
then `sync_catalog` stamps that commit's sha into every entry). That means
`sync_catalog` needs a pin manifest as a second input beyond `plugin.json`
(name/version/description come from the manifest; `sha` cannot, since the
manifest can't know its own future commit). This is a real release-engineering
design problem and belongs in a Phase 0 ADR, not discovered during Phase 3
integration.

---

## Minor Findings & Improvements

7. **Parity gate partially duplicates official validation.** `claude plugin
   validate` already fails on marketplace↔`plugin.json` **version** mismatch.
   The custom parity check in `tools/validate` should cover only what's missing
   (name/description drift, pin-manifest consistency) — don't re-implement what
   the official tool gates.
8. **Hook exit-code semantics are under-documented for plugins.** The plan's
   fail-safe posture (warn + exit 0 unless explicitly gating) is the right
   default; the exact blocking semantics (exit 2, JSON `decision` output)
   should be empirically verified in Phase 0 and recorded in CLAUDE.md with a
   date, per the plan's own doc-caching policy.
9. **Token-ceiling CI gate assumes parseable output.** `claude plugin details`
   does report always-on token cost, but the plan should verify whether a
   `--json`/stable output mode exists before making it a hard CI gate; until
   then treat it as a best-effort gate with a tolerant parser.
10. **Windows consumers.** Constraint 2.1 targets macOS + Linux only, but
    Claude Code runs on Windows and nothing stops a future consumer (or
    collaborator) there. POSIX-shell hooks fail silently on that surface.
    Cheap insurance: make every hook entrypoint Python (already required to
    3.11+), use shell only as an optional thin wrapper, and state the platform
    support policy in each plugin README.
11. **CI needs the `claude` CLI installed and pinned.** The workflow must
    install `@anthropic-ai/claude-code` (pin the version — the validator itself
    evolves). `plugin validate` / `plugin details` are offline; any actual
    session (`claude -p`) needs an API key secret and spends tokens — keep the
    e2e layer non-interactive.
12. **`.claude-plugin/marketplace.json` "generated from manifests"** is a good
    invariant, but note the generator has two inputs once Finding 6 lands:
    plugin manifests (metadata) + pin manifest (release shas).

---

## What the Plan Gets Right (kept as-is)

Objectivity cuts both ways; these were verified correct and are worth calling
out because they're the places similar plans usually fail:

- **Correct structure rules** — `plugin.json` alone in `.claude-plugin/`,
  components at plugin root, marketplace catalog location.
- **Correct consumer keys** — `extraKnownMarketplaces` schema is exact.
- **Correct primitives** — `${CLAUDE_PLUGIN_ROOT}`, `--plugin-dir`,
  `claude plugin validate`, `claude plugin details` token reporting,
  `/reload-plugins` all exist as described.
- **Marketplace-over-template packaging decision** — versioned distribution
  with no file copying is genuinely the only mechanism that meets the stated
  goal, and the plan says so explicitly with rationale.
- **Fail-safe hook default** (warn + exit 0, promote to blocking only after
  measured false-positive rate) — the correct posture for hooks that run in
  every consumer session.
- **"Distillery, not a lab" Phase 2 rule** — only battle-tested components get
  promoted. This is the single best process decision in the plan.
- **Autonomy split in Section 3.2** — CONFIRM-FIRST on MCP servers and blocking
  hooks correctly identifies the two highest-blast-radius changes.
- **Doc-fetch discipline** (Section 6.1: never trust memory for
  frontmatter/schema fields) — this review found exactly one schema error
  (Finding 4) precisely because the spec drifts; the policy is justified.

---

## Sources

- https://code.claude.com/docs/en/plugins — plugin structure, `--plugin-dir`, quickstart
- https://code.claude.com/docs/en/plugins-reference — plugin.json schema, caching/file-resolution, `${CLAUDE_PLUGIN_ROOT}`, hooks config, `plugin details`, plugin settings.json keys
- https://code.claude.com/docs/en/plugin-marketplaces — marketplace.json schema, `extraKnownMarketplaces`, `enabledPlugins`, source `ref`/`sha` pinning, `plugin validate`
- https://code.claude.com/docs/en/discover-plugins — team marketplace auto-prompt, `/reload-plugins`

Doc facts verified 2026-07-03. Per the plan's own policy (Section 5.2), these
should be re-verified with source URLs at the start of implementation — the
plugin spec is actively evolving.
