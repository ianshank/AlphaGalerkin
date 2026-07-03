# Peer Review + Implementation Plan — `claude-code-foundry`

- **Plan ID:** `20260703_claude_code_foundry`
- **Created:** 2026-07-03
- **Reviewed artifact:** "Implementation Plan: `claude-code-foundry`" (constraint-programming
  template; standalone Claude Code plugin repository for reusable subagents, skills, commands,
  hooks, and validation tooling)
- **Status:** REVIEWED — Part A is the objective peer review; Part B is the revised,
  implementation-ready plan that resolves every finding.
- **Grounding:** (1) the current Claude Code plugin specification, verified against
  official docs on 2026-07-02 (URLs in §A.3); (2) prior art already proven in this
  repository (`.claude/` component set, `src/agents/scaffold.py` pure-renderer pattern,
  the schema-versioned document + migration pattern from `BaselineRegistry` /
  `poc/baselines`, the `specs/` markdown-contract convention).

---

## Part A — Peer Review

### A.1 Verdict

**Sound architecture, correct instincts, four defects that must be fixed before Phase 0.**

The plan's core thesis is right: centralize drifting agent/skill copies as a versioned,
schema-validated plugin, keep consumers read-only, gate releases in CI. The scope cap
(3 agents / 3 skills / 2 hooks / 1 command), the schemas-first ordering, and the honest
Usage Notes (rejecting "all components" as a v0.1.0 goal, redefining "backwards
compatible" as "versioned from day one") are all better-than-typical planning discipline.

However, the plan contains **one spec error** (a frontmatter field that does not exist),
**one internal contradiction** (its smoke-test success criterion violates its own
no-API-calls constraint), **one design regression** (copy-mode fallback silently
reintroduces the exact vendoring failure mode the repo exists to kill), and **one
feasibility gap** (offline string-similarity "evals" measure the heuristic, not skill
triggering — false confidence baked into a success criterion). It also under-uses the
native plugin marketplace mechanism, which would shrink the scaffolder substantially,
and ignores directly reusable prior art.

### A.2 What the plan gets right

1. **Mechanically verifiable success criteria** (§1.2) instead of vibes. Every criterion
   maps to a command with an exit code.
2. **The migration criterion is the correct definition of success.** A foundry no repo
   consumes is a fourth copy of the problem (the plan says this itself, Usage Note 3).
3. **Scope gate as a permission rule** (§3.2: adding a 4th agent requires confirmation)
   turns scope creep from a temptation into a policy violation.
4. **Schemas-first ordering** (§6.1: "the schemas ARE the contract") — matches this
   repo's proven spec-driven workflow (`specs/README.md`).
5. **Deterministic CI** (offline evals, no network in hooks, fixture-driven tests) is
   the right default; the defects below are in the *execution* of that intent, not
   the intent.
6. **Config-injection over hardcoding** with an env-override chain and an audit gate —
   the same governance rule enforced across this repo ("every knob is a typed field").

### A.3 Spec verification (the plan's §3.3 time-box, executed)

Verified against official docs (fetched 2026-07-02): plugins reference
(`code.claude.com/docs/en/plugins-reference.md`), marketplaces
(`plugin-marketplaces.md`), skills (`skills.md`), subagents (`sub-agents.md`),
hooks (`hooks-guide.md`), dependencies (`plugin-dependencies.md`).

| Plan claim | Verified reality | Impact |
|---|---|---|
| Layout: `.claude-plugin/plugin.json` + `agents/`, `skills/*/SKILL.md`, `commands/*.md`, `hooks/hooks.json` | ✅ Correct. Components live at plugin **root**; only `plugin.json` goes inside `.claude-plugin/`. | None — plan is right. |
| `plugin.json` fields | ✅ `name` is the only required field. Rich optional set: `version`, `description`, `author`, `displayName`, `dependencies` (with semver constraints), `userConfig` (values substituted as `${user_config.KEY}`), per-component path overrides. Unrecognized fields silently ignored. | `userConfig` is a **native** mechanism for the config-injection goal — plan should use it instead of only scaffold-time templating. |
| `SkillSpec.triggers: list[str]` frontmatter field | ❌ **Not a recognized field.** Documented skill frontmatter: `name`, `description`, `disable-model-invocation`. Triggering is driven by `description` alone. | **Finding F1.** Shipped components must stay spec-pure; eval fixtures go in sidecar files. |
| Agent frontmatter | ✅ `name`, `description`, `tools`, `model`, plus `effort`, `maxTurns`, `disallowedTools`, `skills`, `memory`, `background`. `hooks`/`mcpServers`/`permissionMode` NOT supported for plugin agents. | Schema field set should mirror this exactly. |
| Hooks events | ✅ `hooks/hooks.json`; large event vocabulary (`SessionStart`, `PreToolUse`, `PostToolUse`, `Stop`, …); `${CLAUDE_PLUGIN_ROOT}` and `${CLAUDE_PLUGIN_DATA}` are supported variables. | None. |
| Consumer version pinning ("foundry@vX.Y.Z") | ✅ **Natively supported**: marketplace plugin sources take `ref`/`sha` (github, git URL, git-subdir); explicit `version` in `plugin.json`; plugin `dependencies` accept semver constraints. Caveat: an explicit `version` must be **manually bumped** or updates never propagate. | **Finding F7.** Distribution should be a self-hosted `.claude-plugin/marketplace.json`, not a custom installer. The plan's existing "plugin.json version == git tag" CI criterion neutralizes the bump caveat — keep it. |
| Commands as a distinct component type | ⚠️ `commands/*.md` works but is the **legacy** layout; commands are now skills (docs recommend `skills/` for new plugins). | `/foundry-smoke` ships as a skill; `CommandSpec` becomes an alias/subset, not a fourth schema. |

### A.4 Findings (ranked)

#### F1 — BLOCKER (spec error): `triggers` frontmatter field does not exist; `extra="forbid"` fights the platform

§Level-4 code shows `SkillSpec.triggers: list[str]` with the intent that shipped SKILL.md
frontmatter carries eval fixtures, and `model_config = ConfigDict(extra="forbid")` so
"schema drift fails loudly."

- `triggers` is not part of the skill spec. Whether Claude Code tolerates unknown
  frontmatter keys is undocumented — shipping non-spec keys in the *artifact consumers
  receive* gambles the core product surface on an undocumented behavior.
- `extra="forbid"` on the validation schema means the validator **breaks on every field
  Claude Code adds** (the platform added `displayName`, `defaultEnabled`, `effort`,
  `memory` within months). The platform explicitly ignores unrecognized fields; a
  stricter-than-platform validator will block legitimate components.

**Resolution (adopted in Part B):** shipped component files are spec-pure — only
documented frontmatter keys. Eval/lint fixtures live in a repo-side sidecar
(`evals/skills/<name>.yaml`), never in the shipped artifact. Validation runs dual-mode,
exactly like this repo's `BaselineRegistry` convention: **strict for foundry-authored
fields** (typos in our own metadata fail CI) but **tolerant of unknown platform fields**
(`extra="ignore"` + a warning list), so a Claude Code release never bricks the validator.

#### F2 — BLOCKER (internal contradiction): the smoke-test success criterion violates the no-API-calls constraint

§1.2 criterion 2 requires `claude --print "run smoke command"` to succeed; §4.3 puts it
in the completion gate. §3.1 declares "Any Anthropic API calls at build/test time" OUT OF
SCOPE for CI. `claude --print` is a live model call. As written, the success criteria
cannot all be satisfied by CI, and the plan doesn't say which side wins.

**Resolution:** split the criterion. CI runs a **structural** smoke test: scaffold into a
temp dir, then assert the plugin loads and every component is discovered (manifest parse
+ component enumeration — no model call). The **live** `claude --print` smoke becomes a
release-runbook step, executed manually before tagging — the same pattern this repo uses
for GPU-gated headline runs (`specs/headline_runs.spec.md`: "runbook specs, not
CI-executed tests").

#### F3 — HIGH (design regression): copy-mode fallback reintroduces failure mode (c)

§1.3 failure mode (c): consumer divergence is "prevented by making consumption read-only
(plugin install, **not file copy**)." §2.1 then mandates a copy-mode fallback, §Level-3
ships a `CopyInstaller` — with no divergence countermeasure. A consumer on copy-mode is
exactly the vendored-copy world this repo exists to end, minus the awareness of it.

**Resolution:** copy-mode is legitimate as a compatibility fallback **only if drift is
detectable and updates are mechanical**:
1. Every copied file gets a generated-file header (`# GENERATED by foundry vX.Y.Z — do
   not edit; edits will be overwritten and flagged`).
2. The scaffolder writes `.foundry-lock.json` in the consumer: foundry version + per-file
   sha256.
3. New command `foundry verify --target <dir>` recomputes hashes and exits non-zero on
   drift; `foundry scaffold --upgrade` refuses to clobber locally modified files without
   `--force` (already a §2.1 constraint — extend it with the hash check so "modified" is
   detected, not assumed).
4. The integration test covers the drift path: scaffold → mutate a copied file → `verify`
   fails → `scaffold --upgrade` refuses.

#### F4 — HIGH (feasibility): offline string-similarity "evals" measure the heuristic, not triggering; the threshold is self-referential

§1.2: "precision/recall ≥ threshold defined in eval config" — a success criterion that
defers its own number to a config file is not mechanically verifiable at plan time.
Worse, §Level-3 scores offline mode with a "string-similarity heuristic." Skill
triggering in Claude Code is the *model* reading `description` in context; string
similarity between a trigger phrase and the description is not a proxy for that decision
with any validated correlation. A green eval gate would certify nothing, and a red one
would demand tuning the heuristic — institutionalized false confidence either way.

**Resolution:** replace the v0.1.0 eval gate with a **deterministic description lint**
(`foundry lint-descriptions`) that enforces the properties official skill-authoring
guidance actually recommends and that a program can check: description length within
bounds (floor *and* ceiling), contains at least one concrete "Use when…" trigger clause,
third-person voice, no undefined project-local jargon (configurable denylist), and — per
F1 — the sidecar fixture file exists with ≥ N trigger and ≥ M non-trigger phrases so the
corpus is ready. Model-scored eval mode (live, local-only, opt-in) is demoted to the
v0.2.0 roadmap where its cost/precision trade-off can be assessed against real triggering
telemetry. §1.2's eval criterion is rewritten accordingly (Part B §B.5).

#### F5 — MEDIUM (over-engineering): full ports/adapters DI for a directory-walking CLI

§2.1 mandates Protocol-based DI for **all** I/O — `FileSystem`, `ProcessRunner`, `Clock`
— with a dedicated `ports/` + `adapters/` layer, "no concrete class imports inside
business logic." For a tool whose core job is "walk a tree, parse YAML, apply Pydantic
models," a `FileSystem` Protocol is abstraction without a second implementation:
pytest's `tmp_path` already provides fast, deterministic, *real* filesystem tests, and
the plan's own integration test (scaffold into a temp dir) uses the real filesystem
anyway. The layer directly pressures the plan's own budgets (§3.3: ≤ 20 files per pass).

**Resolution:** keep Protocols where a fake is genuinely needed to make tests
deterministic — `ProcessRunner` (git/claude subprocess calls) and `Clock` — and inject
them at service constructors. Drop the `FileSystem` Protocol; use `pathlib` + `tmp_path`.
This is a soft-constraint amendment, not a hard-constraint violation: §2.2 already
prefers "small components."

#### F6 — MEDIUM (release coupling): v0.1.0 gated on a cross-repo migration PR

§1.2 criterion 3 and §4.3 make the v0.1.0 tag depend on a PR to an external repo
(ianshank/Agents) deleting its vendored copies. That serializes this repo's release on
review latency in another repo, and creates a chicken-and-egg problem: the consumer
should migrate onto a **tagged** version, but the tag waits on the migration.

**Resolution:** two-stage. **v0.1.0** = plugin + tooling green + **self-dogfood** (the
foundry repo consumes its own plugin via its own marketplace entry — the strongest
integration test that requires no external party). **v0.2.0 (adoption)** = the
ianshank/Agents migration PR merged with vendored copies deleted. Usage Note 3's point
survives intact — migration remains the definition of *mission* success — it just stops
being the release gate for the first tag.

#### F7 — MEDIUM (missed leverage): reimplementing distribution the platform already provides

The scaffolder's plugin mode (`TargetInspector → InstallPlanner → PluginInstaller`)
re-derives what `claude plugin install` + a marketplace already do. Verified: a repo can
carry `.claude-plugin/marketplace.json` naming itself as a marketplace with one plugin
entry; consumers add the marketplace and install with a pinned `ref` (git tag) or `sha`.

**Resolution:** foundry **is its own marketplace**. Plugin-mode scaffolding reduces to:
(a) register the marketplace + pinned version in the consumer's settings, (b) write the
sentinel-delimited CLAUDE.md block, (c) print the verification command. `CopyInstaller`
(with F3's lockfile) remains the genuine fallback for environments without plugin
support. This deletes an entire component from §Level-3 and shrinks Phase 3.

#### F8 — MEDIUM (operability): `audit-hardcoded` has no false-positive strategy

A grep-based gate over paths, model names, endpoints, and "magic numbers" will fire on
documentation prose ("train on 9×9, evaluate on 19×19"), example snippets inside skills,
and version strings. Without a waiver mechanism the gate gets progressively weakened or
ignored — the standard failure mode of blunt lint gates.

**Resolution:** (a) scope the audit to machine-consumed surfaces — frontmatter values,
hook scripts, `config/`, `plugin.json` — never markdown prose; (b) inline waiver comment
(`# foundry: allow-hardcoded(<pattern-id>): <justification>`) that the auditor counts and
reports, so waivers are visible and reviewable rather than silent; (c) pattern registry
stays in config (the plan already has this right).

#### F9 — LOW (constraint vs. reality): "hooks never execute network calls" excludes the most valuable hook pattern

The proven consumer pattern — this repo's own `SessionStart` bootstrap
(`.claude/hooks/session_start.sh`) — runs `pip install` so ephemeral web sessions can
lint/test at all. A blanket network ban excludes foundry from shipping the single
highest-value hook its consumers already depend on.

**Resolution:** refine the constraint: hooks that intercept the edit loop
(`PreToolUse`/`PostToolUse`/pre-commit) never touch the network; `SessionStart`
bootstrap hooks may, but must declare it (`network: true` in foundry's hook metadata),
degrade gracefully offline (the AlphaGalerkin hook already models this: warn, don't
fail the session), and be surfaced by `foundry validate` in its report.

#### F10 — LOW (assorted spec/plan hygiene)

- `description: Field(min_length=40)` incentivizes padding, not trigger quality; F4's
  lint rules replace it.
- `foundry docs` appears in the C4 container list and §2.2 but **no phase builds it** —
  assigned to Phase 4 in Part B (it is cheap: render README sections from the already-
  parsed manifest).
- Commands are the legacy layout (§A.3): `/foundry-smoke` ships as a skill;
  `CommandSpec` folds into `SkillSpec` rather than being a fourth schema module.
- §1.2's eval criterion references "threshold defined in eval config" — a success
  criterion must carry its number (fixed in §B.5).

#### F11 — LOW (missed reuse): the plan ignores its author's own proven prior art

Nothing in the plan seeds from what already works in consumer repos, which both slows
Phase 2 and risks shipping untested components:

| Foundry need | Existing, proven artifact to generalize |
|---|---|
| `code-reviewer` agent | `.claude/agents/reviewer.md` (adversarial checklist; de-projectize the AlphaGalerkin-specific rows) |
| `test-writer` agent | `.claude/agents/sqe.md` (test-pyramid + coverage-gate discipline) |
| Skill structure/tone | `.claude/skills/coverage-gate/SKILL.md`, `regression-surface/SKILL.md` |
| SessionStart hook | `.claude/hooks/session_start.sh` (graceful-degradation bootstrap) |
| Schema versioning + migration | `BaselineRegistry` pattern: explicit `*_SCHEMA_VERSION`, `extra="ignore"` forward compat, `migrate_*_document`, Hypothesis idempotence tests |
| Scaffold architecture | `src/agents/scaffold.py`: pure renderer functions + a `ScaffoldPlan` (path→content map) + thin orchestrator with `dry_run` — directly transplantable |
| Generated-doc golden tests | `specs/` AQA convention: assert doc ↔ config agreement |

**Resolution:** Phase 2 generalizes these rather than authoring from scratch; the
scaffolder adopts the `ScaffoldPlan` shape.

### A.5 Review summary table

| # | Severity | One-line | Disposition in Part B |
|---|---|---|---|
| F1 | Blocker | `triggers` frontmatter isn't spec; `extra="forbid"` fights platform evolution | Spec-pure components; sidecar fixtures; dual-mode validation (§B.2, §B.3 P0) |
| F2 | Blocker | `claude --print` smoke contradicts no-API-in-CI | Structural smoke in CI; live smoke as release runbook (§B.3 P3, §B.5) |
| F3 | High | Copy-mode reintroduces vendoring drift | Lockfile + hashes + `foundry verify` + refuse-on-drift (§B.3 P3) |
| F4 | High | String-similarity evals certify nothing; threshold self-referential | Deterministic description lint in v0.1; live evals → v0.2 (§B.3 P4) |
| F5 | Medium | `FileSystem` Protocol is abstraction without a second implementation | Protocols only for `ProcessRunner`/`Clock` (§B.2) |
| F6 | Medium | Release tag hostage to external-repo PR | v0.1.0 self-dogfood; migration = v0.2.0 (§B.3 P5) |
| F7 | Medium | Custom installer duplicates native marketplace | Self-hosted marketplace.json; scaffolder shrinks (§B.3 P3) |
| F8 | Medium | audit-hardcoded lacks waiver/scoping strategy | Scoped surfaces + visible waivers (§B.3 P1) |
| F9 | Low | Blanket hook-network ban excludes bootstrap hooks | Per-event policy + declared `network: true` (§B.3 P2) |
| F10 | Low | min_length=40 padding; `foundry docs` unphased; commands legacy | Lint rules; docs → P4; smoke ships as skill (§B.3) |
| F11 | Low | Ignores proven prior art | Seed Phase 2 from `.claude/`; port BaselineRegistry + ScaffoldPlan patterns (§B.3 P0/P2) |

---

## Part B — Revised Implementation Plan

Everything from the original plan not contradicted below carries forward unchanged
(objective function, permission architecture, error-handling protocol, resource budgets,
C4 Level 1/2 shape, CLAUDE.md seed).

### B.1 Revised repository layout

```
claude-code-foundry/
├── .claude-plugin/
│   ├── plugin.json              # name, version, author, userConfig
│   └── marketplace.json         # self-hosted marketplace: one entry, this repo (F7)
├── agents/                      # spec-pure frontmatter only (F1)
│   ├── code-reviewer.md
│   ├── test-writer.md
│   └── docs-writer.md
├── skills/
│   ├── project-conventions/SKILL.md
│   ├── testing-standards/SKILL.md
│   ├── c4-architecture/SKILL.md
│   └── foundry-smoke/SKILL.md   # the smoke command, as a skill (F10)
├── hooks/
│   ├── hooks.json
│   └── scripts/                 # Python where testability wins (anti-constraint kept)
├── evals/                       # repo-side sidecars, NOT shipped in components (F1)
│   └── skills/<name>.yaml       # trigger / non-trigger phrase fixtures
├── config/
│   └── defaults.yaml            # FOUNDRY_* env override chain (unchanged)
├── src/foundry/
│   ├── cli.py                   # typer entrypoints, thin
│   ├── schemas/                 # AgentSpec, SkillSpec, HookSpec, PluginManifest,
│   │                            #   MarketplaceManifest, FoundryLock — each with
│   │                            #   schema_version + migrate_* (F11: BaselineRegistry pattern)
│   ├── validator.py             # dual-mode: strict-own-fields / tolerant-platform (F1)
│   ├── auditor.py               # pattern registry from config + waiver counting (F8)
│   ├── linter.py                # description lint (F4)
│   ├── scaffold.py              # pure renderers + ScaffoldPlan + orchestrator (F11)
│   ├── verify.py                # consumer drift detection via .foundry-lock.json (F3)
│   ├── docsgen.py               # README generation from manifest (F10)
│   └── runners.py               # ProcessRunner + Clock Protocols + real impls (F5)
└── tests/{unit,integration}/
```

Deleted vs. original: `ports/` + `adapters/` layer (F5), `evaluator/` heuristic scorer
(F4), `PluginInstaller` (F7), standalone `CommandSpec` module (F10).

### B.2 Resolved design decisions

1. **Validation posture (F1):** two Pydantic model families per component —
   `*SpecStrict` (authoring CI: `extra="forbid"` **only over foundry-owned metadata**,
   catches our typos) and `*SpecTolerant` (`extra="ignore"`, used for anything that
   must survive platform evolution). Unknown keys are *reported* (warning list in the
   validation report), never fatal outside foundry-owned namespaces.
2. **Schema versioning (F11):** every persisted document (`plugin.json` mirror,
   `.foundry-lock.json`, eval fixtures) carries `schema_version`; `migrate_*_document`
   functions with Hypothesis idempotence tests, exactly as `poc/baselines` does.
3. **DI scope (F5):** constructor-injected `ProcessRunner` and `Clock` Protocols; real
   `pathlib` filesystem + `tmp_path` in tests. No `FileSystem` abstraction.
4. **Config injection (unchanged + F7):** `config/defaults.yaml` → `FOUNDRY_*` env →
   CLI flags, via Pydantic Settings; **plus** plugin `userConfig` for consumer-side
   values the platform substitutes natively (`${user_config.KEY}`).
5. **CLAUDE.md block (idempotency):** sentinel-delimited
   (`<!-- foundry:begin vX.Y.Z --> … <!-- foundry:end -->`); scaffold replaces only
   between sentinels; golden-file tests for the rendered block.
6. **Hook network policy (F9):** edit-loop hooks (PreToolUse/PostToolUse/pre-commit)
   are network-free, CI-enforced by the auditor; SessionStart bootstrap hooks may
   declare `network: true` and must degrade gracefully offline.

### B.3 Revised phases

**Phase 0 — Contract & spec freeze.**
Record the verified plugin spec (field tables from §A.3 + doc URLs + fetch date) in
`docs/SPEC_NOTES.md` — this satisfies §5.2 (preserve spec details across sessions) and
closes the §3.3 research time-box. Init pyproject/ruff/mypy/pytest/CI skeleton. Write
schemas (B.2 items 1–2), config loader, structlog setup, `runners.py`. *Exit:* empty
plugin validates; schema unit tests green, including one test asserting an unknown
platform frontmatter key is a warning, not an error.

**Phase 1 — Validator + auditor.**
`foundry validate` (walk tree → parse frontmatter → dual-mode schemas → aggregated
structured error report, non-zero exit on any error). `foundry audit-hardcoded` with
config-resident pattern registry, scoped to machine-consumed surfaces, waiver comments
counted in the report (F8). *Exit:* both self-run green on this repo; a seeded
violation fixture proves each fails correctly.

**Phase 2 — Component set (3 agents / 4 skills / 2 hooks; smoke is the 4th skill).**
Generalize from proven artifacts (F11): `code-reviewer` ← AlphaGalerkin `reviewer.md`;
`test-writer` ← `sqe.md`; `docs-writer` new. Skills: `project-conventions`,
`testing-standards`, `c4-architecture`, `foundry-smoke`. Hooks: pre-commit validate +
post-edit lint (`PostToolUse` matcher on Edit|Write), Python scripts. All frontmatter
spec-pure; every skill gets an `evals/skills/<name>.yaml` sidecar with ≥ 5 trigger and
≥ 5 non-trigger phrases. *Exit:* `foundry validate` + `audit-hardcoded` green over the
real component set. Scope gate unchanged: a 4th agent still requires confirmation.

**Phase 3 — Distribution + scaffolder.**
`.claude-plugin/marketplace.json` naming this repo as marketplace + plugin source (F7).
`foundry scaffold --target X`: plugin mode = register marketplace pinned to a tag +
CLAUDE.md sentinel block; copy mode = ScaffoldPlan-rendered copies **with generated-file
headers + `.foundry-lock.json` hashes** (F3). `foundry verify --target X` drift check.
Integration tests: (a) scaffold → validate round-trip in `tmp_path`; (b) drift path —
mutate a copied file → verify fails → `scaffold --upgrade` refuses without `--force`;
(c) **structural smoke** — scaffolded target's components enumerate correctly, no model
call (F2). *Exit:* self-dogfood — this repo consumes its own plugin via its own
marketplace entry.

**Phase 4 — Description lint + docs.**
`foundry lint-descriptions` (F4): length window, "Use when…" trigger clause present,
third-person voice, configurable jargon denylist, sidecar fixture existence + minimum
corpus size. `foundry docs` (F10): README component tables rendered from the parsed
manifest, golden-file tested. *Exit:* both green over the Phase 2 set; CI wired.

**Phase 5 — Release v0.1.0 (F6).**
CHANGELOG; CI job asserting `plugin.json version == git tag == CHANGELOG top entry`;
release workflow. **Live smoke runbook** (release-blocking, manual): fresh clone →
`pipx install -e .` → scaffold into `mktemp -d` → `claude --print` runs
`/foundry-smoke` (F2). Tag v0.1.0. *Post-tag (v0.2.0 roadmap, not release gates):*
migrate ianshank/Agents and delete its vendored copies (the adoption criterion);
model-scored live eval mode; `foundry upgrade` consumer update command.

### B.4 CI pipeline (revised)

```
lint (ruff) → typecheck (mypy --strict) → unit → integration (scaffold round-trip,
drift path, structural smoke) → foundry validate → foundry audit-hardcoded →
foundry lint-descriptions → version-consistency (tag == plugin.json == CHANGELOG)
→ [on tag] release
```

Seven gates as before, with `eval --offline` replaced by `lint-descriptions` (F4) and
`version-consistency` added.

### B.5 Rewritten success criteria (§1.2 replacement)

```
v0.1.0 succeeds when:
- [ ] `foundry validate` exits 0 over the shipped component set (dual-mode: strict on
      foundry-owned fields, tolerant+warning on unknown platform fields)
- [ ] `foundry audit-hardcoded` exits 0 with zero unwaivered findings; every waiver
      carries a justification
- [ ] `foundry lint-descriptions` exits 0: every skill description passes the
      deterministic lint AND has a sidecar fixture with ≥5 trigger / ≥5 non-trigger
      phrases
- [ ] Integration suite green: scaffold→validate round-trip, copy-mode drift detection,
      structural smoke (component enumeration, no model call)
- [ ] Self-dogfood: this repo consumes its own plugin via its own marketplace entry
- [ ] pytest ≥85% coverage on src/foundry; ruff + mypy --strict zero errors
- [ ] plugin.json version == git tag == CHANGELOG top entry (CI-enforced)
- [ ] Release runbook executed manually: live `claude --print` /foundry-smoke in a
      fresh scaffolded consumer (documented with date + Claude Code version)

v0.2.0 (adoption) succeeds when:
- [ ] ianshank/Agents consumes foundry@v0.1.x and its vendored agent/skill files are
      DELETED in a merged PR linking back here
```

### B.6 Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Plugin spec moves again (it did repeatedly in 2026) | High | `docs/SPEC_NOTES.md` with fetch dates; tolerant validation mode (F1); re-verify spec at each minor release |
| Explicit `version` field forgotten at release → consumers never see updates | Medium | version-consistency CI gate (B.4) makes the tag fail, not the consumers |
| Copy-mode consumers drift anyway | Medium | `foundry verify` in consumer CI is the countermeasure; document it in the scaffolded CLAUDE.md block |
| Description lint passes but skills still don't trigger well | Medium | Acknowledged residual risk of dropping heuristic evals (F4) — the honest position; live eval mode in v0.2 closes it with real signal |
| Scope creep past 3/3/2/1 | Low | Permission-architecture gate retained verbatim |
```
