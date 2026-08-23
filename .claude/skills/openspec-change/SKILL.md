---
name: openspec-change
description: Scaffold an OpenSpec change package for a change that touches a charter Requirement — proposal, design, tasks, and the delta spec. Use when scope, claims, capabilities, gates or deviations change; use spec-new instead for a feature that changes no Requirement.
---

# Scaffold an OpenSpec change package

The charter is supreme, and `specs/` sits below it (`openspec/project.md` sets the order:
charter → `ARCHITECTURE.md` → `CLAUDE.md` → `specs/`). `spec-new` scaffolds a *feature* spec.
Nothing scaffolded the layer above it, so charter changes were being made by hand.

## Which system

| The change… | Use |
|---|---|
| adds/removes a `src/` package, or changes what is in scope | **this** — *Scope Integrity* |
| makes, corrects or retracts a numeric claim | **this** — *Evidence-Backed Claims* |
| registers a new PoC scenario | **this** — *Capability Register Accuracy* |
| adds or changes a coverage gate | **this** — *Quality Gate Fidelity* |
| accepts a deviation | **this** — *Accepted Deviation Disclosure* |
| ships a feature touching none of the above | `spec-new` |
| both | both — the substrate work needed a feature spec *and* a charter delta |

## Steps

1. **Pick a change id**: kebab-case, names the outcome not the mechanism
   (`element-local-substrate`, not `add-substrate-module`).

2. **Create the four files.** Mirror `openspec/changes/project-charter-alignment/`, which is
   the worked example:

   ```
   openspec/changes/<id>/
   ├── proposal.md                              # Why / What Changes / Impact
   ├── design.md                                # the decisions worth arguing about
   ├── tasks.md                                 # ordered checkboxes
   └── specs/project-charter/spec.md            # the delta
   ```

3. **`proposal.md`** — `## Why` states the problem with evidence, not intent. Cite a committed
   artifact or a spike under `evidence/spikes/`. Add `## What Changes`, `## Impact`, and — this
   is the one people skip — an explicit section naming what the change does **not** do.
   Bundling is the failure mode this convention exists to correct.

4. **`design.md`** — only the decisions a reviewer could reasonably disagree with, each with
   its reason. Skip anything obvious. If a measurement settled a decision, give the number.

5. **`tasks.md`** — ordered checkboxes, critical path called out, plus a *deferred* section so
   the boundary from the proposal is visible in the work plan too.

6. **The delta** uses OpenSpec verbs and touches only what changes:

   ```markdown
   ## MODIFIED Requirements
   ### Requirement: Scope Integrity
   ...what is now true...
   #### Scenario: <the failure this prevents>
   - GIVEN … WHEN … THEN … SHALL …
   ```

   State plainly that the other Requirements are untouched. **Do not edit
   `openspec/specs/project-charter/spec.md`** — the delta is a proposal; it is applied when
   the change lands.

7. **Verify**:
   ```bash
   pytest tests/docs/ -v
   python scripts/check_doc_links.py
   ```

8. **Archive on completion.** When every task is checked and the delta has landed, move the
   directory to `openspec/changes/archive/<id>/`. `project-charter-alignment` is a documented
   exception — it is cited from append-only `CHANGELOG.md`.

## Honesty rules

- **A Requirement without a guard is a wish.** If a scenario is a *review* obligation rather
  than a mechanical check, say so in the delta. Claiming enforcement that does not exist is
  precisely what the charter was written to stop.
- **Every deviation states a retirement condition.** Without one it is a permanent exemption
  wearing a temporary label.
- **Cite artifacts that contain what you claim.** A comparison claim citing a file holding one
  arm passes the existence guard and evidences nothing.
