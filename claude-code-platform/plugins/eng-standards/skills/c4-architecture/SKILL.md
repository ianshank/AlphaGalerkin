---
name: c4-architecture
description: Standards for authoring C4 architecture documentation in Mermaid — Context, Container, and Component levels, dynamic views for key flows, and diagram hygiene rules. Use when creating or updating architecture docs — trigger phrases include "C4 diagram", "architecture doc", "document the architecture", "add a component diagram", "system context", or "sequence diagram for this flow".
---

# C4 Architecture Documentation

## Levels and When Each Is Required

Author top-down. One diagram per level, one Mermaid block per diagram.

1. **Context (L1)** — required for every system. Shows the system as one box, its users (Person), and external systems it talks to. Answers: who uses this and what does it depend on?
2. **Container (L2)** — required for every deployable system. Shows the runnable/deployable units (services, CLIs, databases, model servers) and the protocols between them. Answers: what runs where and how do the pieces communicate?
3. **Component (L3)** — required per container that has non-trivial internal structure (more than ~3 collaborating modules). Shows the major modules/packages inside one container and their responsibilities. Do NOT diagram every class.
4. **Dynamic views** — required for each key flow (the happy path of every headline use case, plus any flow with retry/fallback branching). Use a Mermaid `sequenceDiagram` showing the ordered interactions across containers/components.

Code-level (L4) diagrams are not authored — the code is the diagram.

## Authoring Rules

- Every diagram lives in a markdown file under the project's architecture docs directory, preceded by a one-paragraph prose summary stating what the diagram shows and what changed since the last revision.
- Keep each diagram under ~10 elements. If it exceeds 10, split by boundary or promote a cluster to its own child diagram.
- Every element gets a description string: what it is and its single responsibility. No empty descriptions.
- Update the affected diagram in the same PR as the structural change. An architecture change without a diagram update is an incomplete PR.

## Mermaid Conventions

- Use the C4 blocks: `C4Context`, `C4Container`, `C4Component` — not generic `graph TD` for C4 levels.
- Elements: `Person(alias, "Label", "description")`, `System(...)`, `Container(alias, "Label", "technology", "description")`, `Component(...)`, `ContainerDb(...)` for stores, `System_Ext(...)` for externals.
- Relationships: `Rel(from, to, "verb phrase", "protocol/technology")`. Labels are active verbs — "submits job to", "reads checkpoints from", "streams frames to" — never bare nouns like "data" or "API".
- Boundaries: wrap owned elements in `System_Boundary` / `Container_Boundary`; externals stay outside the boundary. One boundary level per diagram — do not nest boundaries twice.
- Aliases are snake_case and stable across levels: the container named `training_service` at L2 keeps that alias when it appears as a boundary at L3.
- Direction of `Rel` is the direction of the request/call, not the data. If both matter, use `BiRel` sparingly with a label naming both actions.

## Review Checklist (apply before merging)

- [ ] Each level present that the rules above require; one diagram per Mermaid block.
- [ ] ≤ ~10 elements per diagram; boundaries used instead of clutter.
- [ ] All `Rel` labels are verb phrases with a technology/protocol argument.
- [ ] Descriptions non-empty; aliases stable and snake_case.
- [ ] Dynamic view exists for each headline flow touched by the change.

See references/mermaid-conventions.md for complete skeletons for each C4 level, a sequenceDiagram template, naming conventions, and the full review checklist.
