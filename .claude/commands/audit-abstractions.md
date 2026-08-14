---
description: Audit @abstractmethod / Protocol members for missing call sites (dead abstractions like F1's get_reward).
argument-hint: "[package path, default src]"
---

Run the abstract-method audit over `$ARGUMENTS` (default `src`). Report abstract methods with no
call site and protocol members with no reader — dead abstractions of the F1 (`PDEGame.get_reward`)
class.

```bash
python -m scripts.audit_abstractions ${ARGUMENTS:-src}
```

Interpretation:
- **Report mode is non-blocking** — triage the hits, don't batch-fix. Known pre-existing dead
  abstractions live in the domain PoCs (`src/backend`).
- `src/mcts` and `src/refinement` **are clean and must stay clean** — run with
  `--fail-on-missing` and treat any hit as a blocker.
- `src/pde` has **one known baseline hit** (`PDEGame.get_result`, `src/pde/game.py:457`) and so
  is run *without* `--fail-on-missing`. Treat any hit beyond that one as a blocker. See the
  `abstract-method-audit` skill and `docs/CODE_HYGIENE_AUDIT.md` **B17**.
- A hit is fixed by wiring the method to a call site, deleting it (and its docstring), or confirming
  the protocol member has a reader.

See the `abstract-method-audit` skill for the heuristic and policy.
