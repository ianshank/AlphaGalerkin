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
  abstractions live in the domain PoCs (`src/reentry`, `src/intercept`, `src/backend`).
- The refinement surfaces **must stay clean**: `src/mcts`, `src/pde`, `src/refinement`, `src/thermo`.
  For those, run with `--fail-on-missing` and treat any hit as a blocker.
- A hit is fixed by wiring the method to a call site, deleting it (and its docstring), or confirming
  the protocol member has a reader.

See the `abstract-method-audit` skill for the heuristic and policy.
