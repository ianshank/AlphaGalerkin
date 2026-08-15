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
- `src/mcts`, `src/refinement` and `src/pde` **are clean and must stay clean** — CI runs them
  with `--fail-on-missing` (`.github/workflows/ci.yml`, `lint` job), so treat any hit as a
  blocker:

  ```bash
  python -m scripts.audit_abstractions src/mcts src/refinement src/pde --fail-on-missing
  ```

- `src/training` has **one accepted baseline** (`BaseLoss.forward`,
  `src/training/losses/base.py:40` — a Protocol member with no reader), recorded in
  `docs/CODE_HYGIENE_AUDIT.md` §7.3. It is *not* in the blocking set; run it without
  `--fail-on-missing` and treat anything beyond that one hit as a blocker.
- A hit is fixed by wiring the method to a call site, deleting it (and its docstring), or confirming
  the protocol member has a reader. `PDEGame.get_result` (`docs/CODE_HYGIENE_AUDIT.md` **B17**) is
  the worked example of the delete path — and of extracting the one genuinely-wanted piece
  (`termination_reason`) rather than wiring a dead struct up for its own sake.

See the `abstract-method-audit` skill for the heuristic and policy.
