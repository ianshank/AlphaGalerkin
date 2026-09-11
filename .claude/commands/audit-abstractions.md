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

Interpretation (this mirrors the `typecheck` job in `.github/workflows/ci.yml`; when they
disagree, the workflow wins and this file is stale):
- **Report mode is non-blocking** — triage the hits, don't batch-fix. The only package with a
  known, untriaged backlog is `src/backend` (the domain-PoC `BackendInterface` members,
  `docs/CODE_HYGIENE_AUDIT.md` B10); CI runs it with `continue-on-error`.
- **The four refinement roots are gated hard, in ONE invocation** — the audit resolves call
  sites within the union of the roots it is given, so scanning them one at a time is strictly
  stricter and flags cross-package readers as missing (`src/research` drives
  `src/refinement`'s `RefinementSubstrate`):

  ```bash
  python -m scripts.audit_abstractions src/mcts src/refinement src/pde src/research --fail-on-missing
  ```

- **Every other package except `src/backend` is gated hard too**, again in one combined
  invocation (`$(ls -d src/*/ | grep -v 'src/backend/') --fail-on-missing`). The former
  `src/training` "accepted baseline" (`BaseLoss.forward`) is not a baseline any more: the combined
  scan finds its reader outside `src/training`, which is exactly why the roots must be passed
  together. A hit in *any* of these packages is a blocker.
- A hit is fixed by wiring the method to a call site, deleting it (and its docstring), or confirming
  the protocol member has a reader. `PDEGame.get_result` (`docs/CODE_HYGIENE_AUDIT.md` **B17**) is
  the worked example of the delete path — and of extracting the one genuinely-wanted piece
  (`termination_reason`) rather than wiring a dead struct up for its own sake.

See the `abstract-method-audit` skill for the heuristic and policy.
