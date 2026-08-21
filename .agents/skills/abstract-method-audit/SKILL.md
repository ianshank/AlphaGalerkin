---
name: abstract-method-audit
description: Audit a package's @abstractmethod and Protocol members for missing call sites — dead abstractions that every subclass implements but nothing invokes. Use before adding a new ABC/Protocol or reviewing a diff that touches one, to catch the F1-class defect (an abstract method spec'd in a docstring that never actually runs).
---

# abstract-method-audit — find abstractions that never run

An `@abstractmethod` overridden by every subclass but *called* by nothing is dead — the module
docstring describes an algorithm that does not execute. That is how `PDEGame.get_reward` (F1)
survived: abstract, universally implemented, invoked nowhere in `src/`. A `Protocol` member declared
by a caller but read by no callee is the same class of silent contract break (F0: `n_players`).

## Run it

```bash
# Report mode (non-blocking) — default. Scans src/ by default.
python -m scripts.audit_abstractions src

# One package
python -m scripts.audit_abstractions src/mcts

# Blocking mode (CI / pre-merge for a package that must be clean)
python -m scripts.audit_abstractions src/mcts --fail-on-missing
```

## How it decides

- An `@abstractmethod` `foo` is **called** iff `.foo(` appears anywhere under the scanned roots.
  Overrides (`def foo(`) do not match, so only genuine call sites count.
- An abstract **property** (`@property @abstractmethod`) or a `Protocol` member is **read** iff the
  attribute form `.name` appears. (Properties are read, not called — the tool distinguishes them.)
- Dunder / framework hooks (`__init__`, `__enter__`, …) are never flagged.

The heuristic is deliberately simple so its output is trustworthy. It can miss a member accessed only
via `getattr`; it counts a call site anywhere in the tree (not just outside the defining module); and
because the match is by member *name*, if two classes declare the same member name and only one has a
caller, the tool credits both (a name-collision false negative). Findings are de-duplicated by the
fully-qualified `(file, class, name)` key, so distinct declarations are each reported when flagged.
Treat it as a *screen*, not a proof — a hit is a strong signal, a clean run is reassurance.

## Policy

- **`src/mcts`, `src/refinement` and `src/pde` are clean and must stay clean.** All three are
  gated with `--fail-on-missing` **in CI** (`.github/workflows/ci.yml`, `lint` job), so a newly
  dead abstraction on these surfaces fails the build:

      python -m scripts.audit_abstractions src/mcts src/refinement src/pde --fail-on-missing

  `src/mcts`/`src/refinement` were cleared by the F0/F1 fixes (`get_reward`, `n_players` gained
  call sites). `src/pde` was cleared by **B17**: `PDEGame.get_result` — declared abstract,
  documented as lifecycle step 4, implemented by every concrete game, called by nothing — was
  deleted along with its `PDEResult` struct. It was replaced by `PDEGame.termination_reason`,
  which has a real consumer (`AlphaGalerkinSolver` records it under
  `METADATA_KEY_TERMINATION_REASON`). That is the canonical worked example of this skill's
  triage: the fix was *not* to wire the dead method up for its own sake — `PDEResult` carried
  six fields no caller read and lacked the fields the five real terminal paths need — but to
  delete it and extract the one part that something genuinely wanted.
- **`src/training` has one accepted baseline** (recorded 2026-08, `docs/CODE_HYGIENE_AUDIT.md`
  §7.3) — it is *not* in the blocking set. Run it without `--fail-on-missing` and expect exactly:

      BaseLoss.forward  (src/training/losses/base.py:40)

  Reported as a *Protocol member with no reader*, not an abstract method. Treat any hit beyond
  this one as a blocker. (`BaseTrainer.compute_loss/generate_data/evaluate` used to be dead
  `@abstractmethod`s here; they were demoted to concrete `step()` hooks, so they no longer
  appear.)
- **The rest of `src/` is report-only.** The same CI job runs `audit_abstractions src/` with
  `continue-on-error`, because the domain PoCs (`src/backend`) carry a known untriaged backlog.
  Treat its output as advisory.
- **Then: fully blocking.** Once that backlog is triaged, promote the report-only step to
  `--fail-on-missing src`.

When a hit is real, the fix is one of: wire the method to a call site (F1 → Option 1), delete it and
rewrite the docstring, or (for a protocol member) confirm the callee reads it.
