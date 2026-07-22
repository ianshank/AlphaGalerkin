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

- **Now (one release): non-blocking report.** Run it, triage the hits, don't batch-fix. The current
  `src/` baseline has known dead abstractions in domain PoCs (`src/backend`) — those are
  pre-existing and out of scope for the refinement-engine work.
- **`src/mcts`, `src/pde`, `src/refinement` must stay clean** (`--fail-on-missing`) —
  these are the surfaces the F0/F1 fixes touched.
- **Then: blocking.** Once the domain-PoC backlog is triaged, wire `--fail-on-missing src` into CI.

When a hit is real, the fix is one of: wire the method to a call site (F1 → Option 1), delete it and
rewrite the docstring, or (for a protocol member) confirm the callee reads it.
