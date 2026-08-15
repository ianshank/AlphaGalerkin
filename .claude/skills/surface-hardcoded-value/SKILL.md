---
name: surface-hardcoded-value
description: Replace a bare numeric literal with a named constant or typed Pydantic field without changing any computed value. Use when a review flags a magic number, when adding a tunable, or when reconciling the same literal across call sites — covers frozen migration literals, splitting by semantic, copying mutable constants, and the value-identity check that proves nothing moved.
---

# surface-hardcoded-value — name a literal without moving a number

The project rule is "every knob a typed Pydantic field or named constant"
(`CLAUDE.md`; the `reviewer` agent treats a bare magic number as a finding). The rule is easy;
applying it without changing behaviour is not. **Zero numeric change is the acceptance criterion** —
a "cleanup" that shifts a training trajectory or a solver tolerance is a defect, not a cleanup.

Work through these in order. Steps 1–2 decide *whether* and *how*; step 5 proves you were right.

## Step 1 — Classify the literal before touching it

| If the literal is… | Then | Why |
|---|---|---|
| A **frozen historical default** (schema migration, checkpoint upgrade, on-disk format) | **Leave it. Add a freeze comment + a drift-alarm test.** | A `1.0.0 → 1.1.0` migration must inject the defaults *v1.1.0 shipped with*, forever. Binding it to a live constant means retuning that constant silently rewrites what old artifacts migrate to. Precedent: `src/training/checkpoint_migration.py` + `tests/training/test_checkpoint_migration.py::TestMigrationDefaultFreeze`. |
| The **same value, two different meanings** | **Two constants, not one.** | Precedent: Gumbel's six `1e-8`s were three inert division guards (`GUMBEL_NORMALIZATION_EPSILON`) and three log-argument floors that set a zero-prior action's ≈−18.4 score (`GUMBEL_LOG_PRIOR_FLOOR`). One constant couples an inert number to an algorithmic one. |
| The **same meaning, different values** across call sites | **Preserve each site's live value.** Name them separately, or surface as a config field defaulted to what that site used. | Precedent: LR-scheduler knobs existed in three copies (0.01/1e-6 twice, 0.1/0.1 once). Unifying would have rewritten one trainer's LR trajectory. |
| A **mutable container** (`list`, `dict`, `set`) | Hand out **copies**: `list(CONST)` / `default_factory=lambda: list(CONST)`. | Sharing the module-level object lets one config mutate every other config and the constant itself. |
| A genuinely single-purpose scalar | Name it, or make it a typed field. | The easy case. |

**Do not unify two values just because they are numerically equal.** Check whether they are the
same *knob* (`DEFAULT_PICOGK_BOUNDARY_TOLERANCE` 1e-5 vs `DEFAULT_BOUNDARY_PROJECTION_TOL` 1e-5 are
equal and unrelated — a classification band vs a Newton convergence tolerance).

## Step 2 — Pick the home

- **Cross-package numeric** → `src/constants.py`, with a docstring saying what it means and what it
  is *not* (cross-reference any numerically-equal-but-unrelated neighbour).
- **Package-internal** → a module-level constant in the owning module (precedent:
  `GUMBEL_*` in `src/mcts/gumbel.py`, `_SOFTMAX_NORMALIZER_FLOOR` in `src/mcts/evaluator.py`).
- **User-tunable** → a typed Pydantic field with validators (`gt`, `ge`, `le`) and a `description`.
  Mirror the constraints of any sibling field for the same knob.
- If a *mirrored* implementation already names it, **reuse the exact name** so the mirror stays
  greppable (`_SOFTMAX_NORMALIZER_FLOOR` matches `src/integrations/lm_studio/evaluator.py`).

## Step 3 — Backwards compatibility for new config fields

Before adding a Pydantic field, check the model's `extra` policy and its parse paths:
- `extra="forbid"` models reject unknown keys — adding a field is fine, but check nothing
  *round-trips* a dict built from an older schema into a stricter sibling.
- Give every new field a default equal to the value the call site used. Old YAMLs and old
  checkpoints must parse unchanged and produce identical numbers.
- If Hydra-addressable, add the key to the shipped YAML too, or overrides need a `+` prefix.
- Adding a field with a default needs **no** checkpoint-migration bump; changing a default does.

## Step 4 — Leave the tests and docstrings alone

Literals in **tests** are assertions — replacing them with the constant makes the test tautological
(it would then pass for any value). Leave them. Same for docstring examples.

## Step 5 — Prove zero numeric change

Assert it, don't eyeball it:

```bash
python -c "
from src.constants import <CONST>
from <module> import <thing>
assert <CONST> == <the literal you replaced>
# construct the config/objects and assert the resolved values equal the old literals
# for mutable constants, prove copy semantics:
a = Cfg(); a.field.append(999); assert Cfg().field == <original>, 'mutable default leaked!'
"
```

Then run the Regression Surface rows for every touched path (use the `regression-surface` skill)
and the per-module coverage gates (`coverage-gate`). Report the actual results.

## Step 6 — Record it

Add a `CHANGELOG.md` `[Unreleased]` entry under **Changed**, stating explicitly that defaults are
unchanged. If the literal was a latent divergence (the same knob with different values), say which
values existed and which each call site kept — that is the part a reviewer cannot reconstruct from
the diff.
