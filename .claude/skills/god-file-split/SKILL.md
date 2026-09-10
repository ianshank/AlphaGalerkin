---
name: god-file-split
description: Split a flat Python module into a package with import-compatible re-exports. Use before converting any god file (baselines.py, trainer.py, losses/physics.py, operators.py) — encodes the PR #140 recipe: public-name freeze (not raw dir()), one-directional __all__, del submodule names, mypy override glob, coverage --include lockstep, and grep of the old path with and without a directory prefix.
---

# god-file-split — package a flat module without breaking callers

A "just move the file" PR will go red on four independent axes that look
unrelated: extra public names (`__path__`, submodule attributes), a mypy
override that does not cascade, a coverage `--include` that measures the
re-export shim at 100% while the implementation is untraced, and docs that
still cite the old `.py` path after it became a package. PR #140
(`src/pde/operators/` was a flat module; see that package's `__init__.py`
docstring) is the worked example. Follow this recipe **before**
any production code moves.

Related: `add-coverage-gate` (coverage form), `pr-preflight` (the verify
block), `abstract-method-audit` (do not split a Protocol into unread
members).

## Step 0 — Freeze the public surface **before** the split

```bash
python - <<'PY'
import importlib
mod = importlib.import_module("src.<pkg>.<module>")
public = sorted(n for n in dir(mod) if not n.startswith("_"))
print("\n".join(public))
print("--- count", len(public))
PY
```

Paste that set into a test as a `frozenset` (precedent:
`tests/pde/test_operators.py::_OPERATORS_PACKAGE_PUBLIC_API` /
`TestOperatorsPackagePublicAPI`).

**Not** raw `dir()`. Becoming a package unavoidably adds `__path__` and,
once you write it, `__all__`. Claiming "`dir()` is byte-identical" is the
overclaim peer review already killed on PR #140. The real guarantee is:

```python
public_names = {n for n in dir(pkg) if not n.startswith("_")}
assert public_names == _FROZEN_PUBLIC_API
```

Include **leaked** imports (`np`, `torch`, `ABC`, `annotations`,
`dataclass`, …). Callers and `from module import *` see them today; dropping
them is an API change, not a cleanup. `from __future__ import annotations`
leaks the name `annotations` — keep it in the freeze even if `__all__`
omits it (see one-directional `__all__` below).

Do **not** start the split until this test exists and is green against the
flat module. A freeze written after the move cannot fail.

## Step 1 — Package layout and `__init__.py` re-exports

Convert `src/<pkg>/<module>.py` → `src/<pkg>/<module>/`. Keep the import
path `from src.<pkg>.<module> import Name` valid.

Mirror the **exact** top-of-file import block of the old module (down to
incidental leaked names) in `__init__.py`, then import each extracted
symbol from its new submodule.

### `del` submodule names

`from src.<pkg>.<module>.foo import Bar` binds `foo` as an attribute of the
package. The old flat module never exposed that. Delete the bindings after
the imports (the submodules stay importable and cached in `sys.modules`):

```python
# Precedent: src/pde/operators/__init__.py
del (
    schemas,
    predicates,
    # ... every submodule imported above
)
```

If you skip this, the public-name freeze fails with extra names
(`schemas`, `pinn`, …).

### `__all__` is one-directional

```python
assert _FROZEN_PUBLIC_API - {"annotations"} <= set(pkg.__all__)
```

`__all__` may list extra explicit re-exports (a private `_helper` a
foreign test imports). It is **not** required to equal `dir()`'s public
names. The invariant is: every frozen public name is in `__all__`, so
`from pkg import *` cannot silently drop one. `annotations` (the
future-import leak) is the documented exception.

## Step 2 — mypy overrides do not cascade

`pyproject.toml` `[[tool.mypy.overrides]]` module globs **do not** apply
to submodules. If the flat module had:

```toml
[[tool.mypy.overrides]]
module = ["src.research.baselines"]
disable_error_code = ["no-untyped-call"]
```

the package needs **both** `"src.research.baselines"` and
`"src.research.baselines.*"`. Compare the mypy error set on the package
before and after (`mypy src/<pkg> --strict --ignore-missing-imports`).

## Step 3 — Coverage `--include` lockstep (the silent false-pass)

This is how a split goes green while measuring nothing — the B31
omit-collision class.

A glob `*/src/<pkg>/<module>.py` matches the package's `__init__.py` only.
After the move, executed lines live in `*/src/<pkg>/<module>/*`. Leaving
the include as-is reports 100% of a re-export shim.

**Required include after a `.py` → package conversion:**

```
--include="*/src/<pkg>/<module>.py,*/src/<pkg>/<module>/*"
```

Keep the `.py` glob so the `__init__.py` shim cannot rot unread. Add the
directory glob so solver/loss files are actually measured.

If CI also lists the old path in CLAUDE.md's Regression Surface, update
**both** in the same commit. A docs-only fix leaves CI measuring the shim;
a CI-only fix leaves the documented command wrong (B8).

Native runner form, never `--cov=path.py` (coverage 7.x silently drops
file-path specs) and never dotted `--cov=src.pkg.module` (torch C-extension
collision on this repo).

## Step 4 — Grep the old path, both ways

```bash
# The file name, anywhere
rg -n "<module>\.py" --glob '!**/__pycache__/**'
# The same name after a directory prefix (ARCHITECTURE.md, CLAUDE.md, specs)
rg -n "src/.*/<module>\.py" docs/ CLAUDE.md ARCHITECTURE.md specs/ openspec/ .github/
python scripts/check_doc_links.py
```

Update `ARCHITECTURE.md` citations. The charter scope register is
**package-granular** (`src/<pkg>/`); a nested `src/<pkg>/<module>/` row
cannot be added (`tests/docs/test_architecture_map.py` enumerates
`src/*/__init__.py` only). Do not invent a Scope Integrity row for a
subpackage.

## Step 5 — Registry / import-cycle landmines

If other modules import a registry object from the old module
(`SOLVER_REGISTRY`, `@register_loss`), bind that object in `__init__.py`
**before** any registrant module is imported. Do **not** import
registrants from the new package `__init__` if the old module did not —
they import it.

Lazy `TYPE_CHECKING` + in-method imports that break a cycle stay lazy.
Hoisting them to module level is how you recreate the cycle the split
was supposed to leave alone.

## Step 6 — Verify (one process)

```bash
# Public-name freeze + the package's own tests
pytest tests/<pkg>/test_<module>.py -q

# Combined surface that imported the old path (one pytest process)
pytest <the plan's combined list> -q -m "not gpu_required"

# Coverage include must list both globs; fail-under is the existing gate
python -m coverage run --branch \
  --include="*/src/<pkg>/<module>.py,*/src/<pkg>/<module>/*" \
  -m pytest <same list> -q -p no:cov
python -m coverage report \
  --include="*/src/<pkg>/<module>.py,*/src/<pkg>/<module>/*" \
  --fail-under=<existing>

ruff check src/<pkg>/<module>/
ruff format --check src/<pkg>/<module>/
mypy src/<pkg> --strict --ignore-missing-imports
python scripts/check_doc_links.py
```

Do **not** set `COVERAGE_CORE`. Do not claim raw `dir()` identity in the
PR body.

## Out of scope for this skill

- Calling `super().__init__()` as part of a class split (behavior change;
  see `src/training/trainer.py`).
- Adding a charter scope-register row for a subpackage or a root module.
- Unifying leaked names away (`np` → not re-exported) — that is an API
  break, land it separately if at all.
