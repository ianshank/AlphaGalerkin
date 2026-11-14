---
name: retire-a-deprecation
description: Move a call off an API a dependency has deprecated (a DeprecationWarning, FutureWarning or "not recommended" UserWarning from torch, onnx, numpy, pydantic, ...) before its removal turns a gate red — reproduce the warning a suite was hiding, guard it with a premise test plus no-warning tests, check the replacement over the dependency's whole supported range, and never silence it. Use when CI logs show a deprecation raised on a code path this repo owns.
---

# retire-a-deprecation — prepare for the removal, don't hide it

A deprecation warning is a dated failure. B50 is the worked example: every export in
CI's ONNX step warned that torch would remove `from_dynamic_axes_to_dynamic_shapes`.
The step had been a hard gate since B45, so the warning was a scheduled red build. No
test had seen it, because the suite's export helper ran under
`warnings.simplefilter("ignore")`.

Related: `harden-a-guard` (plant the old call back), `close-hygiene-row` (record it),
`pr-preflight`.

## Step 0 — Find who swallows it

A warning that shows in CI's log but in no test result is being swallowed. Look for
the swallower before writing anything:

```bash
grep -rnE 'simplefilter\("ignore"|filterwarnings|-W ignore' tests/<area> pyproject.toml conftest.py
```

Don't remove it as a side effect: it may hide unrelated noise. Write new tests that
record warnings themselves (Step 2).

## Step 1 — Read the deprecated path in the dependency's source

Find the exact condition that reaches the deprecated function. It is often narrower
than the warning suggests. B50's conversion ran only when `dynamo=True` and
`dynamic_shapes is None`, which is why passing both arguments silences it (Step 5).
Read the installed copy: `python -c "import <mod>; print(<mod>.__file__)"`.

## Step 2 — Reproduce, recording

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")  # inner filter wins over -W and pytest's filterwarnings
    run_the_real_path()
flagged = [str(w.message) for w in caught if MARKER in str(w.message)]
```

Put the markers in one shared module with a filter helper, as
`tests/support/onnx_export.py` does, so every suite matches the same strings.

## Step 3 — A premise test, then the guards

1. **Premise.** Call the deprecated API directly. Assert that at least one marker
   fires, and that every warning mentioning the API is matched by a marker. Don't
   assert an exact count: releases differ (torch 2.8 emits one of B50's two warnings,
   2.14 both). When the dependency rewords a message, the no-warning tests would pass
   on nothing; the premise fails instead and says what to update. See
   `test_premise_torch_flags_dynamic_axes_with_our_markers` in
   `tests/deployment/test_onnx_runtime_regressions.py`.
2. **No-warning tests.** Add one per distinct route through the code: a tensor model,
   a model behind an adapter, the real model. Each must fail on the old code (TDD red).

## Step 4 — Check the replacement over the whole supported range

The replacement must work from the dependency's **floor** in `pyproject.toml` to CI's
pin, not just on the version you have. Read the dependency's source at each release
tag:

```bash
gh api "repos/pytorch/pytorch/contents/<path>?ref=v2.6.0" --jq .content | base64 -d | grep -n "<symbol>"
```

B50 shipped without this and broke the floor. String axis names in `dynamic_shapes`
are converted only from torch 2.7; on 2.6, `torch.export` rejects them. Meanwhile the
deprecated route is reached only where `torch.onnx.export` defaults to the dynamo
exporter, which is 2.9 and later. So:

- **Change the call only where the deprecated path is actually reached.** Elsewhere,
  keep the historical call byte for byte; it never warned there.
- **Gate on a capability, not a version string.** Read it from the public API at import
  time, before any test monkeypatches the function, for example
  `inspect.signature(torch.onnx.export).parameters["dynamo"].default`. Put it in a
  module constant that tests patch to exercise both branches.

## Step 5 — Pin the full call

Record the backend call with a fake and assert the kwargs dict **equals** the expected
one. Two weaker defects are silent on the newest dependency, and only the pin catches
them: passing the old and the new argument together (the new one wins, so no warning),
and dropping a flag the new path needs on older versions. See
`TestExportDispatch::test_tensor_output_trace_call_is_pinned` in
`tests/deployment/test_export_adapters.py`.

## Step 6 — Never

- `-W ignore`, a `filterwarnings` entry, or a widened tolerance. These hide the removal
  instead of preparing for it.
- Pinning the old backend to dodge the new one. B48: the legacy ONNX exporter exports
  this model with outputs off by about 0.1.

## Step 7 — Prove and record

Plant the pre-fix call and both Step 5 variants with `harden-a-guard`'s runner, `python -m scripts.plant_mutations`.
Then close the row with `close-hygiene-row`, stating the versions the change was
measured on (Step 4's range is the claim; the versions you ran are the evidence).
