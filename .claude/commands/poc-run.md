---
description: Run an AlphaGalerkin PoC scenario from a config file (CPU-safe by default).
argument-hint: <config-path or scenario-name>
---

Run a PoC scenario. If `$ARGUMENTS` looks like a path, use `--config`; otherwise treat it as a
scenario name for `--scenario`.

```bash
python -m src.poc.cli run --config "$ARGUMENTS"    # when $ARGUMENTS is a YAML path
# or
python -m src.poc.cli run --scenario "$ARGUMENTS"  # when $ARGUMENTS is a registered name
```

Prefer the CPU-safe demo configs (e.g. `config/scenarios/*_cpu.yaml`). GPU/LLM arms auto-gate and
skip when CUDA / `LM_STUDIO_URL` are unavailable. Report the scenario status and any threshold
pass/fail from the result.
