---
description: Run the AlphaGalerkin centaur research-loop harness over a problem manifest.
argument-hint: <research-loop config yaml> [output-dir]
---

Run the research-loop orchestrator over a problem manifest, persisting the discovery ledger.

```bash
python -m src.agents.cli research --config "$ARGUMENTS" --output-dir outputs/agents/research
```

Use a CPU-safe manifest (e.g. `config/agents/research_loop_cpu.yaml`) unless CUDA + an LLM backend
are available. The result JSON lands under `outputs/agents/research/<run_id>/result.json` and is
diffable via `python -m src.poc.cli diff`.
