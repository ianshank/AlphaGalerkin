# AGENT.md - Third-Party Integrations (`src/integrations/`)

## Persona

**Name**: Integration Engineer
**Expertise**: HTTP clients, OpenAI-compatible APIs, retry/backoff, JSON-mode decoding, optional dependency gating, fail-loud preflight checks
**Mindset**: Treat every external service as eventually unreliable. Validate the contract at the boundary (Pydantic), fail loud on misuse, never silently fabricate output. Each integration ships behind its own optional extra so the base install is never weighed down.

## Module Overview

`src/integrations/` houses third-party-service adapters that AlphaGalerkin can opt into. Each subpackage maps one external surface (cloud LLM, local LLM, hosted dataset, etc.) onto an existing AlphaGalerkin protocol or component.

Currently shipped:

- **`lm_studio/`** — OpenAI-compatible local LLM client (LM Studio, llama.cpp server, vLLM, etc.) plus an `LMStudioEvaluator` that implements the MCTS `Evaluator` protocol (`src/mcts/evaluator.py::Evaluator`). Used by the `llm_prior_ablation` PoC scenario to drive MCTS basis selection with a generalist LLM prior. Optional extra: `[lm-studio]`.
- **`openai_compat/`** — backend-profile registry shared by every OpenAI-wire-compatible server. `LMStudioConfig.backend` selects a profile; `apply_backend_defaults` fills the endpoint/model/VRAM policy for fields the user left unset (explicit values always win). The concrete client/preflight/evaluator stay in `lm_studio/` — `openai_compat/` only carries the per-backend *configuration*.
- **`eval_harness/`** — adapter onto the external `langfuse-eval-harness` (`github.com/ianshank/Agents`) LLM-evaluation framework. Wraps the LLM-prior MCTS basis-selection layer as a harness `callable` target (`target.run_basis_cell`) + greedy-oracle dataset (`dataset.BasisOracleDataset`) + scorers (`FinalResidualScorer`, `PolicyTopKScorer`), traces via Langfuse, and bridges results back into the `src/poc/baselines` regression gate via `sink.ScenarioResultSink`. `register_all` (and the `_entrypoint` shim) register the adapters with the harness registries; everything imported there is torch-free. Optional extras: `[eval-harness]` (the harness, pinned to a commit SHA — not on PyPI) and `[eval-harness-langfuse]` (live Langfuse, pinned `langfuse>=2,<3`). CI runs `--offline` (`NullLangfuseClient`).

### Tested-server matrix

All three backends speak the OpenAI wire protocol and are served by the single `openai` SDK shipped by the `[lm-studio]` extra — switching is a `backend:` (and usually a `model:`) change, no code edit.

| Backend (`backend:`) | Default endpoint | Default `vram_check_mode` | OpenAI SDK | Status | Notes |
|---|---|---|---|---|---|
| `lm_studio` (default) | `http://127.0.0.1:1234/v1` | `local` | `openai>=1.40,<2.0` | tested (mocked CPU + manual GPU) | Reference backend; colocated single-box; historical default. |
| `vllm` | `http://127.0.0.1:8000/v1` | `off` | `openai>=1.40,<2.0` | tested (mocked CPU); GPU smoke pending | `vllm serve <hf-id>`; HF model ids; remote-by-default so the local VRAM floor is skipped. |
| `llama_cpp` | `http://127.0.0.1:8080/v1` | `off` | `openai>=1.40,<2.0` | tested (mocked CPU); GPU smoke pending | `./llama-server -m model.gguf`; GGUF; remote-by-default. |

To pin a backend to a colocated GPU and re-enable the free-VRAM floor, set `vram_check_mode: local` explicitly. To add a backend, call `register_backend(BackendProfile(...))` — no client/preflight/evaluator change.

## Conventions & Constraints

1. **Optional dependency gating.** Every subpackage's third-party SDK is declared under `[project.optional-dependencies]` in `pyproject.toml`. SDK imports happen inside class `__init__` or function bodies, never at module top level, so a missing dependency fails only when the integration is actually constructed.
2. **No hardcoded values.** Every URL, model name, timeout, retry count, backoff base, temperature, and threshold is a Pydantic field with a default and a docstring.
3. **GPU-required integrations fail loud.** Use `src/poc/device.py::resolve_device` for device resolution; never silently fall back to CPU when GPU was requested.
4. **Typed exception hierarchy.** Each integration raises typed exceptions (`<Integration>Error`, with parse/connection/preflight subclasses). Callers can catch precisely. The integration code never silently lies about results.
5. **Structured logging.** Every outbound call emits a `structlog` event including a request hash, latency, token counts when applicable, parse success flag, and retry count. No `print`.
6. **Mockable in CI.** Tests monkey-patch the third-party SDK at the boundary so CPU CI never imports the optional dependency. GPU-required end-to-end tests carry `@pytest.mark.gpu_required` and are auto-skipped on CPU runners via the root `conftest.py` hook.
7. **Preflight on construct (default on).** Each integration validates its dependency surface before accepting traffic (server reachable, model loaded, free VRAM sufficient, etc.).

## Adding a new integration

1. Create `src/integrations/<name>/` with `__init__.py`, `config.py`, `schema.py`, `client.py`, plus whatever else maps onto an AlphaGalerkin protocol (`evaluator.py`, `trainer.py`, ...).
2. Add the SDK to `pyproject.toml::[project.optional-dependencies]` under a new key (matching the subpackage name).
3. Add a Regression Surface row to `CLAUDE.md`.
4. Land tests under `tests/integrations/<name>/` with mocked-SDK CPU coverage; mark any real-server smoke test with `@pytest.mark.gpu_required` (or `@pytest.mark.integration` if no GPU is needed).
