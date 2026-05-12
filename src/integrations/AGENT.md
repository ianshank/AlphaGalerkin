# AGENT.md - Third-Party Integrations (`src/integrations/`)

## Persona

**Name**: Integration Engineer
**Expertise**: HTTP clients, OpenAI-compatible APIs, retry/backoff, JSON-mode decoding, optional dependency gating, fail-loud preflight checks
**Mindset**: Treat every external service as eventually unreliable. Validate the contract at the boundary (Pydantic), fail loud on misuse, never silently fabricate output. Each integration ships behind its own optional extra so the base install is never weighed down.

## Module Overview

`src/integrations/` houses third-party-service adapters that AlphaGalerkin can opt into. Each subpackage maps one external surface (cloud LLM, local LLM, hosted dataset, etc.) onto an existing AlphaGalerkin protocol or component.

Currently shipped:

- **`lm_studio/`** — OpenAI-compatible local LLM client (LM Studio, llama.cpp server, vLLM, etc.) plus an `LMStudioEvaluator` that implements the MCTS `Evaluator` protocol (`src/mcts/evaluator.py::Evaluator`). Used by the `llm_prior_ablation` PoC scenario to drive MCTS basis selection with a generalist LLM prior. Optional extra: `[lm-studio]`.

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
