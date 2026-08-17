"""Third-party service integrations for AlphaGalerkin.

Each subpackage is an optional surface gated behind a Pydantic config and a
dedicated ``[project.optional-dependencies]`` extra in ``pyproject.toml``.
The base install stays slim; integrations are opt-in.

Available subpackages:
    - ``lm_studio``: OpenAI-compatible local LLM (e.g. Qwen-14B in LM Studio)
      used as an MCTS policy prior. Gated behind ``[lm-studio]`` extra.
    - ``openai_compat``: per-backend configuration profiles shared by every
      OpenAI-wire-compatible server (LM Studio, vLLM, llama.cpp).
    - ``eval_harness``: adapter onto the external ``langfuse-eval-harness``
      (github.com/ianshank/Agents) for tracing + labelled scoring of the
      LLM-prior MCTS layer. Gated behind the ``[eval-harness]`` extra.
"""
