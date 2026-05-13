"""Third-party service integrations for AlphaGalerkin.

Each subpackage is an optional surface gated behind a Pydantic config and a
dedicated ``[project.optional-dependencies]`` extra in ``pyproject.toml``.
The base install stays slim; integrations are opt-in.

Available subpackages:
    - ``lm_studio``: OpenAI-compatible local LLM (e.g. Qwen-14B in LM Studio)
      used as an MCTS policy prior. Gated behind ``[lm-studio]`` extra.
"""
