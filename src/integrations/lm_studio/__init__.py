"""LM Studio (and any OpenAI-compatible local LLM server) integration.

Provides an MCTS-compatible policy prior backed by a locally-served LLM such
as Qwen-14B running in LM Studio. The integration is opt-in via the
``[lm-studio]`` optional extra in ``pyproject.toml``.

Public surface:
    - ``LMStudioConfig``: Pydantic configuration with no hardcoded values.
    - ``LMStudioClient``: synchronous client wrapping the ``openai`` SDK
      against the LM Studio endpoint, with bounded retries and structured
      logging.
    - ``LMStudioEvaluator``: implements ``src/mcts/evaluator.py::Evaluator``.
    - ``LMStudioPolicyResponse``: Pydantic response model.
    - ``PreflightReport`` / ``check_lm_studio_server``: dependency surface
      validation (server reachable + model present + free VRAM sufficient).
    - Typed exception hierarchy: ``LMStudioError`` and subclasses.
"""

from __future__ import annotations

from src.integrations.lm_studio.client import LMStudioClient
from src.integrations.lm_studio.config import LMStudioConfig
from src.integrations.lm_studio.evaluator import LMStudioEvaluator
from src.integrations.lm_studio.preflight import PreflightReport, check_lm_studio_server
from src.integrations.lm_studio.prompt import build_policy_prompt, prompt_hash
from src.integrations.lm_studio.schema import (
    LMStudioActionSpaceMismatchError,
    LMStudioConnectionError,
    LMStudioError,
    LMStudioParseError,
    LMStudioPolicyResponse,
    LMStudioPreflightError,
)

__all__ = [
    "LMStudioActionSpaceMismatchError",
    "LMStudioClient",
    "LMStudioConfig",
    "LMStudioConnectionError",
    "LMStudioError",
    "LMStudioEvaluator",
    "LMStudioParseError",
    "LMStudioPolicyResponse",
    "LMStudioPreflightError",
    "PreflightReport",
    "build_policy_prompt",
    "check_lm_studio_server",
    "prompt_hash",
]
