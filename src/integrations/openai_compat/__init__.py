"""OpenAI-compatible LLM backend profiles (LM Studio, vLLM, llama.cpp-server).

This subpackage holds the backend-agnostic layer that lets the single
``openai``-SDK client serve every OpenAI-wire-compatible local server. The
concrete client / preflight / evaluator still live in
``src.integrations.lm_studio`` (that package is the reference backend and the
back-compat surface); here we only carry the per-backend *configuration*
profiles and the helper that applies them.

Public surface:
    - ``BackendProfile``: per-backend default endpoint/model/VRAM policy.
    - ``BACKEND_REGISTRY`` / ``register_backend`` / ``get_backend`` /
      ``list_backends``: registry of known backends.
    - ``apply_backend_defaults``: fill unset config fields from the selected
      backend's profile (explicit user values always win).
"""

from __future__ import annotations

from src.integrations.openai_compat.registry import (
    BACKEND_REGISTRY,
    BackendProfile,
    apply_backend_defaults,
    get_backend,
    list_backends,
    register_backend,
)

__all__ = [
    "BACKEND_REGISTRY",
    "BackendProfile",
    "apply_backend_defaults",
    "get_backend",
    "list_backends",
    "register_backend",
]
