"""Backend-profile registry for OpenAI-compatible local LLM servers.

The :class:`~src.integrations.lm_studio.client.LMStudioClient` speaks the
OpenAI wire protocol, which LM Studio, vLLM, and llama.cpp-server all expose.
The only thing that differs between them is *configuration*: the default
endpoint port, the canonical model identifier, and whether a local free-VRAM
floor is meaningful (it is not for a remote server).

This module makes that difference data-driven and discoverable:

- :class:`BackendProfile` — the per-backend default bundle (no hardcoded
  values leak into client code; they live here as named fields).
- :data:`BACKEND_REGISTRY` + :func:`register_backend` / :func:`get_backend` —
  a small registry mirroring the project's other ``@register_*`` patterns.
- :func:`apply_backend_defaults` — fills the fields a user *left unset* on an
  ``LMStudioConfig`` from the selected backend's profile. Anything the user
  set explicitly (tracked via Pydantic's ``model_fields_set``) always wins, so
  the transform is a no-op for fully-specified configs and for the historical
  ``lm_studio`` default.

Adding a new OpenAI-compatible backend is one :func:`register_backend` call;
no client, preflight, or evaluator code changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

# ``VramCheckMode`` is needed at runtime because it types a Pydantic *field*
# below; importing the canonical alias here keeps a single source of truth.
# ``lm_studio.config`` has no dependency on this module, so there is no cycle.
from src.integrations.lm_studio.config import VramCheckMode

if TYPE_CHECKING:
    from src.integrations.lm_studio.config import LMStudioConfig

# Fields that :func:`apply_backend_defaults` may fill from a profile. Each is
# a config field whose sensible value depends on the backend rather than on
# the user's intent.
_PROFILE_FILLED_FIELDS: tuple[str, ...] = ("base_url", "model", "vram_check_mode")


class BackendProfile(BaseModel):
    """Default endpoint/model/VRAM policy for one OpenAI-compatible backend.

    Every value is an explicit, documented field — these are the canonical
    defaults that would otherwise be magic numbers/strings in client code.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    name: str = Field(..., description="Backend identifier (matches LLMBackend).", min_length=1)
    default_base_url: str = Field(
        ...,
        description="Canonical OpenAI-compatible endpoint for this backend.",
        min_length=1,
    )
    default_model: str = Field(
        ...,
        description="A representative served model id for this backend.",
        min_length=1,
    )
    default_vram_check_mode: VramCheckMode = Field(
        ...,
        description=(
            "Whether the local free-VRAM floor is probed by default. 'local' "
            "for a colocated single-box server, 'off' for a backend that is "
            "typically reached over the network."
        ),
    )
    openai_extra: str = Field(
        default="lm-studio",
        description=(
            "pip extra that installs this backend's client dependency. All "
            "three OpenAI-compatible backends share the single 'openai' SDK "
            "shipped by the [lm-studio] extra."
        ),
    )
    description: str = Field(default="", description="Human-readable backend note.")


# Built-in profiles. The ``lm_studio`` profile's values are exactly the
# historical ``LMStudioConfig`` defaults, so selecting it (the default) and
# applying defaults is a guaranteed no-op — preserving backwards compatibility.
_BUILTIN_PROFILES: tuple[BackendProfile, ...] = (
    BackendProfile(
        name="lm_studio",
        default_base_url="http://127.0.0.1:1234/v1",
        default_model="qwen2.5-14b-instruct",
        default_vram_check_mode="local",
        description="LM Studio desktop server (default; colocated single-box).",
    ),
    BackendProfile(
        name="vllm",
        default_base_url="http://127.0.0.1:8000/v1",
        default_model="Qwen/Qwen2.5-14B-Instruct",
        default_vram_check_mode="off",
        description="vLLM OpenAI-compatible server (vllm serve ...; HF model ids).",
    ),
    BackendProfile(
        name="llama_cpp",
        default_base_url="http://127.0.0.1:8080/v1",
        default_model="qwen2.5-14b-instruct",
        default_vram_check_mode="off",
        description="llama.cpp server (./llama-server; GGUF, OpenAI-compatible).",
    ),
)

BACKEND_REGISTRY: dict[str, BackendProfile] = {}


def register_backend(profile: BackendProfile) -> BackendProfile:
    """Register a backend profile, returning it for convenient chaining.

    Raises:
        ValueError: A profile with the same ``name`` is already registered.

    """
    if profile.name in BACKEND_REGISTRY:
        raise ValueError(f"backend {profile.name!r} is already registered")
    BACKEND_REGISTRY[profile.name] = profile
    return profile


def get_backend(name: str) -> BackendProfile:
    """Look up a registered backend profile by name.

    Raises:
        KeyError: ``name`` is not registered (message lists known backends).

    """
    try:
        return BACKEND_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(BACKEND_REGISTRY)) or "<none>"
        raise KeyError(f"unknown LLM backend {name!r}; registered backends: {known}") from exc


def list_backends() -> list[str]:
    """Return the sorted names of every registered backend."""
    return sorted(BACKEND_REGISTRY)


def apply_backend_defaults(config: LMStudioConfig) -> LMStudioConfig:
    """Return a copy of ``config`` with backend defaults filled for unset fields.

    Only fields the user did *not* explicitly set (per Pydantic's
    ``model_fields_set``) are overwritten from the selected backend's profile.
    For the default ``lm_studio`` backend the profile equals the historical
    field defaults, so this is a no-op; for a user who sets ``backend: vllm``
    and nothing else, ``base_url`` / ``model`` / ``vram_check_mode`` become the
    vLLM canonical values.

    The returned config has ``preflight_on_construct`` untouched and is safe to
    hand to preflight / the client.
    """
    profile = get_backend(config.backend)
    explicit = config.model_fields_set
    updates: dict[str, object] = {}
    if "base_url" not in explicit:
        updates["base_url"] = profile.default_base_url
    if "model" not in explicit:
        updates["model"] = profile.default_model
    if "vram_check_mode" not in explicit:
        updates["vram_check_mode"] = profile.default_vram_check_mode
    if not updates:
        return config
    return config.model_copy(update=updates)


def _register_builtins() -> None:
    """Idempotently register the built-in backend profiles."""
    for profile in _BUILTIN_PROFILES:
        if profile.name not in BACKEND_REGISTRY:
            BACKEND_REGISTRY[profile.name] = profile


_register_builtins()
