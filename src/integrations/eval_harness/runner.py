"""Programmatic entry point: load a harness ``EvalConfig`` YAML and run it.

``run_eval`` registers the AlphaGalerkin adapters, loads the config (with env-var
interpolation), selects the Langfuse client (the dependency-free
``NullLangfuseClient`` when ``offline``, else the real ``SDKLangfuseClient`` which
reads ``LANGFUSE_*`` from the environment), and executes the engine.

``eval_harness`` is imported lazily so this module is importable on a base install.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from eval_harness.core.types import RunResult


def run_eval(
    config_path: str,
    *,
    offline: bool = True,
    env: Mapping[str, str] | None = None,
    overrides: Iterable[str] | None = None,
) -> RunResult:
    """Load and execute a harness ``EvalConfig`` YAML.

    Args:
        config_path: Path to a harness ``EvalConfig`` YAML.
        offline: When ``True`` (default) use the in-memory ``NullLangfuseClient``
            (no network, no keys). When ``False`` build the real
            ``SDKLangfuseClient`` (requires ``langfuse`` + ``LANGFUSE_*`` env vars).
        env: Environment mapping for ``${VAR:-default}`` interpolation; defaults
            to ``os.environ``.
        overrides: Optional ``key.path=value`` config overrides.

    Returns:
        The harness ``RunResult``.

    """
    from eval_harness.config import load_config  # noqa: PLC0415
    from eval_harness.engine import EvalEngine  # noqa: PLC0415
    from eval_harness.langfuse_client import NullLangfuseClient  # noqa: PLC0415

    from src.integrations.eval_harness.plugins import register_all  # noqa: PLC0415

    register_all()
    eval_config = load_config(
        config_path,
        overrides=list(overrides) if overrides is not None else None,
        env=dict(env) if env is not None else os.environ,
    )
    client: Any
    if offline:
        client = NullLangfuseClient()
    else:  # pragma: no cover - live Langfuse needs LANGFUSE_* keys (integration only)
        from eval_harness.langfuse_client import SDKLangfuseClient  # noqa: PLC0415

        client = SDKLangfuseClient()
    engine = EvalEngine.from_config(eval_config, langfuse_client=client)
    return engine.run()


__all__ = ["run_eval"]
