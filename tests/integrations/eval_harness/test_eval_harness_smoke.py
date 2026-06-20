"""Manual end-to-end smoke against a live LM Studio + Langfuse (GPU/integration).

Auto-skipped on CPU CI via the root ``conftest.py`` ``gpu_required`` hook, and
additionally skipped unless ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` /
``LM_STUDIO_URL`` are set (mirrors the LM Studio smoke gating). Exercises the
real ``SDKLangfuseClient`` online path end to end.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("eval_harness")

pytestmark = [pytest.mark.gpu_required, pytest.mark.integration]

_REQUIRED_ENV = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LM_STUDIO_URL")


def _env_ready() -> bool:
    return all(os.environ.get(key) for key in _REQUIRED_ENV)


@pytest.mark.skipif(not _env_ready(), reason="requires LANGFUSE_* + LM_STUDIO_URL")
def test_online_eval_smoke(tmp_path: Path) -> None:
    config = {
        "schema_version": "1.0",
        "run": {"name": "basis_eval_smoke", "seed": 0},
        "dataset": {
            "type": "basis_oracle",
            "params": {
                "pde_families": ["poisson"],
                "seeds": [0],
                "arm": "llm",
                "max_basis_functions": 3,
                "n_candidate_bases": 8,
                "target_residual": 1e-2,
                "max_rollouts": 32,
                "n_simulations": 8,
                "topk": 3,
                "lm_studio": {"base_url": os.environ["LM_STUDIO_URL"]},
            },
        },
        "target": {
            "type": "callable",
            "params": {"function": "src.integrations.eval_harness.target:run_basis_cell"},
        },
        "scorers": [
            {"type": "final_residual", "params": {"target_residual": 1e-2}},
            {"type": "policy_topk", "params": {"k": 3}},
        ],
        "sinks": [
            {"type": "scenario_result", "params": {"output_dir": str(tmp_path)}},
            {"type": "langfuse", "params": {}},
        ],
        "gate": {"rules": [{"score": "final_residual", "metric": "mean", "max": 1.0}]},
    }
    config_path = tmp_path / "smoke.json"
    config_path.write_text(json.dumps(config))

    from src.integrations.eval_harness.runner import run_eval

    # YAML loader also reads JSON (yaml.safe_load); online client exercises Langfuse.
    result = run_eval(str(config_path), offline=False)
    assert "final_residual" in result.aggregate
    assert (tmp_path / "results" / result.run_id).is_dir()
