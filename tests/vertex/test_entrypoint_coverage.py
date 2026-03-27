"""Coverage tests for Vertex AI training entrypoint.

Targets uncovered lines in src/vertex/entrypoint.py:
    - parse_args
    - setup_logging
    - init_distributed (single node path)
    - load_training_config
    - create_vertex_config_from_env
    - GracefulShutdownHandler
    - run_training with mocked trainer
    - init_wandb_for_vertex
"""

from __future__ import annotations

import signal
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.vertex.entrypoint import (
    GracefulShutdownHandler,
    init_distributed,
    init_wandb_for_vertex,
    load_training_config,
    run_training,
    setup_logging,
)
from src.vertex.multi_node import DistributedContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_node_ctx() -> DistributedContext:
    return DistributedContext(
        world_size=1,
        rank=0,
        local_rank=0,
        master_addr="127.0.0.1",
        master_port=29500,
    )


def _multi_node_ctx() -> DistributedContext:
    return DistributedContext(
        world_size=2,
        rank=0,
        local_rank=0,
        master_addr="127.0.0.1",
        master_port=29500,
        num_nodes=2,
    )


# ---------------------------------------------------------------------------
# Tests: setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_info_level(self) -> None:
        setup_logging(debug=False)

    def test_debug_level(self) -> None:
        setup_logging(debug=True)


# ---------------------------------------------------------------------------
# Tests: init_distributed
# ---------------------------------------------------------------------------


class TestInitDistributed:
    def test_single_node_skips(self) -> None:
        ctx = _single_node_ctx()
        # Should not call dist.init_process_group
        init_distributed(ctx, backend="gloo")

    @patch("src.vertex.entrypoint.dist")
    def test_multi_node_calls_init(self, mock_dist: MagicMock) -> None:
        ctx = _multi_node_ctx()
        init_distributed(ctx, backend="gloo")
        mock_dist.init_process_group.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: load_training_config
# ---------------------------------------------------------------------------


class TestLoadTrainingConfig:
    def test_load_yaml(self, tmp_path: Path) -> None:
        config = {"training": {"batch_size": 32, "total_steps": 100}}
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(config, f)

        result = load_training_config(str(cfg_path))
        assert result["training"]["batch_size"] == 32


# ---------------------------------------------------------------------------
# Tests: GracefulShutdownHandler
# ---------------------------------------------------------------------------


class TestGracefulShutdownHandler:
    def test_init_no_callback(self) -> None:
        handler = GracefulShutdownHandler()
        assert handler.should_shutdown is False

    def test_shutdown_flag(self) -> None:
        callback = MagicMock()
        handler = GracefulShutdownHandler(checkpoint_callback=callback)

        # Simulate signal
        handler._handle_signal(signal.SIGTERM, None)

        assert handler.should_shutdown is True
        callback.assert_called_once()

    def test_callback_exception_handled(self) -> None:
        callback = MagicMock(side_effect=RuntimeError("save failed"))
        handler = GracefulShutdownHandler(checkpoint_callback=callback)

        # Should not raise
        handler._handle_signal(signal.SIGINT, None)
        assert handler.should_shutdown is True


# ---------------------------------------------------------------------------
# Tests: run_training
# ---------------------------------------------------------------------------


class TestRunTraining:
    @patch("src.vertex.entrypoint.VertexTrainer")
    def test_run_training_success(self, mock_trainer_cls: MagicMock) -> None:
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "step": 100,
            "loss": 0.01,
            "metrics": {"accuracy": 0.95},
        }
        mock_trainer_cls.return_value = mock_trainer

        from src.vertex.config import VertexStorageConfig, VertexTrainingConfig

        vertex_config = VertexTrainingConfig(
            project_id="test-project",
            staging_bucket="gs://test-bucket",
            storage=VertexStorageConfig(bucket_name="test-bucket"),
        )
        ctx = _single_node_ctx()
        checkpoint_manager = MagicMock()

        result = run_training(
            config={"training": {"total_steps": 100}},
            vertex_config=vertex_config,
            ctx=ctx,
            checkpoint_manager=checkpoint_manager,
        )

        assert result["status"] == "completed"

    @patch("src.vertex.entrypoint.VertexTrainer")
    def test_run_training_with_resume(self, mock_trainer_cls: MagicMock) -> None:
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"step": 50, "loss": 0.02}
        mock_trainer_cls.return_value = mock_trainer

        from src.vertex.config import VertexStorageConfig, VertexTrainingConfig

        vertex_config = VertexTrainingConfig(
            project_id="test-project",
            staging_bucket="gs://test-bucket",
            storage=VertexStorageConfig(bucket_name="test-bucket"),
        )
        ctx = _single_node_ctx()
        cm = MagicMock()

        result = run_training(
            config={},
            vertex_config=vertex_config,
            ctx=ctx,
            checkpoint_manager=cm,
            resume_path="gs://bucket/ckpt.pt",
        )

        mock_trainer.load_checkpoint.assert_called_once()

    @patch("src.vertex.entrypoint.VertexTrainer")
    def test_trainer_ref_set(self, mock_trainer_cls: MagicMock) -> None:
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"step": 1}
        mock_trainer_cls.return_value = mock_trainer

        from src.vertex.config import VertexStorageConfig, VertexTrainingConfig

        vertex_config = VertexTrainingConfig(
            project_id="test-project",
            staging_bucket="gs://test-bucket",
            storage=VertexStorageConfig(bucket_name="test-bucket"),
        )
        ctx = _single_node_ctx()
        ref: dict[str, Any] = {}

        run_training(
            config={},
            vertex_config=vertex_config,
            ctx=ctx,
            checkpoint_manager=MagicMock(),
            trainer_ref=ref,
        )

        assert "trainer" in ref


# ---------------------------------------------------------------------------
# Tests: init_wandb_for_vertex
# ---------------------------------------------------------------------------


class TestInitWandB:
    def test_non_main_process_returns_none(self) -> None:
        ctx = DistributedContext(
            world_size=2, rank=1, local_rank=1,
            master_addr="127.0.0.1", master_port=29500,
        )
        result = init_wandb_for_vertex({}, ctx)
        assert result is None

    @patch.dict("os.environ", {"WANDB_MODE": "disabled"}, clear=False)
    def test_disabled_mode_returns_none(self) -> None:
        ctx = _single_node_ctx()
        result = init_wandb_for_vertex({}, ctx)
        assert result is None

    @patch.dict("os.environ", {}, clear=False)
    def test_no_api_key_returns_none(self) -> None:
        import os
        # Ensure no WANDB_API_KEY
        os.environ.pop("WANDB_API_KEY", None)
        ctx = _single_node_ctx()
        result = init_wandb_for_vertex({}, ctx)
        assert result is None
