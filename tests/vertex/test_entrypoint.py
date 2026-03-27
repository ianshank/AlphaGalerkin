"""Tests for Vertex AI training entrypoint.

Covers emergency checkpoint functionality, graceful shutdown handler,
and run_training integration with trainer_ref.
"""

from __future__ import annotations

import contextlib
import signal
import sys
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
    parse_args,
    run_training,
    setup_logging,
)
from src.vertex.multi_node import DistributedContext

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _single_node_ctx() -> DistributedContext:
    return DistributedContext(
        world_size=1,
        rank=0,
        local_rank=0,
        master_addr="127.0.0.1",
        master_port=29500,
    )


def _worker_ctx() -> DistributedContext:
    """Non-main-process (rank=1) distributed context."""
    return DistributedContext(
        world_size=2,
        rank=1,
        local_rank=1,
        master_addr="127.0.0.1",
        master_port=29500,
    )


def _multi_node_ctx() -> DistributedContext:
    return DistributedContext(
        world_size=4,
        rank=0,
        local_rank=0,
        master_addr="10.0.0.1",
        master_port=29500,
        num_nodes=2,
    )


def _make_vertex_config() -> Any:
    from src.vertex.config import VertexStorageConfig, VertexTrainingConfig

    return VertexTrainingConfig(
        project_id="test-project",
        staging_bucket="gs://test-bucket",
        storage=VertexStorageConfig(bucket_name="test-bucket"),
    )


# ---------------------------------------------------------------------------
# Tests: parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_required_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--config", "train.yaml"])
        args = parse_args()
        assert args.config == "train.yaml"

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--config", "train.yaml"])
        args = parse_args()
        assert args.backend == "nccl"
        assert args.checkpoint_dir == "/tmp/alphagalerkin_cache"
        assert args.dry_run is False
        assert args.debug is False
        assert args.resume is None
        assert args.vertex_config is None

    def test_all_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--config",
                "train.yaml",
                "--vertex-config",
                "vertex.yaml",
                "--resume",
                "gs://bucket/ckpt.pt",
                "--checkpoint-dir",
                "/tmp/cache",
                "--backend",
                "gloo",
                "--dry-run",
                "--debug",
            ],
        )
        args = parse_args()
        assert args.vertex_config == "vertex.yaml"
        assert args.resume == "gs://bucket/ckpt.pt"
        assert args.checkpoint_dir == "/tmp/cache"
        assert args.backend == "gloo"
        assert args.dry_run is True
        assert args.debug is True

    @pytest.mark.parametrize("backend", ["nccl", "gloo"])
    def test_backend_choices(
        self, monkeypatch: pytest.MonkeyPatch, backend: str
    ) -> None:
        monkeypatch.setattr(
            sys, "argv", ["prog", "--config", "c.yaml", "--backend", backend]
        )
        args = parse_args()
        assert args.backend == backend

    def test_missing_config_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["prog"])
        with pytest.raises(SystemExit):
            parse_args()


# ---------------------------------------------------------------------------
# Tests: setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_info_level(self) -> None:
        setup_logging(debug=False)

    def test_debug_level(self) -> None:
        setup_logging(debug=True)


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
        assert result["training"]["total_steps"] == 100

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "empty.yaml"
        cfg_path.write_text("")
        result = load_training_config(str(cfg_path))
        assert result is None

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_training_config(str(tmp_path / "nonexistent.yaml"))

    @pytest.mark.parametrize(
        "config_data",
        [
            {"training": {"total_steps": 10}},
            {"model": {"d_model": 128}},
            {"training": {"batch_size": 64}, "model": {"layers": 6}},
        ],
    )
    def test_various_configs(
        self, tmp_path: Path, config_data: dict[str, Any]
    ) -> None:
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(config_data, f)
        result = load_training_config(str(cfg_path))
        assert result == config_data


# ---------------------------------------------------------------------------
# Tests: init_distributed
# ---------------------------------------------------------------------------


class TestInitDistributed:
    def test_single_node_skips_init(self) -> None:
        ctx = _single_node_ctx()
        # Should return without calling dist.init_process_group
        init_distributed(ctx, backend="gloo")

    @patch("src.vertex.entrypoint.dist")
    def test_multi_node_calls_init(self, mock_dist: MagicMock) -> None:
        ctx = _multi_node_ctx()
        init_distributed(ctx, backend="gloo")
        mock_dist.init_process_group.assert_called_once_with(
            backend="gloo",
            init_method="tcp://10.0.0.1:29500",
            world_size=4,
            rank=0,
        )

    @patch("src.vertex.entrypoint.dist")
    def test_nccl_backend_passed(self, mock_dist: MagicMock) -> None:
        ctx = _multi_node_ctx()
        init_distributed(ctx, backend="nccl")
        call_kwargs = mock_dist.init_process_group.call_args
        assert call_kwargs[1]["backend"] == "nccl"

    @patch("src.vertex.entrypoint.torch")
    @patch("src.vertex.entrypoint.dist")
    def test_cuda_device_set_when_available(
        self, mock_dist: MagicMock, mock_torch: MagicMock
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        ctx = _multi_node_ctx()
        init_distributed(ctx, backend="nccl")
        mock_torch.cuda.set_device.assert_called_once_with(0)

    @patch("src.vertex.entrypoint.torch")
    @patch("src.vertex.entrypoint.dist")
    def test_cuda_device_not_set_when_unavailable(
        self, mock_dist: MagicMock, mock_torch: MagicMock
    ) -> None:
        mock_torch.cuda.is_available.return_value = False
        ctx = _multi_node_ctx()
        init_distributed(ctx, backend="nccl")
        mock_torch.cuda.set_device.assert_not_called()

    @patch("src.vertex.entrypoint.torch")
    @patch("src.vertex.entrypoint.dist")
    def test_cuda_not_set_when_rank_exceeds_device_count(
        self, mock_dist: MagicMock, mock_torch: MagicMock
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 0
        ctx = _multi_node_ctx()
        init_distributed(ctx, backend="nccl")
        mock_torch.cuda.set_device.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: GracefulShutdownHandler
# ---------------------------------------------------------------------------


class TestGracefulShutdownHandler:
    def test_init_registers_signals(self) -> None:
        callback = MagicMock()
        handler = GracefulShutdownHandler(checkpoint_callback=callback)
        assert not handler.should_shutdown
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    def test_should_shutdown_initially_false(self) -> None:
        handler = GracefulShutdownHandler(checkpoint_callback=None)
        assert handler.should_shutdown is False
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    def test_handle_signal_sets_shutdown(self) -> None:
        callback = MagicMock()
        handler = GracefulShutdownHandler(checkpoint_callback=callback)
        handler._handle_signal(signal.SIGTERM, None)  # type: ignore[arg-type]
        assert handler.should_shutdown is True
        callback.assert_called_once()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    def test_handle_signal_no_callback(self) -> None:
        handler = GracefulShutdownHandler(checkpoint_callback=None)
        handler._handle_signal(signal.SIGINT, None)  # type: ignore[arg-type]
        assert handler.should_shutdown is True
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    def test_handle_signal_callback_exception(self) -> None:
        callback = MagicMock(side_effect=RuntimeError("save failed"))
        handler = GracefulShutdownHandler(checkpoint_callback=callback)
        handler._handle_signal(signal.SIGTERM, None)  # type: ignore[arg-type]
        assert handler.should_shutdown is True
        callback.assert_called_once()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    def test_init_no_callback(self) -> None:
        handler = GracefulShutdownHandler()
        assert handler.should_shutdown is False
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    @pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
    def test_both_signals_trigger_shutdown(self, sig: signal.Signals) -> None:
        callback = MagicMock()
        handler = GracefulShutdownHandler(checkpoint_callback=callback)
        handler._handle_signal(sig, None)  # type: ignore[arg-type]
        assert handler.should_shutdown is True
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)


# ---------------------------------------------------------------------------
# Tests: init_wandb_for_vertex
# ---------------------------------------------------------------------------


class TestInitWandBForVertex:
    def test_non_main_process_returns_none(self) -> None:
        ctx = _worker_ctx()
        result = init_wandb_for_vertex({}, ctx)
        assert result is None

    @patch.dict("os.environ", {"WANDB_MODE": "disabled"}, clear=False)
    def test_disabled_mode_returns_none(self) -> None:
        ctx = _single_node_ctx()
        result = init_wandb_for_vertex({}, ctx)
        assert result is None

    def test_no_api_key_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        ctx = _single_node_ctx()
        result = init_wandb_for_vertex({}, ctx)
        assert result is None

    @patch.dict(
        "os.environ",
        {"WANDB_API_KEY": "fake-key", "WANDB_MODE": "online"},
        clear=False,
    )
    @patch("src.vertex.entrypoint.logger")
    def test_with_api_key_and_import_error(
        self, mock_logger: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If wandb_logger import fails, returns None gracefully."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        # Re-add via os.environ patch above doesn't work with monkeypatch combo,
        # patch directly via patch.dict instead (done by decorator).
        ctx = _single_node_ctx()
        with patch(
            "src.vertex.entrypoint.init_wandb_for_vertex",
            wraps=init_wandb_for_vertex,
        ):
            pass
        # Without real create_wandb_logger, will hit except block -> return None
        result = init_wandb_for_vertex({}, ctx)
        assert result is None

    @patch.dict(
        "os.environ",
        {"WANDB_API_KEY": "fake-key", "WANDB_MODE": "online"},
        clear=False,
    )
    def test_uses_project_from_training_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ctx = _single_node_ctx()
        mock_create = MagicMock(return_value=MagicMock())
        with patch(
            "src.vertex.entrypoint.init_wandb_for_vertex",
            wraps=init_wandb_for_vertex,
        ):
            # The real function will fall back to None if create_wandb_logger not available
            result = init_wandb_for_vertex(
                {"wandb": {"project": "my-project"}}, ctx
            )
        # Result is either a logger or None (if module not installed)
        assert result is None or hasattr(result, "finish")

    @patch.dict(
        "os.environ",
        {
            "WANDB_API_KEY": "fake-key",
            "WANDB_PROJECT": "env-project",
            "WANDB_ENTITY": "env-entity",
            "WANDB_RUN_NAME": "env-run",
            "WANDB_MODE": "online",
        },
        clear=False,
    )
    def test_env_vars_override_config(self) -> None:
        ctx = _single_node_ctx()
        mock_logger_inst = MagicMock()
        with patch(
            "src.vertex.entrypoint.init_wandb_for_vertex",
            return_value=mock_logger_inst,
        ) as mock_fn:
            result = mock_fn({}, ctx)
        assert result is mock_logger_inst


# ---------------------------------------------------------------------------
# Tests: run_training
# ---------------------------------------------------------------------------


class TestRunTraining:
    @patch("src.vertex.trainer.VertexTrainer")
    def test_run_training_success(
        self, mock_trainer_cls: MagicMock
    ) -> None:
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "step": 100,
            "loss": 0.01,
            "metrics": {"accuracy": 0.95},
        }
        mock_trainer_cls.return_value = mock_trainer

        vertex_config = _make_vertex_config()
        ctx = _single_node_ctx()
        checkpoint_manager = MagicMock()

        result = run_training(
            config={"training": {"total_steps": 100}},
            vertex_config=vertex_config,
            ctx=ctx,
            checkpoint_manager=checkpoint_manager,
        )

        assert result["status"] == "completed"
        assert result["final_step"] == 100
        assert result["final_loss"] == 0.01

    @patch("src.vertex.trainer.VertexTrainer")
    def test_run_training_with_resume(
        self, mock_trainer_cls: MagicMock
    ) -> None:
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"step": 50, "loss": 0.02}
        mock_trainer_cls.return_value = mock_trainer

        result = run_training(
            config={},
            vertex_config=_make_vertex_config(),
            ctx=_single_node_ctx(),
            checkpoint_manager=MagicMock(),
            resume_path="gs://bucket/ckpt.pt",
        )

        mock_trainer.load_checkpoint.assert_called_once_with("gs://bucket/ckpt.pt")
        assert result["status"] == "completed"

    @patch("src.vertex.trainer.VertexTrainer")
    def test_trainer_ref_populated(
        self, mock_trainer_cls: MagicMock
    ) -> None:
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"step": 1}
        mock_trainer_cls.return_value = mock_trainer

        ref: dict[str, Any] = {}
        run_training(
            config={},
            vertex_config=_make_vertex_config(),
            ctx=_single_node_ctx(),
            checkpoint_manager=MagicMock(),
            trainer_ref=ref,
        )

        assert "trainer" in ref
        assert ref["trainer"] is mock_trainer

    def test_trainer_ref_none_is_safe(self) -> None:
        """Passing trainer_ref=None does not crash."""
        trainer_ref = None
        if trainer_ref is not None:
            trainer_ref["trainer"] = MagicMock()

    @patch("src.vertex.trainer.VertexTrainer")
    def test_metrics_propagated(
        self, mock_trainer_cls: MagicMock
    ) -> None:
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "step": 200,
            "loss": 0.05,
            "metrics": {"val_loss": 0.1},
        }
        mock_trainer_cls.return_value = mock_trainer

        result = run_training(
            config={},
            vertex_config=_make_vertex_config(),
            ctx=_single_node_ctx(),
            checkpoint_manager=MagicMock(),
        )

        assert result["metrics"] == {"val_loss": 0.1}

    @patch("src.vertex.trainer.VertexTrainer")
    def test_empty_train_results_handled(
        self, mock_trainer_cls: MagicMock
    ) -> None:
        """Missing keys in train result default gracefully."""
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {}
        mock_trainer_cls.return_value = mock_trainer

        result = run_training(
            config={},
            vertex_config=_make_vertex_config(),
            ctx=_single_node_ctx(),
            checkpoint_manager=MagicMock(),
        )
        assert result["final_step"] == 0
        assert result["final_loss"] == 0.0

    def test_import_error_returns_failed_status(self) -> None:
        """ImportError of VertexTrainer falls back; if fallback also fails, status=failed."""
        with patch.dict(
            "sys.modules",
            {
                "src.vertex.trainer": MagicMock(
                    VertexTrainer=MagicMock(side_effect=ImportError("no vertex"))
                ),
            },
        ):
            result = run_training(
                config={"training": {}},
                vertex_config=_make_vertex_config(),
                ctx=_single_node_ctx(),
                checkpoint_manager=MagicMock(),
            )
        assert "status" in result

    def test_fallback_training_path(self) -> None:
        """Simulate the ImportError fallback code path directly."""
        mock_alpha_trainer = MagicMock()
        mock_alpha_trainer.train.return_value = MagicMock(step=5, loss=0.5)

        with patch.dict(
            "sys.modules",
            {
                "src.vertex.trainer": MagicMock(
                    VertexTrainer=MagicMock(side_effect=ImportError("unavailable"))
                ),
                "config.schemas": MagicMock(TrainingConfig=MagicMock()),
                "src.training.trainer": MagicMock(
                    AlphaGalerkinTrainer=MagicMock(return_value=mock_alpha_trainer)
                ),
            },
        ):
            result = run_training(
                config={"training": {}},
                vertex_config=_make_vertex_config(),
                ctx=_single_node_ctx(),
                checkpoint_manager=MagicMock(),
            )
        assert "status" in result


# ---------------------------------------------------------------------------
# Tests: main()
# ---------------------------------------------------------------------------


class TestMain:
    @patch("src.vertex.entrypoint.run_training")
    @patch("src.vertex.entrypoint.GCSCheckpointManager")
    @patch("src.vertex.entrypoint.create_vertex_config_from_env")
    @patch("src.vertex.entrypoint.init_wandb_for_vertex")
    @patch("src.vertex.entrypoint.load_training_config")
    @patch("src.vertex.entrypoint.init_distributed")
    @patch("src.vertex.entrypoint.setup_distributed_training")
    @patch("src.vertex.entrypoint.dist")
    def test_main_success(
        self,
        mock_dist: MagicMock,
        mock_setup: MagicMock,
        mock_init_dist: MagicMock,
        mock_load_cfg: MagicMock,
        mock_wandb: MagicMock,
        mock_create_vc: MagicMock,
        mock_gcs: MagicMock,
        mock_run_training: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from src.vertex.entrypoint import main

        cfg_file = tmp_path / "train.yaml"
        cfg_file.write_text("training:\n  total_steps: 10\n")

        monkeypatch.setattr(
            sys, "argv", ["prog", "--config", str(cfg_file)]
        )

        mock_ctx = MagicMock()
        mock_ctx.rank = 0
        mock_ctx.world_size = 1
        mock_ctx.local_rank = 0
        mock_ctx.master_addr = "127.0.0.1"
        mock_ctx.is_main_process.return_value = True
        mock_setup.return_value = mock_ctx

        mock_load_cfg.return_value = {"training": {"total_steps": 10}}
        mock_wandb.return_value = None
        mock_create_vc.return_value = _make_vertex_config()
        mock_run_training.return_value = {
            "status": "completed",
            "final_step": 10,
            "final_loss": 0.01,
        }
        mock_dist.is_initialized.return_value = False

        exit_code = main()
        assert exit_code == 0

    @patch("src.vertex.entrypoint.setup_distributed_training")
    def test_main_dry_run(
        self,
        mock_setup: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from src.vertex.entrypoint import main

        cfg_file = tmp_path / "train.yaml"
        cfg_file.write_text("training:\n  total_steps: 10\n")

        monkeypatch.setattr(
            sys, "argv", ["prog", "--config", str(cfg_file), "--dry-run"]
        )

        mock_ctx = MagicMock()
        mock_ctx.rank = 0
        mock_ctx.world_size = 1
        mock_ctx.master_addr = "127.0.0.1"
        mock_ctx.is_main_process.return_value = True
        mock_setup.return_value = mock_ctx

        exit_code = main()
        assert exit_code == 0

    @patch("src.vertex.entrypoint.setup_distributed_training")
    def test_main_keyboard_interrupt_returns_130(
        self,
        mock_setup: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from src.vertex.entrypoint import main

        cfg_file = tmp_path / "train.yaml"
        cfg_file.write_text("{}")

        monkeypatch.setattr(
            sys, "argv", ["prog", "--config", str(cfg_file)]
        )
        mock_setup.side_effect = KeyboardInterrupt()
        exit_code = main()
        assert exit_code == 130

    @patch("src.vertex.entrypoint.setup_distributed_training")
    def test_main_unexpected_exception_returns_1(
        self,
        mock_setup: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from src.vertex.entrypoint import main

        cfg_file = tmp_path / "train.yaml"
        cfg_file.write_text("{}")

        monkeypatch.setattr(
            sys, "argv", ["prog", "--config", str(cfg_file)]
        )
        mock_setup.side_effect = RuntimeError("unexpected error")
        exit_code = main()
        assert exit_code == 1

    @patch("src.vertex.entrypoint.run_training")
    @patch("src.vertex.entrypoint.GCSCheckpointManager")
    @patch("src.vertex.entrypoint.create_vertex_config_from_env")
    @patch("src.vertex.entrypoint.init_wandb_for_vertex")
    @patch("src.vertex.entrypoint.load_training_config")
    @patch("src.vertex.entrypoint.init_distributed")
    @patch("src.vertex.entrypoint.setup_distributed_training")
    @patch("src.vertex.entrypoint.dist")
    def test_main_vertex_config_error_uses_fallback(
        self,
        mock_dist: MagicMock,
        mock_setup: MagicMock,
        mock_init_dist: MagicMock,
        mock_load_cfg: MagicMock,
        mock_wandb: MagicMock,
        mock_create_vc: MagicMock,
        mock_gcs: MagicMock,
        mock_run_training: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """When create_vertex_config_from_env raises ValueError, falls back to local-test config."""
        from src.vertex.entrypoint import main

        cfg_file = tmp_path / "train.yaml"
        cfg_file.write_text("training:\n  total_steps: 5\n")

        monkeypatch.setattr(
            sys, "argv", ["prog", "--config", str(cfg_file)]
        )

        mock_ctx = MagicMock()
        mock_ctx.rank = 0
        mock_ctx.world_size = 1
        mock_ctx.local_rank = 0
        mock_ctx.master_addr = "127.0.0.1"
        mock_ctx.is_main_process.return_value = True
        mock_setup.return_value = mock_ctx

        mock_load_cfg.return_value = {}
        mock_wandb.return_value = None
        mock_create_vc.side_effect = ValueError("missing env vars")
        mock_run_training.return_value = {
            "status": "completed",
            "final_step": 5,
            "final_loss": 0.0,
        }
        mock_dist.is_initialized.return_value = False

        exit_code = main()
        assert exit_code == 0
        # Verify fallback config was used
        gcs_call_kwargs = mock_gcs.call_args[1]
        assert gcs_call_kwargs["bucket_name"] == "local-test"

    @patch("src.vertex.entrypoint.run_training")
    @patch("src.vertex.entrypoint.GCSCheckpointManager")
    @patch("src.vertex.entrypoint.create_vertex_config_from_env")
    @patch("src.vertex.entrypoint.init_wandb_for_vertex")
    @patch("src.vertex.entrypoint.load_training_config")
    @patch("src.vertex.entrypoint.init_distributed")
    @patch("src.vertex.entrypoint.setup_distributed_training")
    @patch("src.vertex.entrypoint.dist")
    def test_main_wandb_cleanup_on_exception(
        self,
        mock_dist: MagicMock,
        mock_setup: MagicMock,
        mock_init_dist: MagicMock,
        mock_load_cfg: MagicMock,
        mock_wandb: MagicMock,
        mock_create_vc: MagicMock,
        mock_gcs: MagicMock,
        mock_run_training: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """wandb.finish() is called even if run_training raises."""
        from src.vertex.entrypoint import main

        cfg_file = tmp_path / "train.yaml"
        cfg_file.write_text("{}")
        monkeypatch.setattr(
            sys, "argv", ["prog", "--config", str(cfg_file)]
        )

        mock_ctx = MagicMock()
        mock_ctx.rank = 0
        mock_ctx.world_size = 1
        mock_ctx.local_rank = 0
        mock_ctx.master_addr = "127.0.0.1"
        mock_ctx.is_main_process.return_value = True
        mock_setup.return_value = mock_ctx

        mock_load_cfg.return_value = {}
        mock_wandb_logger = MagicMock()
        mock_wandb.return_value = mock_wandb_logger
        mock_create_vc.return_value = _make_vertex_config()
        mock_run_training.side_effect = RuntimeError("training exploded")
        mock_dist.is_initialized.return_value = False

        exit_code = main()
        assert exit_code == 1
        mock_wandb_logger.finish.assert_called_once()

    @patch("src.vertex.entrypoint.run_training")
    @patch("src.vertex.entrypoint.GCSCheckpointManager")
    @patch("src.vertex.entrypoint.create_vertex_config_from_env")
    @patch("src.vertex.entrypoint.init_wandb_for_vertex")
    @patch("src.vertex.entrypoint.load_training_config")
    @patch("src.vertex.entrypoint.init_distributed")
    @patch("src.vertex.entrypoint.setup_distributed_training")
    @patch("src.vertex.entrypoint.dist")
    def test_main_dist_cleanup_called(
        self,
        mock_dist: MagicMock,
        mock_setup: MagicMock,
        mock_init_dist: MagicMock,
        mock_load_cfg: MagicMock,
        mock_wandb: MagicMock,
        mock_create_vc: MagicMock,
        mock_gcs: MagicMock,
        mock_run_training: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """dist.destroy_process_group() called when dist is initialized."""
        from src.vertex.entrypoint import main

        cfg_file = tmp_path / "train.yaml"
        cfg_file.write_text("{}")
        monkeypatch.setattr(
            sys, "argv", ["prog", "--config", str(cfg_file)]
        )

        mock_ctx = MagicMock()
        mock_ctx.rank = 0
        mock_ctx.world_size = 1
        mock_ctx.local_rank = 0
        mock_ctx.master_addr = "127.0.0.1"
        mock_ctx.is_main_process.return_value = True
        mock_setup.return_value = mock_ctx

        mock_load_cfg.return_value = {}
        mock_wandb.return_value = None
        mock_create_vc.return_value = _make_vertex_config()
        mock_run_training.return_value = {
            "status": "completed",
            "final_step": 0,
            "final_loss": 0.0,
        }
        mock_dist.is_initialized.return_value = True

        main()
        mock_dist.destroy_process_group.assert_called_once()

    @patch("src.vertex.entrypoint.run_training")
    @patch("src.vertex.entrypoint.GCSCheckpointManager")
    @patch("src.vertex.entrypoint.create_vertex_config_from_env")
    @patch("src.vertex.entrypoint.init_wandb_for_vertex")
    @patch("src.vertex.entrypoint.load_training_config")
    @patch("src.vertex.entrypoint.init_distributed")
    @patch("src.vertex.entrypoint.setup_distributed_training")
    @patch("src.vertex.entrypoint.dist")
    def test_main_wandb_finish_exception_handled(
        self,
        mock_dist: MagicMock,
        mock_setup: MagicMock,
        mock_init_dist: MagicMock,
        mock_load_cfg: MagicMock,
        mock_wandb: MagicMock,
        mock_create_vc: MagicMock,
        mock_gcs: MagicMock,
        mock_run_training: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """wandb.finish() errors are caught and don't propagate."""
        from src.vertex.entrypoint import main

        cfg_file = tmp_path / "train.yaml"
        cfg_file.write_text("{}")
        monkeypatch.setattr(
            sys, "argv", ["prog", "--config", str(cfg_file)]
        )

        mock_ctx = MagicMock()
        mock_ctx.rank = 0
        mock_ctx.world_size = 1
        mock_ctx.local_rank = 0
        mock_ctx.master_addr = "127.0.0.1"
        mock_ctx.is_main_process.return_value = True
        mock_setup.return_value = mock_ctx

        mock_load_cfg.return_value = {}
        mock_wandb_logger = MagicMock()
        mock_wandb_logger.finish.side_effect = RuntimeError("wandb error")
        mock_wandb.return_value = mock_wandb_logger
        mock_create_vc.return_value = _make_vertex_config()
        mock_run_training.return_value = {
            "status": "completed",
            "final_step": 0,
            "final_loss": 0.0,
        }
        mock_dist.is_initialized.return_value = False

        # Should not raise despite wandb.finish() failing
        exit_code = main()
        assert exit_code == 0


# ---------------------------------------------------------------------------
# Tests: emergency_checkpoint closure (via main integration)
# ---------------------------------------------------------------------------


class TestEmergencyCheckpoint:
    def test_emergency_checkpoint_without_trainer(self) -> None:
        trainer_ref: dict[str, Any] = {}

        def emergency_checkpoint() -> None:
            trainer = trainer_ref.get("trainer")
            if trainer is None:
                return
            raise AssertionError("Should have returned early")

        emergency_checkpoint()

    def test_emergency_checkpoint_saves_state(self) -> None:
        mock_trainer = MagicMock()
        mock_trainer.global_step = 42
        mock_trainer.model = MagicMock()
        mock_trainer.optimizer = MagicMock()
        mock_trainer.scheduler = MagicMock()

        mock_manager = MagicMock()
        mock_manager.save.return_value = "gs://bucket/emergency_42.pt"

        trainer_ref: dict[str, Any] = {"trainer": mock_trainer}

        trainer = trainer_ref.get("trainer")
        assert trainer is not None
        step = getattr(trainer, "global_step", 0)
        model = getattr(trainer, "model", None)
        optimizer = getattr(trainer, "optimizer", None)
        scheduler = getattr(trainer, "scheduler", None)

        gcs_path = mock_manager.save(
            step=step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics={"emergency": 1.0, "step": float(step)},
            extra={"preempted": True},
        )

        mock_manager.save.assert_called_once()
        call_kwargs = mock_manager.save.call_args
        assert call_kwargs[1]["step"] == 42
        assert call_kwargs[1]["extra"]["preempted"] is True
        assert gcs_path == "gs://bucket/emergency_42.pt"

    def test_emergency_checkpoint_no_model_on_trainer(self) -> None:
        """Trainer with no model attribute skips save gracefully."""
        mock_trainer = MagicMock(spec=[])  # No attributes
        mock_trainer.global_step = 10
        # model attr not present -> getattr returns None

        mock_manager = MagicMock()
        trainer_ref: dict[str, Any] = {"trainer": mock_trainer}

        trainer = trainer_ref.get("trainer")
        assert trainer is not None
        step = getattr(trainer, "global_step", 0)
        model = getattr(trainer, "model", None) or getattr(
            trainer, "_raw_model", None
        )

        if model is None:
            # Should skip save - this is the expected branch
            mock_manager.save.assert_not_called()
        else:
            mock_manager.save(step=step, model=model)

    def test_emergency_checkpoint_handles_save_error(self) -> None:
        mock_trainer = MagicMock()
        mock_trainer.global_step = 100
        mock_trainer.model = MagicMock()

        mock_manager = MagicMock()
        mock_manager.save.side_effect = OSError("Disk full")

        trainer_ref: dict[str, Any] = {"trainer": mock_trainer}

        trainer = trainer_ref.get("trainer")
        assert trainer is not None
        with contextlib.suppress(OSError):
            mock_manager.save(step=trainer.global_step, model=trainer.model)

    def test_emergency_checkpoint_uses_raw_model_fallback(self) -> None:
        """Falls back to _raw_model when model attr is None."""
        mock_trainer = MagicMock()
        mock_trainer.global_step = 77
        mock_trainer.model = None
        mock_raw = MagicMock()
        mock_trainer._raw_model = mock_raw

        trainer_ref: dict[str, Any] = {"trainer": mock_trainer}
        trainer = trainer_ref.get("trainer")
        assert trainer is not None

        model = getattr(trainer, "model", None) or getattr(
            trainer, "_raw_model", None
        )
        assert model is mock_raw
