"""End-to-end integration tests for distributed training.

Uses torch.multiprocessing.spawn with 2 processes to validate:
  - Model initialization across ranks
  - Gradient synchronization via all_reduce
  - Checkpoint save (rank 0) and load (rank 1) with verification

Backend: GLOO (works on CPU, no CUDA required).
"""

from __future__ import annotations

import os
import pathlib
import socket
from typing import Any

import pytest
import torch
import torch.distributed
import torch.multiprocessing
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Return an OS-assigned free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class _ToyModel(nn.Module):
    """Minimal linear model for testing gradient synchronization."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Module-level worker function (must NOT be nested — spawn cannot pickle lambdas)
# ---------------------------------------------------------------------------


def _worker_fn(
    rank: int,
    world_size: int,
    checkpoint_dir: str,
    port: int,
    result_dict: Any,
) -> None:
    """Worker executed in each spawned process.

    Steps:
    1. Initialise torch.distributed with GLOO backend.
    2. Build identical toy model on every rank.
    3. Perform a forward + backward pass with rank-specific input.
    4. All-reduce gradients and verify they match across ranks.
    5. Rank 0 saves a checkpoint; rank 1 loads it and verifies weights.
    """
    # --- Environment setup --------------------------------------------------
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    torch.distributed.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
    )

    try:
        # --- Model initialization -------------------------------------------
        torch.manual_seed(42)  # same seed → identical initial weights
        model = _ToyModel()

        # --- Rank-specific forward/backward pass ----------------------------
        # Each rank uses a different input so gradients differ before sync.
        x = torch.ones(2, 4) * (rank + 1.0)
        y = torch.zeros(2, 2)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()

        # Capture pre-sync gradient from rank 0's perspective
        pre_sync_grad = model.fc.weight.grad.clone()

        # --- Gradient all-reduce (mean) ------------------------------------
        for param in model.parameters():
            if param.grad is not None:
                torch.distributed.all_reduce(
                    param.grad,
                    op=torch.distributed.ReduceOp.SUM,
                )
                param.grad /= world_size

        post_sync_grad = model.fc.weight.grad.clone()

        # All ranks should now have the same gradient.
        # Broadcast rank-0 gradient to verify equality.
        reference_grad = post_sync_grad.clone()
        torch.distributed.broadcast(reference_grad, src=0)
        assert torch.allclose(post_sync_grad, reference_grad, atol=1e-5), (
            f"Rank {rank}: gradient mismatch after all_reduce. "
            f"got {post_sync_grad}, expected {reference_grad}"
        )

        # Confirm sync actually changed something (gradients were different before)
        if world_size > 1:
            assert not torch.allclose(pre_sync_grad, post_sync_grad, atol=1e-7), (
                f"Rank {rank}: gradients unchanged after all_reduce — "
                "sync may not have happened"
            )

        # --- Checkpoint: rank 0 saves, rank 1 loads and verifies -----------
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")

        if rank == 0:
            # Save using torch.save (weights_only-compatible: state_dict + scalars only)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": 1,
                    "loss": loss.item(),
                },
                ckpt_path,
            )

        # Barrier ensures rank 0 has finished writing before rank 1 reads.
        torch.distributed.barrier()

        if rank == 1:
            ckpt = torch.load(ckpt_path, weights_only=True)  # noqa: S614 — local test file
            loaded_model = _ToyModel()
            loaded_model.load_state_dict(ckpt["model_state_dict"])

            # Weights in the checkpoint should match the model that rank 0 saved.
            for (name, orig_param), (_, loaded_param) in zip(
                model.named_parameters(), loaded_model.named_parameters()
            ):
                assert torch.allclose(orig_param.data, loaded_param.data, atol=1e-6), (
                    f"Checkpoint weight mismatch for '{name}': "
                    f"original={orig_param.data}, loaded={loaded_param.data}"
                )
            assert ckpt["epoch"] == 1

        # Signal success for this rank
        result_dict[rank] = "ok"

    finally:
        torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_distributed_e2e_2_processes(tmp_path: pathlib.Path) -> None:
    """Test 2-process distributed training round-trip with GLOO backend.

    Validates:
    - Process group initialisation with GLOO
    - Gradient all-reduce synchronisation
    - Checkpoint save (rank 0) → load (rank 1) consistency
    """
    port = _find_free_port()
    world_size = 2

    # Use a manager dict so worker processes can write results back.
    ctx = torch.multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    result_dict = manager.dict()

    try:
        torch.multiprocessing.spawn(
            _worker_fn,
            args=(world_size, str(tmp_path), port, result_dict),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )
    except RuntimeError as exc:
        pytest.skip(f"torch.multiprocessing.spawn not supported on this platform: {exc}")

    # Verify both processes reported success
    assert len(result_dict) == world_size, (
        f"Expected {world_size} results, got {len(result_dict)}: {dict(result_dict)}"
    )
    for rank in range(world_size):
        assert result_dict[rank] == "ok", (
            f"Rank {rank} did not complete successfully: {result_dict.get(rank)}"
        )


def test_find_free_port_returns_valid_port() -> None:
    """Sanity-check that _find_free_port returns a usable port number."""
    port = _find_free_port()
    assert 1024 <= port <= 65535
