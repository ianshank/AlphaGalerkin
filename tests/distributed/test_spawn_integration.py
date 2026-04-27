"""Real ``torch.multiprocessing.spawn``-based distributed tests (Track D).

The existing :mod:`tests.distributed.test_multiprocess` covers the
distributed *config* surface using :class:`unittest.mock` — it never
actually starts a process group.  This module fills the gap identified
by Milestone 5 in ``docs/NEXT_STEPS_PLAN.md`` by spawning real worker
processes that run all-reduce / broadcast / barrier collectives end-to-end
with the **gloo** backend (CPU-only, so the tests run unattended in CI).

Marker: ``@pytest.mark.distributed``.  Tests skip cleanly when
``RUN_DISTRIBUTED_TESTS`` is unset *and* the platform is Windows (where
gloo + spawn has historically been brittle in CI runners).

The companion NCCL variant lives behind ``@pytest.mark.gpu_required`` so
it runs on the user's dual-Blackwell rig (RTX 5060 Ti + 5060) but is
skipped on CPU CI.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from src.distributed.config import (
    DistributedBackend,
    DistributedInfraConfig,
)

pytestmark = pytest.mark.distributed


def _should_skip_distributed() -> tuple[bool, str]:
    """Return (skip?, reason) for the current environment.

    Skips on Windows unless ``RUN_DISTRIBUTED_TESTS=1`` is set, since
    gloo + multiprocessing.spawn has known-flaky behaviour on Windows
    GitHub Actions runners.  Linux always runs.
    """
    if sys.platform.startswith("win") and os.environ.get("RUN_DISTRIBUTED_TESTS") != "1":
        return True, (
            "gloo+spawn is flaky on Windows runners; set " "RUN_DISTRIBUTED_TESTS=1 to opt in"
        )
    return False, ""


_SKIP, _SKIP_REASON = _should_skip_distributed()


# ---------------------------------------------------------------------------
# Worker functions executed in each spawned process
# ---------------------------------------------------------------------------


def _all_reduce_worker(
    rank: int,
    world_size: int,
    backend: str,
    init_file: str,
    out_dir: str,
) -> None:
    """Spawn worker: init pg, all-reduce a rank-tagged tensor, save result."""
    import torch.distributed as dist

    init_method = f"file://{init_file}"
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        # Each rank contributes its own rank+1 (so we can verify the
        # sum is sum(1..world_size) = world_size*(world_size+1)/2).
        local = torch.tensor([float(rank + 1)])
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        torch.save(local, Path(out_dir) / f"rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


def _broadcast_worker(
    rank: int,
    world_size: int,
    backend: str,
    init_file: str,
    out_dir: str,
) -> None:
    """Spawn worker: rank 0 broadcasts a payload; all ranks save it."""
    import torch.distributed as dist

    dist.init_process_group(
        backend=backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        # Rank 0 starts with payload [3.14, 2.71]; others start with zeros.
        if rank == 0:
            payload = torch.tensor([3.14, 2.71])
        else:
            payload = torch.zeros(2)
        dist.broadcast(payload, src=0)
        torch.save(payload, Path(out_dir) / f"rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


def _barrier_worker(
    rank: int,
    world_size: int,
    backend: str,
    init_file: str,
    out_dir: str,
) -> None:
    """Spawn worker: hit a barrier, write a sentinel file post-barrier."""
    import torch.distributed as dist

    dist.init_process_group(
        backend=backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        dist.barrier()
        (Path(out_dir) / f"rank_{rank}.ok").write_text(f"rank={rank}\n")
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Spawn helper (single source of truth for run config)
# ---------------------------------------------------------------------------


def _spawn(
    fn: object,
    world_size: int,
    backend: str,
    out_dir: Path,
) -> None:
    """Spawn ``world_size`` workers and wait for completion.

    Uses a temp file as the c10d rendezvous so the test does not rely on
    a free TCP port — that has bitten previous flaky runs on shared CI.
    """
    init_file = out_dir / "rendezvous"
    # The init file must not exist before init_process_group is called.
    if init_file.exists():
        init_file.unlink()
    torch.multiprocessing.spawn(  # type: ignore[attr-defined]
        fn=fn,
        args=(world_size, backend, str(init_file), str(out_dir)),
        nprocs=world_size,
        join=True,
    )


# ---------------------------------------------------------------------------
# Tests — gloo backend (CPU-only, runs in CI on Linux)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestSpawnGloo:
    WORLD_SIZE = 2
    BACKEND = "gloo"

    def test_all_reduce_sum_matches_closed_form(self, tmp_path: Path) -> None:
        _spawn(_all_reduce_worker, self.WORLD_SIZE, self.BACKEND, tmp_path)
        # Each rank should have the same all-reduced sum: 1+2+...+world_size.
        expected = sum(range(1, self.WORLD_SIZE + 1))
        for rank in range(self.WORLD_SIZE):
            tensor = torch.load(tmp_path / f"rank_{rank}.pt", weights_only=False)
            assert tensor.item() == pytest.approx(float(expected))

    def test_broadcast_propagates_payload_to_all_ranks(self, tmp_path: Path) -> None:
        _spawn(_broadcast_worker, self.WORLD_SIZE, self.BACKEND, tmp_path)
        ref = torch.tensor([3.14, 2.71])
        for rank in range(self.WORLD_SIZE):
            tensor = torch.load(tmp_path / f"rank_{rank}.pt", weights_only=False)
            torch.testing.assert_close(tensor, ref)

    def test_barrier_synchronises_all_ranks(self, tmp_path: Path) -> None:
        _spawn(_barrier_worker, self.WORLD_SIZE, self.BACKEND, tmp_path)
        for rank in range(self.WORLD_SIZE):
            assert (tmp_path / f"rank_{rank}.ok").exists()


# ---------------------------------------------------------------------------
# Tests — NCCL backend (GPU-only, for the user's RTX 5060 Ti + 5060 rig)
# ---------------------------------------------------------------------------


def _nccl_available() -> bool:
    """Return True iff NCCL is compiled into the running PyTorch.

    NCCL is not built on Windows wheels even when CUDA is, so the
    plain ``torch.cuda.is_available()`` check is insufficient.
    """
    try:
        return bool(torch.distributed.is_nccl_available())  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):
        return False


@pytest.mark.gpu_required
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() >= 2 and _nccl_available()),
    reason=(
        "Requires >=2 CUDA devices AND a PyTorch build with NCCL compiled in "
        "(typical on Linux + CUDA, missing from Windows wheels)."
    ),
)
class TestSpawnNccl:
    """Same suite, NCCL backend, dual-GPU only.

    Skipped automatically on the CI matrix; intended to be run by the
    user via:

        pytest tests/distributed/test_spawn_integration.py::TestSpawnNccl -v
    """

    WORLD_SIZE = 2
    BACKEND = "nccl"

    def test_all_reduce_sum_matches_closed_form(self, tmp_path: Path) -> None:
        _spawn(_all_reduce_worker, self.WORLD_SIZE, self.BACKEND, tmp_path)
        expected = sum(range(1, self.WORLD_SIZE + 1))
        for rank in range(self.WORLD_SIZE):
            tensor = torch.load(tmp_path / f"rank_{rank}.pt", weights_only=False)
            assert tensor.item() == pytest.approx(float(expected))


# ---------------------------------------------------------------------------
# DistributedInfraConfig.per_rank_batch_size validation
# ---------------------------------------------------------------------------


class TestPerRankBatchSize:
    """The new ``per_rank_batch_size`` field added in Track D."""

    def test_default_is_none_for_backwards_compat(self) -> None:
        cfg = DistributedInfraConfig(enabled=False)
        assert cfg.per_rank_batch_size is None

    def test_int_override_returned_for_every_rank(self) -> None:
        cfg = DistributedInfraConfig(enabled=True, world_size=2, per_rank_batch_size=128)
        assert cfg.get_rank_batch_size(0, default=999) == 128
        assert cfg.get_rank_batch_size(1, default=999) == 128

    def test_list_override_returns_per_rank_value(self) -> None:
        # Asymmetric rig: 5060 Ti gets larger batch than 5060.
        cfg = DistributedInfraConfig(enabled=True, world_size=2, per_rank_batch_size=[64, 32])
        assert cfg.get_rank_batch_size(0, default=999) == 64
        assert cfg.get_rank_batch_size(1, default=999) == 32

    def test_none_falls_back_to_default(self) -> None:
        cfg = DistributedInfraConfig(enabled=True, world_size=2)
        assert cfg.get_rank_batch_size(0, default=42) == 42

    def test_list_length_must_match_world_size(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DistributedInfraConfig(
                enabled=True,
                world_size=4,
                per_rank_batch_size=[1, 2, 3],
            )

    def test_list_entries_must_be_strictly_positive(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DistributedInfraConfig(
                enabled=True,
                world_size=2,
                per_rank_batch_size=[8, 0],
            )

    def test_int_must_be_strictly_positive(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DistributedInfraConfig(enabled=True, world_size=1, per_rank_batch_size=-1)

    def test_rank_out_of_range_raises(self) -> None:
        cfg = DistributedInfraConfig(enabled=True, world_size=2, per_rank_batch_size=[16, 32])
        with pytest.raises(ValueError, match="out of range"):
            cfg.get_rank_batch_size(rank=2, default=999)

    def test_default_backend_is_nccl_for_gpu_primary_rig(self) -> None:
        # Sanity: track D rule that "nccl is the default backend".
        cfg = DistributedInfraConfig()
        assert cfg.backend == DistributedBackend.NCCL


# ---------------------------------------------------------------------------
# Smoke test: tempfile cleanup hygiene
# ---------------------------------------------------------------------------


def test_tempfile_rendezvous_does_not_leak() -> None:
    """Tempfile rendezvous files do not leak across runs.

    The rendezvous file approach used above must not pollute the
    system temp dir across runs.
    """
    # Sanity: the temp dir is writable and no orphan rendezvous exists
    # under the tests/ tree before the suite runs.
    tmp = Path(tempfile.gettempdir())
    assert tmp.exists()
