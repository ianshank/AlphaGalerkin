"""Tests for model checkpoint safety and secure loading."""

import os
import pickle
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn

from src.training.checkpoint import (
    CHECKPOINT_VERSION,
    SAFE_CHECKPOINT_GLOBALS,
    CheckpointManager,
    load_checkpoint_with_config,
    load_model_only,
    load_torch_checkpoint,
)

# How long the "slow" load stalls inside the allowlist window waiting for the
# other thread. Without the lock the other thread finishes immediately and this
# never elapses; with the lock the other thread is blocked on acquire, so this
# is the (one-off, per-test) cost of proving no tear-down happened.
_WINDOW_HANDSHAKE_TIMEOUT_S = 0.5

# Ceiling on any thread join in these tests. Exceeding it means a deadlock, not
# slowness -- every load here is a few hundred bytes off tmpfs.
_THREAD_JOIN_TIMEOUT_S = 30.0

pytestmark = pytest.mark.security


class MaliciousPickle:
    """A malicious object that executes code when unpickled."""

    def __reduce__(self) -> tuple[Any, ...]:
        return (os.system, ('echo "vulnerable" > /dev/null',))


class MarkerPickle:
    """A ``MaliciousPickle`` that leaves *evidence* when unpickled.

    Same shape and same blocked GLOBAL (``posix.system``) as
    :class:`MaliciousPickle`, but it touches a file instead of writing to
    ``/dev/null``. That turns "did the payload run?" into a filesystem
    assertion, which is the only way to catch a fallback that silently
    unpickles after the safe load rejected the file. The side effect stays
    inside pytest's ``tmp_path``.
    """

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[Any, ...]:
        return (os.system, (f"touch {self.marker}",))


@pytest.fixture
def malicious_checkpoint(tmp_path: Path) -> Path:
    """Create a mock malicious checkpoint using standard pickle."""
    ckpt_path = tmp_path / "malicious.pt"
    state = {"weights": MaliciousPickle()}
    torch.save(state, ckpt_path)
    return ckpt_path


@pytest.fixture
def marker_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    """A checkpoint whose payload creates a marker file when unpickled.

    Returns:
        ``(checkpoint_path, marker_path)``. The marker does not exist yet;
        if it exists after a load, the payload executed.

    """
    marker = tmp_path / "PAYLOAD_EXECUTED"
    ckpt_path = tmp_path / "marker.pt"
    # A real state dict alongside the payload, shaped to load cleanly into the
    # ``nn.Linear(1, 1)`` these tests use -- so the file is exactly what a loader
    # would happily consume once unpickled, and nothing but the safety check
    # stands between the payload and execution.
    torch.save(
        {
            "model_state_dict": {"weight": torch.zeros(1, 1), "bias": torch.zeros(1)},
            "weights": MarkerPickle(marker),
        },
        ckpt_path,
    )
    assert not marker.exists()
    return ckpt_path, marker


def _minimal_trainer(tmp_path: Path) -> Any:
    """Build the smallest real ``BaseTrainer`` that can attempt a load.

    Mirrors ``tests/training/test_base_trainer.py::ConcreteTrainer``; defined
    here so the security suite exercises the shipped public method rather than
    the private helper underneath it.
    """
    from src.training.base_trainer import BaseTrainer, BaseTrainerConfig

    class _Config(BaseTrainerConfig):
        pass

    class _Trainer(BaseTrainer[_Config]):
        def compute_loss(self, batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
            loss = self.model(batch).sum()
            return loss, {"loss": float(loss)}

        def generate_data(self) -> torch.Tensor:
            return torch.randn(2, 2)

    return _Trainer(nn.Linear(2, 2), _Config(name="sec"), "cpu", tmp_path / "ckpt")


def test_torch_load_weights_only() -> None:
    """Verify torch.load is explicitly called with weights_only=True where applicable."""
    # Check that when CheckpointManager loads weights, it specifies weights_only=True
    with patch("torch.load") as mock_load:
        mock_load.return_value = {"model_state_dict": {}}
        mock_model = nn.Linear(1, 1)

        load_model_only(mock_model, "dummy.pt", strict=False)

        mock_load.assert_called_once_with("dummy.pt", map_location="cpu", weights_only=True)


def test_safe_load_failure_does_not_retry_unsafely() -> None:
    """A *failing* safe load must not be retried with weights_only=False.

    ``test_torch_load_weights_only`` above mocks ``torch.load`` to *succeed*, so
    the failure branch never runs -- which is precisely why the fallback ACE path
    survived it. Here the safe load raises, and the assertion is on what happens
    next: exactly one call, and no unpickle.
    """
    with patch("torch.load") as mock_load:
        mock_load.side_effect = pickle.UnpicklingError("Unsupported global: posix.system")
        mock_model = nn.Linear(1, 1)

        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            load_model_only(mock_model, "dummy.pt", strict=False)

        assert mock_load.call_count == 1, "safe-load failure triggered a retry"
        assert all(call.kwargs.get("weights_only") is True for call in mock_load.call_args_list), (
            "torch.load was reached with weights_only != True"
        )


def test_malicious_checkpoint_rejected(malicious_checkpoint: Path) -> None:
    """Verify that a malicious checkpoint is rejected when loaded with weights_only=True."""
    with pytest.raises(pickle.UnpicklingError):
        torch.load(malicious_checkpoint, weights_only=True)


def test_checkpoint_path_validation(tmp_path: Path) -> None:
    """Verify untrusted checkpoint paths are validated against traversal."""
    manager = CheckpointManager(checkpoint_dir=str(tmp_path))
    with pytest.raises((ValueError, FileNotFoundError, PermissionError, RuntimeError)):
        manager.load(path="../../../etc/shadow")


def test_traversal_rejected_even_when_target_is_readable(tmp_path: Path) -> None:
    """Traversal is refused by path policy, not by an incidental read failure.

    ``test_checkpoint_path_validation`` above passes for the right reason only if
    the rejection is independent of file permissions: as root, ``/etc/shadow`` is
    readable and the pre-fix code reached ``torch.load``. Here the traversal target
    is a real, world-readable, perfectly loadable checkpoint, so the only thing that
    can reject it is the containment check.
    """
    root = tmp_path / "ckpts"
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "trusted_looking.pt"
    torch.save({"model_state_dict": {}, "step": 0, "version": CHECKPOINT_VERSION}, target)
    assert target.is_file()

    manager = CheckpointManager(checkpoint_dir=root)

    with pytest.raises(ValueError, match="outside checkpoint directory"):
        manager.load(path=f"../outside/{target.name}")
    with pytest.raises(ValueError, match="outside checkpoint directory"):
        manager.load(path=target)


def test_external_load_requires_explicit_opt_in(tmp_path: Path) -> None:
    """A checkpoint outside the managed directory loads only with allow_external."""
    root = tmp_path / "ckpts"
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "resume.pt"
    torch.save({"model_state_dict": {}, "step": 11, "version": CHECKPOINT_VERSION}, target)

    manager = CheckpointManager(checkpoint_dir=root)

    with pytest.raises(ValueError):
        manager.load(path=target)
    assert manager.load(path=target, allow_external=True).step == 11


class TestNoUnsafePickleFallback:
    """Regression tests for the ``load_model_only`` fallback ACE path.

    Before 2026-08-21 every loader here either called ``weights_only=False``
    outright or (in ``load_model_only``) caught *any* exception from the safe
    load and retried with ``weights_only=False``. The failure mode was inverted:
    a malicious pickle was executed *precisely because* it failed the safe check.

    These tests use a real on-disk payload rather than a mocked ``torch.load``.
    Mocking is what let the defect survive -- a mock that returns successfully
    never reaches the fallback branch at all.
    """

    def test_load_model_only_rejects_payload_without_executing_it(
        self, marker_checkpoint: tuple[Path, Path]
    ) -> None:
        """The demonstrated ACE path: safe load fails, payload must NOT run."""
        ckpt, marker = marker_checkpoint
        model = nn.Linear(1, 1)

        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            load_model_only(model, ckpt, strict=False)

        assert not marker.exists(), (
            "ARBITRARY CODE EXECUTION: the payload ran. A checkpoint that fails "
            "the weights_only=True check was unpickled anyway."
        )

    def test_load_checkpoint_with_config_rejects_payload(
        self, marker_checkpoint: tuple[Path, Path]
    ) -> None:
        """Sibling loader: same file, same rejection, no execution."""
        ckpt, marker = marker_checkpoint

        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            load_checkpoint_with_config(ckpt)

        assert not marker.exists(), "payload executed via load_checkpoint_with_config"

    def test_manager_load_rejects_payload(self, tmp_path: Path) -> None:
        """A payload *inside* checkpoint_dir passes containment but not the unpickle.

        The ``allow_external`` containment check added in 77e928c bounds *which*
        files are opened; it says nothing about what is inside a file that is
        legitimately in the directory (e.g. written by a compromised training
        job, or restored from a poisoned artifact store).
        """
        marker = tmp_path / "PAYLOAD_EXECUTED"
        ckpt_dir = tmp_path / "ckpts"
        manager = CheckpointManager(checkpoint_dir=ckpt_dir)
        torch.save(
            {
                "model_state_dict": {},
                "step": 1,
                "version": CHECKPOINT_VERSION,
                "payload": MarkerPickle(marker),
            },
            ckpt_dir / "checkpoint_00000001.pt",
        )

        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            manager.load()

        assert not marker.exists(), "payload executed via CheckpointManager.load"

    def test_base_trainer_load_checkpoint_rejects_payload(self, tmp_path: Path) -> None:
        """``BaseTrainer.load_checkpoint`` had no containment of any kind.

        Unlike ``CheckpointManager`` it has no ``checkpoint_dir`` policy, so the
        safe default is the only thing standing between an arbitrary ``--resume``
        path and code execution. Driven through the real public method rather
        than the shared helper, so the wiring is covered as well as the policy.
        """
        trainer = _minimal_trainer(tmp_path)
        marker = tmp_path / "PAYLOAD_EXECUTED"
        ckpt = tmp_path / "trainer.pt"
        torch.save({"global_step": 1, "payload": MarkerPickle(marker)}, ckpt)

        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            trainer.load_checkpoint(ckpt)

        assert not marker.exists(), "payload executed via BaseTrainer.load_checkpoint"

    def test_base_trainer_load_training_state_rejects_payload(self, tmp_path: Path) -> None:
        """The sibling optimizer/scheduler/scaler load path gets the same policy."""
        trainer = _minimal_trainer(tmp_path)
        marker = tmp_path / "PAYLOAD_EXECUTED"
        state = tmp_path / "training_state.pt"
        torch.save({"global_step": 1, "payload": MarkerPickle(marker)}, state)

        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            trainer._load_training_state(state)

        assert not marker.exists(), "payload executed via BaseTrainer._load_training_state"

    def test_error_message_points_at_the_opt_in(self, marker_checkpoint: tuple[Path, Path]) -> None:
        """A rejected load tells the operator the supported escape hatch."""
        ckpt, _ = marker_checkpoint
        model = nn.Linear(1, 1)

        with pytest.raises(RuntimeError, match="allow_unsafe_pickle=True") as exc:
            load_model_only(model, ckpt, strict=False)

        assert "Refusing to retry" in str(exc.value)


class TestUnsafePickleOptIn:
    """Both states of the explicit ``allow_unsafe_pickle`` opt-in."""

    def test_default_is_safe(self, marker_checkpoint: tuple[Path, Path]) -> None:
        """Opt-in defaults to off: the payload is rejected."""
        ckpt, marker = marker_checkpoint
        model = nn.Linear(1, 1)

        with pytest.raises(RuntimeError):
            load_model_only(model, ckpt, strict=False)
        assert not marker.exists()

    def test_opt_in_reaches_pickle(self, marker_checkpoint: tuple[Path, Path]) -> None:
        """``allow_unsafe_pickle=True`` really does unpickle.

        This asserts the escape hatch is a *functioning* one and not dead code --
        the flag must be the only thing that reaches pickle, so that the safe
        default is meaningful rather than incidental.
        """
        ckpt, marker = marker_checkpoint
        model = nn.Linear(1, 1)

        # strict=False: the payload checkpoint's state dict does not match Linear.
        load_model_only(model, ckpt, strict=False, allow_unsafe_pickle=True)

        assert marker.exists(), (
            "allow_unsafe_pickle=True did not reach pickle -- the opt-in is dead "
            "code, so the safe default proves nothing."
        )

    def test_first_party_checkpoints_never_need_the_opt_in(self, tmp_path: Path) -> None:
        """Every checkpoint this repo writes loads under the safe default.

        This is the evidence the fallback removal rests on: if a real
        ``CheckpointManager.save`` payload -- optimizer + scheduler +
        ``config.model_dump()`` + metrics -- needed ``weights_only=False``, then
        removing the fallback would break resume-from-checkpoint. It does not.
        """
        from config.schemas import AlphaGalerkinConfig

        model = nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        scheduler.step()

        manager = CheckpointManager(checkpoint_dir=tmp_path / "ckpts")
        manager.save(
            step=1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=AlphaGalerkinConfig(),
            metrics={"loss": 0.5},
        )

        # No allow_unsafe_pickle anywhere: the safe path must be sufficient.
        assert manager.load().step == 1


def _load_outcome(path: Path) -> str:
    """Run a safe load and reduce it to ``"ok"`` or a short failure tag.

    Returning a string rather than letting the exception escape keeps the
    assertion in the *main* thread, where a failure is reported instead of
    printed to stderr and swallowed.
    """
    try:
        load_torch_checkpoint(path)
    except Exception as exc:  # Intentionally broad: classify the outcome, do not handle.
        return f"{type(exc).__name__}: {str(exc)[:80]}"
    return "ok"


class TestSafeGlobalsWindowIsSerialized:
    """The allowlist window is process-global, so concurrent loads must not race.

    ``torch.serialization.safe_globals.__enter__`` calls
    ``torch._weights_only_unpickler._add_safe_globals``, which rebinds a
    *module-global* set; ``__exit__`` subtracts from that same global. Two
    overlapping windows therefore tear each other down: the first ``__exit__``
    strips the allowlist while the other load is still unpickling.

    This is reachable in shipped code -- ``src/poc/runner.py`` (the documented
    ``--parallel`` flag) and ``src/agents/research_loop.py`` both reach this
    loader from a ``ThreadPoolExecutor`` -- and becomes live the moment a
    checkpoint carries a ``datetime``, which is exactly what the allowlist
    exists for.
    """

    @staticmethod
    def _datetime_checkpoint(tmp_path: Path) -> Path:
        """A checkpoint that loads *only* while the allowlist window is open."""
        path = tmp_path / "with_datetime.pt"
        torch.save({"created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}, path)
        return path

    def test_payload_really_depends_on_the_window(self, tmp_path: Path) -> None:
        """Precondition for the race test: bare ``weights_only=True`` rejects it.

        Without this the race test could pass for the wrong reason -- a payload
        that loads with or without the allowlist proves nothing about tear-down.
        """
        ckpt = self._datetime_checkpoint(tmp_path)

        with pytest.raises(pickle.UnpicklingError):
            torch.load(ckpt, map_location="cpu", weights_only=True)

        assert load_torch_checkpoint(ckpt)["created_at"].year == 2026

    def test_concurrent_loads_do_not_tear_down_each_others_window(self, tmp_path: Path) -> None:
        """Two overlapping loads must both succeed.

        The handshake is deterministic in both directions rather than timing
        dependent:

        * The "slow" thread stalls *inside* its window (the patched
          ``torch.load`` runs after ``safe_globals.__enter__``) until the other
          thread reports done.
        * Unlocked, the other thread runs straight through -- its ``__exit__``
          removes the globals -- and the slow thread's real ``torch.load`` then
          dies with ``UnpicklingError`` wrapped as ``RuntimeError: Failed to
          load checkpoint``. Verified: that is exactly what HEAD~ produced.
        * Locked, the other thread is parked on ``_SAFE_GLOBALS_LOCK`` and never
          reaches its window, so the wait times out and both loads succeed.
        """
        ckpt = self._datetime_checkpoint(tmp_path)
        original_load = torch.load
        slow_registered = threading.Event()
        slow_inside_window = threading.Event()
        other_done = threading.Event()
        slow_ident: dict[str, int] = {}
        results: dict[str, str] = {}

        def patched(*args: Any, **kwargs: Any) -> Any:
            if threading.get_ident() == slow_ident.get("id"):
                slow_inside_window.set()
                other_done.wait(_WINDOW_HANDSHAKE_TIMEOUT_S)
            return original_load(*args, **kwargs)

        def slow() -> None:
            slow_ident["id"] = threading.get_ident()
            slow_registered.set()
            results["slow"] = _load_outcome(ckpt)

        def other() -> None:
            slow_inside_window.wait(_THREAD_JOIN_TIMEOUT_S)
            results["other"] = _load_outcome(ckpt)
            other_done.set()

        with patch("src.training.checkpoint.torch.load", side_effect=patched):
            t_slow = threading.Thread(target=slow, name="slow-load")
            t_other = threading.Thread(target=other, name="other-load")
            t_slow.start()
            assert slow_registered.wait(_THREAD_JOIN_TIMEOUT_S), "slow thread never started"
            t_other.start()
            t_slow.join(_THREAD_JOIN_TIMEOUT_S)
            t_other.join(_THREAD_JOIN_TIMEOUT_S)

        assert not t_slow.is_alive() and not t_other.is_alive(), (
            "a load thread never finished -- the serialization lock deadlocked"
        )
        assert results == {"slow": "ok", "other": "ok"}, (
            f"a concurrent load tore down the in-flight allowlist window: {results}"
        )

    def test_nested_load_does_not_deadlock(self, tmp_path: Path) -> None:
        """Re-entering the window from the same thread must not hang.

        Nothing re-enters :func:`load_torch_checkpoint` today -- the guarded
        region is a single ``torch.load`` and ``weights_only=True`` cannot
        execute arbitrary code -- which is why a plain ``threading.Lock`` would
        also be *correct*. It would not be *safe*: any future nested load would
        become an unkillable self-deadlock. ``RLock`` makes that case terminate.

        The assertion is deliberately only "it terminates". ``RLock`` does not
        make nesting semantically correct: the inner ``safe_globals.__exit__``
        still strips the allowlist from the outer, in-flight load. If a nested
        load is ever introduced, this test is the place to tighten.
        """
        ckpt = self._datetime_checkpoint(tmp_path)
        original_load = torch.load
        depth = {"n": 0}
        outcome: dict[str, str] = {}

        def patched(*args: Any, **kwargs: Any) -> Any:
            depth["n"] += 1
            if depth["n"] == 1:
                # Re-enter the guarded region from this same thread.
                _load_outcome(ckpt)
            return original_load(*args, **kwargs)

        def run() -> None:
            outcome["result"] = _load_outcome(ckpt)

        with patch("src.training.checkpoint.torch.load", side_effect=patched):
            thread = threading.Thread(target=run, name="nested-load")
            thread.start()
            thread.join(_THREAD_JOIN_TIMEOUT_S)

        assert not thread.is_alive(), (
            "a nested load deadlocked -- _SAFE_GLOBALS_LOCK must stay re-entrant"
        )
        assert depth["n"] >= 2, "the nested load never happened; the test is vacuous"
        assert "result" in outcome


class TestSafeGlobalsAllowlist:
    """The narrow ``weights_only=True`` allowlist that replaces the fallback.

    ``BaseTrainer.save_checkpoint`` stores ``config.model_dump()``, and
    ``BaseModuleConfig.created_at`` is a ``datetime``, so that one first-party
    payload genuinely does not load under a bare ``weights_only=True``. The fix
    is to admit the three pure-data datetime constructors it needs -- not to
    relax the whole file to ``weights_only=False``.
    """

    def test_allowlist_is_exactly_the_datetime_trio(self) -> None:
        """Pin the allowlist so widening it is a deliberate, reviewed act.

        Each entry must be a pure data constructor. Adding anything callable
        that can reach other code (``os.system``, ``builtins.eval``,
        ``subprocess.Popen``) would reopen the hole this module just closed.
        """
        assert set(SAFE_CHECKPOINT_GLOBALS) == {datetime, timezone, timedelta}

    def test_base_trainer_checkpoint_round_trips_under_safe_default(self, tmp_path: Path) -> None:
        """The regression the allowlist exists for: resume must still work.

        Without the allowlist this raises, and ``BaseTrainer`` checkpoints
        become unloadable -- which is exactly the breakage that would tempt
        someone to reinstate the unsafe fallback.
        """
        trainer = _minimal_trainer(tmp_path)
        trainer.global_step = 7
        path = trainer.save_checkpoint()

        restored = _minimal_trainer(tmp_path)
        assert restored.load_checkpoint(path) == 7

    def test_allowlist_does_not_admit_the_payload(self, tmp_path: Path) -> None:
        """Widening for datetime must not widen for anything else.

        A checkpoint carrying *both* a legitimate ``datetime`` config and a
        malicious global proves the allowlist is a whitelist, not a switch that
        loosens the unpickler generally.
        """
        trainer = _minimal_trainer(tmp_path)
        marker = tmp_path / "PAYLOAD_EXECUTED"
        ckpt = tmp_path / "mixed.pt"
        torch.save(
            {
                "global_step": 1,
                "config": trainer.config.model_dump(),  # contains a real datetime
                "payload": MarkerPickle(marker),
            },
            ckpt,
        )

        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            trainer.load_checkpoint(ckpt)

        assert not marker.exists(), "allowlist admitted a non-allowlisted global"
