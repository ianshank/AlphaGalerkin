"""Every trainer must be able to read back a checkpoint it just wrote.

This is a security test file rather than a correctness one because of *how* the
failures it guards were introduced and how they were found.

From torch 2.6 ``torch.load`` defaults to ``weights_only=True``, whose unpickler
rejects any global it was not told about. A config that keeps a non-primitive --
an ``Enum`` member, a ``datetime``, a ``Path``, a raw dataclass instance -- makes
the *safe* load fail on a *valid* checkpoint. That is not merely a bug: it is the
condition that turns an unsafe fallback into the normal path, which is exactly
how ``scripts/decode_video.py`` came to unpickle untrusted files as routine
behaviour. A loader whose safe path cannot succeed is a loader whose unsafe path
becomes routine.

Three of these were live at once (``VideoCompressionTrainer``, ``OperatorTrainer``,
``DistributedTrainer``) and none was caught by the existing suite -- the
distributed one was actively *masked* by a module-level
``torch.serialization.add_safe_globals`` in its own test file. So the guard here
is a real round-trip per writer, with no process-global registration anywhere in
this module, deliberately.

Add a case here whenever a class gains a ``save_checkpoint``.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from src.training.checkpoint import load_torch_checkpoint

pytestmark = pytest.mark.security

REPO_ROOT = Path(__file__).resolve().parents[2]

# The single audited unrestricted-pickle call: ``load_torch_checkpoint``'s
# ``allow_unsafe_pickle`` branch, which logs a warning before deserializing.
# Recorded without a line number -- pinning one would make this test fail on
# any edit above it, training the next reader to update the expectation
# rather than to ask why it moved.
ALLOWED_UNSAFE_LOAD_FILE = "src/training/checkpoint.py"


def test_no_unrestricted_pickle_load_outside_the_opt_in() -> None:
    """The repo-wide claim in `checkpoint.py`'s Security Note, machine-checked.

    That note documents a grep for a reader to run. A documented grep is a claim
    that rots the moment someone adds a loader, and the round-3/round-4 history
    is precisely a sequence of such claims going stale between reviews. So the
    invariant is asserted here instead: exactly one `torch.load` in the shipped
    tree may pass `weights_only=False`, and it is the audited opt-in branch.

    Parsed with `ast` rather than grepped, so docstrings and `--help` strings
    that merely *describe* the flag do not count -- there are a dozen of those,
    and a substring grep makes the check look failed when it has not.
    """
    roots = ("src", "scripts", "dashboard")
    offenders: list[str] = []

    for root in roots:
        for py in (REPO_ROOT / root).rglob("*.py"):
            tree = ast.parse(py.read_text(), filename=str(py))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (isinstance(func, ast.Attribute) and func.attr == "load"):
                    continue
                if not (isinstance(func.value, ast.Name) and func.value.id == "torch"):
                    continue
                for kw in node.keywords:
                    if kw.arg == "weights_only" and (
                        isinstance(kw.value, ast.Constant) and kw.value.value is False
                    ):
                        offenders.append(str(py.relative_to(REPO_ROOT)))

    assert offenders == [ALLOWED_UNSAFE_LOAD_FILE], (
        "unrestricted pickle loads outside the audited opt-in: "
        f"{sorted(set(offenders) - {ALLOWED_UNSAFE_LOAD_FILE})}"
    )


def _assert_loads(payload: dict[str, Any], tmp_path: Path, **load_kw: Any) -> dict[str, Any]:
    """Save a payload and load it exactly as production does."""
    path = tmp_path / "roundtrip.pt"
    torch.save(payload, path)
    loaded = load_torch_checkpoint(path, map_location="cpu", **load_kw)
    assert isinstance(loaded, dict)
    return loaded


class TestOperatorTrainerRoundTrip:
    """`src/training/operator_trainer.py` — stored a raw dataclass and a `Path`."""

    def test_config_payload_contains_only_primitives(self) -> None:
        """The fix is in what is *written*, so assert that rather than the load.

        Stringifying at save time means the payload needs no allowlist entry,
        which is strictly better than admitting a first-party class to a
        process-global window.
        """
        from src.training.operator_trainer import TrainingConfig, _config_to_plain_data

        plain = _config_to_plain_data(TrainingConfig())
        assert not is_dataclass(plain)
        for key, value in plain.items():
            assert isinstance(value, str | int | float | bool | type(None) | list | dict), (
                f"{key} is {type(value).__name__}, which weights_only=True will reject"
            )

    def test_round_trips_through_the_chokepoint(self, tmp_path: Path) -> None:
        from src.training.operator_trainer import TrainingConfig, _config_to_plain_data

        loaded = _assert_loads(
            {
                "epoch": 1,
                "model_state_dict": {"w": torch.zeros(2)},
                "optimizer_state_dict": {},
                "best_val_loss": 0.5,
                "history": {},
                "config": _config_to_plain_data(TrainingConfig()),
            },
            tmp_path,
        )
        assert loaded["config"]["checkpoint_dir"]

    def test_the_old_payload_shape_still_fails(self, tmp_path: Path) -> None:
        """Pins the defect, so the fix cannot be reverted silently.

        Saving the dataclass instance is what used to happen; if this ever starts
        passing, torch's default changed and the guard needs re-examining rather
        than deleting.
        """
        from src.training.operator_trainer import TrainingConfig

        path = tmp_path / "old.pt"
        torch.save({"config": TrainingConfig(), "model_state_dict": {}}, path)
        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            load_torch_checkpoint(path, map_location="cpu")

    def test_asdict_alone_would_not_have_been_enough(self, tmp_path: Path) -> None:
        """`asdict` leaves `checkpoint_dir` a `PosixPath`, which is still rejected.

        Worth pinning because `asdict` is the obvious fix and it does not work --
        the first attempt at this used it and still failed.
        """
        from src.training.operator_trainer import TrainingConfig

        path = tmp_path / "asdict.pt"
        torch.save({"config": asdict(TrainingConfig()), "model_state_dict": {}}, path)
        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            load_torch_checkpoint(path, map_location="cpu")


class TestDistributedTrainerRoundTrip:
    """`src/distributed/trainer.py` — an enum inside `distributed_config`."""

    def test_safe_distributed_globals_is_complete(self) -> None:
        """Reflect over the module rather than restating the list."""
        import enum

        from src.distributed import config as dist_config
        from src.distributed.config import SAFE_DISTRIBUTED_GLOBALS

        defined = {
            obj
            for name in dir(dist_config)
            if isinstance(obj := getattr(dist_config, name), type)
            and issubclass(obj, enum.Enum)
            and obj.__module__ == dist_config.__name__
        }
        assert defined == set(SAFE_DISTRIBUTED_GLOBALS)

    def test_round_trips_with_the_allowlist(self, tmp_path: Path) -> None:
        from src.distributed.config import SAFE_DISTRIBUTED_GLOBALS, DistributedConfig

        loaded = _assert_loads(
            {
                "step": 1,
                "model_state_dict": {"w": torch.zeros(2)},
                "optimizer_state_dict": {},
                "distributed_config": DistributedConfig().model_dump(),
            },
            tmp_path,
            extra_safe_globals=SAFE_DISTRIBUTED_GLOBALS,
        )
        assert loaded["distributed_config"]["backend"]

    def test_is_rejected_without_the_allowlist(self, tmp_path: Path) -> None:
        """Proves the allowlist is load-bearing, not decorative.

        This is the assertion the suite lacked: the production loader was broken
        for months while tests passed, because a test module registered the enum
        process-wide at import. Nothing in *this* file may do that.

        The explicit de-registration below is not paranoia. Any module in the
        same pytest process can register globals at import time, and this
        assertion is exactly the kind that such a registration silently
        satisfies -- reproducing, inside the test suite, the failure mode the
        test exists to prevent. Depending on *not* being masked is not the same
        as asserting it, so the precondition is established rather than assumed.

        (Empirically it is usually already absent even when another module
        registers it, because ``safe_globals.__exit__`` removes by value and so
        clobbers the pre-existing registration -- see the narrowing-leak note in
        ``src/training/checkpoint.py``. That is an accident of ordering, not a
        guarantee, which is the point.)
        """
        from src.distributed.config import DistributedBackend, DistributedConfig

        torch.serialization._weights_only_unpickler._remove_safe_globals([DistributedBackend])

        path = tmp_path / "dist.pt"
        torch.save({"distributed_config": DistributedConfig().model_dump()}, path)
        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            load_torch_checkpoint(path, map_location="cpu")


class TestVideoCompressionTrainerRoundTrip:
    """`src/video_compression/training/trainer.py` — a `datetime`, and *only* that."""

    def test_round_trips_on_the_base_allowlist_alone(self, tmp_path: Path) -> None:
        """No `extra_safe_globals`, deliberately.

        The config this trainer holds is `video_compression.config.TrainingConfig`,
        not `CodecConfig`: its only non-primitive is `created_at`, already covered
        by `SAFE_CHECKPOINT_GLOBALS`. Passing `SAFE_CODEC_GLOBALS` here would widen
        a process-global window for globals the payload never contains.
        """
        from src.video_compression.config import TrainingConfig

        loaded = _assert_loads(
            {
                "step": 1,
                "epoch": 0,
                "best_rd_loss": 0.3,
                "model_state": {"w": torch.zeros(2)},
                "optimizer_state": {},
                "scheduler_state": {},
                "config": TrainingConfig(name="rt").model_dump(),
            },
            tmp_path,
        )
        assert loaded["config"]["name"] == "rt"

    def test_its_config_carries_no_enum(self) -> None:
        """Pins the claim the code comment makes, so the two cannot drift.

        A comment previously asserted this payload embedded `CodecConfig` and
        "mode fields are enums". It does not, and the extras it justified were
        unnecessary.
        """
        from src.video_compression.config import TrainingConfig

        dump = TrainingConfig(name="rt").model_dump()
        non_primitive = {
            k: type(v).__name__
            for k, v in dump.items()
            if not isinstance(v, str | int | float | bool | type(None) | list | dict)
        }
        assert non_primitive == {"created_at": "datetime"}
