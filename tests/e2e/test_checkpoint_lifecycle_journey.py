"""E2E: trainer -> checkpoint -> inspector -> evaluator, through the shipped tools.

Guards CLAUDE.md's *"Trained evaluator"* and *"Checkpoint deserialization safety"*
Regression Surface rows.

What this adds over the existing suites: ``tests/security/test_checkpoint_safety.py``
and ``tests/scripts/test_cli_pickle_flags.py`` both exercise the loaders
**in-process**. Neither observes the exit code a shell sees -- which is the whole
signal a CI step or a scripting caller reads, and which
``scripts/inspect_checkpoint.py`` got wrong: it caught every exception, printed
``Error:``, and returned ``None``, so it exited **0 whether the checkpoint
deserialized or not**. That is fixed; these tests pin the fixed contract.

Scope, stated because it is narrower than "the checkpoint lifecycle" sounds:
the shipped ``train_fast`` config trains a **Go** model, so the consumer
exercised here is the Go evaluator (``scripts/evaluate_model.py``). The PDE
trained-evaluator path (``AlphaGalerkinSolver(evaluator="trained")``) needs a
checkpoint with the basis-selection encoding, and **no shipped command produces
one** -- see ``test_pde_game_training_is_a_known_gap`` below, which pins that
finding rather than leaving it as a comment.

Device: the training run and the evaluator are both pinned to ``e2e_device``, so
this file runs unchanged on a CUDA host. The inspector deliberately reads at
``map_location="cpu"``, which on a CUDA host makes this a genuine cross-device
round trip: trained on the accelerator, inspected on the host.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Final

import pytest

from tests.e2e.conftest import (
    E2E_BENCHMARK_TIMEOUT_S,
    E2E_TRAINING_TIMEOUT_S,
    CLIRunnerType,
)

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

#: Hydra config the fixture trains with. The shipped "fast" preset.
TRAIN_CONFIG_NAME: Final[str] = "train_fast"

#: Experiment name declared by ``config/train_fast.yaml``; the trainer writes
#: checkpoints to ``<checkpoint_dir>/<experiment_name>/``.
TRAIN_EXPERIMENT_NAME: Final[str] = "alphagalerkin_fast_test"

#: Minimal training budget. ``checkpoint_interval`` is lowered too, but a final
#: checkpoint is written regardless of it (``src/training/trainer.py``), so the
#: fixture does not depend on the interval dividing the step count.
TRAIN_TOTAL_STEPS: Final[str] = "2"
TRAIN_SELF_PLAY_GAMES: Final[str] = "1"
TRAIN_CHECKPOINT_INTERVAL: Final[str] = "1"
TRAIN_MCTS_SIMULATIONS: Final[str] = "2"

#: Evaluation budget for the Go consumer. Smallest values that still play a game.
EVAL_N_GAMES: Final[str] = "1"
EVAL_BOARD_SIZE: Final[str] = "9"
EVAL_N_SIMS: Final[str] = "2"
EVAL_N_POSITIONS: Final[str] = "2"

#: Key every checkpoint this repo writes must carry, and the one the migration
#: registry dispatches on.
CHECKPOINT_VERSION_KEY: Final[str] = "version"

#: Marker written by the hostile payload. Its absence after a refused load is
#: what proves nothing executed; the exit code alone would not.
PAYLOAD_MARKER: Final[str] = "e2e_pickle_marker"


def _hostile_payload(path: Path, marker: Path) -> None:
    """Write a pickle that creates *marker* if it is ever executed.

    Args:
        path: Destination ``.pt`` file.
        marker: File the payload would create. Must not exist beforehand.

    """
    import os

    class _Payload:
        def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
            return (os.makedirs, (str(marker),))

    path.write_bytes(pickle.dumps(_Payload()))


@pytest.fixture(scope="module")
def trained_checkpoint(
    cli_runner: CLIRunnerType,
    e2e_device: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Train for two steps and return the checkpoint the trainer wrote.

    ``checkpoint_dir`` is overridden -- **not** ``hydra.run.dir``, which does not
    relocate checkpoints: ``config/train.yaml``'s ``checkpoint_dir`` resolves
    against the process cwd, so without this override the run would write
    ``checkpoints/`` into the working tree and violate the tier's
    "every output under tmp_path" rule.

    Fails loudly rather than skipping when no checkpoint appears: a skip here
    would silently retire every test in the file.

    Returns:
        Path to a written ``.pt`` checkpoint.

    """
    checkpoint_dir = tmp_path_factory.mktemp("checkpoints")
    result = cli_runner(
        "scripts.train",
        [
            f"--config-name={TRAIN_CONFIG_NAME}",
            f"training.total_steps={TRAIN_TOTAL_STEPS}",
            f"training.n_self_play_games={TRAIN_SELF_PLAY_GAMES}",
            f"training.checkpoint_interval={TRAIN_CHECKPOINT_INTERVAL}",
            f"mcts.n_simulations={TRAIN_MCTS_SIMULATIONS}",
            f"checkpoint_dir={checkpoint_dir}",
            f"device={e2e_device}",
        ],
        E2E_TRAINING_TIMEOUT_S,
        None,
    )
    if result.returncode != 0:
        pytest.fail(f"scripts.train did not exit 0:\n{result.output[-3000:]}")

    written = sorted((checkpoint_dir / TRAIN_EXPERIMENT_NAME).glob("*.pt"))
    if not written:
        pytest.fail(f"scripts.train exited 0 but wrote no checkpoint under {checkpoint_dir}")
    return written[0]


def test_trainer_writes_a_versioned_checkpoint(trained_checkpoint: Path) -> None:
    """The trainer's output loads through the safe loader and carries a version.

    ``map_location="cpu"`` is explicit: on a CUDA host the tensors were saved on
    the accelerator, and loading them without it would make the assertion depend
    on which process holds a CUDA context.
    """
    from src.training.checkpoint import load_torch_checkpoint

    payload = load_torch_checkpoint(trained_checkpoint, map_location="cpu")
    assert isinstance(payload, dict)
    assert CHECKPOINT_VERSION_KEY in payload
    assert "model_state_dict" in payload


def test_inspect_checkpoint_exits_zero_and_reports_keys(
    trained_checkpoint: Path,
    cli_runner: CLIRunnerType,
) -> None:
    """The inspector reads a real checkpoint, exits 0, and prints its keys."""
    result = cli_runner(
        "scripts.inspect_checkpoint",
        [str(trained_checkpoint)],
        E2E_BENCHMARK_TIMEOUT_S,
        None,
    )
    assert result.returncode == 0, result.output
    assert "Keys:" in result.output
    assert CHECKPOINT_VERSION_KEY in result.output


def test_inspect_checkpoint_exits_nonzero_on_a_payload_it_refuses(
    cli_runner: CLIRunnerType,
    tmp_path: Path,
) -> None:
    """A file the safe loader refuses exits non-zero, and nothing executes.

    Both halves matter. The exit code is the part that was broken -- the script
    reported ``Error:`` and exited 0 -- and it is the only signal a shell reads.
    The marker check is what proves the refusal was real rather than a message
    printed after the payload had already run.

    The ``allow_unsafe_pickle`` escape hatch is deliberately NOT exercised here;
    proving it still deserializes is ``tests/security/test_checkpoint_safety.py``'s
    job, and duplicating it would put a second copy of that decision in a file
    whose subject is the exit code.
    """
    hostile = tmp_path / "hostile.pt"
    marker = tmp_path / PAYLOAD_MARKER
    _hostile_payload(hostile, marker)
    assert not marker.exists()

    result = cli_runner(
        "scripts.inspect_checkpoint",
        [str(hostile)],
        E2E_BENCHMARK_TIMEOUT_S,
        None,
    )
    assert result.returncode != 0, (
        "inspect_checkpoint exited 0 on a checkpoint it could not load -- "
        "the defect this test exists for"
    )
    assert not marker.exists(), "the payload executed despite the safe loader"


def test_evaluate_model_consumes_the_trained_checkpoint(
    trained_checkpoint: Path,
    cli_runner: CLIRunnerType,
    e2e_device: str,
) -> None:
    """The shipped Go evaluator runs against the shipped trainer's output.

    ``--device`` is always passed: ``scripts/evaluate_model.py`` defaults it to
    ``"cuda" if torch.cuda.is_available() else "cpu"`` at argv-construction time,
    so on a CUDA host an unpinned run would silently use the accelerator
    regardless of ``E2E_DEVICE``.

    Note the flag is ``--model``, not ``--checkpoint``.
    """
    result = cli_runner(
        "scripts.evaluate_model",
        [
            "--model",
            str(trained_checkpoint),
            "--n-games",
            EVAL_N_GAMES,
            "--board-size",
            EVAL_BOARD_SIZE,
            "--device",
            e2e_device,
            "--n-sims",
            EVAL_N_SIMS,
            "--n-positions",
            EVAL_N_POSITIONS,
        ],
        E2E_TRAINING_TIMEOUT_S,
        None,
    )
    assert result.returncode == 0, result.output


@pytest.mark.xfail(
    strict=True,
    reason=(
        "No shipped command produces a checkpoint the PDE trained-evaluator path can "
        "consume. `scripts/train.py +game=pde_basis` crashes in "
        "src/modeling/embeddings.py with a tensor-shape mismatch, because "
        "config/train_fast.yaml's operator.input_channels is Go-shaped (17) and nothing "
        "reconciles it with the selected game's encoding. Recorded as a strict xfail so "
        "it flips VISIBLY the moment that is fixed, rather than sitting in a comment."
    ),
)
def test_pde_game_training_is_a_known_gap(
    cli_runner: CLIRunnerType,
    e2e_device: str,
    tmp_path: Path,
) -> None:
    """Training with the PDE basis game should produce a usable checkpoint.

    It does not. This test states the intended contract and is expected to fail;
    ``strict=True`` means it also fails the build if it ever starts passing
    without this marker being removed.

    Separately noted while writing this: CLAUDE.md's Multi-Game Commands section
    documents ``python -m scripts.train game=go``, which errors -- ``game`` is
    not a key in ``config/train.yaml``, so Hydra requires ``+game=go``.
    """
    result = cli_runner(
        "scripts.train",
        [
            f"--config-name={TRAIN_CONFIG_NAME}",
            "+game=pde_basis",
            f"training.total_steps={TRAIN_TOTAL_STEPS}",
            f"training.n_self_play_games={TRAIN_SELF_PLAY_GAMES}",
            f"mcts.n_simulations={TRAIN_MCTS_SIMULATIONS}",
            f"checkpoint_dir={tmp_path}",
            f"device={e2e_device}",
        ],
        E2E_TRAINING_TIMEOUT_S,
        None,
    )
    assert result.returncode == 0, result.output[-2000:]
