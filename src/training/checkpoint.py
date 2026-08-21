"""Checkpoint management for training state persistence.

Provides save/load functionality for:
- Model weights
- Optimizer state
- Learning rate scheduler state
- Training step and metrics
- Configuration

Security Note:
    Every loader in this module deserializes with `weights_only=True` by
    default. That is not a compromise: it covers every checkpoint AlphaGalerkin
    writes. `save_model_only()`, `CheckpointManager.save()` (optimizer +
    scheduler + `config.model_dump()` + metrics) and
    `BaseTrainer._save_training_state()` (including AMP `GradScaler` state)
    contain only tensors, plain scalars/strings, `dict`/`list` and
    `OrderedDict`, which the `weights_only` unpickler accepts unaided --
    `src/distributed/` has loaded full training state this way all along.

    `BaseTrainer.save_checkpoint()` is the one exception. It stores
    `config.model_dump()`, and `templates.config.BaseModuleConfig` carries a
    `created_at: datetime` field, so the payload references `datetime.datetime`,
    `datetime.timezone` and `datetime.timedelta`. Those three -- and *only*
    those three, enumerated by walking the pickle opcodes of every first-party
    save path -- are admitted via `torch.serialization.safe_globals`
    (`SAFE_CHECKPOINT_GLOBALS`). They are pure data constructors: they build a
    value from its arguments and cannot invoke anything else. Allowlisting them
    keeps `weights_only=True` in force, which is strictly better than relaxing
    to `weights_only=False` for the whole file.

    **A failed safe load is never a reason to retry unsafely.** Until
    2026-08-21 `load_model_only()` caught *any* exception from the
    `weights_only=True` load and retried with `weights_only=False`, so a
    malicious pickle was executed *precisely because* it failed the safety
    check -- a demonstrated arbitrary-code-execution path. There is no
    automatic fallback anywhere in this module any more: a checkpoint that
    fails the safe load is rejected with `RuntimeError`.

    Unrestricted pickle deserialization is still reachable, but only by an
    explicit, per-call `allow_unsafe_pickle=True` -- for a genuinely legacy or
    third-party file whose provenance the *operator* has established. No
    first-party checkpoint needs it. It cannot be reached by a checkpoint's own
    contents.

    `CheckpointManager.load()` additionally bounds *which* files can be opened
    at all: a caller-supplied path is resolved against `checkpoint_dir` and
    must stay inside it, so an untrusted path (e.g. `"../../../etc/shadow"`)
    raises `ValueError`. Loading from elsewhere is an explicit opt-in via
    `allow_external=True`.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.constants import CHECKPOINT_BEST

if TYPE_CHECKING:
    from config.schemas import AlphaGalerkinConfig

logger = structlog.get_logger(__name__)

# Checkpoint format version for compatibility checking
CHECKPOINT_VERSION = "1.1.0"

# The complete set of non-tensor globals appearing in any checkpoint this repo
# writes, obtained by disassembling the pickle stream of every first-party save
# path (``BaseTrainer.save_checkpoint`` / ``_save_training_state``,
# ``CheckpointManager.save``, ``save_model_only``). Only ``BaseTrainer``
# checkpoints need any of them, via ``BaseModuleConfig.created_at: datetime``.
#
# Each entry is a pure data constructor -- it builds a value from its arguments
# and calls nothing else -- so admitting it does not weaken ``weights_only=True``
# into a code-execution path the way ``weights_only=False`` would. Anything not
# on this list (``posix.system``, ``builtins.eval``, arbitrary classes) stays
# blocked. Keep this list minimal and re-derive it before adding to it.
SAFE_CHECKPOINT_GLOBALS: tuple[type, ...] = (datetime, timezone, timedelta)

# Appended to every safe-load failure so an operator hitting a genuinely legacy
# file learns the supported escape hatch instead of reaching for a source patch.
UNSAFE_PICKLE_HINT = (
    "Refusing to retry with weights_only=False: a checkpoint that fails the safe "
    "load is exactly the checkpoint that must not be unpickled. If this file is "
    "legacy or third-party and you have independently established its provenance, "
    "re-save it with a current AlphaGalerkin, or pass allow_unsafe_pickle=True."
)


def load_torch_checkpoint(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
    allow_unsafe_pickle: bool = False,
) -> Any:
    """Deserialize a checkpoint file, safe by default.

    The single ``torch.load`` chokepoint for AlphaGalerkin checkpoint loading;
    :mod:`src.training.base_trainer` routes through it too, so the policy is
    stated once rather than re-argued at each call site. ``weights_only=True``
    is the default and every AlphaGalerkin-written checkpoint loads under it,
    given the three pure-data ``SAFE_CHECKPOINT_GLOBALS`` (see the module
    Security Note). Failure of the safe load raises -- it never
    escalates to pickle deserialization, which is the inversion that made the
    old ``load_model_only`` fallback an arbitrary-code-execution path.

    Args:
        path: Checkpoint file to read. Callers are responsible for any path
            containment policy (see :meth:`CheckpointManager.load`).
        map_location: Device mapping forwarded to ``torch.load``.
        allow_unsafe_pickle: Deserialize with ``weights_only=False``. Only ever
            pass ``True`` for a file whose provenance a human operator has
            established; a malicious checkpoint executes arbitrary code at
            deserialization time. Never derive this from untrusted input, and
            never set it in response to a safe-load failure you did not
            diagnose.

    Returns:
        The deserialized checkpoint object.

    Raises:
        RuntimeError: If the checkpoint cannot be deserialized. Normalised so
            callers see one deterministic type instead of a platform- and
            format-dependent pickle error.

    """
    if allow_unsafe_pickle:
        logger.warning(
            "checkpoint_unsafe_pickle_enabled",
            path=str(path),
            hint="weights_only=False: arbitrary code may execute. Trusted sources only.",
        )
        return torch.load(path, map_location=map_location, weights_only=False)

    try:
        # Scoped rather than a process-global ``add_safe_globals``: the widened
        # allowlist applies to this load only and cannot leak into unrelated
        # ``torch.load`` calls elsewhere in the process.
        with torch.serialization.safe_globals(list(SAFE_CHECKPOINT_GLOBALS)):
            return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        logger.error("checkpoint_load_failed", path=str(path), error=str(e))
        raise RuntimeError(f"Failed to load checkpoint: {e}\n\n{UNSAFE_PICKLE_HINT}") from e


@dataclass
class CheckpointState:
    """Complete training state from a checkpoint.

    Attributes:
        step: Training step number.
        model_state_dict: Model weights.
        optimizer_state_dict: Optimizer state.
        scheduler_state_dict: LR scheduler state.
        config: Training configuration.
        metrics: Training metrics at checkpoint time.
        timestamp: When checkpoint was created.
        version: Checkpoint format version.

    """

    step: int
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None = None
    scheduler_state_dict: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    version: str = CHECKPOINT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step": self.step,
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "scheduler_state_dict": self.scheduler_state_dict,
            "config": self.config,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        """Create from dictionary."""
        # Migrate old checkpoint formats
        from src.training.checkpoint_migration import migrate_checkpoint

        data = migrate_checkpoint(data, CHECKPOINT_VERSION)

        return cls(
            step=data["step"],
            model_state_dict=data["model_state_dict"],
            optimizer_state_dict=data.get("optimizer_state_dict"),
            scheduler_state_dict=data.get("scheduler_state_dict"),
            config=data.get("config"),
            metrics=data.get("metrics", {}),
            timestamp=data.get("timestamp", ""),
            version=data.get("version", "0.0.0"),
        )


class CheckpointManager:
    """Manages saving and loading of training checkpoints.

    Features:
    - Automatic checkpoint naming with step numbers
    - Best model tracking
    - Checkpoint rotation (keep N most recent)
    - Atomic saves (write to temp, then rename)
    - Version compatibility checking
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        max_checkpoints: int = 5,
        keep_best: bool = True,
        best_metric: str = "loss",
        best_mode: str = "min",
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints.
            max_checkpoints: Maximum number of checkpoints to keep.
            keep_best: Whether to keep best checkpoint separately.
            best_metric: Metric to use for best model selection.
            best_mode: "min" or "max" for best metric comparison.

        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.best_metric = best_metric
        self.best_mode = best_mode

        self._best_value: float | None = None

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "checkpoint_manager_initialized",
            checkpoint_dir=str(self.checkpoint_dir),
            max_checkpoints=max_checkpoints,
        )

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        config: AlphaGalerkinConfig | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save a checkpoint.

        Args:
            step: Current training step.
            model: Model to save.
            optimizer: Optimizer state to save.
            scheduler: LR scheduler state to save.
            config: Configuration to save.
            metrics: Current training metrics.

        Returns:
            Path to saved checkpoint.

        """
        metrics = metrics or {}

        # Create checkpoint state
        state = CheckpointState(
            step=step,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict() if optimizer else None,
            scheduler_state_dict=scheduler.state_dict() if scheduler else None,
            config=config.model_dump() if config else None,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            version=CHECKPOINT_VERSION,
        )

        # Save checkpoint atomically
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step:08d}.pt"
        temp_path = checkpoint_path.with_suffix(".pt.tmp")

        torch.save(state.to_dict(), temp_path)
        temp_path.replace(checkpoint_path)  # Works on Windows even if target exists

        logger.info(
            "checkpoint_saved",
            path=str(checkpoint_path),
            step=step,
            metrics=metrics,
        )

        # Update best checkpoint if applicable
        if self.keep_best and self.best_metric in metrics:
            self._update_best(checkpoint_path, metrics[self.best_metric])

        # Rotate old checkpoints
        self._rotate_checkpoints()

        return checkpoint_path

    def load(
        self,
        path: Path | str | None = None,
        load_best: bool = False,
        allow_external: bool = False,
        allow_unsafe_pickle: bool = False,
    ) -> CheckpointState:
        """Load a checkpoint.

        Deserialization is ``weights_only=True`` (see the module Security Note).
        The path is additionally resolved against ``self.checkpoint_dir`` (relative
        paths are interpreted relative to it, not to the process CWD) and must
        resolve to a location inside that directory, so an untrusted ``path`` such
        as ``"../../../etc/shadow"`` is rejected rather than opened.

        Args:
            path: Specific checkpoint path (None for latest). Relative paths are
                resolved against ``checkpoint_dir``.
            load_best: Whether to load best checkpoint.
            allow_external: Permit a checkpoint outside ``checkpoint_dir`` (the
                explicit resume-from-elsewhere workflow). Only pass ``True`` for
                paths supplied by a trusted operator, never for attacker-influenced
                input.
            allow_unsafe_pickle: Deserialize with ``weights_only=False``. No
                checkpoint this repo writes needs it; see :func:`load_torch_checkpoint`.

        Returns:
            CheckpointState with loaded data.

        Raises:
            FileNotFoundError: If no checkpoint found.
            ValueError: If the path escapes ``checkpoint_dir`` while
                ``allow_external`` is False, or the checkpoint version is
                incompatible.
            RuntimeError: If the checkpoint cannot be deserialized.

        """
        if load_best:
            path = self.checkpoint_dir / CHECKPOINT_BEST
        elif path is None:
            path = self.get_latest()

        if path is None:
            raise FileNotFoundError("No checkpoint found")

        # Relative paths belong to this manager's directory, not the CWD.
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.checkpoint_dir / candidate

        # Path.resolve() is non-strict, so containment can be (and is) checked
        # before existence: a missing *in-dir* checkpoint still raises
        # FileNotFoundError, while an escaping path never reaches torch.load.
        resolved = candidate.resolve()
        root = self.checkpoint_dir.resolve()
        if not allow_external and not resolved.is_relative_to(root):
            logger.error(
                "checkpoint_path_outside_directory",
                path=str(resolved),
                checkpoint_dir=str(root),
            )
            raise ValueError(
                f"Checkpoint path {resolved} is outside checkpoint directory {root}. "
                "Pass allow_external=True to load a checkpoint from another location."
            )
        path = resolved

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Safe by default. load_torch_checkpoint normalises failures to RuntimeError and
        # never escalates a failed safe load into a pickle load.
        data = load_torch_checkpoint(
            path, map_location="cpu", allow_unsafe_pickle=allow_unsafe_pickle
        )

        # Check version compatibility
        version = data.get("version", "0.0.0")
        if not self._is_compatible(version):
            raise ValueError(
                f"Checkpoint version {version} is not compatible with "
                f"current version {CHECKPOINT_VERSION}"
            )

        state = CheckpointState.from_dict(data)

        logger.info(
            "checkpoint_loaded",
            path=str(path),
            step=state.step,
            version=state.version,
        )

        return state

    def restore(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        path: Path | str | None = None,
        load_best: bool = False,
        strict: bool = True,
        allow_external: bool = False,
        allow_unsafe_pickle: bool = False,
    ) -> int:
        """Restore model and optimizer from checkpoint.

        Convenience method that loads checkpoint and applies states.

        Args:
            model: Model to restore.
            optimizer: Optimizer to restore.
            scheduler: Scheduler to restore.
            path: Specific checkpoint path (relative paths resolve against
                ``checkpoint_dir``).
            load_best: Whether to load best checkpoint.
            strict: Whether to require exact model state match.
            allow_external: Permit a checkpoint outside ``checkpoint_dir``.
                Forwarded to :meth:`load` — see its security note.
            allow_unsafe_pickle: Forwarded to :meth:`load`. Defaults to safe.

        Returns:
            Training step from checkpoint.

        """
        state = self.load(
            path=path,
            load_best=load_best,
            allow_external=allow_external,
            allow_unsafe_pickle=allow_unsafe_pickle,
        )

        # Restore model
        model.load_state_dict(state.model_state_dict, strict=strict)

        # Restore optimizer
        if optimizer is not None and state.optimizer_state_dict is not None:
            optimizer.load_state_dict(state.optimizer_state_dict)

        # Restore scheduler
        if scheduler is not None and state.scheduler_state_dict is not None:
            scheduler.load_state_dict(state.scheduler_state_dict)

        logger.info(
            "training_state_restored",
            step=state.step,
            has_optimizer=state.optimizer_state_dict is not None,
            has_scheduler=state.scheduler_state_dict is not None,
        )

        return state.step

    def get_latest(self) -> Path | None:
        """Get path to latest checkpoint.

        Returns:
            Path to latest checkpoint, or None if none exist.

        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None

        # Sort by step number
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[1]))
        return checkpoints[-1]

    def get_all_checkpoints(self) -> list[Path]:
        """Get all checkpoint paths sorted by step.

        Returns:
            List of checkpoint paths.

        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[1]))
        return checkpoints

    def _update_best(self, checkpoint_path: Path, metric_value: float) -> None:
        """Update best checkpoint if metric improved.

        Args:
            checkpoint_path: Path to current checkpoint.
            metric_value: Current metric value.

        """
        is_better = False

        if (
            self._best_value is None
            or (self.best_mode == "min" and metric_value < self._best_value)
            or (self.best_mode == "max" and metric_value > self._best_value)
        ):
            is_better = True

        if is_better:
            self._best_value = metric_value
            best_path = self.checkpoint_dir / CHECKPOINT_BEST

            # Copy checkpoint to best.pt
            shutil.copy2(checkpoint_path, best_path)

            logger.info(
                "best_checkpoint_updated",
                metric=self.best_metric,
                value=metric_value,
            )

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints limit."""
        checkpoints = self.get_all_checkpoints()

        # Keep only max_checkpoints most recent
        if len(checkpoints) > self.max_checkpoints:
            for old_ckpt in checkpoints[: -self.max_checkpoints]:
                old_ckpt.unlink()
                logger.debug("old_checkpoint_removed", path=str(old_ckpt))

    def _is_compatible(self, version: str) -> bool:
        """Check if checkpoint version is compatible.

        Args:
            version: Checkpoint version string.

        Returns:
            True if compatible.

        """
        # Parse versions
        try:
            ckpt_major = int(version.split(".")[0])
            curr_major = int(CHECKPOINT_VERSION.split(".")[0])
            # Major version must match
            return ckpt_major == curr_major
        except (ValueError, IndexError):
            return False

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        """Save additional metadata to checkpoint directory.

        Args:
            metadata: Metadata dictionary.

        """
        metadata_path = self.checkpoint_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def load_metadata(self) -> dict[str, Any]:
        """Load metadata from checkpoint directory.

        Returns:
            Metadata dictionary (empty if not found).

        """
        metadata_path = self.checkpoint_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {}


def save_model_only(
    model: nn.Module,
    path: Path | str,
    config: AlphaGalerkinConfig | None = None,
) -> None:
    """Save only model weights (for deployment).

    Args:
        model: Model to save.
        path: Output path.
        config: Optional configuration.

    """
    state = {
        "model_state_dict": model.state_dict(),
        "config": config.model_dump() if config else None,
        "version": CHECKPOINT_VERSION,
        "timestamp": datetime.now().isoformat(),
    }

    torch.save(state, path)
    logger.info("model_saved", path=str(path))


def load_model_only(
    model: nn.Module,
    path: Path | str,
    strict: bool = True,
    allow_unsafe_pickle: bool = False,
) -> None:
    """Load only model weights.

    Deserialization is ``weights_only=True``. This function used to catch *any*
    exception from that safe load and retry with ``weights_only=False``, which
    meant a malicious pickle was executed *because* it tripped the safety check.
    There is no such fallback: a checkpoint that fails the safe load is rejected.

    Args:
        model: Model to load into.
        path: Checkpoint path.
        strict: Whether to require exact state match.
        allow_unsafe_pickle: Deserialize with ``weights_only=False``. Every
            checkpoint :func:`save_model_only` writes loads safely without it;
            see :func:`load_torch_checkpoint`.

    Raises:
        RuntimeError: If the checkpoint cannot be safely deserialized.

    """
    state = load_torch_checkpoint(path, map_location="cpu", allow_unsafe_pickle=allow_unsafe_pickle)
    model.load_state_dict(state["model_state_dict"], strict=strict)
    logger.info("model_loaded", path=str(path))


def load_checkpoint_with_config(
    path: Path | str,
    device: str = "cpu",
    allow_unsafe_pickle: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load checkpoint and extract configuration.

    This is the recommended way to load checkpoints when you need both
    model weights and the saved configuration. It handles:
    - Multiple checkpoint formats (full state vs weights-only)
    - Configuration extraction from checkpoint
    - Proper device mapping
    - Logging and error handling

    Args:
        path: Path to checkpoint file.
        device: Target device for loading (default: cpu for safety).
        allow_unsafe_pickle: Deserialize with ``weights_only=False``. Defaults
            to safe; see :func:`load_torch_checkpoint`.

    Returns:
        Tuple of (checkpoint_dict, config_dict or None).

    Raises:
        FileNotFoundError: If checkpoint path doesn't exist.
        RuntimeError: If checkpoint is corrupted or incompatible.

    Example:
        >>> checkpoint, config = load_checkpoint_with_config("model.pt")
        >>> if config and "operator" in config:
        ...     op_config = OperatorConfig(**config["operator"])
        >>> model.load_state_dict(checkpoint["model_state_dict"])

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info("loading_checkpoint", path=str(path), device=device)

    checkpoint = load_torch_checkpoint(
        path, map_location=device, allow_unsafe_pickle=allow_unsafe_pickle
    )

    # Extract config if present
    config = checkpoint.get("config")
    if config is not None:
        logger.debug("checkpoint_config_found", config_keys=list(config.keys()))
    else:
        logger.warning("checkpoint_config_not_found", path=str(path))

    return checkpoint, config


def create_model_from_checkpoint(
    path: Path | str,
    device: str = "cpu",
    model_class: type | None = None,
    config_class: type | None = None,
    strict: bool = True,
    allow_unsafe_pickle: bool = False,
) -> tuple[Any, dict[str, Any] | None]:
    """Create and load model from checkpoint with proper configuration.

    This is the highest-level utility for loading a trained model. It:
    1. Loads the checkpoint
    2. Extracts the configuration (or uses defaults)
    3. Creates the model with the correct architecture
    4. Loads the weights
    5. Moves to the target device

    Args:
        path: Path to checkpoint file.
        device: Target device (cuda, cpu, etc.).
        model_class: Model class to instantiate (default: AlphaGalerkinModel).
        config_class: Config class for model (default: OperatorConfig).
        strict: Whether to require exact state dict match.
        allow_unsafe_pickle: Deserialize with ``weights_only=False``. Defaults
            to safe. This function is on the trained-evaluator path
            (``FNetEvaluator``), so the safe default is what stops an
            attacker-supplied checkpoint from executing code at load time.

    Returns:
        Tuple of (loaded_model, config_dict or None).

    Raises:
        FileNotFoundError: If checkpoint doesn't exist.
        RuntimeError: If loading fails.

    Example:
        >>> model, config = create_model_from_checkpoint(
        ...     "checkpoints/best.pt",
        ...     device="cuda"
        ... )
        >>> model.eval()
        >>> output = model(input_tensor)

    """
    # Import here to avoid circular imports
    if model_class is None:
        from src.modeling.model import AlphaGalerkinModel

        model_class = AlphaGalerkinModel
    if config_class is None:
        from config.schemas import OperatorConfig

        config_class = OperatorConfig

    # Load checkpoint and config
    checkpoint, config_dict = load_checkpoint_with_config(
        path, device="cpu", allow_unsafe_pickle=allow_unsafe_pickle
    )

    # Create model config from checkpoint or use defaults
    if config_dict is not None and "operator" in config_dict:
        logger.info("using_checkpoint_config")
        try:
            model_config = config_class(**config_dict["operator"])
        except Exception as e:
            logger.warning(
                "checkpoint_config_parse_failed",
                error=str(e),
                fallback="default_config",
            )
            model_config = config_class()
    else:
        logger.info("using_default_config")
        model_config = config_class()

    # Create and load model
    model = model_class(model_config)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    else:
        # Legacy format: checkpoint is the state dict directly
        model.load_state_dict(checkpoint, strict=strict)

    # Move to target device and set to eval mode
    model.to(device)
    model.eval()

    logger.info(
        "model_created_from_checkpoint",
        path=str(path),
        device=device,
        config_source="checkpoint" if config_dict else "default",
    )

    return model, config_dict
