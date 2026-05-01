"""Filesystem (and optional GCS) storage for trained zoo entries.

The codec zoo stores one directory per entry under ``storage_root``,
each containing:

- ``checkpoint.pt`` — model + optimizer + scheduler state.
- ``entry.json`` — :class:`ModelZooEntryConfig` snapshot at training
  time (so the manifest can be regenerated from artifacts alone).
- ``metrics.json`` — final training metrics (loss, bpp, psnr, ms_ssim,
  etc.).

This module deliberately does **not** subclass
:class:`src.distributed.model_zoo.ModelZoo`: that registry is built
around AlphaZero curriculum semantics (win rate, ELO, best-model
selection) which do not apply to a static R-D grid. The two registries
remain orthogonal; nothing prevents wrapping them together later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
import torch

from src.video_compression.zoo.config import (
    ModelZooEntryConfig,
    StorageBackend,
)

logger = structlog.get_logger(__name__)

CHECKPOINT_FILENAME: str = "checkpoint.pt"
ENTRY_FILENAME: str = "entry.json"
METRICS_FILENAME: str = "metrics.json"


@dataclass(frozen=True)
class EntryArtifacts:
    """Locator + metrics for a trained zoo entry."""

    entry_id: str
    entry_dir: Path
    checkpoint_path: Path
    metrics: dict[str, float]
    saved_at: str  # ISO-8601


class VideoCodecZoo:
    """Filesystem-backed registry of trained zoo entries.

    Args:
        storage_root: Local directory (filesystem backend) or
            ``gs://bucket/prefix`` URI (GCS backend, opt-in).
        backend: Storage backend. GCS support requires the optional
            ``[vertex]`` extra.

    """

    def __init__(
        self,
        storage_root: str | Path,
        *,
        backend: StorageBackend = StorageBackend.FILESYSTEM,
    ) -> None:
        self.backend = backend
        if backend is StorageBackend.FILESYSTEM:
            self.root = Path(storage_root)
            self.root.mkdir(parents=True, exist_ok=True)
        elif backend is StorageBackend.GCS:
            # Defer the import so the optional extra is only required
            # when GCS is explicitly requested. ``src.vertex.storage``
            # is the canonical GCS-backed checkpoint module; we import
            # it for its side-effect of failing loud when the
            # ``[vertex]`` extra is missing.
            import importlib

            importlib.import_module("src.vertex.storage")
            self.root = Path(str(storage_root))  # treated as URI
        else:  # pragma: no cover - exhaustive enum
            raise ValueError(f"unsupported backend: {backend!r}")

        self._log = logger.bind(
            component="VideoCodecZoo",
            backend=backend.value,
            storage_root=str(self.root),
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def entry_dir(self, entry_id: str) -> Path:
        return self.root / entry_id

    def checkpoint_path(self, entry_id: str) -> Path:
        return self.entry_dir(entry_id) / CHECKPOINT_FILENAME

    def metrics_path(self, entry_id: str) -> Path:
        return self.entry_dir(entry_id) / METRICS_FILENAME

    def has_entry(self, entry_id: str) -> bool:
        return self.checkpoint_path(entry_id).exists()

    def list_entries(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            d.name for d in self.root.iterdir()
            if d.is_dir() and (d / CHECKPOINT_FILENAME).exists()
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save_entry(
        self,
        entry: ModelZooEntryConfig,
        state_dict: dict[str, Any],
        metrics: dict[str, float],
    ) -> EntryArtifacts:
        """Persist a trained entry's checkpoint + metadata.

        Args:
            entry: The entry config that was trained.
            state_dict: Torch state-dict bundle. Conventionally:
                ``{"model": ..., "optimizer": ..., "scheduler": ...,
                "step": int, "lambda_rd": float, "config": ...}``.
            metrics: Final training metrics (must be JSON-serializable
                floats).

        Returns:
            :class:`EntryArtifacts` describing the saved location.

        """
        if self.backend is not StorageBackend.FILESYSTEM:
            raise NotImplementedError(
                f"save_entry is implemented only for filesystem; got "
                f"{self.backend!r}",
            )

        entry_dir = self.entry_dir(entry.entry_id)
        entry_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = entry_dir / CHECKPOINT_FILENAME
        torch.save(state_dict, ckpt_path)

        with (entry_dir / ENTRY_FILENAME).open("w", encoding="utf-8") as fh:
            json.dump(entry.to_yaml_dict(), fh, indent=2, sort_keys=True, default=str)

        # Validate metrics are JSON-friendly floats.
        clean: dict[str, float] = {}
        for k, v in metrics.items():
            if not isinstance(k, str):
                raise TypeError(f"metric key must be str; got {type(k).__name__}")
            clean[k] = float(v)

        saved_at = datetime.now(timezone.utc).isoformat()
        with (entry_dir / METRICS_FILENAME).open("w", encoding="utf-8") as fh:
            json.dump(
                {"metrics": clean, "saved_at": saved_at},
                fh,
                indent=2,
                sort_keys=True,
            )

        self._log.info(
            "zoo.entry.saved",
            entry_id=entry.entry_id,
            lambda_rd=entry.lambda_rd,
            metrics=clean,
        )
        return EntryArtifacts(
            entry_id=entry.entry_id,
            entry_dir=entry_dir,
            checkpoint_path=ckpt_path,
            metrics=clean,
            saved_at=saved_at,
        )

    def load_state_dict(
        self,
        entry_id: str,
        *,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """Load the raw state-dict bundle for an entry."""
        path = self.checkpoint_path(entry_id)
        if not path.exists():
            raise FileNotFoundError(
                f"no checkpoint for entry_id={entry_id!r} at {path}",
            )
        bundle = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(bundle, dict):
            raise TypeError(
                f"checkpoint at {path} is not a dict; got "
                f"{type(bundle).__name__}",
            )
        return bundle

    def load_metrics(self, entry_id: str) -> dict[str, float]:
        path = self.metrics_path(entry_id)
        if not path.exists():
            raise FileNotFoundError(f"no metrics for entry_id={entry_id!r}")
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            raise TypeError(
                f"metrics for entry_id={entry_id!r} is not a dict",
            )
        return {str(k): float(v) for k, v in metrics.items()}
