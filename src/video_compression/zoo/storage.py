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
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, SupportsFloat

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
            # Keep the original URI string verbatim. ``Path("gs://x/y")``
            # would normalize the double-slash to a single slash on
            # POSIX, mangling the URI. Phase 2-D will replace ``self.root``
            # with a dedicated URI wrapper; the str form is the safe
            # interim representation that preserves the original value
            # for logs and future path composition.
            self.root = str(storage_root)  # type: ignore[assignment]
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
        if self.backend is not StorageBackend.FILESYSTEM:
            raise NotImplementedError(
                f"has_entry is implemented only for the filesystem backend; got {self.backend!r}",
            )
        return self.checkpoint_path(entry_id).exists()

    def list_entries(self) -> list[str]:
        if self.backend is not StorageBackend.FILESYSTEM:
            raise NotImplementedError(
                f"list_entries is implemented only for the filesystem backend; "
                f"got {self.backend!r}",
            )
        if not self.root.exists():
            return []
        return sorted(
            d.name for d in self.root.iterdir() if d.is_dir() and (d / CHECKPOINT_FILENAME).exists()
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save_entry(
        self,
        entry: ModelZooEntryConfig,
        state_dict: dict[str, Any],
        metrics: Mapping[str, SupportsFloat],
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
                f"save_entry is implemented only for filesystem; got {self.backend!r}",
            )

        entry_dir = self.entry_dir(entry.entry_id)
        entry_dir.mkdir(parents=True, exist_ok=True)

        # Validate metrics are JSON-friendly floats first; we want to
        # fail before we touch the filesystem.
        clean: dict[str, float] = {}
        for k, v in metrics.items():
            if not isinstance(k, str):
                raise TypeError(f"metric key must be str; got {type(k).__name__}")
            clean[k] = float(v)

        # Atomic write: stage every artifact at a sibling ``.tmp`` path
        # then ``Path.replace`` it onto the canonical name. ``replace``
        # is atomic on POSIX and on Windows for same-volume moves, so
        # readers never observe a half-written checkpoint, entry.json,
        # or metrics.json.
        ckpt_path = entry_dir / CHECKPOINT_FILENAME
        ckpt_tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        torch.save(state_dict, ckpt_tmp)
        ckpt_tmp.replace(ckpt_path)

        entry_path = entry_dir / ENTRY_FILENAME
        entry_tmp = entry_path.with_suffix(entry_path.suffix + ".tmp")
        with entry_tmp.open("w", encoding="utf-8") as fh:
            json.dump(entry.to_yaml_dict(), fh, indent=2, sort_keys=True, default=str)
        entry_tmp.replace(entry_path)

        saved_at = datetime.now(timezone.utc).isoformat()
        metrics_path = entry_dir / METRICS_FILENAME
        metrics_tmp = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
        with metrics_tmp.open("w", encoding="utf-8") as fh:
            json.dump(
                {"metrics": clean, "saved_at": saved_at},
                fh,
                indent=2,
                sort_keys=True,
            )
        metrics_tmp.replace(metrics_path)

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
        weights_only: bool = True,
    ) -> dict[str, Any]:
        """Load the raw state-dict bundle for an entry.

        Security note:
            ``torch.load`` performs pickle deserialization and can
            execute arbitrary code when ``weights_only=False``. The
            default here is ``weights_only=True`` (matching torch ≥2.6
            guidance) so callers loading checkpoints from an untrusted
            ``storage_root`` stay safe. Pass ``weights_only=False``
            explicitly only when the bundle contains non-tensor objects
            (e.g. an entire optimizer/scheduler) and the source is
            known-trusted.

        """
        if self.backend is not StorageBackend.FILESYSTEM:
            raise NotImplementedError(
                f"load_state_dict is implemented only for filesystem; got {self.backend!r}",
            )
        path = self.checkpoint_path(entry_id)
        if not path.exists():
            raise FileNotFoundError(
                f"no checkpoint for entry_id={entry_id!r} at {path}",
            )
        bundle = torch.load(
            path,
            map_location=map_location,
            weights_only=weights_only,
        )
        if not isinstance(bundle, dict):
            raise TypeError(
                f"checkpoint at {path} is not a dict; got {type(bundle).__name__}",
            )
        return bundle

    def load_metrics(self, entry_id: str) -> dict[str, float]:
        if self.backend is not StorageBackend.FILESYSTEM:
            raise NotImplementedError(
                f"load_metrics is implemented only for filesystem; got {self.backend!r}",
            )
        path = self.metrics_path(entry_id)
        if not path.exists():
            raise FileNotFoundError(f"no metrics for entry_id={entry_id!r}")
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise TypeError(
                f"metrics file at {path} is not a dict; got {type(payload).__name__}",
            )
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            raise TypeError(
                f"metrics for entry_id={entry_id!r} is not a dict",
            )
        return {str(k): float(v) for k, v in metrics.items()}
