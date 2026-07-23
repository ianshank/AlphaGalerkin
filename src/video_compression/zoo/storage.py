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

import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, SupportsFloat

import structlog
import torch

if TYPE_CHECKING:
    from google.cloud.storage import Bucket

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
    """Locator + metrics for a trained zoo entry.

    For the filesystem backend ``entry_dir`` / ``checkpoint_path`` are local
    :class:`~pathlib.Path` objects; for the GCS backend they are ``gs://`` URI
    strings (GCS objects have no filesystem path).
    """

    entry_id: str
    entry_dir: Path | str
    checkpoint_path: Path | str
    metrics: dict[str, float]
    saved_at: str  # ISO-8601


def parse_gcs_uri(uri_str: str) -> tuple[str, str]:
    """Parse a gs:// URI into bucket and prefix."""
    if not uri_str.startswith("gs://"):
        raise ValueError(f"not a gs:// URI: {uri_str}")
    stripped = uri_str[5:]
    if not stripped:
        raise ValueError("missing bucket")
    parts = stripped.split("/", 1)
    bucket = parts[0]
    if not bucket:
        raise ValueError("missing bucket")
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix.rstrip("/")


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
        self._gcs_bucket_name: str = ""
        self._gcs_prefix: str = ""
        self._gcs_bucket_obj: Bucket | None = None
        if backend is StorageBackend.FILESYSTEM:
            self.root = Path(storage_root)
            self.root.mkdir(parents=True, exist_ok=True)
        elif backend is StorageBackend.GCS:
            # Keep the original URI string verbatim. ``Path("gs://x/y")`` would
            # normalize the double-slash on POSIX, mangling the URI.
            uri = str(storage_root)
            self.root = uri  # type: ignore[assignment]
            self._gcs_bucket_name, self._gcs_prefix = parse_gcs_uri(uri)
        else:  # pragma: no cover - exhaustive enum
            raise ValueError(f"unsupported backend: {backend!r}")

        self._log = logger.bind(
            component="VideoCodecZoo",
            backend=backend.value,
            storage_root=str(self.root),
        )

    # ------------------------------------------------------------------
    # GCS helpers
    # ------------------------------------------------------------------
    def _bucket(self) -> Bucket:
        """Return the GCS bucket, creating the client lazily.

        Raises:
            ImportError: When the ``[vertex]`` extra (google-cloud-storage) is
                not installed.

        """
        if self._gcs_bucket_obj is None:
            try:
                from google.cloud import storage
            except ImportError as exc:  # pragma: no cover - exercised when extra missing
                raise ImportError(
                    "google-cloud-storage is required for the GCS zoo backend. "
                    "Install with: pip install 'alphagalerkin[vertex]'"
                ) from exc
            self._gcs_bucket_obj = storage.Client().bucket(self._gcs_bucket_name)
        return self._gcs_bucket_obj

    def _blob_name(self, entry_id: str, filename: str) -> str:
        """Build the object name ``<prefix>/<entry_id>/<filename>``."""
        parts = [p for p in (self._gcs_prefix, entry_id, filename) if p]
        return "/".join(parts)

    def _gcs_uri(self, entry_id: str, filename: str = "") -> str:
        """Build a ``gs://`` URI for an entry dir or a file within it."""
        name = self._blob_name(entry_id, filename)
        return f"gs://{self._gcs_bucket_name}/{name}"

    @staticmethod
    def _clean_metrics(metrics: Mapping[str, SupportsFloat]) -> dict[str, float]:
        """Validate metric keys are ``str`` and coerce values to ``float``."""
        clean: dict[str, float] = {}
        for k, v in metrics.items():
            if not isinstance(k, str):
                raise TypeError(f"metric key must be str; got {type(k).__name__}")
            clean[k] = float(v)
        return clean

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
        if self.backend is StorageBackend.GCS:
            return self._bucket().blob(self._blob_name(entry_id, CHECKPOINT_FILENAME)).exists()
        return self.checkpoint_path(entry_id).exists()

    def list_entries(self) -> list[str]:
        if self.backend is StorageBackend.GCS:
            return self._list_entries_gcs()
        if not self.root.exists():
            return []
        return sorted(
            d.name for d in self.root.iterdir() if d.is_dir() and (d / CHECKPOINT_FILENAME).exists()
        )

    def _list_entries_gcs(self) -> list[str]:
        """List entry IDs by scanning for ``checkpoint.pt`` blobs under the prefix."""
        bucket = self._bucket()
        prefix = f"{self._gcs_prefix}/" if self._gcs_prefix else ""
        suffix = f"/{CHECKPOINT_FILENAME}"
        entries: set[str] = set()
        for blob in bucket.client.list_blobs(bucket, prefix=prefix):
            name = blob.name
            if not name.endswith(suffix):
                continue
            relative = name[len(prefix) :] if prefix else name
            entry_id = relative[: -len(suffix)]
            if entry_id and "/" not in entry_id:
                entries.add(entry_id)
        return sorted(entries)

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
        # Validate metrics are JSON-friendly floats first; we want to fail
        # before we touch any backend.
        clean = self._clean_metrics(metrics)

        if self.backend is StorageBackend.GCS:
            return self._save_entry_gcs(entry, state_dict, clean)

        entry_dir = self.entry_dir(entry.entry_id)
        entry_dir.mkdir(parents=True, exist_ok=True)

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

    def _save_entry_gcs(
        self,
        entry: ModelZooEntryConfig,
        state_dict: dict[str, Any],
        clean: dict[str, float],
    ) -> EntryArtifacts:
        """GCS save: upload each artifact directly to its final object.

        GCS objects are immutable and a single upload is object-atomic (the
        object is visible only after the upload fully succeeds), so — unlike the
        filesystem path — there is **no** stage-to-``.tmp``-then-rename dance
        (GCS has no in-place rename). Direct upload to the canonical name is the
        correct atomic primitive.
        """
        bucket = self._bucket()
        saved_at = datetime.now(timezone.utc).isoformat()

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        bucket.blob(self._blob_name(entry.entry_id, CHECKPOINT_FILENAME)).upload_from_string(
            buffer.getvalue(),
            content_type="application/octet-stream",
        )

        bucket.blob(self._blob_name(entry.entry_id, ENTRY_FILENAME)).upload_from_string(
            json.dumps(entry.to_yaml_dict(), indent=2, sort_keys=True, default=str),
            content_type="application/json",
        )

        bucket.blob(self._blob_name(entry.entry_id, METRICS_FILENAME)).upload_from_string(
            json.dumps({"metrics": clean, "saved_at": saved_at}, indent=2, sort_keys=True),
            content_type="application/json",
        )

        checkpoint_uri = self._gcs_uri(entry.entry_id, CHECKPOINT_FILENAME)
        self._log.info(
            "zoo.entry.uploaded",
            entry_id=entry.entry_id,
            lambda_rd=entry.lambda_rd,
            gcs_uri=checkpoint_uri,
            metrics=clean,
        )
        return EntryArtifacts(
            entry_id=entry.entry_id,
            entry_dir=self._gcs_uri(entry.entry_id),
            checkpoint_path=checkpoint_uri,
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
        if self.backend is StorageBackend.GCS:
            bundle = self._load_state_dict_gcs(
                entry_id,
                map_location=map_location,
                weights_only=weights_only,
            )
            if not isinstance(bundle, dict):
                raise TypeError(
                    f"checkpoint for entry_id={entry_id!r} is not a dict; "
                    f"got {type(bundle).__name__}",
                )
            return bundle

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
        if self.backend is StorageBackend.GCS:
            payload = self._load_json_gcs(entry_id, METRICS_FILENAME)
            location = self._gcs_uri(entry_id, METRICS_FILENAME)
        else:
            path = self.metrics_path(entry_id)
            if not path.exists():
                raise FileNotFoundError(f"no metrics for entry_id={entry_id!r}")
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            location = str(path)
        if not isinstance(payload, dict):
            raise TypeError(
                f"metrics file at {location} is not a dict; got {type(payload).__name__}",
            )
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            raise TypeError(
                f"metrics for entry_id={entry_id!r} is not a dict",
            )
        return {str(k): float(v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # GCS download helpers
    # ------------------------------------------------------------------
    def _download_blob(self, entry_id: str, filename: str) -> bytes:
        """Download an entry blob's bytes, raising FileNotFoundError if absent."""
        blob = self._bucket().blob(self._blob_name(entry_id, filename))
        if not blob.exists():
            raise FileNotFoundError(
                f"no {filename} for entry_id={entry_id!r} at {self._gcs_uri(entry_id, filename)}",
            )
        return blob.download_as_bytes()

    def _load_state_dict_gcs(
        self,
        entry_id: str,
        *,
        map_location: str | torch.device | None,
        weights_only: bool,
    ) -> Any:
        data = self._download_blob(entry_id, CHECKPOINT_FILENAME)
        return torch.load(
            io.BytesIO(data),
            map_location=map_location,
            weights_only=weights_only,
        )

    def _load_json_gcs(self, entry_id: str, filename: str) -> Any:
        data = self._download_blob(entry_id, filename)
        return json.loads(data.decode("utf-8"))
