"""Shared fixtures for codec-zoo tests, including an in-memory GCS fake.

The real ``mock_gcs_client`` in ``tests/vertex/conftest.py`` is a bare
``MagicMock`` (it only needs ``client.bucket(...)`` to exist for the vertex
tests). The GCS zoo backend exercises upload / download / exists / list, so
these tests use a small in-memory fake that models object semantics: uploads
become immediately-visible immutable objects, mirroring GCS's object-atomic
write guarantee.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest


class _FakeBlob:
    def __init__(self, store: dict[str, bytes], name: str) -> None:
        self._store = store
        self.name = name

    def upload_from_string(self, data: str | bytes, content_type: str | None = None) -> None:
        self._store[self.name] = data.encode("utf-8") if isinstance(data, str) else bytes(data)

    def exists(self) -> bool:
        return self.name in self._store

    def download_as_bytes(self) -> bytes:
        return self._store[self.name]


class _FakeClient:
    def __init__(self, store: dict[str, bytes]) -> None:
        self._store = store

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(self._store, name, self)

    def list_blobs(self, bucket: Any, prefix: str = "") -> list[_FakeBlob]:
        return [_FakeBlob(self._store, n) for n in sorted(self._store) if n.startswith(prefix)]


class _FakeBucket:
    def __init__(self, store: dict[str, bytes], name: str, client: _FakeClient) -> None:
        self._store = store
        self.name = name
        self.client = client

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self._store, name)


@pytest.fixture
def gcs_store() -> dict[str, bytes]:
    """Shared in-memory object store backing the fake GCS client."""
    return {}


@pytest.fixture
def fake_gcs(gcs_store: dict[str, bytes]) -> Iterator[dict[str, bytes]]:
    """Patch ``google.cloud.storage.Client`` with the in-memory fake.

    Yields the underlying object store so tests can assert on raw blob keys.
    """
    with patch("google.cloud.storage.Client", return_value=_FakeClient(gcs_store)):
        yield gcs_store
