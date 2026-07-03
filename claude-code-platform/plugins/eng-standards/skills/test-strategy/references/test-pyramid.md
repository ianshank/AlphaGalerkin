# Test Strategy — Worked Examples

## 1. Unit Test with Fixtures (fakes via constructor injection)

```python
# conftest.py
import pytest


class InMemoryStore:
    """Hand-written fake satisfying the CheckpointStore Protocol."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def save(self, key: str, payload: bytes) -> None:
        self.data[key] = payload

    def load(self, key: str) -> bytes:
        return self.data[key]


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def trainer(store: InMemoryStore) -> Trainer:
    return Trainer(store=store, config=TrainerConfig(checkpoint_every=1))
```

```python
# test_trainer.py
import pytest


def test_checkpoint_written_after_step(trainer: Trainer, store: InMemoryStore) -> None:
    trainer.step()
    assert len(store.data) == 1


def test_load_missing_checkpoint_raises(trainer: Trainer) -> None:
    with pytest.raises(KeyError):
        trainer.resume("does_not_exist")


@pytest.mark.parametrize("every,steps,expected", [(1, 3, 3), (2, 3, 1), (5, 3, 0)])
def test_checkpoint_cadence(store: InMemoryStore, every: int, steps: int, expected: int) -> None:
    trainer = Trainer(store=store, config=TrainerConfig(checkpoint_every=every))
    for _ in range(steps):
        trainer.step()
    assert len(store.data) == expected
```

Notes: fakes are plain classes, not mock objects with `assert_called_with` — assert on observable state. Parametrize replaces three near-identical tests.

## 2. Hypothesis Property Test (pure logic)

```python
from hypothesis import given, settings
from hypothesis import strategies as st


@given(doc=st.dictionaries(st.text(min_size=1), st.integers() | st.text()))
def test_migration_is_idempotent(doc: dict) -> None:
    once = migrate_document(doc)
    twice = migrate_document(once)
    assert once == twice


@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False,
                              min_value=-1e6, max_value=1e6), min_size=1)
)
def test_normalize_output_in_unit_range(values: list[float]) -> None:
    result = normalize(values)
    assert all(0.0 <= v <= 1.0 for v in result)


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=25)
def test_sampler_deterministic_for_seed(seed: int) -> None:
    assert draw_samples(n=8, seed=seed) == draw_samples(n=8, seed=seed)
```

Choose invariants, not examples: idempotence, round-trip (`decode(encode(x)) == x`), bounds, monotonicity, determinism-under-seed. Constrain floats explicitly (`allow_nan=False`) unless NaN handling is the behavior under test.

## 3. Subprocess Contract Test (CLI/scripts)

```python
import json
import subprocess
import sys
from pathlib import Path


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mypackage.cli", *args],
        capture_output=True, text=True, cwd=cwd, timeout=60,
    )


def test_run_subcommand_writes_result_json(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("name: smoke\nsteps: 2\n")

    proc = run_cli("run", "--config", str(config), "--output-dir", str(tmp_path), cwd=tmp_path)

    assert proc.returncode == 0, proc.stderr
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "completed"


def test_invalid_config_exits_nonzero_with_message(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("steps: -1\n")

    proc = run_cli("run", "--config", str(config), cwd=tmp_path)

    assert proc.returncode != 0
    assert "steps" in proc.stderr  # actionable error names the bad field
```

Contract = stdin/argv in, stdout/stderr/exit-code/files out. Never import the CLI module and call `main()` for contract tests — that skips argument parsing, entry-point wiring, and exit-code behavior. Always set `timeout`.

## 4. Marker Gating for GPU / External Tests

```python
# conftest.py (repo root) — auto-skip on capability, never skipif(True)
import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu_required" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
```

```toml
# pyproject.toml — register markers so typos fail loudly
[tool.pytest.ini_options]
markers = [
    "gpu_required: needs a CUDA device; auto-skipped when unavailable",
    "live_server: needs a running external endpoint; gated on env var",
]
```

```python
# The gated test — plus its always-running mocked twin
import os
import pytest


@pytest.mark.gpu_required
@pytest.mark.live_server
@pytest.mark.skipif(not os.environ.get("SERVER_URL"), reason="SERVER_URL not set")
def test_real_server_round_trip() -> None:
    client = Client(base_url=os.environ["SERVER_URL"])
    assert client.complete(prompt="ping", seed=0).ok


def test_client_round_trip_mocked(fake_server: FakeServer) -> None:
    """CPU twin of the smoke test above; runs in every CI job."""
    client = Client(base_url=fake_server.url)
    assert client.complete(prompt="ping", seed=0).ok
```

Rules: every gated test has a mocked twin that runs everywhere; skips carry a reason naming the missing capability; run marker-excluded suites in CI with `-m "not gpu_required"` so selection is explicit.

## Determinism Checklist

- [ ] `random.seed`, `np.random.seed`, framework `manual_seed` called (or a seeded fixture used) before any stochastic code.
- [ ] No `time.time()` / `datetime.now()` assertions — inject a clock or freeze time.
- [ ] No network in unit/integration tests — fake at the client boundary.
- [ ] `tmp_path` for all file output; never the working directory.
- [ ] Test passes under `pytest -p no:randomly` and with `--randomly-seed` shuffled — no inter-test ordering dependence.
