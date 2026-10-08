"""Every committed run sidecar's ``config_hash`` is reproducible from its recorded config.

The hash-pin protocol (``claims-ledger``, ``run-provenance``) says a re-run of the
recorded configuration must hash to the committed ``config_hash``. Nothing checked
it for the arena: its scenario config nests a ``SubstrateConfig``, whose
auto-populated ``created_at`` was folded into the hash, so two loads of the same
YAML hashed differently and the recorded value was unreproducible by
construction (fixed in ``src/templates/config.py::config_hash``).
``tests/scripts/test_run_adaptive_vs_uniform.py`` pinned one sidecar; this pins
every one, through the harness that wrote it.

The property is checked against a *fresh* construction: the recorded config with
:data:`~src.templates.config.VOLATILE_CONFIG_FIELDS` removed, rebuilt the way a
re-run would build it. Rebuilding from the recorded dict as-is would replay the
recorded timestamp and pass on exactly the defect this exists for.

A new harness must say how its hash recomputes (``HASH_RECOMPUTERS``) or carry an
exemption with a reason (``UNRECOMPUTABLE_SIDECARS``, which ships empty and expires
itself in both directions).

Killed mutations, each by a named test:
- ``BaseScenarioConfig.compute_hash`` reverted to hashing the raw dump ->
  ``test_a_fresh_construction_reproduces_the_recorded_hash[mcts_classical_amr_arena]``;
- a sidecar whose recorded hash was edited by hand ->
  ``TestPlantedDefects::test_a_hand_edited_hash_is_caught``;
- a sidecar from a harness with no recompute rule ->
  ``TestPlantedDefects::test_an_undeclared_harness_is_caught``;
- an empty scan -> ``test_the_scan_sees_the_committed_sidecars``;
- an exemption for a sidecar that recomputes ->
  ``TestPlantedDefects::test_a_stale_exemption_is_caught``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

import pytest

from src.templates.config import stable_config_payload

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: Where committed run sidecars live.
RESULTS_DIR: Final[Path] = REPO_ROOT / "results"

#: Committed sidecars this repo has today; the scan must see at least this many.
MIN_SIDECARS: Final[int] = 4

HashRecomputer = Callable[[Mapping[str, Any]], str]


def _scenario_config_hash(config: Mapping[str, Any]) -> str:
    """Rebuild a PoC scenario config through the same dispatch the scenario CLI uses."""
    from src.poc.config import load_config_from_dict

    return load_config_from_dict(dict(config)).compute_hash()


def _adaptive_vs_uniform_hash(config: Mapping[str, Any]) -> str:
    """What ``scripts.run_adaptive_vs_uniform.main`` hashes and records as ``config``.

    ``config_from_args`` keeps every option except ``RUN_MODE_FLAGS``
    (``--output``, ``--proposal-grade``): where a run writes is recorded under
    ``artifacts``, so a scratch-path reproduction hashes like the committed run.
    """
    from scripts.run_adaptive_vs_uniform import AdaptiveVsUniformConfig

    return AdaptiveVsUniformConfig(**config).compute_hash()


#: How each harness's recorded config is rebuilt and hashed: the same class and
#: call the harness used when it wrote ``config_hash``.
HASH_RECOMPUTERS: Final[dict[str, HashRecomputer]] = {
    "scripts.run_mcts_classical_amr_arena": _scenario_config_hash,
    "scripts.run_adaptive_vs_uniform": _adaptive_vs_uniform_hash,
    # Gate 1: the recorded config is the post-setup LookaheadVsGreedyConfig the
    # scenario hashed (RunRecorder.start(config_hash=self.config.compute_hash())).
    "scripts.run_lookahead_vs_greedy": _scenario_config_hash,
}

#: Sidecar file name -> reason it cannot be recomputed. Ships empty; an entry
#: needs a reason of at least ``MIN_EXEMPTION_REASON_CHARS`` characters.
UNRECOMPUTABLE_SIDECARS: Final[dict[str, str]] = {}
MIN_EXEMPTION_REASON_CHARS: Final[int] = 40


def _sidecars(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*.run.json"))


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path.name} is not a JSON object"
    return data


def hash_violations(
    results_dir: Path,
    *,
    recomputers: Mapping[str, HashRecomputer] = HASH_RECOMPUTERS,
    exemptions: Mapping[str, str] = UNRECOMPUTABLE_SIDECARS,
) -> list[str]:
    """Every way a sidecar under *results_dir* fails the reproducible-hash rule."""
    problems: list[str] = []
    for path in _sidecars(results_dir):
        sidecar = _load(path)
        harness = sidecar.get("harness")
        recompute = recomputers.get(str(harness))
        exempt = path.name in exemptions
        if recompute is None:
            if not exempt:
                problems.append(
                    f"{path.name}: harness {harness!r} declares no hash recompute rule; "
                    "add it to HASH_RECOMPUTERS or exempt the sidecar with a reason"
                )
            continue
        fresh = stable_config_payload(sidecar.get("config", {}))
        recomputed = recompute(fresh)
        recorded = sidecar.get("config_hash")
        if recomputed != recorded and not exempt:
            problems.append(
                f"{path.name}: recorded config_hash {recorded!r}, but a fresh construction "
                f"of its recorded config hashes to {recomputed!r}; re-record it"
            )
        if recomputed == recorded and exempt:
            problems.append(f"{path.name}: exempted, but its hash now recomputes; drop it")
    for name, reason in exemptions.items():
        if not (results_dir / name).is_file():
            problems.append(f"{name}: exempted, but no such sidecar exists")
        if len(reason.strip()) < MIN_EXEMPTION_REASON_CHARS:
            problems.append(
                f"{name}: exemption reason is shorter than {MIN_EXEMPTION_REASON_CHARS}"
            )
    return problems


def test_the_scan_sees_the_committed_sidecars() -> None:
    assert len(_sidecars(RESULTS_DIR)) >= MIN_SIDECARS


@pytest.mark.parametrize(
    "path", _sidecars(RESULTS_DIR), ids=lambda p: p.name.removesuffix(".run.json")
)
def test_a_fresh_construction_reproduces_the_recorded_hash(path: Path) -> None:
    sidecar = _load(path)
    recompute = HASH_RECOMPUTERS.get(str(sidecar.get("harness")))
    if recompute is None:
        assert path.name in UNRECOMPUTABLE_SIDECARS, (
            f"{path.name}: harness {sidecar.get('harness')!r} has no recompute rule"
        )
        return
    recomputed = recompute(stable_config_payload(sidecar["config"]))
    assert recomputed == sidecar["config_hash"], (
        f"{path.name} records {sidecar['config_hash']!r}; a re-run of its config hashes to "
        f"{recomputed!r}. Re-record it with its harness's --proposal-grade."
    )


def test_the_committed_tree_has_no_violations() -> None:
    assert hash_violations(RESULTS_DIR) == []


class TestPlantedDefects:
    """The rule run against planted copies, so every kill stays live in CI."""

    @pytest.fixture
    def results_copy(self, tmp_path: Path) -> Path:
        for path in _sidecars(RESULTS_DIR):
            shutil.copy(path, tmp_path / path.name)
        return tmp_path

    def test_a_hand_edited_hash_is_caught(self, results_copy: Path) -> None:
        victim = _sidecars(results_copy)[0]
        data = _load(victim)
        data["config_hash"] = "0" * len(data["config_hash"])
        victim.write_text(json.dumps(data), encoding="utf-8")
        assert any(victim.name in problem for problem in hash_violations(results_copy))

    def test_an_undeclared_harness_is_caught(self, results_copy: Path) -> None:
        victim = _sidecars(results_copy)[0]
        data = _load(victim)
        data["harness"] = "scripts.run_something_new"
        victim.write_text(json.dumps(data), encoding="utf-8")
        assert any("declares no hash recompute rule" in p for p in hash_violations(results_copy))

    def test_a_stale_exemption_is_caught(self, results_copy: Path) -> None:
        name = _sidecars(results_copy)[0].name
        reason = "x" * MIN_EXEMPTION_REASON_CHARS
        problems = hash_violations(results_copy, exemptions={name: reason})
        assert any("now recomputes" in problem for problem in problems)

    def test_an_exemption_needs_a_real_sidecar_and_a_reason(self, results_copy: Path) -> None:
        problems = hash_violations(results_copy, exemptions={"gone.run.json": "short"})
        assert any("no such sidecar" in problem for problem in problems)
        assert any("reason is shorter" in problem for problem in problems)

    def test_a_timestamp_folded_into_the_hash_is_caught(
        self, results_copy: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The defect this guard exists for, planted in the scenario hash."""
        import hashlib

        from src.poc.config import BaseScenarioConfig

        def raw_dump_hash(self: BaseScenarioConfig) -> str:
            payload = json.dumps(self.model_dump(), sort_keys=True, default=str)
            return hashlib.sha256(payload.encode()).hexdigest()[:16]

        monkeypatch.setattr(BaseScenarioConfig, "compute_hash", raw_dump_hash)
        problems = hash_violations(results_copy)
        assert any("mcts_classical_amr_arena" in problem for problem in problems)
