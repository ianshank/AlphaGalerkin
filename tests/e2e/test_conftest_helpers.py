"""Hermetic tests for the E2E tier's own fixtures.

``tests/e2e/conftest.py`` advertises two safety properties that CLAUDE.md
repeats, and neither was exercised by anything:

* ``pin_scenario_yaml`` "refuses to pin a key the config does not declare, so
  the pin cannot silently no-op";
* ``_scenario_mapping`` "refuses to guess which one the caller meant" when a
  document has an ambiguous shape.

Both call sites in the tier pass only declared keys against single-entry
configs, so deleting either check failed nothing -- a fixture whose guarantee is
documented, relied upon, and untested. These drive the helpers directly on
synthetic YAML: no subprocess, no scenario run, milliseconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.e2e.conftest import ScenarioYamlKeyError, ScenarioYamlPinnerType, _scenario_mapping

pytestmark = pytest.mark.e2e

#: A minimal single-entry config in the `scenarios:` list shape.
_LIST_SHAPED: dict[str, object] = {
    "scenarios": [{"name": "demo", "description": "d", "device": "cpu", "n_steps": 3}]
}

#: The same scenario in the bare-mapping shape `src/poc/config.py` also accepts.
_BARE_SHAPED: dict[str, object] = {"name": "demo", "description": "d", "device": "cpu"}


def _write(tmp_path: Path, document: object, filename: str = "scenario.yaml") -> Path:
    path = tmp_path / filename
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


class TestScenarioMappingShapes:
    """Both accepted shapes resolve; every ambiguous one raises."""

    def test_a_single_entry_list_resolves_to_that_entry(self, tmp_path: Path) -> None:
        path = _write(tmp_path, _LIST_SHAPED)
        mapping = _scenario_mapping(yaml.safe_load(path.read_text()), path)
        assert mapping["name"] == "demo"

    def test_a_bare_mapping_resolves_to_itself(self, tmp_path: Path) -> None:
        """The branch that is unreachable with the two shipped configs."""
        path = _write(tmp_path, _BARE_SHAPED)
        mapping = _scenario_mapping(yaml.safe_load(path.read_text()), path)
        assert mapping["name"] == "demo"

    def test_a_two_entry_list_refuses_to_guess(self, tmp_path: Path) -> None:
        """Picking ``scenarios[0]`` silently would pin the wrong scenario."""
        document = {"scenarios": [dict(_BARE_SHAPED), {"name": "other", "description": "d"}]}
        path = _write(tmp_path, document)
        with pytest.raises(ScenarioYamlKeyError, match="exactly one entry"):
            _scenario_mapping(yaml.safe_load(path.read_text()), path)

    def test_an_empty_scenarios_list_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path, {"scenarios": []})
        with pytest.raises(ScenarioYamlKeyError, match="exactly one entry"):
            _scenario_mapping(yaml.safe_load(path.read_text()), path)

    def test_a_non_list_scenarios_key_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path, {"scenarios": {"name": "demo"}})
        with pytest.raises(ScenarioYamlKeyError, match="exactly one entry"):
            _scenario_mapping(yaml.safe_load(path.read_text()), path)

    def test_a_document_with_neither_shape_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path, {"unrelated": 1})
        with pytest.raises(ScenarioYamlKeyError, match="bare scenario mapping"):
            _scenario_mapping(yaml.safe_load(path.read_text()), path)

    def test_a_non_mapping_entry_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path, {"scenarios": ["just a string"]})
        with pytest.raises(ScenarioYamlKeyError, match="not a mapping"):
            _scenario_mapping(yaml.safe_load(path.read_text()), path)


class TestPinRefusesAnUndeclaredKey:
    """The advertised safety net: a pin that would no-op must fail loudly."""

    def test_pinning_a_declared_key_rewrites_it(
        self, tmp_path: Path, pin_scenario_yaml: ScenarioYamlPinnerType
    ) -> None:
        """The conditional half -- tightening must not break the real use."""
        src = _write(tmp_path, _LIST_SHAPED)
        pinned = pin_scenario_yaml(src, device="cuda:0", n_steps=1)
        entry = yaml.safe_load(pinned.read_text())["scenarios"][0]
        assert entry["device"] == "cuda:0"
        assert entry["n_steps"] == 1

    def test_pinning_an_undeclared_key_raises_and_names_it(
        self, tmp_path: Path, pin_scenario_yaml: ScenarioYamlPinnerType
    ) -> None:
        """A renamed or removed field must fail here, not pin nothing.

        Without this the fixture would happily write a key the scenario config
        does not read -- the journey would then "pin" a device that never
        reaches the run, and pass for the wrong reason.
        """
        src = _write(tmp_path, _LIST_SHAPED)
        with pytest.raises(ScenarioYamlKeyError, match="renamed_device"):
            pin_scenario_yaml(src, renamed_device="cpu")

    def test_the_copy_does_not_touch_the_source(
        self, tmp_path: Path, pin_scenario_yaml: ScenarioYamlPinnerType
    ) -> None:
        """Pinning must never edit a committed config in the working tree."""
        src = _write(tmp_path, _LIST_SHAPED)
        before = src.read_bytes()
        pinned = pin_scenario_yaml(src, device="cuda:0")
        assert pinned != src
        assert src.read_bytes() == before

    def test_two_pins_of_one_source_do_not_collide(
        self, tmp_path: Path, pin_scenario_yaml: ScenarioYamlPinnerType
    ) -> None:
        """Each call gets its own directory; the fixture is session-scoped."""
        src = _write(tmp_path, _LIST_SHAPED)
        first = pin_scenario_yaml(src, device="cpu")
        second = pin_scenario_yaml(src, device="cuda:0")
        assert first != second
        assert yaml.safe_load(first.read_text())["scenarios"][0]["device"] == "cpu"
        assert yaml.safe_load(second.read_text())["scenarios"][0]["device"] == "cuda:0"
