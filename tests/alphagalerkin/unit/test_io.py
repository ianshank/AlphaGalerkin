"""Tests for I/O utilities (utils/io.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.alphagalerkin.utils.io import load_yaml, resolve_device, save_yaml


class TestLoadYaml:
    """load_yaml reads YAML files."""

    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("key: value\nnested:\n  a: 1\n")

        data = load_yaml(yaml_file)

        assert data["key"] == "value"
        assert data["nested"]["a"] == 1

    def test_returns_empty_dict_for_empty_file(
        self,
        tmp_path: Path,
    ) -> None:
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        data = load_yaml(yaml_file)

        assert data == {}

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.yaml"

        with pytest.raises(FileNotFoundError):
            load_yaml(missing)

    def test_loads_list_values(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "list.yaml"
        yaml_file.write_text("items:\n  - a\n  - b\n  - c\n")

        data = load_yaml(yaml_file)

        assert data["items"] == ["a", "b", "c"]

    def test_loads_numeric_values(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "numbers.yaml"
        yaml_file.write_text("int_val: 42\nfloat_val: 3.14\n")

        data = load_yaml(yaml_file)

        assert data["int_val"] == 42
        assert data["float_val"] == pytest.approx(3.14)


class TestSaveYaml:
    """save_yaml writes YAML files."""

    def test_round_trip(self, tmp_path: Path) -> None:
        original = {"key": "value", "count": 42, "items": [1, 2, 3]}
        yaml_file = tmp_path / "out.yaml"

        save_yaml(original, yaml_file)
        loaded = load_yaml(yaml_file)

        assert loaded == original

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "a" / "b" / "c" / "config.yaml"
        data = {"x": 1}

        save_yaml(data, deep_path)

        assert deep_path.exists()
        assert load_yaml(deep_path) == data

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "overwrite.yaml"
        save_yaml({"old": True}, yaml_file)
        save_yaml({"new": True}, yaml_file)

        data = load_yaml(yaml_file)

        assert "old" not in data
        assert data["new"] is True

    def test_handles_nested_dicts(self, tmp_path: Path) -> None:
        original = {"level1": {"level2": {"level3": "deep"}}}
        yaml_file = tmp_path / "nested.yaml"

        save_yaml(original, yaml_file)
        loaded = load_yaml(yaml_file)

        assert loaded == original


class TestResolveDevice:
    """resolve_device maps 'auto' and pass-through strings."""

    def test_explicit_cpu(self) -> None:
        assert resolve_device("cpu") == "cpu"

    def test_explicit_cuda(self) -> None:
        assert resolve_device("cuda:0") == "cuda:0"

    def test_auto_returns_string(self) -> None:
        result = resolve_device("auto")

        # On any machine it should return one of the known backends.
        assert result in ("cpu", "cuda", "mps")

    def test_passthrough_arbitrary_string(self) -> None:
        # Non-auto strings are returned as-is.
        assert resolve_device("xla:0") == "xla:0"

    def test_auto_on_cpu_only(self) -> None:
        import torch

        # If CUDA is not available the result must be cpu or mps.
        result = resolve_device("auto")
        if not torch.cuda.is_available():
            assert result in ("cpu", "mps")
