"""Unit tests for ``scripts/measure_shape.py`` on synthetic trees.

Every metric function is exercised on a small fake repository built under
``tmp_path`` (the real ``src/`` is never read here), the ruff subprocess is
mocked everywhere except one deliberate smoke test, and the CLI is driven end
to end through ``main(argv)``. The live-tree comparison lives in
``tests/docs/test_shape_baseline.py``.
"""

from __future__ import annotations

import json
import subprocess
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import ValidationError

from scripts import measure_shape as ms
from scripts.measure_shape import (
    METRIC_NAMES,
    REGENERATE_HINT,
    SHAPE_BASELINE_SCHEMA_VERSION,
    TABLE_KEY_SEPARATOR,
    MirrorReport,
    RuffFinding,
    ShapeBaseline,
    ShapeConfig,
    Site,
    Violation,
    adhoc_device_resolution_sites,
    compare,
    compare_count,
    content_hash,
    count_complexity_findings,
    count_magic_values,
    dead_import_error_guards,
    dump_baseline,
    format_report,
    git_head_sha,
    hard_dependencies,
    lazy_first_party_imports,
    library_print_findings,
    load_baseline,
    load_config,
    magic_value_table,
    main,
    measure,
    mirror_divergence,
    normalise_distribution_name,
    orphan_modules,
    parse_ruff_json,
    python_files_under,
    ruff_version,
    run_ruff,
    shim_files_without_deprecation_warning,
    write_baseline,
)

FAKE_PYPROJECT: Final[str] = """
[project]
name = "fake"
dependencies = [
  "torch>=2.6.0",
  "numpy>=1.24",
  "PyYAML>=6.0",
  "hydra-core>=1.3",
  "Some.Dist_Name[extra]>=1 ; python_version >= '3.10'",
]
"""

FAKE_A: Final[str] = '''
"""Module a. Kept for backwards-compatibility with old callers."""

from typing import TYPE_CHECKING

import torch

import src.device

if TYPE_CHECKING:
    from src.c import C

try:
    import numpy
except ImportError:
    numpy = None

try:
    import optuna
except ImportError:
    optuna = None

try:
    import numpy as np2
    from PIL import Image
except (ImportError, RuntimeError):
    Image = None

try:
    import torch.nn
except ModuleNotFoundError:
    pass

try:
    x = 1
except ImportError:
    x = 2


def lazy():
    from src.b import g
    import src.pkg.mod
    import os

    if TYPE_CHECKING:
        from src.c import D
    return g, os


def pick():
    first = "cuda" if torch.cuda.is_available() else "cpu"
    second = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    third = torch.device("cuda")
    not_counted = "cpu" if torch.cuda.is_available() else "cuda"
    other = torch.device("cpu")
    return first, second, third, not_counted, other


def compare(n: int) -> bool:
    return n > 3
'''

FAKE_B: Final[str] = """
import warnings


# back-compat alias; warns
def g():
    warnings.warn("use h", DeprecationWarning, stacklevel=2)
"""

#: Expected values on the fake tree, with the config in :func:`fake_config`.
EXPECTED_LAZY: Final[int] = 2
EXPECTED_DEVICE_SITES: Final[int] = 3
EXPECTED_ORPHANS: Final[tuple[str, ...]] = ("src.d", "src.lib", "src.pkg.other")
EXPECTED_DEAD_GUARDS: Final[int] = 2
EXPECTED_SHIMS: Final[tuple[str, ...]] = ("src/a.py",)
EXPECTED_MIRROR: Final[tuple[int, int, int]] = (1, 1, 1)

FAKE_SHA: Final[str] = "a" * 40


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    _write(root, "pyproject.toml", FAKE_PYPROJECT)
    _write(root, "src/__init__.py", "")
    _write(
        root,
        "src/device.py",
        """
        import torch


        def resolve() -> str:
            return "cuda" if torch.cuda.is_available() else "cpu"
        """,
    )
    _write(root, "src/a.py", FAKE_A)
    _write(root, "src/b.py", FAKE_B)
    _write(root, "src/c.py", "C = 1\nD = 2\n")
    _write(root, "src/d.py", "# nobody imports me\n")
    _write(
        root,
        "src/cli.py",
        """
        def main():
            print("cli")


        if __name__ == "__main__":
            main()
        """,
    )
    _write(root, "src/lib.py", "def f():\n    print('lib')\n")
    _write(root, "src/pkg/__init__.py", "")
    _write(root, "src/pkg/mod.py", "thing = 1\n")
    _write(root, "src/pkg/other.py", "from . import mod\nfrom .mod import thing\n")
    _write(root, "scripts/run.py", "from src.pkg import mod\nfrom src import a\n")
    _write(root, "dashboard/app.py", "import src.a\n")
    _write(root, "hf_space/src/a.py", (root / "src/a.py").read_text(encoding="utf-8"))
    _write(root, "hf_space/src/b.py", "# diverged\n")
    _write(root, "hf_space/src/only.py", "")
    (root / "src/__pycache__").mkdir()
    (root / "src/__pycache__/junk.py").write_text("print('cache')\n", encoding="utf-8")
    return root


@pytest.fixture
def fake_config() -> ShapeConfig:
    return ShapeConfig(cli_entry_modules=("src.cli",))


def _findings() -> list[RuffFinding]:
    return [
        RuffFinding("C901", "src/a.py", 30),
        RuffFinding("PLR0913", "src/a.py", 31),
        RuffFinding("PLR2004", "src/a.py", 40),
        RuffFinding("PLR2004", "src/a.py", 41),
        RuffFinding("PLR2004", "src/b.py", 3),
        RuffFinding("T201", "src/cli.py", 2),
        RuffFinding("T201", "src/lib.py", 2),
        RuffFinding("E501", "src/a.py", 1),
    ]


def _fake_runner(root: Path, target: str, rules: Sequence[str]) -> list[RuffFinding]:
    return _findings()


@pytest.fixture
def mocked_toolchain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the three subprocess collaborators ``measure`` resolves at call time."""
    monkeypatch.setattr(ms, "run_ruff", _fake_runner)
    monkeypatch.setattr(ms, "ruff_version", lambda: "ruff 9.9.9")
    monkeypatch.setattr(ms, "git_head_sha", lambda root: (FAKE_SHA, False))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestShapeConfig:
    def test_defaults_name_the_repository_layout(self) -> None:
        config = ShapeConfig()
        assert config.src_root == "src"
        assert "src" in config.importer_roots
        assert config.mirror_root == "hf_space/src"

    def test_ruff_rules_is_the_deduplicated_union_in_order(self) -> None:
        config = ShapeConfig(
            complexity_rules=("C901", "PLR0913"), magic_value_rules=("PLR2004", "C901")
        )
        assert config.ruff_rules == ("C901", "PLR0913", "PLR2004", "T201")

    @pytest.mark.parametrize(
        "field", ["complexity_rules", "magic_value_rules", "print_rules", "importer_roots"]
    )
    def test_an_empty_rule_set_is_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError, match="measures nothing"):
            ShapeConfig(**{field: ()})

    def test_a_bad_shim_regex_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not a valid regex"):
            ShapeConfig(shim_pattern="(unclosed")

    def test_a_custom_shim_regex_is_kept(self) -> None:
        """Defaults are not validated by pydantic, so the success path needs an explicit value."""
        assert ShapeConfig(shim_pattern="legacy").shim_pattern == "legacy"

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ShapeConfig(not_a_field=1)

    def test_load_config_none_gives_defaults(self) -> None:
        assert load_config(None) == ShapeConfig()

    def test_load_config_empty_file_gives_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_config(path) == ShapeConfig()

    def test_load_config_rejects_a_non_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(path)

    def test_load_config_overrides_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "override.yaml"
        path.write_text("cli_entry_modules: [src.cli]\nsrc_root: lib\n", encoding="utf-8")
        config = load_config(path)
        assert config.cli_entry_modules == ("src.cli",)
        assert config.src_root == "lib"


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


def test_python_files_under_skips_caches_and_is_sorted(fake_repo: Path) -> None:
    files = python_files_under(fake_repo / "src")
    names = [p.relative_to(fake_repo).as_posix() for p in files]
    assert "src/__pycache__/junk.py" not in names
    assert names == sorted(names)
    assert "src/a.py" in names


def test_python_files_under_a_missing_root_is_empty(tmp_path: Path) -> None:
    assert python_files_under(tmp_path / "absent") == []


# ---------------------------------------------------------------------------
# ruff-backed metrics
# ---------------------------------------------------------------------------


class TestRuffMetrics:
    def test_complexity_counts_only_the_configured_rules(self) -> None:
        config = ShapeConfig()
        assert count_complexity_findings(_findings(), config.complexity_rules) == 2
        assert count_complexity_findings(_findings(), ("E501",)) == 1
        assert count_complexity_findings([], config.complexity_rules) == 0

    def test_magic_value_table_is_per_file_and_rule_and_sorted(self) -> None:
        table = magic_value_table(_findings(), ("PLR2004",))
        assert table == {
            f"src/a.py{TABLE_KEY_SEPARATOR}PLR2004": 2,
            f"src/b.py{TABLE_KEY_SEPARATOR}PLR2004": 1,
        }
        assert list(table) == sorted(table)
        assert count_magic_values(_findings(), ("PLR2004",)) == 3

    def test_library_prints_exclude_files_with_a_main_block(self, fake_repo: Path) -> None:
        config = ShapeConfig()
        prints = library_print_findings(
            _findings(), fake_repo, config.print_rules, config.main_block_name
        )
        assert [p.path for p in prints] == ["src/lib.py"]

    def test_a_comment_mentioning_main_does_not_hide_a_library_print(self, tmp_path: Path) -> None:
        """The adjacent weaker defect the first mutation run found.

        Appendix A split CLI from library prints by ``"__main__" in text``; a
        probe whose *comment* said ``no __main__ here`` was classified as a CLI
        module and its print vanished from the count. The block is an AST
        notion and is detected as one.
        """
        root = tmp_path / "r"
        _write(root, "src/lib.py", "def f():\n    print('x')  # no __main__ block here\n")
        _write(
            root,
            "src/cli.py",
            'def f():\n    print("y")\n\n\nif "__main__" == __name__:\n    f()\n',
        )
        findings = [RuffFinding("T201", "src/lib.py", 2), RuffFinding("T201", "src/cli.py", 2)]
        prints = library_print_findings(findings, root, ("T201",), "__main__")
        assert [p.path for p in prints] == ["src/lib.py"]

    def test_library_prints_ignore_other_codes(self, fake_repo: Path) -> None:
        only_other = [f for f in _findings() if f.code != "T201"]
        assert library_print_findings(only_other, fake_repo, ("T201",), "__main__") == []

    def test_library_prints_read_each_file_once(self, fake_repo: Path) -> None:
        """Two findings in one file share a verdict; both are reported."""
        twice = [RuffFinding("T201", "src/lib.py", 2), RuffFinding("T201", "src/lib.py", 9)]
        assert library_print_findings(twice, fake_repo, ("T201",), "__main__") == twice

    def test_parse_ruff_json_relativises_paths_and_skips_codeless_and_cache_entries(
        self, fake_repo: Path
    ) -> None:
        payload = [
            {
                "code": "T201",
                "filename": str(fake_repo / "src" / "lib.py"),
                "location": {"row": 2, "column": 5},
            },
            {"code": None, "filename": str(fake_repo / "src" / "a.py"), "location": {"row": 1}},
            {
                "code": "T201",
                "filename": str(fake_repo / "src" / "__pycache__" / "junk.py"),
                "location": {"row": 1},
            },
        ]
        findings = parse_ruff_json(json.dumps(payload), fake_repo)
        assert findings == [RuffFinding("T201", "src/lib.py", 2)]

    def test_run_ruff_raises_when_ruff_cannot_run(
        self, monkeypatch: pytest.MonkeyPatch, fake_repo: Path
    ) -> None:
        def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="boom")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(RuntimeError, match="boom"):
            run_ruff(fake_repo, "src", ("T201",))

    def test_run_ruff_parses_a_successful_run(
        self, monkeypatch: pytest.MonkeyPatch, fake_repo: Path
    ) -> None:
        seen: dict[str, object] = {}

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            seen["command"] = command
            seen["cwd"] = kwargs["cwd"]
            payload = [
                {
                    "code": "T201",
                    "filename": str(fake_repo / "src" / "lib.py"),
                    "location": {"row": 2},
                }
            ]
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout=json.dumps(payload), stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        findings = run_ruff(fake_repo, "src", ("T201", "PLR2004"))
        assert findings == [RuffFinding("T201", "src/lib.py", 2)]
        command = seen["command"]
        assert isinstance(command, list)
        assert "--select" in command and "T201,PLR2004" in command
        assert "--exit-zero" in command
        assert seen["cwd"] == str(fake_repo)

    def test_run_ruff_real_subprocess_smoke(self, fake_repo: Path) -> None:
        """The one non-mocked ruff call: the JSON shape ruff emits is what we parse."""
        findings = run_ruff(fake_repo, "src", ("T201", "PLR2004"))
        by_code: dict[str, set[str]] = {}
        for finding in findings:
            by_code.setdefault(finding.code, set()).add(finding.path)
        assert by_code["T201"] == {"src/cli.py", "src/lib.py"}
        assert by_code["PLR2004"] == {"src/a.py"}

    def test_ruff_version_strips_output_and_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outputs = iter(["ruff 1.2.3\n", ""])

        def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout=next(outputs), stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert ruff_version() == "ruff 1.2.3"
        assert ruff_version() == "unknown"


# ---------------------------------------------------------------------------
# AST metrics
# ---------------------------------------------------------------------------


class TestLazyImports:
    def test_counts_function_body_first_party_imports_only(
        self, fake_repo: Path, fake_config: ShapeConfig
    ) -> None:
        sites = lazy_first_party_imports(fake_repo, fake_config)
        assert len(sites) == EXPECTED_LAZY
        assert {s.detail for s in sites} == {"src.b", "src.pkg.mod"}
        assert all(s.path == "src/a.py" for s in sites)

    def test_type_checking_and_module_level_imports_are_not_lazy(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        _write(
            root,
            "src/m.py",
            """
            from typing import TYPE_CHECKING
            import src.x

            def f():
                if TYPE_CHECKING:
                    from src.y import Y
                import json
            """,
        )
        assert lazy_first_party_imports(root, ShapeConfig()) == []

    def test_first_party_package_is_configurable(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        _write(root, "src/m.py", "def f():\n    import mypkg.sub\n    import src.q\n")
        sites = lazy_first_party_imports(root, ShapeConfig(first_party_package="mypkg"))
        assert [s.detail for s in sites] == ["mypkg.sub"]


class TestDeviceSites:
    def test_matches_both_shapes_and_exempts_the_canonical_module(
        self, fake_repo: Path, fake_config: ShapeConfig
    ) -> None:
        sites = adhoc_device_resolution_sites(fake_repo, fake_config)
        assert len(sites) == EXPECTED_DEVICE_SITES
        assert all(s.path == "src/a.py" for s in sites)
        kinds = sorted(s.detail for s in sites)
        assert kinds == ["bare-device-call", "conditional-expression", "conditional-expression"]

    def test_a_conditional_whose_body_is_a_device_call_counts_once(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        _write(
            root,
            "src/m.py",
            'import torch\nd = torch.device("cuda") if torch.cuda.is_available() else None\n',
        )
        sites = adhoc_device_resolution_sites(root, ShapeConfig())
        assert [s.detail for s in sites] == ["conditional-expression"]

    def test_without_the_exemption_the_canonical_module_is_counted(self, fake_repo: Path) -> None:
        config = ShapeConfig(canonical_device_modules=())
        sites = adhoc_device_resolution_sites(fake_repo, config)
        assert "src/device.py" in {s.path for s in sites}

    def test_a_non_torch_is_available_is_not_a_site(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        _write(root, "src/m.py", 'import jax\nd = "cuda" if jax.cuda.is_available() else "cpu"\n')
        assert adhoc_device_resolution_sites(root, ShapeConfig()) == []


class TestOrphans:
    def test_finds_modules_no_production_root_imports(
        self, fake_repo: Path, fake_config: ShapeConfig
    ) -> None:
        assert tuple(orphan_modules(fake_repo, fake_config)) == EXPECTED_ORPHANS

    def test_cli_entry_modules_are_excluded_by_rule(self, fake_repo: Path) -> None:
        with_cli = orphan_modules(fake_repo, ShapeConfig(cli_entry_modules=()))
        assert "src.cli" in with_cli
        assert len(with_cli) == len(EXPECTED_ORPHANS) + 1

    def test_from_import_reaches_the_submodule(
        self, fake_repo: Path, fake_config: ShapeConfig
    ) -> None:
        """``from src.pkg import mod`` names ``src.pkg`` but reaches ``src.pkg.mod``."""
        assert "src.pkg.mod" not in orphan_modules(fake_repo, fake_config)

    def test_a_type_checking_import_still_counts_as_reaching(
        self, fake_repo: Path, fake_config: ShapeConfig
    ) -> None:
        assert "src.c" not in orphan_modules(fake_repo, fake_config)

    def test_tests_are_not_importers(self, fake_repo: Path, fake_config: ShapeConfig) -> None:
        _write(fake_repo, "tests/test_d.py", "import src.d\n")
        assert "src.d" in orphan_modules(fake_repo, fake_config)
        wider = ShapeConfig(cli_entry_modules=("src.cli",), importer_roots=("src", "tests"))
        assert "src.d" not in orphan_modules(fake_repo, wider)


class TestDeadGuards:
    def test_hard_dependencies_normalise_names_and_drop_extras(self, fake_repo: Path) -> None:
        deps = hard_dependencies(fake_repo / "pyproject.toml")
        assert deps == {"torch", "numpy", "pyyaml", "hydra_core", "some_dist_name"}

    def test_hard_dependencies_rejects_an_unparseable_specifier(self, tmp_path: Path) -> None:
        path = tmp_path / "pyproject.toml"
        path.write_text('[project]\ndependencies = ["  ", ]\n', encoding="utf-8")
        with pytest.raises(ValueError, match="unparseable"):
            hard_dependencies(path)

    def test_hard_dependencies_missing_section_is_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "pyproject.toml"
        path.write_text("[tool.x]\ny = 1\n", encoding="utf-8")
        assert hard_dependencies(path) == frozenset()

    def test_normalisation(self) -> None:
        assert normalise_distribution_name("Hydra-Core") == "hydra_core"
        assert normalise_distribution_name("a.b-c__d") == "a_b_c_d"

    def test_only_guards_whose_every_import_is_hard_are_dead(
        self, fake_repo: Path, fake_config: ShapeConfig
    ) -> None:
        sites = dead_import_error_guards(fake_repo, fake_config)
        assert len(sites) == EXPECTED_DEAD_GUARDS
        assert {s.detail for s in sites} == {"numpy", "torch.nn"}

    def test_an_injected_dependency_set_is_honoured(
        self, fake_repo: Path, fake_config: ShapeConfig
    ) -> None:
        sites = dead_import_error_guards(fake_repo, fake_config, hard_deps=frozenset({"optuna"}))
        assert [s.detail for s in sites] == ["optuna"]

    def test_aliases_map_import_names_to_distributions(
        self, fake_repo: Path, fake_config: ShapeConfig
    ) -> None:
        _write(
            fake_repo, "src/y.py", "try:\n    import yaml\nexcept ImportError:\n    yaml = None\n"
        )
        sites = dead_import_error_guards(fake_repo, fake_config)
        assert "src/y.py" in {s.path for s in sites}

    def test_handlers_that_do_not_name_an_import_error_are_not_guards(self, tmp_path: Path) -> None:
        """A bare ``except:`` and an ``except ValueError:`` around a hard import are not counted.

        Neither is an *import* guard by name: the first is a catch-all, the
        second guards something else. Counting them would inflate the metric
        with sites the ``except ImportError`` cleanup cannot touch.
        """
        root = tmp_path / "r"
        _write(root, "pyproject.toml", FAKE_PYPROJECT)
        _write(
            root,
            "src/m.py",
            """
            try:
                import numpy
            except:  # noqa: E722
                numpy = None
            try:
                import torch
            except ValueError:
                torch = None
            try:
                import numpy as second
            except (ValueError, ImportError):
                second = None
            """,
        )
        sites = dead_import_error_guards(root, ShapeConfig())
        assert [(s.line, s.detail) for s in sites] == [(10, "numpy")]


class TestShims:
    def test_pattern_without_warning_marker(
        self, fake_repo: Path, fake_config: ShapeConfig
    ) -> None:
        assert (
            tuple(shim_files_without_deprecation_warning(fake_repo, fake_config)) == EXPECTED_SHIMS
        )

    def test_match_is_case_insensitive(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        _write(root, "src/m.py", "# BACKWARDS COMPAT shim\n")
        assert shim_files_without_deprecation_warning(root, ShapeConfig()) == ["src/m.py"]


class TestMirror:
    def test_classifies_identical_diverged_and_mirror_only(
        self, fake_repo: Path, fake_config: ShapeConfig
    ) -> None:
        report = mirror_divergence(fake_repo, fake_config)
        assert (report.identical, report.diverged, report.mirror_only) == EXPECTED_MIRROR
        assert report.diverged_paths == ("hf_space/src/b.py",)

    def test_a_missing_mirror_is_all_zeros(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        _write(root, "src/a.py", "")
        assert mirror_divergence(root, ShapeConfig()) == MirrorReport(0, 0, 0)


class TestContentHash:
    def test_is_deterministic_and_sensitive_to_bytes_and_names(
        self, fake_repo: Path, fake_config: ShapeConfig
    ) -> None:
        first = content_hash(fake_repo, fake_config)
        assert first == content_hash(fake_repo, fake_config)
        assert len(first) == 64
        (fake_repo / "src/d.py").write_text("# nobody imports me!\n", encoding="utf-8")
        second = content_hash(fake_repo, fake_config)
        assert second != first
        (fake_repo / "src/d.py").rename(fake_repo / "src/d2.py")
        assert content_hash(fake_repo, fake_config) != second

    def test_ignores_files_outside_src_root(
        self, fake_repo: Path, fake_config: ShapeConfig
    ) -> None:
        before = content_hash(fake_repo, fake_config)
        _write(fake_repo, "scripts/new.py", "x = 1\n")
        _write(fake_repo, "hf_space/src/z.py", "x = 1\n")
        assert content_hash(fake_repo, fake_config) == before


# ---------------------------------------------------------------------------
# Baseline document + comparison
# ---------------------------------------------------------------------------


def _baseline(**overrides: object) -> ShapeBaseline:
    payload: dict[str, object] = {
        "generated_from": FAKE_SHA,
        "content_hash": "0" * 64,
        "metrics": dict.fromkeys(METRIC_NAMES, 1),
        "tables": {"magic_values": {"src/a.py::PLR2004": 1}},
    }
    payload.update(overrides)
    return ShapeBaseline.model_validate(payload)


class TestShapeBaseline:
    def test_measure_assembles_every_metric(
        self, fake_repo: Path, fake_config: ShapeConfig, mocked_toolchain: None
    ) -> None:
        baseline = measure(fake_repo, fake_config)
        assert tuple(baseline.metrics) == METRIC_NAMES
        assert baseline.metrics == {
            "complexity_findings": 2,
            "magic_value_findings": 3,
            "library_print_findings": 1,
            "lazy_first_party_imports": EXPECTED_LAZY,
            "adhoc_device_resolution_sites": EXPECTED_DEVICE_SITES,
            "orphan_modules": len(EXPECTED_ORPHANS),
            "dead_import_error_guards": EXPECTED_DEAD_GUARDS,
            "shim_files_without_deprecation_warning": len(EXPECTED_SHIMS),
            "mirror_diverged_files": EXPECTED_MIRROR[1],
        }
        assert baseline.generated_from == FAKE_SHA
        assert baseline.tool_versions["ruff"] == "ruff 9.9.9"
        assert baseline.tool_versions["measure_shape_schema"] == str(SHAPE_BASELINE_SCHEMA_VERSION)
        assert baseline.content_hash == content_hash(fake_repo, fake_config)
        assert sum(baseline.tables["magic_values"].values()) == 3

    def test_measure_accepts_explicit_collaborators(
        self, fake_repo: Path, fake_config: ShapeConfig
    ) -> None:
        baseline = measure(
            fake_repo,
            fake_config,
            ruff_runner=lambda root, target, rules: [],
            ruff_version_probe=lambda: "ruff x",
            git_probe=lambda root: ("b" * 40, True),
        )
        assert baseline.metrics["complexity_findings"] == 0
        assert baseline.git_dirty is True
        assert baseline.tool_versions["ruff"] == "ruff x"

    def test_round_trip_through_yaml_with_header(self, tmp_path: Path) -> None:
        baseline = _baseline()
        path = tmp_path / "nested" / "shape.yaml"
        write_baseline(baseline, path)
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# Generated by `python -m scripts.measure_shape write`")
        assert load_baseline(path) == baseline
        assert dump_baseline(baseline) == text

    def test_load_rejects_a_non_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("[1, 2]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_baseline(path)

    def test_missing_metric_is_rejected(self) -> None:
        metrics = dict.fromkeys(METRIC_NAMES, 1)
        del metrics["orphan_modules"]
        with pytest.raises(ValidationError, match="missing=\\['orphan_modules'\\]"):
            _baseline(metrics=metrics)

    def test_unexpected_metric_is_rejected(self) -> None:
        metrics = dict.fromkeys(METRIC_NAMES, 1)
        metrics["tenth"] = 0
        with pytest.raises(ValidationError, match="unexpected=\\['tenth'\\]"):
            _baseline(metrics=metrics)

    def test_negative_metric_is_rejected(self) -> None:
        metrics = dict.fromkeys(METRIC_NAMES, 1)
        metrics["orphan_modules"] = -1
        with pytest.raises(ValidationError, match="non-negative"):
            _baseline(metrics=metrics)

    def test_unknown_table_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown tables"):
            _baseline(tables={"other": {}})

    def test_negative_table_entry_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="negative counts"):
            _baseline(tables={"magic_values": {"k": -2}})

    def test_future_schema_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="newer than this tool"):
            _baseline(schema_version=SHAPE_BASELINE_SCHEMA_VERSION + 1)

    def test_bad_hash_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _baseline(content_hash="not-a-sha")

    def test_git_head_sha_reports_unknown_when_git_is_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def raise_oserror(*args: object, **kwargs: object) -> None:
            raise OSError("no git")

        monkeypatch.setattr(subprocess, "run", raise_oserror)
        assert git_head_sha(tmp_path) == ("unknown", False)

    def test_git_head_sha_reports_unknown_outside_a_repository(self, tmp_path: Path) -> None:
        """Real git, real failure: ``rev-parse HEAD`` in a fresh ``git init`` has no HEAD."""
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        assert git_head_sha(tmp_path) == ("unknown", False)

    def test_git_head_sha_reads_sha_and_dirty_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        outputs = iter([FAKE_SHA + "\n", " M src/a.py\n"])

        def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout=next(outputs), stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert git_head_sha(Path(".")) == (FAKE_SHA, True)


class TestCompare:
    def test_equal_is_clean(self) -> None:
        assert compare_count("m", 3, 3, src_unchanged=True) is None

    def test_growth_is_a_violation_regardless_of_the_tree(self) -> None:
        for unchanged in (True, False):
            violation = compare_count("m", 3, 4, src_unchanged=unchanged)
            assert violation == Violation("m", 3, 4, "grew")
            assert "grew" in violation.message and REGENERATE_HINT in violation.message

    def test_improvement_on_an_unchanged_tree_is_hand_edited(self) -> None:
        violation = compare_count("m", 4, 3, src_unchanged=True)
        assert violation == Violation("m", 4, 3, "hand_edited")
        assert "edited by hand" in violation.message and REGENERATE_HINT in violation.message

    def test_improvement_on_a_changed_tree_is_clean(self) -> None:
        assert compare_count("m", 4, 3, src_unchanged=False) is None

    def test_compare_covers_metrics_and_table_entries(self) -> None:
        recorded = _baseline()
        grown = dict.fromkeys(METRIC_NAMES, 1)
        grown["orphan_modules"] = 2
        current = _baseline(
            metrics=grown,
            tables={"magic_values": {"src/a.py::PLR2004": 1, "src/new.py::PLR2004": 1}},
        )
        violations = compare(recorded, current)
        assert [(v.metric, v.kind) for v in violations] == [
            ("orphan_modules", "grew"),
            ("magic_values[src/new.py::PLR2004]", "grew"),
        ]

    def test_compare_flags_hand_edited_table_entries_when_src_is_unchanged(self) -> None:
        recorded = _baseline(tables={"magic_values": {"src/a.py::PLR2004": 5}})
        current = _baseline(tables={"magic_values": {"src/a.py::PLR2004": 1}})
        violations = compare(recorded, current)
        assert [(v.metric, v.kind) for v in violations] == [
            ("magic_values[src/a.py::PLR2004]", "hand_edited")
        ]

    def test_compare_accepts_a_removed_table_entry_on_a_changed_tree(self) -> None:
        recorded = _baseline(tables={"magic_values": {"src/a.py::PLR2004": 5}})
        current = _baseline(content_hash="1" * 64, tables={"magic_values": {}})
        assert compare(recorded, current) == []

    def test_format_report_without_a_baseline(self) -> None:
        text = format_report(None, _baseline(), ())
        assert "=== Repository shape ===" in text
        assert "recorded" not in text
        assert "OK:" not in text

    def test_format_report_with_a_clean_comparison(self) -> None:
        text = format_report(_baseline(), _baseline(), ())
        assert "unchanged" in text
        assert "OK: no metric grew" in text

    def test_format_report_lists_violations(self) -> None:
        violation = Violation("orphan_modules", 1, 2, "grew")
        text = format_report(_baseline(), _baseline(content_hash="1" * 64), [violation])
        assert "src/ is changed" in text
        assert "1 violation(s):" in text
        assert violation.message in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_write_then_check_round_trip(
        self,
        fake_repo: Path,
        mocked_toolchain: None,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config_path = tmp_path / "shape.yaml"
        config_path.write_text("cli_entry_modules: [src.cli]\n", encoding="utf-8")
        out = tmp_path / "baseline.yaml"
        common = ["--root", str(fake_repo), "--config", str(config_path), "--log-level", "DEBUG"]
        assert main([*common, "write", "--output", str(out)]) == 0
        assert load_baseline(out).metrics["orphan_modules"] == len(EXPECTED_ORPHANS)
        assert main([*common, "check", "--baseline", str(out)]) == 0
        assert "OK: no metric grew" in capsys.readouterr().out

    def test_check_fails_on_a_hand_raised_count(
        self,
        fake_repo: Path,
        mocked_toolchain: None,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        out = tmp_path / "baseline.yaml"
        common = ["--root", str(fake_repo), "--log-level", "ERROR"]
        assert main([*common, "write", "--output", str(out)]) == 0
        document = yaml.safe_load(out.read_text(encoding="utf-8"))
        document["metrics"]["complexity_findings"] += 10
        out.write_text(yaml.safe_dump(document), encoding="utf-8")
        assert main([*common, "check", "--baseline", str(out)]) == 1
        captured = capsys.readouterr().out
        assert "edited by hand" in captured and REGENERATE_HINT in captured

    def test_check_fails_on_a_hand_lowered_count(
        self,
        fake_repo: Path,
        mocked_toolchain: None,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        out = tmp_path / "baseline.yaml"
        common = ["--root", str(fake_repo), "--log-level", "ERROR"]
        assert main([*common, "write", "--output", str(out)]) == 0
        document = yaml.safe_load(out.read_text(encoding="utf-8"))
        document["metrics"]["complexity_findings"] = 0
        out.write_text(yaml.safe_dump(document), encoding="utf-8")
        assert main([*common, "check", "--baseline", str(out)]) == 1
        assert "grew" in capsys.readouterr().out

    def test_check_accepts_an_improvement_after_a_real_change(
        self, fake_repo: Path, mocked_toolchain: None, tmp_path: Path
    ) -> None:
        out = tmp_path / "baseline.yaml"
        common = ["--root", str(fake_repo), "--log-level", "ERROR"]
        assert main([*common, "write", "--output", str(out)]) == 0
        (fake_repo / "src/d.py").unlink()  # one orphan fewer, and the hash moves
        assert main([*common, "check", "--baseline", str(out)]) == 0

    def test_check_exits_2_on_a_missing_or_malformed_baseline(
        self,
        fake_repo: Path,
        mocked_toolchain: None,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        common = ["--root", str(fake_repo), "--log-level", "ERROR"]
        assert main([*common, "check", "--baseline", str(tmp_path / "absent.yaml")]) == 2
        assert "cannot read baseline" in capsys.readouterr().err
        malformed = tmp_path / "malformed.yaml"
        malformed.write_text("metrics: {}\n", encoding="utf-8")
        assert main([*common, "check", "--baseline", str(malformed)]) == 2

    def test_a_command_is_required(self) -> None:
        with pytest.raises(SystemExit):
            main([])

    def test_log_level_is_validated(self, fake_repo: Path) -> None:
        with pytest.raises(SystemExit):
            main(["--root", str(fake_repo), "--log-level", "LOUD", "write"])

    def test_site_is_a_plain_record(self) -> None:
        site = Site("src/a.py", 3)
        assert site.detail == ""
