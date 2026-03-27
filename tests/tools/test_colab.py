"""Coverage tests for the Colab notebook generator.

Targets uncovered lines in src/tools/colab.py:
    - generate_colab_notebook (success/file-not-found)
    - _update_colab_badge (badge insertion, title update, no markdown cell)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools.colab import generate_colab_notebook, _update_colab_badge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_source_notebook(path: Path, cells: list[dict] | None = None) -> None:
    """Write a minimal .ipynb file."""
    if cells is None:
        cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# AlphaGalerkin Demo\n", "Some description.\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["import torch\n"],
            },
        ]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 2,
        "metadata": {},
        "cells": cells,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(nb, f)


# ---------------------------------------------------------------------------
# Tests: generate_colab_notebook
# ---------------------------------------------------------------------------


class TestGenerateColabNotebook:
    def test_generates_output(self, tmp_path: Path) -> None:
        source = tmp_path / "source.ipynb"
        target = tmp_path / "output" / "colab.ipynb"
        _make_source_notebook(source)

        generate_colab_notebook(
            source_path=source,
            target_path=target,
            repo_name="AlphaGalerkin",
            repo_url="https://github.com/test/AlphaGalerkin.git",
        )

        assert target.exists()
        with open(target) as f:
            nb = json.load(f)
        # First cell should be the setup cell
        assert nb["cells"][0]["metadata"]["id"] == "colab-setup-cell"

    def test_source_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            generate_colab_notebook(
                source_path=tmp_path / "nonexistent.ipynb",
                target_path=tmp_path / "out.ipynb",
            )

    def test_removes_existing_setup_cell(self, tmp_path: Path) -> None:
        source = tmp_path / "src.ipynb"
        cells = [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "colab-setup-cell"},
                "outputs": [],
                "source": ["# Old setup\n"],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# AlphaGalerkin\n"],
            },
        ]
        _make_source_notebook(source, cells)

        target = tmp_path / "out.ipynb"
        generate_colab_notebook(source_path=source, target_path=target)

        with open(target) as f:
            nb = json.load(f)
        # Should have exactly one setup cell (replaced, not duplicated)
        setup_cells = [c for c in nb["cells"] if c.get("metadata", {}).get("id") == "colab-setup-cell"]
        assert len(setup_cells) == 1

    def test_no_markdown_cells(self, tmp_path: Path) -> None:
        """Notebook with only code cells should still generate."""
        source = tmp_path / "code_only.ipynb"
        cells = [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["x = 1\n"],
            },
        ]
        _make_source_notebook(source, cells)

        target = tmp_path / "out.ipynb"
        generate_colab_notebook(source_path=source, target_path=target)
        assert target.exists()


# ---------------------------------------------------------------------------
# Tests: _update_colab_badge
# ---------------------------------------------------------------------------


class TestUpdateColabBadge:
    def test_badge_inserted(self) -> None:
        cells = [
            {
                "cell_type": "markdown",
                "source": ["# AlphaGalerkin\n", "Description.\n"],
            },
        ]
        _update_colab_badge(cells, "Colab.ipynb", "AlphaGalerkin")

        source = cells[0]["source"]
        assert "(Colab Version)" in source[0]
        assert any("colab-badge.svg" in line for line in source)

    def test_no_duplicate_badge(self) -> None:
        cells = [
            {
                "cell_type": "markdown",
                "source": [
                    "# AlphaGalerkin (Colab Version)\n",
                    '<a href="..."><img src="colab-badge.svg"/></a>\n',
                ],
            },
        ]
        _update_colab_badge(cells, "Colab.ipynb", "AlphaGalerkin")
        badge_count = sum(1 for line in cells[0]["source"] if "colab-badge.svg" in line)
        assert badge_count == 1

    def test_no_markdown_cells(self) -> None:
        cells = [
            {"cell_type": "code", "source": ["x = 1"]},
        ]
        _update_colab_badge(cells, "Colab.ipynb", "AlphaGalerkin")
        # Should not modify anything

    def test_empty_source(self) -> None:
        cells = [
            {"cell_type": "markdown", "source": []},
        ]
        _update_colab_badge(cells, "Colab.ipynb", "AlphaGalerkin")
        # Should not crash
