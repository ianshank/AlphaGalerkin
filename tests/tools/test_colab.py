"""Tests for Colab notebook generation utilities.

Covers generate_colab_notebook and _update_colab_badge functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools.colab import _update_colab_badge, generate_colab_notebook


@pytest.fixture
def sample_notebook(tmp_path: Path) -> Path:
    """Create a minimal notebook file for testing."""
    notebook = {
        "cells": [
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
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    path = tmp_path / "source.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    return path


class TestGenerateColabNotebook:
    """Tests for generate_colab_notebook function."""

    def test_source_not_found_raises(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when source doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Source notebook not found"):
            generate_colab_notebook(
                source_path=tmp_path / "nonexistent.ipynb",
                target_path=tmp_path / "output.ipynb",
            )

    def test_generates_output_file(self, sample_notebook: Path, tmp_path: Path) -> None:
        """Creates the target notebook file."""
        target = tmp_path / "output" / "colab.ipynb"
        generate_colab_notebook(source_path=sample_notebook, target_path=target)
        assert target.exists()

    def test_output_is_valid_json(self, sample_notebook: Path, tmp_path: Path) -> None:
        """Output file is valid JSON."""
        target = tmp_path / "colab.ipynb"
        generate_colab_notebook(source_path=sample_notebook, target_path=target)
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
        assert "cells" in data

    def test_setup_cell_prepended(self, sample_notebook: Path, tmp_path: Path) -> None:
        """Setup cell is inserted as first cell."""
        target = tmp_path / "colab.ipynb"
        generate_colab_notebook(source_path=sample_notebook, target_path=target)
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
        first_cell = data["cells"][0]
        assert first_cell["metadata"].get("id") == "colab-setup-cell"
        assert first_cell["cell_type"] == "code"

    def test_setup_cell_contains_clone(self, sample_notebook: Path, tmp_path: Path) -> None:
        """Setup cell contains repo clone logic."""
        target = tmp_path / "colab.ipynb"
        generate_colab_notebook(source_path=sample_notebook, target_path=target)
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
        setup_source = "".join(data["cells"][0]["source"])
        assert "git clone" in setup_source

    def test_no_duplicate_setup_cell(self, sample_notebook: Path, tmp_path: Path) -> None:
        """Running twice doesn't duplicate setup cell."""
        target = tmp_path / "colab.ipynb"
        generate_colab_notebook(source_path=sample_notebook, target_path=target)
        # Run again with the output as source
        target2 = tmp_path / "colab2.ipynb"
        generate_colab_notebook(source_path=target, target_path=target2)
        with open(target2, encoding="utf-8") as f:
            data = json.load(f)
        # Count setup cells
        setup_cells = [
            c for c in data["cells"]
            if c.get("metadata", {}).get("id") == "colab-setup-cell"
        ]
        assert len(setup_cells) == 1

    def test_colab_badge_added(self, sample_notebook: Path, tmp_path: Path) -> None:
        """Colab badge is added to first markdown cell."""
        target = tmp_path / "colab.ipynb"
        generate_colab_notebook(source_path=sample_notebook, target_path=target)
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
        # Find first markdown cell (after setup code cell)
        for cell in data["cells"]:
            if cell["cell_type"] == "markdown":
                source_text = "".join(cell["source"])
                assert "colab-badge.svg" in source_text
                break

    def test_title_updated(self, sample_notebook: Path, tmp_path: Path) -> None:
        """Title gets '(Colab Version)' appended."""
        target = tmp_path / "colab.ipynb"
        generate_colab_notebook(source_path=sample_notebook, target_path=target)
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
        for cell in data["cells"]:
            if cell["cell_type"] == "markdown":
                assert "(Colab Version)" in cell["source"][0]
                break

    def test_creates_parent_directories(self, sample_notebook: Path, tmp_path: Path) -> None:
        """Creates parent directories for target path."""
        target = tmp_path / "deep" / "nested" / "colab.ipynb"
        generate_colab_notebook(source_path=sample_notebook, target_path=target)
        assert target.exists()


class TestUpdateColabBadge:
    """Tests for _update_colab_badge helper."""

    def test_adds_badge_to_markdown(self) -> None:
        """Adds badge to first markdown cell."""
        cells = [
            {
                "cell_type": "markdown",
                "source": ["# AlphaGalerkin Demo\n", "Description.\n"],
            },
        ]
        _update_colab_badge(cells, "notebook.ipynb", "AlphaGalerkin")
        source_text = "".join(cells[0]["source"])
        assert "colab-badge.svg" in source_text

    def test_no_markdown_cells(self) -> None:
        """Does nothing when no markdown cells exist."""
        cells = [
            {
                "cell_type": "code",
                "source": ["print('hello')\n"],
            },
        ]
        _update_colab_badge(cells, "notebook.ipynb", "AlphaGalerkin")
        # Should not raise

    def test_empty_markdown_source(self) -> None:
        """Does nothing when markdown cell has empty source."""
        cells = [
            {
                "cell_type": "markdown",
                "source": [],
            },
        ]
        _update_colab_badge(cells, "notebook.ipynb", "AlphaGalerkin")
        # Should not raise

    def test_badge_not_duplicated(self) -> None:
        """Badge is not added if already present."""
        cells = [
            {
                "cell_type": "markdown",
                "source": [
                    "# Title\n",
                    '<img src="colab-badge.svg"/>\n',
                    "Description.\n",
                ],
            },
        ]
        original_len = len(cells[0]["source"])
        _update_colab_badge(cells, "notebook.ipynb", "AlphaGalerkin")
        assert len(cells[0]["source"]) == original_len

    def test_title_updated_with_colab_version(self) -> None:
        """Appends (Colab Version) to title."""
        cells = [
            {
                "cell_type": "markdown",
                "source": ["# AlphaGalerkin Demo\n"],
            },
        ]
        _update_colab_badge(cells, "notebook.ipynb", "AlphaGalerkin")
        assert "(Colab Version)" in cells[0]["source"][0]

    def test_title_not_double_updated(self) -> None:
        """Does not append (Colab Version) twice."""
        cells = [
            {
                "cell_type": "markdown",
                "source": ["# AlphaGalerkin Demo (Colab Version)\n"],
            },
        ]
        _update_colab_badge(cells, "notebook.ipynb", "AlphaGalerkin")
        count = cells[0]["source"][0].count("(Colab Version)")
        assert count == 1
