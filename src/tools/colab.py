"""Utilities for generating Colab-compatible notebooks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def generate_colab_notebook(
    source_path: Path | str = "notebooks/AlphaGalerkin_Demo.ipynb",
    target_path: Path | str = "notebooks/AlphaGalerkin_Colab.ipynb",
    repo_name: str = "AlphaGalerkin",
    repo_url: str = "https://github.com/ianshank/AlphaGalerkin.git",
) -> None:
    """Generate a Colab-compatible notebook from a source notebook.

    Args:
        source_path: Path to the source notebook.
        target_path: Path to save the generated notebook.
        repo_name: Name of the repository directory.
        repo_url: URL of the git repository.

    Raises:
        FileNotFoundError: If source_path does not exist.

    """
    source = Path(source_path)
    target = Path(target_path)

    if not source.exists():
        msg = f"Source notebook not found: {source}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("generating_colab_notebook", source=str(source), target=str(target))

    with open(source, encoding="utf-8") as f:
        notebook = json.load(f)

    # Define the setup cell
    setup_source = [
        "# @title Colab Environment Setup\n",
        "# @markdown Run this cell to clone the repository and install dependencies.\n",
        "\n",
        "import os\n",
        "import sys\n",
        "from pathlib import Path\n",
        "\n",
        "# 1. Clone Repository\n",
        "if 'google.colab' in sys.modules:\n",
        '    repo_name = "AlphaGalerkin"\n',
        '    repo_url = "https://github.com/ianshank/AlphaGalerkin.git"\n',
        "    \n",
        "    if not Path(repo_name).exists():\n",
        '        print(f"Cloning {repo_name}...")\n',
        "        !git clone {repo_url}\n",
        "    else:\n",
        '        print(f"{repo_name} already exists. Pulling latest changes...")\n',
        "        !cd {repo_name} && git pull\n",
        "    \n",
        "    # 2. Install Dependencies\n",
        '    print("Installing dependencies...")\n',
        "    !pip install -q einops jaxtyping pydantic hydra-core structlog wandb scipy\n",
        "    \n",
        "    # 3. Setup Path and Working Directory\n",
        "    project_root = Path(os.getcwd()) / repo_name\n",
        "    \n",
        "    # Change dir to repo root so relative paths work\n",
        "    os.chdir(project_root)\n",
        '    print(f"Working directory changed to: {os.getcwd()}")\n',
        "    \n",
        "    # Ensure src is in python path\n",
        "    if str(project_root) not in sys.path:\n",
        "        sys.path.insert(0, str(project_root))\n",
        '        print(f"Added {project_root} to sys.path")\n',
        "else:\n",
        '    print("Not running in Colab. Skipping setup.")',
    ]

    setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "colab-setup-cell"},
        "outputs": [],
        "source": setup_source,
    }

    # Remove existing setup cell if it exists (to avoid duplicates on re-runs)
    cells = notebook["cells"]
    if cells and cells[0].get("metadata", {}).get("id") == "colab-setup-cell":
        cells.pop(0)

    # Prepend the fresh setup cell
    cells.insert(0, setup_cell)

    # Update badge in the first markdown cell
    _update_colab_badge(cells, target.name, repo_name)

    # Ensure target directory exists
    target.parent.mkdir(parents=True, exist_ok=True)

    with open(target, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    logger.info("colab_notebook_generated", path=str(target))


def _update_colab_badge(cells: list[dict[str, Any]], filename: str, repo_name: str) -> None:
    """Update or add the Open in Colab badge."""
    # Find first markdown cell
    markdown_cell = None
    for cell in cells:
        if cell["cell_type"] == "markdown":
            markdown_cell = cell
            break

    if not markdown_cell:
        return

    source = markdown_cell["source"]
    if not source:
        return

    # Update title if present
    if source[0].startswith("# AlphaGalerkin") and "(Colab Version)" not in source[0]:
        source[0] = source[0].strip() + " (Colab Version)\n"

    # Define the badge line
    colab_base = "https://colab.research.google.com"
    github_path = f"github/ianshank/{repo_name}/blob/main/notebooks/{filename}"
    badge_url = f"{colab_base}/assets/colab-badge.svg"
    badge_html = (
        f'\n<a href="{colab_base}/{github_path}" target="_parent">'
        f'<img src="{badge_url}" alt="Open In Colab"/></a>\n'
    )

    # Check if badge already exists
    has_badge = any("colab-badge.svg" in line for line in source)

    if not has_badge:
        # Insert after title
        source.insert(1, badge_html)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_colab_notebook()
