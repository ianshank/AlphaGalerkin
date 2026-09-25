"""The project's public front doors: ``README.md`` plus every page in the docs-site nav.

Guards that police what the project *tells readers* -- performance claims, the PicoGK
disclosure -- apply to the same set of documents, so the set is defined once. The nav is read
from ``mkdocs.yml`` rather than listed by hand, so a page added to the site is covered the
moment it ships, and moving a claim from ``README.md`` onto a site page is not an evasion.

Dated history (``docs/archive/``, ADRs, plans, reviews) and the ledgers (``CLAUDE.md``,
``CHANGELOG.md``) are deliberately not front doors: they record what was believed at a date.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import yaml

#: The repository's landing page, always a front door.
README: Final[str] = "README.md"

#: The MkDocs configuration whose ``nav`` defines the published site.
MKDOCS_CONFIG: Final[str] = "mkdocs.yml"

#: MkDocs' default when ``docs_dir`` is not set.
DEFAULT_DOCS_DIR: Final[str] = "docs"

_MARKDOWN_SUFFIX: Final[str] = ".md"
_URL_MARKER: Final[str] = "://"
_PYTHON_TAG_PREFIX: Final[str] = "tag:yaml.org,2002:python/"


class _TolerantSafeLoader(yaml.SafeLoader):
    """A ``SafeLoader`` that reads ``!!python/...`` tags as ``None`` instead of refusing.

    ``mkdocs.yml`` names a fence formatter with ``!!python/name:``; the plain safe loader
    raises on it, and the unsafe one would import it. Neither is wanted for reading ``nav``.
    """


_TolerantSafeLoader.add_multi_constructor(_PYTHON_TAG_PREFIX, lambda loader, suffix, node: None)


def _leaves(node: object) -> Iterator[str]:
    """Every string leaf of a nested ``nav`` structure."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _leaves(value)
    elif isinstance(node, list):
        for item in node:
            yield from _leaves(item)


def mkdocs_nav_pages(repo_root: Path) -> list[str]:
    """Repository-relative paths of every local Markdown page in the site nav.

    Args:
        repo_root: The repository root holding ``mkdocs.yml``.

    Returns:
        Pages in nav order, prefixed with ``docs_dir``; external links are skipped.

    """
    text = (repo_root / MKDOCS_CONFIG).read_text(encoding="utf-8")
    # A SafeLoader subclass: python/ tags resolve to None, nothing is imported or executed.
    config = yaml.load(text, Loader=_TolerantSafeLoader)
    docs_dir = str(config.get("docs_dir", DEFAULT_DOCS_DIR)).rstrip("/")
    return [
        f"{docs_dir}/{page}"
        for page in _leaves(config.get("nav", []))
        if page.endswith(_MARKDOWN_SUFFIX) and _URL_MARKER not in page
    ]


def front_door_documents(repo_root: Path) -> list[str]:
    """``README.md`` followed by the docs-site nav pages, each once.

    Args:
        repo_root: The repository root.

    Returns:
        Repository-relative paths, ``README.md`` first.

    """
    return list(dict.fromkeys([README, *mkdocs_nav_pages(repo_root)]))
