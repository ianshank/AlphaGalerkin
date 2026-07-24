#!/usr/bin/env python3
"""Offline internal-link checker for the repository's Markdown docs.

Extracts ``[text](link)`` links from tracked Markdown files and verifies that
every *internal* link resolves — accepting either **file-relative** or
**repo-root-relative** targets, which matches the conventions used across this
repo. External URLs, anchors, and ``mailto:`` links are skipped (no network).

Fenced and inline code spans are stripped first so type annotations like
``Registry[Base]`` inside code blocks are not mistaken for links.

Usage::

    python scripts/check_doc_links.py            # check all tracked *.md
    python scripts/check_doc_links.py a.md b.md  # check specific files

Exit code is non-zero if any internal link is broken, so it is usable as a
pre-commit hook or CI step.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Paths excluded from checking: immutable history, templates with intentional
# placeholders, the nested sub-project, and the Hugging Face mirror.
EXCLUDE_PREFIXES = (
    "CHANGELOG.md",
    "docs/templates/",
    "docs/archive/",
    "claude-code-platform/",
    "hf_space/",
)

_FENCED = re.compile(r"```.*?```", re.DOTALL)
_INLINE = re.compile(r"`[^`]*`")
_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _tracked_markdown() -> list[str]:
    out = subprocess.check_output(["git", "ls-files", "*.md"], text=True, cwd=REPO_ROOT)
    return [f for f in out.splitlines() if not f.startswith(EXCLUDE_PREFIXES)]


def _links(text: str) -> list[str]:
    text = _FENCED.sub("", text)
    text = _INLINE.sub("", text)
    return [m.group(1).strip() for m in _LINK.finditer(text)]


def _is_broken(link: str, source: Path) -> bool:
    if link.startswith(("http://", "https://", "#", "mailto:", "tel:")):
        return False
    target = link.split("#", 1)[0].split("?", 1)[0].strip()
    if not target:
        return False
    file_rel = (source.parent / target).resolve()
    root_rel = (REPO_ROOT / target.lstrip("/")).resolve()
    return not (file_rel.exists() or root_rel.exists())


def main(argv: list[str]) -> int:
    """Check *argv* (or all tracked Markdown) and return a shell exit code."""
    files = argv or _tracked_markdown()
    broken: list[tuple[str, str]] = []
    for f in files:
        p = REPO_ROOT / f
        if not p.exists():
            continue
        for link in _links(p.read_text(encoding="utf-8", errors="ignore")):
            if _is_broken(link, p):
                broken.append((f, link))
    if broken:
        print("Broken internal links found:")
        for f, link in broken:
            print(f"  {f}  ->  {link}")
        return 1
    print(f"OK: internal links resolve in {len(files)} Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
