"""SUB-11 (submission.txt task list): docs link checks, repository-wide --
broader than test_readme_and_docs_links.py's SUB-7/SUB-8-scoped check of
just the pages that session rewrote. Every relative link in every
committed markdown file must resolve to a real path, including historical
reports/results/**/*.md -- a judge or reviewer following any link in this
repository should never hit a 404.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MARKDOWN_LINK = re.compile(r"\]\(([^)]+)\)")

EXCLUDED_DIR_PARTS = {"node_modules", ".venv", "experiments", ".git"}


def _all_markdown_files() -> list[Path]:
    return [
        path
        for path in PROJECT_ROOT.rglob("*.md")
        if not EXCLUDED_DIR_PARTS & set(path.relative_to(PROJECT_ROOT).parts)
    ]


def _broken_links(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    base = path.parent
    broken = []
    for match in MARKDOWN_LINK.finditer(text):
        link = match.group(1)
        if link.startswith(("http://", "https://", "#", "mailto:")):
            continue
        target = link.split("#", 1)[0]
        if not target:
            continue
        if not (base / target).resolve().exists():
            broken.append(link)
    return broken


@pytest.fixture(scope="module")
def markdown_files() -> list[Path]:
    files = _all_markdown_files()
    assert len(files) > 30, "sanity check: expected the repo-wide markdown scan to find many files"
    return files


def test_no_broken_relative_links_anywhere_in_the_repository(markdown_files: list[Path]) -> None:
    all_broken = []
    for path in markdown_files:
        for link in _broken_links(path):
            all_broken.append(f"{path.relative_to(PROJECT_ROOT)} -> {link}")
    assert not all_broken, "broken relative links found:\n" + "\n".join(all_broken)
