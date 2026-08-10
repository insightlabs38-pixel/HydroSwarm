"""SUB-7/SUB-8: static verification that README.md and the docs/ landing
pages don't ship broken relative links, and that the judge-first README
restructure (submission.txt SS3.1) actually landed in the required order.
Not a full site-wide link crawl (that's SUB-11's release-test scope) --
just the pages this session rewrote, so a future edit can't silently
reintroduce a dangling link or reorder the opening narrative back to
research-first.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MARKDOWN_LINK = re.compile(r"\]\(([^)]+)\)")

CHECKED_FILES = ["README.md", "docs/README.md", "docs/FINAL_SYSTEM.md"]


def _relative_links(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    links = []
    for match in MARKDOWN_LINK.finditer(text):
        link = match.group(1)
        if link.startswith(("http://", "https://", "#")):
            continue
        target = link.split("#", 1)[0]
        if target:
            links.append(target)
    return links


@pytest.mark.parametrize("name", CHECKED_FILES)
def test_relative_links_resolve(name: str) -> None:
    path = PROJECT_ROOT / name
    base = path.parent
    for link in _relative_links(path):
        resolved = (base / link).resolve()
        assert resolved.exists(), f"{name} links to missing file: {link} (resolved: {resolved})"


def test_readme_opening_sections_are_judge_first_ordered() -> None:
    """submission.txt SS3.1's required opening order: one-line value prop,
    screenshot, problem, operator workflow, why different, strongest
    measured results, try it -- before any deep research/benchmark
    detail."""
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", text, flags=re.MULTILINE)

    required_order = [
        "The problem",
        "Operator workflow",
        "Why HydroSwarm is different",
        "Strongest measured results",
        "Try it",
    ]
    positions = [headings.index(section) for section in required_order]
    assert positions == sorted(positions), (
        f"README section order violates the judge-first requirement: {headings}"
    )


def test_readme_screenshot_appears_before_the_problem_section() -> None:
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    screenshot_index = text.index("docs/screenshots/operator-overview.png")
    problem_index = text.index("## The problem")
    assert screenshot_index < problem_index


def test_readme_historical_benchmark_is_collapsed_and_labeled_superseded() -> None:
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "<details>" in text
    assert "superseded by the frozen HydroCore-v4" in text


DIAGRAM_FILES = [
    "judge-product-flow.mmd",
    "authority-architecture.mmd",
    "hydrocore-v4.mmd",
    "model-lifecycle.mmd",
    "reference-incident-flow.mmd",
    "offline-deployment.mmd",
]


@pytest.mark.parametrize("name", DIAGRAM_FILES)
def test_diagram_source_exists_and_is_a_real_flowchart(name: str) -> None:
    """submission.txt SS14's diagram-source existence check (SS2062):
    every named .mmd source must exist and contain real flowchart content,
    not an empty placeholder."""
    path = PROJECT_ROOT / "docs" / "diagrams" / name
    text = path.read_text(encoding="utf-8")
    assert "flowchart" in text
    assert len(text.strip().splitlines()) > 5


def test_each_diagram_is_embedded_in_at_least_one_doc() -> None:
    """A diagram source with no embedding anywhere is dead weight -- verify
    each one is actually rendered somewhere a reader would see it."""
    docs_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (PROJECT_ROOT / "docs").rglob("*.md")
    )
    docs_text += (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for name in DIAGRAM_FILES:
        assert name in docs_text, f"{name} is not referenced/embedded in any doc"


def test_final_system_doc_states_the_real_frozen_model_hash() -> None:
    import json

    runtime_manifest = json.loads(
        (PROJECT_ROOT / "models" / "hydrocore-v4-release" / "runtime_manifest.json").read_text()
    )
    text = (PROJECT_ROOT / "docs" / "FINAL_SYSTEM.md").read_text(encoding="utf-8")
    assert runtime_manifest["model_sha256"] in text
    assert runtime_manifest["normalization_hash"] in text
