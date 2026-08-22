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
    screenshot, current-system story, strongest measured results, try it --
    before deep research/historical detail. The docs/v5-final-documentation-
    rebase renamed the original SS3.1 headings ("The problem", "Operator
    workflow", "Why HydroSwarm is different") into one consolidated "What
    HydroSwarm does" section; this checks the current heading names for the
    same judge-first ordering intent, not the pre-rebase ones."""
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", text, flags=re.MULTILINE)

    required_order = [
        "What HydroSwarm does",
        "Final system: HydroCore-v5",
        "Final locked evaluation",
        "Try the current V5 source",
    ]
    positions = [headings.index(section) for section in required_order]
    assert positions == sorted(positions), (
        f"README section order violates the judge-first requirement: {headings}"
    )
    # Deep research/historical material must stay after the judge-first
    # sections above, not interleaved with them.
    assert headings.index("Historical research") > max(positions)


def test_readme_screenshot_appears_before_the_current_system_section() -> None:
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"!\[HydroSwarm first-launch gateway[^\]]*\]\((docs/screenshots/[^)]+)\)", text)
    assert match, "README must retain a canonical hero screenshot"
    assert (PROJECT_ROOT / match.group(1)).is_file(), "README hero screenshot is missing"
    screenshot_index = match.start()
    section_index = text.index("## What HydroSwarm does")
    assert screenshot_index < section_index


def test_readme_historical_research_labels_v4_as_superseded() -> None:
    """The docs/v5-final-documentation-rebase replaced README's collapsed
    `<details>` historical-benchmark table (a big pre-V5 results dump that
    needed collapsing) with one short "Historical research" paragraph --
    there's no longer a large table to collapse, but it must still honestly
    label HydroCore-v4 as historical/superseded rather than a current claim."""
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    section = text[text.index("## Historical research") :]
    assert "HydroCore-v4" in section
    assert "historical" in section.lower()
    assert "superseded" in section.lower()
    assert "not current V5 claims" in section


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


def test_readme_and_devpost_tell_the_same_story() -> None:
    """README and Devpost must share the current headline product concepts."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    devpost = (PROJECT_ROOT / "docs" / "DEVPOST.md").read_text(encoding="utf-8")
    for phrase in [
        "HydroCore-v5",
        "REFERENCE INCIDENT",
        "WNTR",
        r"deterministic",
        r"human[ -]approval",
    ]:
        pattern = re.compile(phrase, re.IGNORECASE)
        assert pattern.search(readme), f"README missing shared concept: {phrase}"
        assert pattern.search(devpost), f"DEVPOST missing shared concept: {phrase}"


def test_final_system_doc_states_the_real_frozen_model_hash() -> None:
    """FINAL_SYSTEM.md is the current authority for the frozen HydroCore-v5
    finalist (see its own "Current authority" banner) -- must check the v5
    release manifest, not the superseded v4 one. v5's manifest has no
    single `normalization_hash` field (that's v4-runtime-specific); its
    checked identity fields are model_sha256 and feature_schema_hash."""
    import json

    runtime_manifest = json.loads(
        (PROJECT_ROOT / "models" / "hydrocore-v5-release" / "runtime_manifest.json").read_text()
    )
    text = (PROJECT_ROOT / "docs" / "FINAL_SYSTEM.md").read_text(encoding="utf-8")
    assert runtime_manifest["model_sha256"] in text
    assert runtime_manifest["feature_schema_hash"] in text
