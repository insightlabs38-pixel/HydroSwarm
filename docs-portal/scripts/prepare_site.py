#!/usr/bin/env python3
"""Build a curated MkDocs source tree from the frozen HydroSwarm submission."""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit

LINK_RE = re.compile(
    r'(?P<prefix>!?\[[^\]]*\]\()'
    r'(?P<target><[^>]+>|[^)\s]+)'
    r'(?P<title>\s+(?:"[^"]*"|\'[^\']*\'))?'
    r'(?P<suffix>\))'
)

COPY_ASSET_SUFFIXES = {
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".svg",
    ".webp",
}

PRODUCT_TOUR_SCREENSHOTS = (
    "reference-source-localization.png",
    "reference-sampling.png",
    "reference-response-verification.png",
    "reference-approval-boundary.png",
    "model-authority.png",
    "validation.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--portal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-ref", required=True)
    return parser.parse_args()


def normalize_repo_path(path: PurePosixPath) -> str:
    normalized = posixpath.normpath(path.as_posix())
    if normalized == ".":
        return ""
    if normalized.startswith("../") or normalized == "..":
        raise ValueError(f"path escaped repository root: {path}")
    return normalized.lstrip("/")


def add_search_exclusion(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            front_matter = text[4:end]
            if re.search(r"(?m)^search\s*:", front_matter) is None:
                front_matter = front_matter.rstrip() + "\nsearch:\n  exclude: true\n"
            return "---\n" + front_matter + "---\n" + text[end + 5 :]
    return "---\nsearch:\n  exclude: true\n---\n\n" + text


def insert_banner(text: str, status: str, system: str, release: str, frozen_ref: str) -> str:
    if status == "historical":
        banner = (
            '<div class="status-banner status-historical">'
            "<strong>Historical / superseded</strong> · Preserved for transparency. "
            f"This page does not describe the current {system} finalist."
            "</div>"
        )
    else:
        banner = (
            '<div class="status-banner status-current">'
            "<strong>Current frozen system</strong> · "
            f"{system} · {release} · source snapshot <code>{frozen_ref[:12]}</code>"
            "</div>"
        )

    lines = text.splitlines()
    front_end = 0
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                front_end = idx + 1
                break

    for idx in range(front_end, len(lines)):
        if lines[idx].startswith("# "):
            lines[idx + 1 : idx + 1] = ["", banner, ""]
            return "\n".join(lines).rstrip() + "\n"

    lines[front_end:front_end] = [banner, ""]
    return "\n".join(lines).rstrip() + "\n"


def freeze_absolute_repo_urls(text: str, repo: str, frozen_ref: str) -> str:
    replacements = {
        f"https://github.com/{repo}/blob/main/": f"https://github.com/{repo}/blob/{frozen_ref}/",
        f"https://github.com/{repo}/tree/main/": f"https://github.com/{repo}/tree/{frozen_ref}/",
        f"https://raw.githubusercontent.com/{repo}/main/": f"https://raw.githubusercontent.com/{repo}/{frozen_ref}/",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def github_url(repo: str, frozen_ref: str, repo_path: str, is_dir: bool = False) -> str:
    mode = "tree" if is_dir else "blob"
    return f"https://github.com/{repo}/{mode}/{frozen_ref}/{repo_path}"


def copy_asset(
    source_root: Path,
    output_root: Path,
    repo_path: str,
) -> str:
    if not repo_path.startswith("docs/"):
        raise ValueError(f"only docs assets are copied into the site: {repo_path}")
    relative = repo_path.removeprefix("docs/")
    src = source_root / repo_path
    dst = output_root / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return relative


def rewrite_target(
    target: str,
    *,
    page_source: str,
    page_output: str,
    source_root: Path,
    output_root: Path,
    source_to_output: dict[str, str],
    repo: str,
    frozen_ref: str,
) -> str:
    wrapped = target.startswith("<") and target.endswith(">")
    raw = target[1:-1] if wrapped else target
    parsed = urlsplit(raw)

    if parsed.scheme or parsed.netloc or raw.startswith(("mailto:", "tel:", "#")):
        return target

    if not parsed.path:
        return target

    path = parsed.path
    page_source_dir = PurePosixPath(page_source).parent

    if path.startswith("/"):
        repo_path = normalize_repo_path(PurePosixPath(path))
    else:
        repo_path = normalize_repo_path(page_source_dir / PurePosixPath(path))

    if repo_path == "README.md":
        new_path = posixpath.relpath("index.md", PurePosixPath(page_output).parent.as_posix())
        rebuilt = urlunsplit(("", "", new_path, parsed.query, parsed.fragment))
        return f"<{rebuilt}>" if wrapped else rebuilt

    if repo_path in source_to_output:
        output_target = source_to_output[repo_path]
        output_parent = PurePosixPath(page_output).parent.as_posix()
        new_path = posixpath.relpath(output_target, output_parent)
        rebuilt = urlunsplit(("", "", new_path, parsed.query, parsed.fragment))
        return f"<{rebuilt}>" if wrapped else rebuilt

    source_path = source_root / repo_path
    suffix = PurePosixPath(repo_path).suffix.lower()

    if source_path.is_file() and repo_path.startswith("docs/") and suffix in COPY_ASSET_SUFFIXES:
        copied = copy_asset(source_root, output_root, repo_path)
        output_parent = PurePosixPath(page_output).parent.as_posix()
        new_path = posixpath.relpath(copied, output_parent)
        rebuilt = urlunsplit(("", "", new_path, parsed.query, parsed.fragment))
        return f"<{rebuilt}>" if wrapped else rebuilt

    if source_path.is_dir() or path.endswith("/"):
        frozen = github_url(repo, frozen_ref, repo_path, is_dir=True)
    else:
        frozen = github_url(repo, frozen_ref, repo_path, is_dir=False)
    rebuilt = urlunsplit((urlsplit(frozen).scheme, urlsplit(frozen).netloc, urlsplit(frozen).path, parsed.query, parsed.fragment))
    return f"<{rebuilt}>" if wrapped else rebuilt


def rewrite_markdown_links(
    text: str,
    *,
    page_source: str,
    page_output: str,
    source_root: Path,
    output_root: Path,
    source_to_output: dict[str, str],
    repo: str,
    frozen_ref: str,
) -> str:
    fenced = False
    rewritten: list[str] = []

    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            rewritten.append(line)
            continue
        if fenced:
            rewritten.append(line)
            continue

        def replace(match: re.Match[str]) -> str:
            target = rewrite_target(
                match.group("target"),
                page_source=page_source,
                page_output=page_output,
                source_root=source_root,
                output_root=output_root,
                source_to_output=source_to_output,
                repo=repo,
                frozen_ref=frozen_ref,
            )
            return (
                match.group("prefix")
                + target
                + (match.group("title") or "")
                + match.group("suffix")
            )

        rewritten.append(LINK_RE.sub(replace, line))

    return "".join(rewritten)


def validate_local_links(output_root: Path) -> None:
    problems: list[str] = []
    for page in output_root.rglob("*.md"):
        text = page.read_text(encoding="utf-8")
        fenced = False
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("```"):
                fenced = not fenced
                continue
            if fenced:
                continue
            for match in LINK_RE.finditer(line):
                target = match.group("target")
                if target.startswith("<") and target.endswith(">"):
                    target = target[1:-1]
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or target.startswith(("mailto:", "tel:", "#")):
                    continue
                if not parsed.path:
                    continue
                local = (page.parent / parsed.path).resolve()
                if not local.exists():
                    problems.append(
                        f"{page.relative_to(output_root)}:{line_number}: missing local target {target}"
                    )

    if problems:
        raise RuntimeError("broken generated links:\n" + "\n".join(problems))


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    portal_root = args.portal_root.resolve()
    output_root = args.output.resolve()

    manifest_path = portal_root / "content-map.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    repo = manifest["repo"]
    frozen_ref = manifest["frozen_ref"]
    release = manifest["release"]
    system = manifest["system"]

    if frozen_ref != args.frozen_ref:
        raise RuntimeError(
            f"workflow frozen ref {args.frozen_ref} does not match manifest {frozen_ref}"
        )

    pages = manifest["pages"]
    source_to_output: dict[str, str] = {}
    outputs: set[str] = set()

    for page in pages:
        source = normalize_repo_path(PurePosixPath(page["source"]))
        output = normalize_repo_path(PurePosixPath(page["output"]))
        if source in source_to_output:
            raise RuntimeError(f"duplicate source in manifest: {source}")
        if output in outputs:
            raise RuntimeError(f"duplicate output in manifest: {output}")
        source_to_output[source] = output
        outputs.add(output)

        source_file = source_root / source
        if not source_file.is_file():
            raise FileNotFoundError(f"manifest source is missing from frozen checkout: {source}")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    static_files = {
        portal_root / "src" / "index.md": output_root / "index.md",
        portal_root / "src" / "PRODUCT_TOUR.md": output_root / "PRODUCT_TOUR.md",
        portal_root / "src" / "archive" / "index.md": output_root / "archive" / "index.md",
        portal_root / "src" / "assets" / "extra.css": output_root / "assets" / "extra.css",
        source_root / "docs" / "screenshots" / "first-launch-gateway.png": (
            output_root / "screenshots" / "first-launch-gateway.png"
        ),
    }
    for name in PRODUCT_TOUR_SCREENSHOTS:
        static_files[source_root / "docs" / "screenshots" / name] = (
            output_root / "screenshots" / name
        )

    for src, dst in static_files.items():
        if not src.is_file():
            raise FileNotFoundError(f"required static file missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for page in pages:
        source = normalize_repo_path(PurePosixPath(page["source"]))
        output = normalize_repo_path(PurePosixPath(page["output"]))
        status = page["status"]

        text = (source_root / source).read_text(encoding="utf-8")
        text = freeze_absolute_repo_urls(text, repo, frozen_ref)

        if source == "docs/README.md":
            text = re.sub(
                r"(?m)^#\s+HydroSwarm documentation map\s*$",
                "# Documentation Map",
                text,
                count=1,
            )

        if status == "historical":
            text = add_search_exclusion(text)

        text = rewrite_markdown_links(
            text,
            page_source=source,
            page_output=output,
            source_root=source_root,
            output_root=output_root,
            source_to_output=source_to_output,
            repo=repo,
            frozen_ref=frozen_ref,
        )
        text = insert_banner(text, status, system, release, frozen_ref)

        destination = output_root / output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")

    validate_local_links(output_root)

    forbidden = (
        f"https://github.com/{repo}/blob/main/",
        f"https://github.com/{repo}/tree/main/",
        f"https://raw.githubusercontent.com/{repo}/main/",
    )
    for page in output_root.rglob("*.md"):
        text = page.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in text:
                raise RuntimeError(f"mutable main-branch URL remained in {page}: {marker}")

    print(
        f"Prepared {len(pages)} curated pages from {frozen_ref} "
        f"into {output_root}"
    )


if __name__ == "__main__":
    main()
