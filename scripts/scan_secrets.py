"""Scan release-bound text files for common credential and private-key patterns."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


PATTERNS = {
    "aws_access_key": re.compile("A" + "KIA[0-9A-Z]{16}"),
    "aws_temporary_key": re.compile("A" + "SIA[0-9A-Z]{16}"),
    "github_token": re.compile("gh" + r"[pousr]_[A-Za-z0-9]{20,}"),
    "openai_key": re.compile("sk" + r"-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "private_key": re.compile("-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
SKIP_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".zip", ".safetensors"}


def release_files(root: Path) -> list[Path]:
    command = ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    output = subprocess.run(command, cwd=root, check=True, capture_output=True).stdout
    return sorted(root / item.decode("utf-8") for item in output.split(b"\0") if item)


def scan(root: Path) -> dict[str, object]:
    files = release_files(root)
    scanned = 0
    findings: list[dict[str, object]] = []
    for path in files:
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        for line_number, line in enumerate(text.splitlines(), start=1):
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append({
                        "file": path.relative_to(root).as_posix(),
                        "line": line_number,
                        "pattern": name,
                    })
    return {
        "schema_version": 1,
        "status": "pass" if not findings else "fail",
        "files_considered": len(files),
        "text_files_scanned": scanned,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = scan(root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        target = root / args.output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
