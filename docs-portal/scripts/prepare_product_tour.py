#!/usr/bin/env python3
"""Copy the visual product tour and its frozen screenshots into the generated docs tree."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SCREENSHOTS = (
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
    return parser.parse_args()


def copy_required(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"required Product Tour source is missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    portal_root = args.portal_root.resolve()
    output_root = args.output.resolve()

    copy_required(
        portal_root / "src" / "PRODUCT_TOUR.md",
        output_root / "PRODUCT_TOUR.md",
    )

    for name in SCREENSHOTS:
        copy_required(
            source_root / "docs" / "screenshots" / name,
            output_root / "screenshots" / name,
        )

    print(f"Prepared Product Tour with {len(SCREENSHOTS)} frozen screenshots")


if __name__ == "__main__":
    main()
