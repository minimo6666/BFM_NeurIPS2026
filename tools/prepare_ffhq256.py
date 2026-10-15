"""Resize the official aligned FFHQ PNGs to the 256x256 training dataset."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the 70,000-image FFHQ 256x256 training tree"
    )
    parser.add_argument("--source", required=True, help="Official images1024x1024 directory")
    parser.add_argument("--output", required=True, help="Output ffhq256 directory")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resize_one(source, source_root, output_root, overwrite):
    target = output_root / source.relative_to(source_root)
    if target.is_file() and not overwrite:
        with Image.open(target) as existing:
            if existing.size == (256, 256) and existing.mode == "RGB":
                return "skipped"

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with Image.open(source) as image:
        image = image.convert("RGB")
        image = image.resize((256, 256), Image.Resampling.LANCZOS)
        image.save(temporary, format="PNG")
    os.replace(temporary, target)
    return "written"


def main():
    args = parse_args()
    source_root = Path(args.source).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"FFHQ source directory not found: {source_root}")
    if source_root == output_root:
        raise ValueError("--source and --output must be different directories")

    sources = sorted(source_root.glob("*/*.png"))
    if len(sources) != 70_000:
        raise RuntimeError(
            f"Expected 70,000 official FFHQ PNGs, found {len(sources)} in "
            f"{source_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    def convert(path):
        return resize_one(path, source_root, output_root, args.overwrite)

    written = skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(convert, sources), start=1):
            written += result == "written"
            skipped += result == "skipped"
            if index % 1000 == 0 or index == len(sources):
                print(
                    f"processed={index}/70000 written={written} skipped={skipped}",
                    flush=True,
                )

    produced = list(output_root.glob("*/*.png"))
    if len(produced) != 70_000:
        raise RuntimeError(
            f"Output verification failed: expected 70,000 PNGs, got {len(produced)}"
        )
    print(f"FFHQ-256 ready: {output_root}", flush=True)


if __name__ == "__main__":
    main()
