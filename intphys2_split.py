#!/usr/bin/env python3
"""Split IntPhys2 metadata into possible and impossible video-path lists."""

import argparse
import csv
from pathlib import Path


def split_metadata(input_path: Path) -> tuple[Path, Path]:
    possible_path = input_path.with_name("possible.txt")
    impossible_path = input_path.with_name("impossible.txt")

    with input_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise SystemExit(f"Input file is empty: {input_path}")

        with possible_path.open("w", encoding="utf-8", newline="") as possible_file, \
             impossible_path.open("w", encoding="utf-8", newline="") as impossible_file:
            for row in reader:
                file_name = (row.get("file_name") or "").strip()
                condition = (row.get("type") or "").strip()
                if not file_name:
                    continue
                if "_Possible" in condition:
                    possible_file.write(f"{file_name}\n")
                elif "_Impossible" in condition:
                    impossible_file.write(f"{file_name}\n")

    return possible_path, impossible_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split IntPhys2 metadata.csv into plain-text possible and impossible video-path lists."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=Path("datasets/intphys2/Main/metadata.csv"),
        type=Path,
        help="Path to the IntPhys2 metadata.csv file",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    possible_path, impossible_path = split_metadata(input_path)
    print(f"Wrote: {possible_path}")
    print(f"Wrote: {impossible_path}")


if __name__ == "__main__":
    main()
