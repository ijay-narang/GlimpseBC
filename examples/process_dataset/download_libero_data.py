"""Download official LIBERO HDF5 demonstrations from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"
SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("all", *SUITES),
        default="libero_spatial",
        help="Dataset suite to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root that will contain the suite directories.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suites = SUITES if args.suite == "all" else (args.suite,)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns=[f"{suite}/*.hdf5" for suite in suites],
    )
    for suite in suites:
        files = sorted((output_dir / suite).glob("*.hdf5"))
        expected = 90 if suite == "libero_90" else 10
        if len(files) != expected:
            raise RuntimeError(
                f"Expected {expected} HDF5 files for {suite}, found {len(files)} "
                f"under {output_dir / suite}"
            )
        print(f"Verified {suite}: {len(files)} tasks in {output_dir / suite}")


if __name__ == "__main__":
    main()
