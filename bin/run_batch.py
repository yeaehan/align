from __future__ import annotations

import argparse

from .pipeline import BatchProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recursive batch alignment.")
    parser.add_argument("--root-folder", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pyramid-levels", nargs="+", type=float, default=[0.25, 0.5])
    parser.add_argument("--no-nonrigid", action="store_true")
    parser.add_argument("--no-pass3", action="store_true")
    parser.add_argument("--keep-preprocessed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = BatchProcessor(root_folder=args.root_folder, output_root=args.output_root)
    results = processor.run(
        pyramid_levels=args.pyramid_levels,
        enable_nonrigid=not args.no_nonrigid,
        enable_pass3=not args.no_pass3,
        cleanup=not args.keep_preprocessed,
    )
    print(results)


if __name__ == "__main__":
    main()
