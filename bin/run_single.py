from __future__ import annotations

import argparse

from .config import RegistrationConfig
from .pipeline import NextGen4iPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run alignment for one folder.")
    parser.add_argument("--input-folder", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--reference-file", required=True)
    parser.add_argument("--pyramid-levels", nargs="+", type=float, default=[0.25, 0.5])
    parser.add_argument("--no-nonrigid", action="store_true")
    parser.add_argument("--no-pass3", action="store_true")
    parser.add_argument("--keep-preprocessed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RegistrationConfig(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        reference_file=args.reference_file,
        pyramid_levels=args.pyramid_levels,
        enable_nonrigid=not args.no_nonrigid,
        enable_pass3_refinement=not args.no_pass3,
        cleanup_preprocessed=not args.keep_preprocessed,
        apply_advanced_preprocessing=True,
        tissue_mask_percentile=1.0,
        skip_non_reference_dapi=True,
        use_gpu_transforms=True,
        use_clahe_for_registration=True,
    )
    results = NextGen4iPipeline(config).run()
    print(results)


if __name__ == "__main__":
    main()
