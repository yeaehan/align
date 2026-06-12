import argparse
import logging
from pathlib import Path

from align.config import RegistrationConfig, ZStackConfig
from align.io.reader import discover_moving_files, find_reference_file, expand_virtual_files
from align.pipeline.orchestrator import AlignmentPipeline
from align.pipeline.zstack import ZStackAlignmentPipeline

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="CLI for 2D or 3D alignment")
    
    # --- Required arguments ---
    parser.add_argument("--input_folder", required=True, 
                        help="Directory containing all round TIFFs for one sample")
    parser.add_argument("--output_folder", required=True, 
                        help="Where aligned TIFFs will be written")
    parser.add_argument("--ref", default="ch00",
                        help="2D mode: channel name to use as reference (e.g., 'ch00', 'ch01'). Default: ch00")
    parser.add_argument("--3d", dest="is_3d", action="store_true",
                        help="Run 3D z-stack alignment. Default is 2D alignment.")
    
    # --- Optional arguments (overriding config defaults) ---
    parser.add_argument("--n_workers", type=int, default=4, 
                        help="Number of CPU workers (default: 4)")
    parser.add_argument("--no_gpu", action="store_true", 
                        help="Disable GPU processing")
    parser.add_argument("--disable_nonrigid", action="store_true", 
                        help="Skip the optical flow (non-rigid) registration step")
    
    args = parser.parse_args()

    input_dir = Path(args.input_folder)
    if not input_dir.exists():
        logger.error(f"Input folder does not exist: {input_dir}")
        return

    if args.is_3d:
        ref_path, ref_issue = find_reference_file(input_dir)
        if ref_path is None:
            logger.error(f"Could not find 3D reference in {input_dir}: {ref_issue}")
            logger.info("3D mode expects exactly one TIFF with '_ref' in the filename.")
            return

        moving_files = discover_moving_files(ref_path)
        if not moving_files:
            logger.error(f"No moving TIFF files found for 3D alignment in {input_dir}")
            return

        logger.info(f"Using 3D reference file: {ref_path.name}")
        logger.info(f"Found {len(moving_files)} moving files for 3D alignment.")

        config = ZStackConfig(
            reference_zstack_63x=str(ref_path),
            moving_images_20x=moving_files,
            output_folder=args.output_folder,
            pyramid_levels=[0.25, 0.5],
        )

        logger.info("3D configuration loaded successfully:")
        logger.info(f"  Input: {args.input_folder}")
        logger.info(f"  Output: {config.output_folder}")
        logger.info(f"  Reference: {Path(config.reference_zstack_63x).name}")

        pipeline = ZStackAlignmentPipeline(config)
        results = pipeline.run()
        logger.info(f"3D alignment status: {results.get('status')}")
        return
    
    # 1. Find reference file by channel name
    image_extensions = {'.tif', '.tiff', '.TIF', '.TIFF', '.lif', '.LIF'}
    all_physical_files = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix in image_extensions
    ]
    
    if not all_physical_files:
        logger.error(f"No image files found in {input_dir}")
        return
    
    virtual_files = expand_virtual_files(all_physical_files)
    
    # Find reference file matching the channel name
    ref_channel = args.ref.lower()
    reference_file = None
    for vf in virtual_files:
        vf_name = vf.split("\\")[-1].split("/")[-1]
        if ref_channel in vf_name.lower():
            reference_file = vf
            break
    
    if reference_file is None:
        logger.error(f"No file matching channel '{args.ref}' found in {input_dir}")
        logger.info(f"Available files: {virtual_files}")
        return
    
    printable_ref = reference_file.split("\\")[-1].split("/")[-1]
    logger.info(f"Using reference file: {printable_ref}")
    
    moving_files = virtual_files
    
    # 2. Instantiate the dataclass with command-line arguments
    config = RegistrationConfig(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        reference_file=reference_file,
        n_workers=args.n_workers,
        moving_files=moving_files,
        use_gpu=not args.no_gpu,
        use_gpu_transforms=not args.no_gpu,
        enable_nonrigid=not args.disable_nonrigid
    )
    
    logger.info("Configuration loaded successfully:")
    logger.info(f"  Input: {config.input_folder}")
    logger.info(f"  Output: {config.output_folder}")
    logger.info(f"  Reference: {printable_ref}")
    logger.info(f"  GPU Enabled: {config.use_gpu}")
    logger.info(f"  Non-rigid alignment: {config.enable_nonrigid}")
    
    # 3. Initialize and run the pipeline
    pipeline = AlignmentPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()
