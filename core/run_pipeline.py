import argparse
import logging
from align.config import RegistrationConfig
from align.pipeline.orchestrator import AlignmentPipeline

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="CLI for 2D Multi-Round 4i Alignment")
    
    # --- Required arguments ---
    parser.add_argument("--input_folder", required=True, 
                        help="Directory containing all round TIFFs for one sample")
    parser.add_argument("--output_folder", required=True, 
                        help="Where aligned TIFFs will be written")
    parser.add_argument("--reference_file", required=True, 
                        help="Path to the reference DAPI (round 1 ch00)")
    
    # --- Optional arguments (overriding config defaults) ---
    parser.add_argument("--n_workers", type=int, default=4, 
                        help="Number of CPU workers (default: 4)")
    parser.add_argument("--no_gpu", action="store_true", 
                        help="Disable GPU processing")
    parser.add_argument("--disable_nonrigid", action="store_true", 
                        help="Skip the optical flow (non-rigid) registration step")
    
    args = parser.parse_args()

    # 1. Instantiate the dataclass with command-line arguments
    config = RegistrationConfig(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        reference_file=args.reference_file,
        n_workers=args.n_workers,
        use_gpu=not args.no_gpu,
        use_gpu_transforms=not args.no_gpu,
        enable_nonrigid=not args.disable_nonrigid
    )
    
    logger.info("Configuration loaded successfully:")
    logger.info(f"  Input: {config.input_folder}")
    logger.info(f"  Output: {config.output_folder}")
    logger.info(f"  GPU Enabled: {config.use_gpu}")
    logger.info(f"  Non-rigid alignment: {config.enable_nonrigid}")
    
    # 2. Initialize and run the pipeline
    pipeline = AlignmentPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()
