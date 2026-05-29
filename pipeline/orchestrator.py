import logging
from pathlib import Path

from align.config import RegistrationConfig
from align.io.reader import read_2d_as_float, discover_moving_files
from align.io.writer import save_tiff, save_debug_overlay
from align.core.tissue import TissueProcessor
from align.core.preprocessing import preprocess_dapi
from align.core.transforms import warp_affine
from align.registration.rigid import RigidRegistrar
from align.registration.nonrigid import OpticalFlowRegistrar

logger = logging.getLogger(__name__)

class AlignmentPipeline:
    def __init__(self, config: RegistrationConfig):
        self.config = config
        self.rigid_registrar = RigidRegistrar(
            pyramid_levels=config.pyramid_levels,
            max_features=config.max_features
        )
        
        if config.enable_nonrigid:
            self.nonrigid_registrar = OpticalFlowRegistrar(
                pyramid_levels=config.pyramid_levels,
                enable_pass3=config.enable_pass3_refinement,
                use_gpu=config.use_gpu,
            )
        else:
            self.nonrigid_registrar = None

    def run(self):
        logger.info("Starting 2D Alignment Pipeline...")
        ref_path = Path(self.config.reference_file)
        
        if not ref_path.exists():
            logger.error(f"Reference file not found: {ref_path}")
            return

        # 1. Load and prepare reference image
        logger.info(f"Loading reference: {ref_path.name}")
        ref_img = read_2d_as_float(str(ref_path))
        ref_prep = preprocess_dapi(
            ref_img, 
            tophat_radius=self.config.preprocessing_tophat_radius,
            use_gpu=self.config.use_gpu
        )
        ref_mask = TissueProcessor.create_tissue_mask(ref_prep, self.config.tissue_mask_percentile)

        # 2. Find all moving files in the folder
        moving_files = discover_moving_files(ref_path)
        logger.info(f"Found {len(moving_files)} moving files to align.")

        out_dir = Path(self.config.output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 3. Align each file to the reference
        for mov_file in moving_files:
            mov_path = Path(mov_file)
            logger.info(f"--- Aligning {mov_path.name} ---")
            
            mov_img = read_2d_as_float(str(mov_path))
            mov_prep = preprocess_dapi(
                mov_img,
                tophat_radius=self.config.preprocessing_tophat_radius,
                use_gpu=self.config.use_gpu
            )
            mov_mask = TissueProcessor.create_tissue_mask(mov_prep, self.config.tissue_mask_percentile)

            # Rigid Registration
            logger.info("Running Rigid Registration...")
            transform, rigid_ncc, method = self.rigid_registrar.register(
                ref_prep, mov_prep, ref_mask, mov_mask
            )
            logger.info(f"Rigid alignment complete (Method: {method}, NCC: {rigid_ncc:.4f})")
            
            # Warp using affine transform
            aligned_img = warp_affine(mov_img, transform, ref_img.shape, is_mask=False)

            # Non-Rigid Registration (Optical flow)
            if self.config.enable_nonrigid:
                logger.info("Running Non-Rigid Registration (Optical Flow)...")
                mov_mask_w = warp_affine(mov_mask.astype(np.uint8), transform, ref_img.shape, is_mask=True)
                aligned_img, base_ncc, final_ncc = self.nonrigid_registrar.refine(
                    ref_prep, aligned_img, ref_mask, mov_mask_w
                )
                logger.info(f"Non-rigid alignment complete (NCC: {base_ncc:.4f} -> {final_ncc:.4f})")

            # Output Generation
            save_tiff(aligned_img, str(out_dir / f"aligned_{mov_path.name}"))
            save_debug_overlay(ref_img, aligned_img, str(out_dir / f"qc_{mov_path.name}"))
            
        logger.info("Pipeline completed successfully!")