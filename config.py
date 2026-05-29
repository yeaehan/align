"""
align/config.py
---------------
Central configuration dataclasses for both pipelines.

RegistrationConfig  →  2D multi-round 4i alignment (NextGen4iPipeline)
ZStackConfig        →  3D z-stack 20x→63x registration (ZStackAlignmentPipeline)

Both are plain dataclasses: no logic, just typed parameters with defaults.
Keeping them here means every other module imports from one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# 2D MULTI-ROUND 4i PIPELINE CONFIG
# ============================================================================

@dataclass
class RegistrationConfig:
    """
    Configuration for the 2D multi-round 4i alignment pipeline.

    Required
    --------
    input_folder    : directory containing all round TIFFs for one sample
    output_folder   : where aligned TIFFs will be written
    reference_file  : path to the reference DAPI (round 1 ch00)

    Key optional parameters
    -----------------------
    pyramid_levels          : downscale factors for multi-scale rigid registration
                              [0.25, 0.5] means register at 25% then 50% resolution
    enable_nonrigid         : run optical flow after rigid registration
    apply_advanced_preprocessing : GPU top-hat background subtraction on DAPI
    skip_non_reference_dapi : don't save aligned DAPI channels from non-ref rounds
                              (saves disk space — you already have the reference DAPI)
    """

    # --- required ---
    input_folder:  str = ""
    output_folder: str = ""
    reference_file: str = ""

    # --- multi-scale rigid registration ---
    pyramid_levels: List[float] = field(default_factory=lambda: [0.25, 0.5])

    # --- quality gates ---
    min_overlap_ratio: float = 0.7   # minimum tissue overlap to attempt registration
    quality_threshold: float = 0.8   # NCC threshold; below this a warning is logged

    # --- feature-based registration (SIFT) ---
    max_features: int = 15000

    # --- compute ---
    n_workers:   int  = 4
    use_gpu:     bool = True
    use_gpu_transforms: bool = True

    # --- non-rigid optical flow ---
    enable_nonrigid:        bool = True
    enable_pass3_refinement: bool = True   # extra optical flow pass for difficult cases

    # --- preprocessing ---
    apply_advanced_preprocessing:   bool  = True
    preprocessing_tophat_radius:    int   = 64
    preprocessing_light_background: bool  = False

    # --- tissue masking ---
    tissue_mask_percentile: float = 5.0

    # --- DAPI detection patterns (used to identify the nuclear channel) ---
    dapi_patterns: List[str] = field(
        default_factory=lambda: [r'ch00', r'dapi', r'hoechst', r'nucleus']
    )

    # --- CLAHE contrast enhancement for registration (not applied to saved output) ---
    use_clahe_for_registration: bool  = True
    clahe_clip_limit:           float = 3.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    # --- I/O ---
    tiff_tile_size:           int  = 512
    cleanup_preprocessed:     bool = True
    skip_non_reference_dapi:  bool = True

    # --- legacy elastix support (disabled by default) ---
    use_elastix: bool = False

    def __post_init__(self) -> None:
        # ensure output directory exists immediately on config creation
        if self.output_folder:
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        # pyramid levels must be ascending so coarse→fine registration works correctly
        self.pyramid_levels = sorted(self.pyramid_levels)


# ============================================================================
# 3D Z-STACK ALIGNMENT CONFIG
# ============================================================================

@dataclass
class ZStackConfig:
    """
    Configuration for the 3D z-stack 20x→63x registration pipeline.

    The core problem: a 63x confocal z-stack covers a small sub-region of a
    large 20x widefield image. We need to find where that sub-region is in the
    20x image and warp each 20x channel into the 63x coordinate space.

    Required
    --------
    reference_zstack_63x : path to the 63x z-stack TIFF (_ref file)
    moving_images_20x    : list of 20x TIFF paths to align and warp
    output_folder        : where to save the warped outputs

    Key parameters
    --------------
    scale_factor          : computed from pixel sizes; ~3.25 for 20x→63x
    coarse_top_k          : number of template-match candidates to refine
                            increase for ambiguous/repetitive tissue regions
    final_crop_mode       : 'overlap' = crop to intersection of ref and moving
                            'full_reference' = keep full 63x field of view
    """

    # --- required ---
    reference_zstack_63x: str = ""
    moving_images_20x:    List[str] = field(default_factory=list)
    output_folder:        str = ""

    # --- z-plane selection for registration reference ---
    use_max_projection:      bool = True    # MIP is more robust than a single plane
    reference_z_plane:       Optional[int] = None   # used if use_max_projection=False
    max_projection_range:    Optional[Tuple[int, int]] = None  # (z_start, z_end) slice

    # --- physical pixel sizes (micrometers) ---
    # these define the scale_factor computed in __post_init__
    voxel_size_63x:   Tuple[float, float, float] = (0.1, 0.1, 0.49)   # (xy, xy, z)
    pixel_size_20x:   Tuple[float, float]        = (0.32472, 0.32472)  # (x, y)

    # --- registration ---
    pyramid_levels: List[float] = field(default_factory=lambda: [0.25, 0.5])
    max_features:   int         = 12000
    tissue_mask_percentile: float = 5.0

    # --- coarse template matching ---
    coarse_max_dim:          int   = 4096   # proxy image max dimension in pixels
    coarse_top_k:            int   = 5      # candidates to evaluate
    coarse_min_peak_distance: int  = 256    # min pixel distance between candidates
    coarse_min_score:        float = 0.10   # discard candidates below this NCC

    # --- refinement ---
    refine_max_dim:    int   = 4096
    roi_margin_factor: float = 1.4  # how much larger the ROI is vs the template hit

    # --- output ---
    final_crop_mode: str  = "overlap"  # "overlap" or "full_reference"
    save_debug:      bool = True       # saves a magenta/green overlay TIFF for QC
    debug_max_dim:   int  = 4000

    # scale_factor is derived, not user-set
    scale_factor: float = field(init=False)

    def __post_init__(self) -> None:
        # pixel_size_20x / voxel_size_63x(xy) gives the magnification ratio
        self.scale_factor = self.pixel_size_20x[0] / self.voxel_size_63x[0]
        if self.output_folder:
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        logger.info(f"ZStackConfig: scale_factor (20x→63x) = {self.scale_factor:.4f}x")