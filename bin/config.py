from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class RegistrationConfig:
    input_folder: str
    output_folder: str
    reference_file: str
    pyramid_levels: List[float] = field(default_factory=lambda: [0.25, 0.5])
    use_elastix: bool = False
    use_gpu: bool = True
    min_overlap_ratio: float = 0.7
    quality_threshold: float = 0.8
    max_features: int = 15000
    n_workers: int = 4
    enable_nonrigid: bool = True
    apply_advanced_preprocessing: bool = True
    preprocessing_tophat_radius: int = 64
    preprocessing_light_background: bool = False
    tissue_mask_percentile: float = 5.0
    dapi_patterns: List[str] = field(default_factory=lambda: [r"ch00", r"dapi", r"hoechst", r"nucleus"])
    enable_pass3_refinement: bool = True
    tiff_tile_size: int = 512
    cleanup_preprocessed: bool = True
    skip_non_reference_dapi: bool = True
    use_gpu_transforms: bool = True
    use_clahe_for_registration: bool = True
    clahe_clip_limit: float = 3.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    def __post_init__(self) -> None:
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.pyramid_levels = sorted(self.pyramid_levels)
