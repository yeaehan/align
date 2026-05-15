import numpy as np
from skimage.registration import phase_cross_correlation
from .base import BaseAligner
from .transforms import apply_affine_transform


class PhaseCorrelationAligner(BaseAligner):
    """
    Translation-only alignment using Phase Cross Correlation.
    """
    
    def compute_transform(
        self, 
        fixed_image: np.ndarray, 
        moving_image: np.ndarray,
        fixed_mask: np.ndarray | None = None,
        moving_mask: np.ndarray | None = None,
        **kwargs
    ) -> np.ndarray:
        upsample_factor = kwargs.get("upsample_factor", 10)
        
        # skimage natively supports masking
        shift, error, diffphase = phase_cross_correlation(
            fixed_image, 
            moving_image, 
            upsample_factor=upsample_factor,
            reference_mask=fixed_mask,
            moving_mask=moving_mask
        )
        
        # Convert shift (y, x) to a 2x3 affine translation matrix for universal compatibility
        transform = np.eye(2, 3, dtype=np.float32)
        transform[0, 2] = shift[1]  # x translation
        transform[1, 2] = shift[0]  # y translation
        
        return transform

    def apply_transform(self, image: np.ndarray, transform: np.ndarray, **kwargs) -> np.ndarray:
        use_nearest = kwargs.get("use_nearest", False)
        return apply_affine_transform(image, transform, use_nearest=use_nearest)