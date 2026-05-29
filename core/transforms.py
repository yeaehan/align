"""
align/core/transforms.py
------------------------
Matrix and affine transformation utilities for image alignment.
"""

import cv2
import numpy as np

def identity() -> np.ndarray:
    """Return a 2x3 identity affine matrix."""
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

def compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Compose two 2x3 affine transformation matrices."""
    M1 = np.vstack([T1, [0.0, 0.0, 1.0]])
    M2 = np.vstack([T2, [0.0, 0.0, 1.0]])
    M3 = M1 @ M2
    return M3[:2, :].astype(np.float32)

def scale_translation_only(T: np.ndarray, scale: float) -> np.ndarray:
    """Scale the translation components (column 2) of a 2x3 affine matrix."""
    T_out = T.copy()
    T_out[:, 2] *= float(scale)
    return T_out

def upscale_to_full(T: np.ndarray, scale: float) -> np.ndarray:
    """Upscale an affine transform from a pyramid level back to full resolution."""
    return scale_translation_only(T, 1.0 / scale)

def warp_affine(img: np.ndarray, T: np.ndarray, out_shape: tuple, is_mask: bool = False) -> np.ndarray:
    """Apply affine warp to a 2D image."""
    h, w = out_shape[:2]
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    if is_mask:
        res = cv2.warpAffine(img.astype(np.uint8), T, (w, h), flags=interp)
        return res.astype(bool)
    return cv2.warpAffine(img.astype(np.float32), T, (w, h), flags=interp)