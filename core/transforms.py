"""
align/core/transforms.py
------------------------
Matrix and affine transformation utilities for image alignment.
"""

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def identity() -> np.ndarray:
    """Return a 2x3 identity affine matrix."""
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

def compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Compose two 2x3 affine transformation matrices.
    
    Computes T1 @ T2, which applies T2 first, then T1.
    """
    M1 = np.vstack([T1, [0.0, 0.0, 1.0]])
    M2 = np.vstack([T2, [0.0, 0.0, 1.0]])
    M3 = M1 @ M2
    result = M3[:2, :].astype(np.float32)
    
    # Validate result
    if not np.all(np.isfinite(result)):
        logger.error(f"Compose produced invalid matrix. T1: {T1}, T2: {T2}")
        logger.error(f"Result: {result}")
        raise ValueError(f"Transform composition resulted in NaN or Inf values")
    
    return result

def scale_translation_only(T: np.ndarray, scale: float) -> np.ndarray:
    """Scale the translation components (column 2) of a 2x3 affine matrix."""
    T_out = T.copy()
    T_out[:, 2] *= float(scale)
    return T_out

def scale_affine_translation(T: np.ndarray, ratio: float) -> np.ndarray:
    T2 = T.copy().astype(np.float32)
    T2[0, 2] *= ratio
    T2[1, 2] *= ratio
    return T2

def upscale_to_full(T: np.ndarray, scale: float) -> np.ndarray:
    """Upscale an affine transform from a pyramid level back to full resolution."""
    return scale_translation_only(T, 1.0 / scale)

def warp_affine(img: np.ndarray, T: np.ndarray, out_shape: tuple, is_mask: bool = False) -> np.ndarray:
    """Apply affine warp to a 2D image.
    
    Parameters
    ----------
    img : 2D array
        Input image to warp
    T : 2x3 array
        Affine transformation matrix
    out_shape : tuple
        (height, width) or (height, width, channels) of output
    is_mask : bool
        If True, uses nearest neighbor interpolation
        
    Returns
    -------
    Warped image
    
    Raises
    ------
    ValueError
        If transform matrix contains NaN or invalid values
    """
    h, w = out_shape[:2]
    
    # Validate transform matrix
    if not np.all(np.isfinite(T)):
        logger.error(f"Invalid transform matrix detected: {T}")
        raise ValueError(f"Transform contains NaN or Inf values: {T}")
    
    # Log the transform values for debugging
    logger.debug(f"Warp affine: img.shape={img.shape}, T[0,2]={T[0,2]:.3f}, T[1,2]={T[1,2]:.3f}, output_size=({w}, {h})")
    
    # Check for extremely large transform values that could cause overflow
    max_translation = 100000  # reasonable limit for image coordinates
    if np.abs(T[0, 2]) > max_translation or np.abs(T[1, 2]) > max_translation:
        logger.warning(f"Transform translation is very large: T[0,2]={T[0,2]}, T[1,2]={T[1,2]}")
        # This might still work, but it's suspicious
    
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    if is_mask:
        res = cv2.warpAffine(img.astype(np.uint8), T, (w, h), flags=interp)
        return res.astype(bool)
    return cv2.warpAffine(img.astype(np.float32), T, (w, h), flags=interp)


# ============================================================================
# 3X3 MATRIX UTILITIES (for multi-level transform composition)
# ============================================================================

def affine_to_3x3(M: np.ndarray) -> np.ndarray:
    """
    Convert a 2x3 affine matrix to 3x3 homogeneous coordinates.
    
    Parameters
    ----------
    M : 2x3 array
        Affine transformation matrix
    
    Returns
    -------
    3x3 array
        Homogeneous coordinate matrix with bottom row [0, 0, 1]
    """
    return np.vstack([M.astype(np.float32), [0, 0, 1]])


def translation_3x3(tx: float, ty: float) -> np.ndarray:
    """
    Create a 3x3 translation matrix.
    
    Parameters
    ----------
    tx, ty : float
        Translation amounts in x and y directions
    
    Returns
    -------
    3x3 array
        Translation matrix
    """
    T = np.eye(3, dtype=np.float32)
    T[0, 2] = float(tx)
    T[1, 2] = float(ty)
    return T


def scale_3x3(sx: float, sy: float = None) -> np.ndarray:
    """
    Create a 3x3 scaling matrix.
    
    Parameters
    ----------
    sx : float
        Scale factor for x direction
    sy : float, optional
        Scale factor for y direction. If None, uses sx for both.
    
    Returns
    -------
    3x3 array
        Scaling matrix
    """
    if sy is None:
        sy = sx
    S = np.eye(3, dtype=np.float32)
    S[0, 0] = float(sx)
    S[1, 1] = float(sy)
    return S


def compose_affine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compose two 2x3 affine matrices via 3x3 homogeneous representation.
    
    Computes the composition A @ B (applies B first, then A).
    
    Parameters
    ----------
    A, B : 2x3 arrays
        Affine transformation matrices
    
    Returns
    -------
    2x3 array
        Composed transformation
    """
    C = affine_to_3x3(A) @ affine_to_3x3(B)
    return C[:2, :].astype(np.float32)
