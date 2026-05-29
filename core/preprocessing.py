"""
align/core/preprocessing.py
----------------------------
DAPI channel preprocessing before registration.

The goal is to produce an image where:
  - Background fluorescence is removed (top-hat morphological filter)
  - Nuclear edges are sharpened (Laplacian enhancement)
  - Intensity range is normalized to [0, 1]

GPU path  : uses CuPy + cupyx for fast processing of large (16k×22k) images
CPU path  : falls back gracefully when CuPy is unavailable

apply_clahe
    Contrast-limited adaptive histogram equalization.
    Applied before feature detection (SIFT/ORB) and phase correlation,
    NOT saved to output — only used during registration.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# GPU detection — done once at import time
try:
    import cupy as cp
    from cupyx.scipy.ndimage import (
        grey_opening,
        laplace as cp_laplace,
        median_filter as cp_median_filter,
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.debug("CuPy not available; preprocessing will use CPU path")


# ============================================================================
# CLAHE (CONTRAST ENHANCEMENT FOR REGISTRATION)
# ============================================================================

def apply_clahe(
    img_u8: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE to a uint8 image.

    Used before feature detection and phase correlation to normalize
    local contrast across the image. Not applied to saved outputs.

    Parameters
    ----------
    img_u8         : uint8 image
    clip_limit     : threshold for contrast limiting
    tile_grid_size : size of grid for histogram equalization

    Returns
    -------
    uint8 image with enhanced local contrast
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    return clahe.apply(img_u8)


# ============================================================================
# GPU PREPROCESSING
# ============================================================================

def preprocess_dapi_gpu(
    img: "cp.ndarray",
    tophat_radius: int = 64,
    light_background: bool = False,
) -> "cp.ndarray":
    """
    GPU-accelerated DAPI preprocessing pipeline.

    Steps
    -----
    1. Median filter (radius 3) — removes salt-and-pepper noise
    2. Top-hat background subtraction — removes uneven illumination
       Uses grey_opening with a disk-shaped structuring element
    3. Laplacian edge enhancement — sharpens nuclear boundaries
    4. Percentile normalization — clips to [0.5%, 99.8%] and rescales to [0, 1]

    Parameters
    ----------
    img              : CuPy 2D float array, values in [0, 1]
    tophat_radius    : radius of the morphological disk (pixels)
                       64 works for 20x; reduce to 32 for 63x crops
    light_background : if True, invert before and after to handle
                       bright-background images

    Returns
    -------
    CuPy float32 array in [0, 1]
    """
    if not GPU_AVAILABLE:
        raise RuntimeError(
            "preprocess_dapi_gpu called but CuPy is not available. "
            "Use preprocess_dapi_cpu instead."
        )

    # median filter to remove noise spikes
    img = cp_median_filter(img, size=3)

    # build circular structuring element on GPU
    y, x = cp.ogrid[-tophat_radius:tophat_radius + 1,
                    -tophat_radius:tophat_radius + 1]
    kernel = (x ** 2 + y ** 2 <= tophat_radius ** 2)

    if light_background:
        img = 1.0 - img

    # top-hat = image - opening (background estimate)
    background = grey_opening(img, structure=kernel)
    img = img - background

    if light_background:
        img = 1.0 - img

    # Laplacian edge enhancement
    lap = cp_laplace(img)
    lap /= (cp.abs(lap).max() + 1e-8)
    img = img + 0.3 * lap

    # percentile normalization
    vmin = cp.quantile(img, 0.005)
    vmax = cp.quantile(img, 0.998)
    img = cp.clip((img - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)

    return img


# ============================================================================
# CPU PREPROCESSING
# ============================================================================

def preprocess_dapi_cpu(
    img: np.ndarray,
    tophat_radius: int = 64,
    light_background: bool = False,
) -> np.ndarray:
    """
    CPU fallback for DAPI preprocessing.

    Same algorithm as preprocess_dapi_gpu but using scipy + numpy.
    Significantly slower on large images (~10-20x) but functionally identical.

    Parameters
    ----------
    img              : numpy 2D float array, values in [0, 1]
    tophat_radius    : radius of the morphological disk (pixels)
    light_background : invert for bright-background images
    """
    from scipy.ndimage import median_filter, grey_opening, laplace

    img = img.astype(np.float32)
    img = median_filter(img, size=3)

    # build circular structuring element
    y, x = np.ogrid[-tophat_radius:tophat_radius + 1,
                    -tophat_radius:tophat_radius + 1]
    kernel = (x ** 2 + y ** 2 <= tophat_radius ** 2)

    if light_background:
        img = 1.0 - img

    background = grey_opening(img, structure=kernel)
    img = img - background

    if light_background:
        img = 1.0 - img

    lap = laplace(img)
    lap_max = np.abs(lap).max()
    if lap_max > 1e-8:
        lap /= lap_max
    img = img + 0.3 * lap

    vmin = float(np.quantile(img, 0.005))
    vmax = float(np.quantile(img, 0.998))
    img = np.clip((img - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)

    return img.astype(np.float32)


# ============================================================================
# UNIFIED ENTRY POINT
# ============================================================================

def preprocess_dapi(
    img: np.ndarray,
    tophat_radius: int = 64,
    light_background: bool = False,
    use_gpu: bool = True,
    gpu_id: int = 0,
) -> np.ndarray:
    """
    Preprocess a DAPI channel image, dispatching to GPU or CPU automatically.

    Parameters
    ----------
    img              : 2D float32 numpy array in [0, 1]
    tophat_radius    : top-hat structuring element radius
    light_background : set True for bright-field / inverted images
    use_gpu          : attempt GPU processing if available
    gpu_id           : which GPU to use (0-indexed)

    Returns
    -------
    Preprocessed float32 numpy array in [0, 1]
    """
    if use_gpu and GPU_AVAILABLE:
        try:
            with cp.cuda.Device(gpu_id):
                img_gpu = cp.asarray(img.astype(np.float32))
                result  = preprocess_dapi_gpu(img_gpu, tophat_radius, light_background)
                return cp.asnumpy(result)
        except Exception as e:
            logger.warning(f"GPU preprocessing failed, falling back to CPU: {e}")

    return preprocess_dapi_cpu(img, tophat_radius, light_background)