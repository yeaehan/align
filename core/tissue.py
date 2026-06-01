"""
align/core/tissue.py
--------------------
Tissue masking and registration quality metrics.

TissueProcessor
    Creates binary tissue masks from DAPI/nuclear images using Otsu thresholding
    with a percentile fallback for sparse tissue. Also builds Laplacian-weighted
    maps so registration focuses on nuclei-rich areas rather than empty background.

WeightedNCC
    Computes weighted normalized cross-correlation between two images.
    Used throughout the pipeline as the primary registration quality metric.

Why weight by tissue?
    Empty background regions are perfectly correlated by definition (both black),
    which inflates NCC scores and can mislead the optimizer. Weighting by the
    Laplacian contrast inside the tissue mask focuses the metric on the
    structured signal that actually matters for alignment.

Cache note
----------
TissueProcessor._mask_cache is a class-level dict that persists across
pipeline runs within a session. This is intentional for performance (tissue
masking is called many times on the same images at different scales), but
call TissueProcessor.clear_cache() between unrelated batches to avoid
stale entries.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import cv2
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


# ============================================================================
# IMAGE NORMALIZATION HELPERS
# ============================================================================

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Linearly rescale image to [0, 255] uint8.

    If the image is flat (max ≈ min), returns a zero array rather than
    dividing by near-zero.
    """
    img  = img.astype(np.float32)
    mn   = float(np.min(img))
    mx   = float(np.max(img))
    if mx <= mn + 1e-8:
        return np.zeros(img.shape, dtype=np.uint8)
    out = (img - mn) / (mx - mn)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def resize_image(
    img: np.ndarray,
    factor: float,
    is_mask: bool = False,
) -> np.ndarray:
    """
    Resize a 2D image by a scale factor.

    Uses INTER_AREA for downscaling (avoids aliasing) and INTER_CUBIC for
    upscaling. Masks use INTER_NEAREST to preserve binary values.
    """
    if factor == 1.0:
        return img.copy()
    h, w    = img.shape
    new_h   = max(1, int(round(h * factor)))
    new_w   = max(1, int(round(w * factor)))
    interp  = (cv2.INTER_NEAREST if is_mask
               else (cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC))
    src     = img.astype(np.uint8) if is_mask else img.astype(np.float32)
    out     = cv2.resize(src, (new_w, new_h), interpolation=interp)
    return out.astype(bool) if is_mask else out.astype(np.float32)


def resize_to_shape(
    img: np.ndarray,
    out_shape: Tuple[int, int],
    is_mask: bool = False,
) -> np.ndarray:
    """Resize a 2D image to an exact (height, width) shape."""
    out_h, out_w = out_shape
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    src    = img.astype(np.uint8) if is_mask else img.astype(np.float32)
    out    = cv2.resize(src, (out_w, out_h), interpolation=interp)
    return out.astype(bool) if is_mask else out.astype(np.float32)


def center_on_canvas(
    img: np.ndarray,
    canvas_shape: Tuple[int, int],
    is_mask: bool = False,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Place an image centered on a larger canvas.
    
    Useful for aligning images of different sizes by padding with zeros
    before registration on a uniform canvas.
    
    Parameters
    ----------
    img : 2D array
        Input image to center
    canvas_shape : tuple
        (canvas_height, canvas_width) of the output
    is_mask : bool
        If True, treats img as binary mask
    
    Returns
    -------
    canvas : 2D array
        Output canvas with centered image
    offset : tuple
        (top, left) offset where the image was placed
    """
    canvas_h, canvas_w = canvas_shape
    h, w = img.shape
    top = (canvas_h - h) // 2
    left = (canvas_w - w) // 2

    if is_mask:
        canvas = np.zeros((canvas_h, canvas_w), dtype=bool)
        canvas[top:top + h, left:left + w] = img.astype(bool)
    else:
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        canvas[top:top + h, left:left + w] = img.astype(np.float32)

    return canvas, (top, left)


# ============================================================================
# TISSUE PROCESSOR
# ============================================================================

class TissueProcessor:
    """
    Creates binary tissue masks and Laplacian-weighted maps from DAPI images.

    Masking strategy
    ----------------
    1. Downsample to ≤2048px for fast processing
    2. Gaussian blur to suppress noise
    3. Otsu threshold — reliable for most staining densities
    4. Fall back to percentile threshold if Otsu finds <20% tissue coverage
       (happens with sparse staining or low-magnification images)
    5. Morphological close + dilate to fill small holes
    6. Binary fill holes to handle hollow tissue structures
    7. Upsample mask back to original resolution

    Weight map strategy
    -------------------
    The Laplacian of the DAPI image highlights nuclei edges (high-frequency
    structure). Pixels with more nuclear structure get higher weights,
    focusing registration on information-dense regions.
    """

    _mask_cache: Dict[str, np.ndarray] = {}

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the mask cache. Call between unrelated batches."""
        cls._mask_cache.clear()

    @staticmethod
    def create_tissue_mask(
        img: np.ndarray,
        percentile: float = 5.0,
    ) -> np.ndarray:
        """
        Create a binary tissue mask from a DAPI/nuclear image.

        Parameters
        ----------
        img        : 2D float image in [0, 1]
        percentile : used for fallback threshold if Otsu finds sparse tissue

        Returns
        -------
        Boolean mask, same shape as img.
        """
        # cache key: shape + percentile + mean (fast proxy for image identity)
        cache_key = f"{img.shape}_{percentile}_{float(np.mean(img)):.6f}"
        if cache_key in TissueProcessor._mask_cache:
            return TissueProcessor._mask_cache[cache_key].copy()

        logger.debug("Creating tissue mask...")
        original_shape = img.shape
        max_size = 2048

        # downsample for speed
        if max(img.shape) > max_size:
            scale_factor = max_size / max(img.shape)
            img_small = resize_image(img, scale_factor, is_mask=False)
        else:
            img_small = img

        img_u8  = normalize_to_uint8(img_small)
        blurred = cv2.GaussianBlur(img_u8, (5, 5), 0)

        # primary: Otsu threshold
        _, thresh_otsu = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        otsu_coverage = np.sum(thresh_otsu > 0) / thresh_otsu.size

        if otsu_coverage < 0.2:
            # fallback: top-percentile threshold for sparse tissue
            threshold_val = np.percentile(img_u8, 100 - percentile)
            _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
        else:
            thresh = thresh_otsu

        # morphological cleanup
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(cleaned, kernel, iterations=3)
        mask    = ndimage.binary_fill_holes(dilated > 0)

        # restore to original resolution
        if mask.shape != original_shape:
            mask = resize_to_shape(mask.astype(np.uint8), original_shape, is_mask=True)

        mask = mask.astype(bool)

        if mask.shape != original_shape:
            raise RuntimeError(
                f"Tissue mask shape mismatch: got {mask.shape}, expected {original_shape}"
            )

        coverage = np.sum(mask) / mask.size
        logger.debug(f"Tissue mask coverage: {coverage:.1%}")

        TissueProcessor._mask_cache[cache_key] = mask.copy()
        return mask

    @staticmethod
    def create_weight_map(
        dapi_img: np.ndarray,
        base_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Create a Laplacian-based weight map from a DAPI image.

        Pixels inside the tissue mask are weighted by local contrast
        (Laplacian magnitude), scaled to [0.2, 1.0]. Background pixels
        get weight 0.

        Parameters
        ----------
        dapi_img  : 2D float image in [0, 1]
        base_mask : boolean tissue mask, same shape as dapi_img

        Returns
        -------
        float32 weight map in [0, 1], same shape as dapi_img
        """
        img      = np.clip(dapi_img.astype(np.float32), 0.0, 1.0)
        lap      = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
        contrast = np.abs(lap)

        vals = contrast[base_mask]
        if vals.size > 0 and np.max(vals) > 0:
            contrast = np.clip(contrast, 0, np.percentile(vals, 99))
            mx = float(np.max(contrast))
            if mx > 0:
                contrast /= mx
        else:
            contrast = np.zeros_like(contrast)

        weight = np.zeros_like(contrast, dtype=np.float32)
        # baseline weight of 0.2 ensures all tissue pixels contribute
        # contrast component pushes nuclei-rich areas toward 1.0
        weight[base_mask] = 0.2 + 0.8 * contrast[base_mask]
        return weight


# ============================================================================
# REGISTRATION QUALITY METRICS
# ============================================================================

def compute_weighted_ncc(
    ref: np.ndarray,
    mov: np.ndarray,
    weight: np.ndarray,
) -> float:
    """
    Compute weighted normalized cross-correlation between two images.

    NCC = Σ w(r-μr)(m-μm) / sqrt(Σ w(r-μr)² · Σ w(m-μm)²)

    Returns a value in [0, 1]. Returns 0.0 if the weighted sum is too
    small (degenerate overlap) or if either image has zero variance.

    Parameters
    ----------
    ref, mov : 2D float images of the same shape
    weight   : per-pixel weights, same shape (from create_weight_map)
    """
    w_sum = float(np.sum(weight))
    if w_sum < 100:
        return 0.0

    ref_mean = float(np.sum(ref * weight) / w_sum)
    mov_mean = float(np.sum(mov * weight) / w_sum)

    ref_c = ref - ref_mean
    mov_c = mov - mov_mean

    numerator = float(np.sum(weight * ref_c * mov_c))
    ref_var   = float(np.sum(weight * ref_c ** 2))
    mov_var   = float(np.sum(weight * mov_c ** 2))

    if ref_var <= 0 or mov_var <= 0:
        return 0.0

    ncc = numerator / np.sqrt(ref_var * mov_var)
    return float(max(0.0, min(1.0, ncc)))


def score_transform(
    ref: np.ndarray,
    mov: np.ndarray,
    ref_mask: np.ndarray,
    mov_mask: np.ndarray,
    transform: np.ndarray,
) -> float:
    """
    Apply a transform to mov and compute weighted NCC against ref.

    This is the primary way registration methods are evaluated: warp the
    moving image with the candidate transform, measure overlap quality.

    Parameters
    ----------
    ref, mov   : 2D float images
    ref_mask   : binary tissue mask for ref
    mov_mask   : binary tissue mask for mov (pre-warp)
    transform  : 2×3 affine matrix

    Returns
    -------
    NCC in [0, 1]
    """
    from align.core.transforms import warp_affine  # local import avoids circular

    mov_warped = warp_affine(mov, transform, ref.shape, is_mask=False)
    mov_mask_w = warp_affine(mov_mask.astype(np.uint8), transform, ref.shape, is_mask=True)

    valid = ref_mask & mov_mask_w
    if np.sum(valid) < 100:
        return 0.0

    weight = TissueProcessor.create_weight_map(ref, valid)
    return compute_weighted_ncc(ref, mov_warped, weight)
