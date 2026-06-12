"""
align/registration/coarse.py
-----------------------------
Coarse localization for the z-stack pipeline.

The problem: a 63x sub-region image must be located within a large 20x
whole-section image. The scale difference is ~3.25x, meaning the 63x
field-of-view corresponds to only a small patch in the 20x image.

Strategy
--------
1. Build proxy images (both downscaled to ≤4096px) to make template
   matching fast on large images
2. Use the tissue-masked region of the reference as the template
3. Run cv2.matchTemplate on the moving proxy
4. Extract top-k non-overlapping peaks from the response map
5. Return candidate ROI bounding boxes in proxy coordinates

Why top-k?
----------
A single best match can fail in tissue sections with repetitive
structure (e.g. hippocampus, cortical layers). Top-k candidates
are all refined independently and the best one is chosen based on
a composite score (see registration/evaluate.py).

Memory safety note
------------------
We never build the full-scale moving image in memory. The proxy is
built at a scale that keeps the largest dimension ≤ coarse_max_dim.
This avoids OOM on 22k×16k images and also avoids the OpenCV
warpAffine >32k dimension crash.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

from align.config import ZStackConfig
from align.core.tissue import (
    TissueProcessor,
    center_on_canvas,
    normalize_to_uint8,
    resize_image,
    resize_to_shape,
)
from align.core.preprocessing import apply_clahe

logger = logging.getLogger(__name__)


# ============================================================================
# PROXY IMAGE CONSTRUCTION
# ============================================================================

def build_proxy_images(
    ref_full: np.ndarray,
    mov_orig: np.ndarray,
    config: ZStackConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build coarse proxy images for template matching.

    The proxy scale is chosen so that the largest dimension of either
    the ref or the scaled moving image fits within coarse_max_dim.

    Parameters
    ----------
    ref_full : full-resolution MIP of the 63x z-stack (H, W)
    mov_orig : full-resolution 20x anchor image (H, W)
    config   : ZStackConfig (scale_factor, coarse_max_dim)

    Returns
    -------
    (ref_proxy, mov_proxy, proxy_scale)
        ref_proxy   : downscaled reference image
        mov_proxy   : downscaled moving image (also scaled by scale_factor)
        proxy_scale : the scale factor applied to ref_full
    """
    mov_scaled_h = int(round(mov_orig.shape[0] * config.scale_factor))
    mov_scaled_w = int(round(mov_orig.shape[1] * config.scale_factor))

    proxy_scale = min(
        config.coarse_max_dim / max(
            ref_full.shape[0], ref_full.shape[1],
            mov_scaled_h,      mov_scaled_w,
        ),
        1.0,
    )
    proxy_scale = float(max(proxy_scale, 1e-4))

    ref_proxy = resize_image(ref_full, proxy_scale, is_mask=False)
    # moving proxy: scaled by both physical scale_factor AND proxy_scale
    mov_proxy = resize_image(mov_orig, config.scale_factor * proxy_scale, is_mask=False)

    logger.debug(
        f"Proxy: ref={ref_proxy.shape}, mov={mov_proxy.shape}, "
        f"proxy_scale={proxy_scale:.6f}"
    )
    return ref_proxy.astype(np.float32), mov_proxy.astype(np.float32), proxy_scale


# ============================================================================
# TOP-K TEMPLATE MATCHING
# ============================================================================

def topk_template_match(
    ref_proxy: np.ndarray,
    mov_proxy: np.ndarray,
    ref_mask_proxy: np.ndarray,
    roi_margin_factor: float = 1.4,
    top_k: int = 5,
    min_peak_distance: int = 256,
    min_score: float = 0.10,
) -> List[dict]:
    """
    Find top-k candidate locations for the reference template in the moving image.

    Algorithm
    ---------
    1. Extract the bounding box of the reference tissue mask as the template
    2. Run cv2.matchTemplate (TM_CCOEFF_NORMED)
    3. Iteratively extract the k best non-overlapping peaks:
       - Take the global max
       - Suppress a region around it (min_peak_distance radius)
       - Repeat

    Parameters
    ----------
    ref_proxy          : downscaled reference MIP
    mov_proxy          : downscaled moving image
    ref_mask_proxy     : binary tissue mask for ref_proxy
    roi_margin_factor  : how much larger the candidate ROI is vs the template hit
                         1.4 = 40% margin on each side
    top_k              : maximum number of candidates to return
    min_peak_distance  : minimum pixel separation between candidate peaks
    min_score          : discard candidates below this template match score

    Returns
    -------
    List of candidate dicts with keys:
        rank, score, touches_border,
        template_bbox_ref_proxy, hit_bbox_mov_proxy, roi_bbox_mov_proxy
    """
    ref_u8 = apply_clahe(normalize_to_uint8(ref_proxy))
    mov_u8 = apply_clahe(normalize_to_uint8(mov_proxy))

    # extract template from tissue bounding box of reference
    coords = cv2.findNonZero((ref_mask_proxy.astype(np.uint8) * 255))
    if coords is None:
        raise RuntimeError("Reference mask is empty — cannot build template")

    x, y, w, h = cv2.boundingRect(coords)
    margin = max(16, int(0.08 * min(w, h)))

    tx0 = max(0, x - margin)
    ty0 = max(0, y - margin)
    tx1 = min(ref_u8.shape[1], x + w + margin)
    ty1 = min(ref_u8.shape[0], y + h + margin)

    template = ref_u8[ty0:ty1, tx0:tx1]

    if template.shape[0] > mov_u8.shape[0] or template.shape[1] > mov_u8.shape[1]:
        raise RuntimeError(
            f"Template ({template.shape}) larger than moving proxy ({mov_u8.shape}). "
            "Consider increasing coarse_max_dim."
        )

    res      = cv2.matchTemplate(mov_u8, template, cv2.TM_CCOEFF_NORMED).astype(np.float32)
    res_work = res.copy()

    min_peak_distance = max(8, int(min_peak_distance))
    hit_w, hit_h      = template.shape[1], template.shape[0]
    candidates        = []

    for rank in range(int(top_k)):
        _, max_val, _, max_loc = cv2.minMaxLoc(res_work)

        if not np.isfinite(max_val) or float(max_val) < float(min_score):
            break

        hit_x, hit_y = max_loc

        # compute ROI around the hit with margin
        roi_w  = int(round(hit_w * roi_margin_factor))
        roi_h  = int(round(hit_h * roi_margin_factor))
        roi_cx = hit_x + hit_w // 2
        roi_cy = hit_y + hit_h // 2

        roi_x0 = max(0, roi_cx - roi_w // 2)
        roi_y0 = max(0, roi_cy - roi_h // 2)
        roi_x1 = min(mov_u8.shape[1], roi_x0 + roi_w)
        roi_y1 = min(mov_u8.shape[0], roi_y0 + roi_h)
        roi_x0 = max(0, roi_x1 - roi_w)
        roi_y0 = max(0, roi_y1 - roi_h)

        touches_border = bool(
            roi_x0 == 0 or roi_y0 == 0
            or roi_x1 >= mov_u8.shape[1]
            or roi_y1 >= mov_u8.shape[0]
        )

        candidates.append({
            "rank":                    int(rank + 1),
            "score":                   float(max_val),
            "touches_border":          touches_border,
            "template_bbox_ref_proxy": [int(tx0), int(ty0), int(tx1), int(ty1)],
            "hit_bbox_mov_proxy":      [int(hit_x), int(hit_y),
                                        int(hit_x + hit_w), int(hit_y + hit_h)],
            "roi_bbox_mov_proxy":      [int(roi_x0), int(roi_y0),
                                        int(roi_x1), int(roi_y1)],
        })

        # suppress this peak and surrounding region
        sx0 = max(0, hit_x - min_peak_distance)
        sy0 = max(0, hit_y - min_peak_distance)
        sx1 = min(res_work.shape[1], hit_x + min_peak_distance + 1)
        sy1 = min(res_work.shape[0], hit_y + min_peak_distance + 1)
        res_work[sy0:sy1, sx0:sx1] = -1.0

    logger.debug(f"Template matching found {len(candidates)} candidates")
    return candidates


# ============================================================================
# REFINEMENT IMAGE CONSTRUCTION
# ============================================================================

def build_refine_images(
    ref_full: np.ndarray,
    mov_orig: np.ndarray,
    roi_bbox_mov_proxy: List[int],
    proxy_scale: float,
    config: ZStackConfig,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Build a registration-ready image pair from a coarse candidate ROI.

    Takes the ROI in proxy coordinates, converts to original 20x coordinates,
    crops the moving image, then scales both ref and moving crop to a common
    canvas for registration.

    Parameters
    ----------
    ref_full           : full-resolution reference MIP
    mov_orig           : full-resolution 20x anchor image
    roi_bbox_mov_proxy : [x0, y0, x1, y1] in proxy pixel coordinates
    proxy_scale        : proxy_scale returned by build_proxy_images()
    config             : ZStackConfig

    Returns
    -------
    (ref_canvas, mov_canvas, refine_meta)
        Both canvases are the same shape, centered, float32.
        refine_meta contains coordinate bookkeeping needed to reconstruct
        the full original→reference transform after refinement.
    """
    x0p, y0p, x1p, y1p = roi_bbox_mov_proxy
    combined_scale = config.scale_factor * proxy_scale

    # convert proxy coordinates back to original 20x pixel space
    x0o = int(round(x0p / combined_scale))
    y0o = int(round(y0p / combined_scale))
    x1o = int(round(x1p / combined_scale))
    y1o = int(round(y1p / combined_scale))

    x0o = max(0, min(x0o, mov_orig.shape[1] - 1))
    y0o = max(0, min(y0o, mov_orig.shape[0] - 1))
    x1o = max(x0o + 1, min(x1o, mov_orig.shape[1]))
    y1o = max(y0o + 1, min(y1o, mov_orig.shape[0]))

    mov_roi_orig = mov_orig[y0o:y1o, x0o:x1o]

    # scale factors for the refine images
    mov_roi_scaled_h = int(round(mov_roi_orig.shape[0] * config.scale_factor))
    mov_roi_scaled_w = int(round(mov_roi_orig.shape[1] * config.scale_factor))

    refine_scale = min(
        config.refine_max_dim / max(
            ref_full.shape[0], ref_full.shape[1],
            mov_roi_scaled_h,  mov_roi_scaled_w,
        ),
        1.0,
    )
    refine_scale = float(max(refine_scale, 1e-4))

    ref_refine    = resize_image(ref_full,    refine_scale,                         is_mask=False)
    mov_roi_refine = resize_image(mov_roi_orig, config.scale_factor * refine_scale, is_mask=False)

    # center both on a common canvas (required so both have the same shape for registration)
    canvas_shape = (
        max(ref_refine.shape[0], mov_roi_refine.shape[0]),
        max(ref_refine.shape[1], mov_roi_refine.shape[1]),
    )

    ref_canvas, ref_offset = center_on_canvas(ref_refine, canvas_shape)
    mov_canvas, mov_offset = center_on_canvas(mov_roi_refine, canvas_shape)

    refine_meta = {
        "refine_scale":       float(refine_scale),
        "ref_offset":         [int(ref_offset[0]), int(ref_offset[1])],
        "mov_offset":         [int(mov_offset[0]), int(mov_offset[1])],
        "mov_roi_orig_bbox":  [int(x0o), int(y0o), int(x1o), int(y1o)],
        "canvas_shape":       [int(canvas_shape[0]), int(canvas_shape[1])],
        "combined_scale_proxy": float(combined_scale),
    }

    logger.debug(
        f"Refine images: ref={ref_canvas.shape}, mov={mov_canvas.shape}, "
        f"refine_scale={refine_scale:.6f}, roi_orig={[x0o,y0o,x1o,y1o]}"
    )

    return ref_canvas.astype(np.float32), mov_canvas.astype(np.float32), refine_meta


# ============================================================================
# NOTEBOOK API ALIASES & SINGLE-CANDIDATE VERSION
# ============================================================================

def build_coarse_proxy_images_from_orig(
    ref_full: np.ndarray,
    mov_orig: np.ndarray,
    config: ZStackConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Alias for build_proxy_images to match notebook interface.
    
    See build_proxy_images for full documentation.
    """
    return build_proxy_images(ref_full, mov_orig, config)


def masked_template_localization(
    ref_proxy: np.ndarray,
    mov_proxy: np.ndarray,
    ref_mask_proxy: np.ndarray,
    roi_margin_factor: float = 1.4,
) -> dict:
    """
    Single-candidate template matching (single best match only).
    
    Simplified version of topk_template_match that returns only
    the best candidate instead of top-k candidates.
    
    Parameters
    ----------
    ref_proxy : reference image at proxy scale
    mov_proxy : moving image at proxy scale
    ref_mask_proxy : reference tissue mask at proxy scale
    roi_margin_factor : ROI expansion factor (default 1.4)
    
    Returns
    -------
    dict with keys:
        - template_bbox_ref_proxy
        - hit_bbox_mov_proxy
        - roi_bbox_mov_proxy
        - score
    """
    candidates = topk_template_match(
        ref_proxy, mov_proxy, ref_mask_proxy,
        roi_margin_factor=roi_margin_factor,
        top_k=1,
        min_peak_distance=256,
        min_score=0.10,
    )
    
    if not candidates:
        raise RuntimeError("No candidates found during single template matching")
    
    best = candidates[0]
    return {
        "template_bbox_ref_proxy": best["template_bbox_ref_proxy"],
        "hit_bbox_mov_proxy": best["hit_bbox_mov_proxy"],
        "roi_bbox_mov_proxy": best["roi_bbox_mov_proxy"],
        "score": best["score"],
    }


def masked_template_localization_topk(
    ref_proxy: np.ndarray,
    mov_proxy: np.ndarray,
    ref_mask_proxy: np.ndarray,
    roi_margin_factor: float = 1.4,
    top_k: int = 5,
    min_peak_distance: int = 256,
    min_score: float = 0.10,
) -> List[dict]:
    """
    Alias for topk_template_match to match notebook interface.
    
    See topk_template_match for full documentation.
    """
    return topk_template_match(
        ref_proxy, mov_proxy, ref_mask_proxy,
        roi_margin_factor=roi_margin_factor,
        top_k=top_k,
        min_peak_distance=min_peak_distance,
        min_score=min_score,
    )


def build_refine_registration_images_from_orig(
    ref_full: np.ndarray,
    mov_orig: np.ndarray,
    roi_bbox_mov_proxy: List[int],
    proxy_scale: float,
    config: ZStackConfig,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Alias for build_refine_images to match notebook interface.
    
    See build_refine_images for full documentation.
    """
    return build_refine_images(ref_full, mov_orig, roi_bbox_mov_proxy, proxy_scale, config)
