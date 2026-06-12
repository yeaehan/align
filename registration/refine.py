"""
align/registration/refine.py
----------------------------
Transform refinement and composition for z-stack alignment.

After coarse localization, the registration is refined at higher resolution.
This module handles conversion of transforms between different coordinate spaces:

1. refinement_space (downsampled, centered on canvas)
   ↓ (inv scale, remove centering offset)
2. proxy_space (proxy scale)
   ↓ (account for magnification scale_factor)
3. original_space (original 20x image)
   ↓ (geometric warp to reference coordinate frame)
4. reference_space (63x reference frame)

The full chain is reconstructed through matrix composition of the inverse transformations.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from align.core.transforms import (
    affine_to_3x3,
    translation_3x3,
    scale_3x3,
    warp_affine,
)

# ============================================================================
# TRANSFORM COMPOSITION & CONVERSION
# ============================================================================

def full_transform_from_refine_registration(
    M_refine: np.ndarray,
    refine_meta: Dict[str, Any],
    scale_factor_20x_to_63x: float,
) -> np.ndarray:
    """
    Convert a refined transform back to original→reference space.
    
    The refinement happens on centered, downscaled images. This function
    reconstructs the full transform chain:
    
        original_moving (20x) → reference_63x (MIP)
    
    by composing:
    1. M_refine (refinement canvas → refinement canvas)
    2. Inverse centering offsets (canvas → un-centered)
    3. Inverse refine_scale (un-centered at refine_scale → full resolution)
    4. scale_factor_20x_to_63x (20x coordinates → 63x coordinates)
    5. ROI offset (moving ROI origin in full 20x space)
    
    Parameters
    ----------
    M_refine : 2x3 array
        Transform computed by the registrar during refinement
    refine_meta : dict
        Metadata from build_refine_images containing:
        - refine_scale, ref_offset, mov_offset, mov_roi_orig_bbox, canvas_shape
    scale_factor_20x_to_63x : float
        Magnification ratio (pixel_size_20x / voxel_size_63x_xy)
    
    Returns
    -------
    2x3 array
        Transform mapping original moving pixels → reference pixels
    """
    refine_scale = float(refine_meta["refine_scale"])
    ref_top, ref_left = refine_meta["ref_offset"]
    mov_top, mov_left = refine_meta["mov_offset"]
    roi_x0, roi_y0, _, _ = refine_meta["mov_roi_orig_bbox"]
    
    # Build the full transform chain in reverse (undo each step):
    # Start: refined coordinate frame
    # Step 1: Remove moving image centering offset
    # Step 2: Remove reference centering offset
    # Step 3: Unscale from refine_scale back to full resolution
    # Step 4: Apply magnification scale factor (20x → 63x)
    # Step 5: Apply ROI offset (moving crop origin)
    
    # This is equivalent to:
    # T_full = T_ref_unscale @ T_ref_decenter @ M_refine @ T_mov_decenter @ T_mov_unscale @ T_roi_origin
    
    # But we build it as matrix products in the right order:
    M_full = (
        # Step 1: unscale at reference (divide by refine_scale)
        scale_3x3(1.0 / refine_scale)
        @ translation_3x3(-ref_left, -ref_top)  # remove ref centering offset
        @ affine_to_3x3(M_refine)               # the refined transform
        @ translation_3x3(mov_left, mov_top)    # apply moving centering offset
        @ scale_3x3(scale_factor_20x_to_63x * refine_scale)  # scale factor + refine_scale
        @ translation_3x3(-roi_x0, -roi_y0)    # apply ROI origin offset
    )
    
    return M_full[:2, :].astype(np.float32)


def evaluate_registration_candidate(
    ref_reg: np.ndarray,
    mov_reg: np.ndarray,
    ref_mask_reg: np.ndarray,
    mov_mask_reg: np.ndarray,
    M_refine: np.ndarray,
    final_ncc: float,
    coarse_score: float,
    touches_border: bool,
) -> Dict[str, float]:
    warped_mask = warp_affine(
        mov_mask_reg.astype(np.uint8),
        M_refine,
        ref_reg.shape,
        is_mask=True
    )

    overlap = ref_mask_reg & warped_mask
    overlap_pixels = int(np.sum(overlap))
    ref_pixels = max(int(np.sum(ref_mask_reg)), 1)
    mov_pixels = max(int(np.sum(warped_mask)), 1)

    overlap_ratio_ref = overlap_pixels / ref_pixels
    overlap_ratio_mov = overlap_pixels / mov_pixels

    a, b, _ = M_refine[0]
    c, d, _ = M_refine[1]
    det = float(a * d - b * c)
    scale_x = float(np.sqrt(a * a + c * c))
    scale_y = float(np.sqrt(b * b + d * d))

    scale_penalty = abs(scale_x - 1.0) + abs(scale_y - 1.0)
    reflection_penalty = 0.25 if det <= 0 else 0.0
    border_penalty = 0.05 if touches_border else 0.0

    selection_score = float(
        final_ncc
        + 0.35 * overlap_ratio_ref
        + 0.25 * overlap_ratio_mov
        + 0.10 * coarse_score
        - 0.15 * scale_penalty
        - reflection_penalty
        - border_penalty
    )

    return {
        "selection_score": float(selection_score),
        "overlap_pixels": int(overlap_pixels),
        "overlap_ratio_ref": float(overlap_ratio_ref),
        "overlap_ratio_mov": float(overlap_ratio_mov),
        "determinant": float(det),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "scale_penalty": float(scale_penalty),
        "reflection_penalty": float(reflection_penalty),
        "border_penalty": float(border_penalty),
    }
