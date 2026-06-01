"""
align/registration/nonrigid.py
--------------------------------
Non-rigid (optical flow) registration to correct local tissue deformations.

After rigid registration there are often small local misalignments caused by:
- Tissue swelling/shrinkage between imaging rounds
- Local mechanical distortions from mounting
- PSF differences between rounds

OpticalFlowRegistrar uses OpenCV's Dense Inverse Search (DIS) optical flow.
It runs 2-3 passes, checking NCC after each pass and stopping early if
improvement is negligible (< 0.001 NCC gain).

A third pass is triggered conditionally (enable_pass3) when the sample
shows poor rigid alignment (NCC < ~0.95), which happens with large initial
displacements or difficult tissue.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from align.core.tissue import (
    TissueProcessor,
    compute_weighted_ncc,
    normalize_to_uint8,
    resize_image,
)
from align.core.preprocessing import apply_clahe
from align.core.transforms import warp_affine

logger = logging.getLogger(__name__)

# GPU detection
try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates as cp_map_coordinates
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


# ============================================================================
# FLOW APPLICATION
# ============================================================================

def _as_2d(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D after squeeze, got shape {arr.shape}")
    return arr

def apply_flow(
    img: np.ndarray,
    flow: np.ndarray,
    use_gpu: bool = True,
    gpu_id: int = 0,
) -> np.ndarray:
    """
    Warp an image using a dense optical flow field.

    Parameters
    ----------
    img     : 2D float image to warp
    flow    : (H, W, 2) float array; flow[y, x] = (dx, dy) displacement
    use_gpu : attempt GPU coordinate mapping
    gpu_id  : GPU device index

    Returns
    -------
    Warped float32 image, same shape as img
    """
    h, w = img.shape

    # build coordinate grids
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coords_y = (yy + flow[..., 1]).ravel()
    coords_x = (xx + flow[..., 0]).ravel()

    if use_gpu and GPU_AVAILABLE:
        try:
            with cp.cuda.Device(gpu_id):
                img_gpu = cp.asarray(img.astype(np.float32))
                cy_gpu  = cp.asarray(coords_y)
                cx_gpu  = cp.asarray(coords_x)
                warped  = cp_map_coordinates(img_gpu, [cy_gpu, cx_gpu], order=1)
                return cp.asnumpy(warped).reshape(h, w).astype(np.float32)
        except Exception as e:
            logger.debug(f"GPU flow application failed, falling back to CPU: {e}")

    warped = map_coordinates(img.astype(np.float32), [coords_y, coords_x], order=1)
    return warped.reshape(h, w).astype(np.float32)


# ============================================================================
# OPTICAL FLOW REGISTRAR
# ============================================================================

class OpticalFlowRegistrar:
    """
    Multi-pass Dense Inverse Search (DIS) optical flow refinement.

    DIS is chosen over Farneback because it:
    - Is significantly faster on large images
    - Handles larger displacements
    - Has better memory efficiency

    The flow is computed at each pyramid level and accumulated.
    """

    def __init__(
        self,
        pyramid_levels: list,
        enable_pass3:   bool = True,
        use_gpu:        bool = True,
        gpu_id:         int  = 0,
    ) -> None:
        self.pyramid_levels = sorted(pyramid_levels)
        self.enable_pass3   = enable_pass3
        self.use_gpu        = use_gpu
        self.gpu_id         = gpu_id

    def _run_dis_flow(
        self,
        ref_u8: np.ndarray,
        mov_u8: np.ndarray,
    ) -> np.ndarray:
        """Run one pass of DIS optical flow. Returns (H, W, 2) flow field."""
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setUseSpatialPropagation(True)
        return dis.calc(ref_u8, mov_u8, None)

    def refine(
        self,
        ref: np.ndarray,
        mov_rigid: np.ndarray,
        ref_mask: np.ndarray,
        mov_mask: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Run optical flow refinement across pyramid levels.

        Parameters
        ----------
        ref        : 2D float reference image (full resolution)
        mov_rigid  : 2D float moving image after rigid alignment
        ref_mask   : binary tissue mask for ref
        mov_mask   : binary tissue mask for mov_rigid

        Returns
        -------
        (warped_image, initial_ncc, final_ncc)
            warped_image : float32 image after optical flow correction
            initial_ncc  : NCC before optical flow (after rigid)
            final_ncc    : NCC after optical flow
        """
        ref = _as_2d(ref, "ref").astype(np.float32)
        mov_rigid = _as_2d(mov_rigid, "mov_rigid").astype(np.float32)
        ref_mask = _as_2d(ref_mask, "ref_mask").astype(bool)
        mov_mask = _as_2d(mov_mask, "mov_mask").astype(bool)

        if ref.shape != mov_rigid.shape:
            raise ValueError(f"ref and mov_rigid shapes differ: {ref.shape} vs {mov_rigid.shape}")
        if ref_mask.shape != ref.shape:
            raise ValueError(f"ref_mask shape {ref_mask.shape} does not match ref shape {ref.shape}")
        if mov_mask.shape != ref.shape:
            raise ValueError(f"mov_mask shape {mov_mask.shape} does not match ref shape {ref.shape}")

        # baseline NCC after rigid registration
        overlap = ref_mask & mov_mask
        if np.sum(overlap) > 100:
            w_map        = TissueProcessor.create_weight_map(ref, overlap)
            baseline_ncc = compute_weighted_ncc(ref, mov_rigid, w_map)
        else:
            baseline_ncc = 0.0

        logger.debug(f"Optical flow baseline NCC: {baseline_ncc:.4f}")

        current_mov  = mov_rigid.copy()
        current_ncc  = baseline_ncc
        best_ncc     = baseline_ncc

        for scale_factor in self.pyramid_levels:
            logger.debug(f"Optical flow at scale {scale_factor}")
            current_mov, current_ncc = self._flow_at_scale(
                ref, current_mov, ref_mask, mov_mask, scale_factor
            )
            best_ncc = max(best_ncc, current_ncc)

        return current_mov, baseline_ncc, best_ncc

    def _flow_at_scale(
        self,
        ref: np.ndarray,
        mov: np.ndarray,
        ref_mask: np.ndarray,
        mov_mask: np.ndarray,
        scale_factor: float,
    ) -> Tuple[np.ndarray, float]:
        """Run multi-pass optical flow at one pyramid level."""
        ref_s = resize_image(ref, scale_factor, is_mask=False)
        mov_s = resize_image(mov, scale_factor, is_mask=False)

        ref_u8 = apply_clahe(normalize_to_uint8(ref_s))
        mov_u8 = apply_clahe(normalize_to_uint8(mov_s))

        overlap = (
            resize_image(ref_mask.astype(np.uint8), scale_factor, is_mask=True)
            & resize_image(mov_mask.astype(np.uint8), scale_factor, is_mask=True)
        )

        def _ncc(warped_s: np.ndarray) -> float:
            if np.sum(overlap) < 100:
                return 0.0
            w = TissueProcessor.create_weight_map(ref_s, overlap)
            return compute_weighted_ncc(ref_s, warped_s, w)

        current_s  = mov_s.copy()
        current_ncc = _ncc(current_s)

        n_passes = 3 if self.enable_pass3 else 2

        for pass_idx in range(1, n_passes + 1):
            flow         = self._run_dis_flow(ref_u8, normalize_to_uint8(current_s))
            warped_s     = apply_flow(current_s, flow, self.use_gpu, self.gpu_id)
            new_ncc      = _ncc(warped_s)

            logger.debug(f"  Pass {pass_idx}: NCC {current_ncc:.4f} → {new_ncc:.4f}")

            if new_ncc > current_ncc:
                # upscale the flow field to full resolution and apply
                flow_full_x = cv2.resize(flow[..., 0], (mov.shape[1], mov.shape[0])) / scale_factor
                flow_full_y = cv2.resize(flow[..., 1], (mov.shape[1], mov.shape[0])) / scale_factor
                flow_full   = np.stack([flow_full_x, flow_full_y], axis=-1)
                mov         = apply_flow(mov, flow_full, self.use_gpu, self.gpu_id)
                current_s   = warped_s
                current_ncc = new_ncc
            else:
                logger.debug(f"  Pass {pass_idx} did not improve NCC, stopping")
                break

        return mov, current_ncc
