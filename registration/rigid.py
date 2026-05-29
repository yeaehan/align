"""
align/registration/rigid.py
----------------------------
Rigid (affine) registration methods.

Three methods are tried at each pyramid level and the best NCC wins:

PhaseCorrelation
    Fast Fourier-based translation estimation.
    Reliable when tissue coverage is high and there's no rotation.
    Runs first because it's cheapest.

ECCRegistration
    Enhanced Correlation Coefficient (OpenCV findTransformECC).
    Handles small rotations. Uses the result from phase as initialization.
    Can be slow if it doesn't converge.

FeatureBasedRegistration
    SIFT keypoint detection + FLANN matching + RANSAC affine estimation.
    Most robust to large displacements and rotations.
    Run last because it's most expensive.

RigidRegistrar
    Orchestrates all three methods across pyramid levels.
    At each scale: run methods, keep best NCC, accumulate transform.
    Finally rescales the accumulated transform back to full resolution.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

from align.config import RegistrationConfig, ZStackConfig
from align.core.tissue import (
    TissueProcessor,
    compute_weighted_ncc,
    normalize_to_uint8,
    resize_image,
    score_transform,
)
from align.core.transforms import (
    compose,
    identity,
    scale_translation_only,
    upscale_to_full,
    warp_affine,
)
from align.core.preprocessing import apply_clahe

logger = logging.getLogger(__name__)

# type alias — either config type works for rigid registration
_Config = RegistrationConfig | ZStackConfig


# ============================================================================
# PHASE CORRELATION
# ============================================================================

def register_phase(
    ref: np.ndarray,
    mov: np.ndarray,
    ref_mask: np.ndarray,
    mov_mask: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Estimate a pure translation using phase cross-correlation.

    Parameters
    ----------
    ref, mov   : 2D float images (CLAHE-enhanced uint8 expected)
    ref_mask   : binary tissue mask for ref
    mov_mask   : binary tissue mask for mov

    Returns
    -------
    (transform_2x3, ncc_score)
    """
    ref_u8 = apply_clahe(normalize_to_uint8(ref))
    mov_u8 = apply_clahe(normalize_to_uint8(mov))

    shift, _, _ = phase_cross_correlation(ref_u8, mov_u8, upsample_factor=10)

    T = identity()
    T[0, 2] = float(shift[1])   # x translation
    T[1, 2] = float(shift[0])   # y translation

    ncc = score_transform(ref, mov, ref_mask, mov_mask, T)
    return T, ncc


# ============================================================================
# ECC REGISTRATION
# ============================================================================

def register_ecc(
    ref: np.ndarray,
    mov: np.ndarray,
    ref_mask: np.ndarray,
    init_transform: np.ndarray,
    max_iterations: int = 100,
    epsilon: float = 1e-5,
) -> Tuple[np.ndarray, float]:
    """
    Refine a transform using Enhanced Correlation Coefficient (ECC).

    ECC optimizes a Euclidean (rotation + translation) transform.
    It's initialized with init_transform, so it works best after a
    coarse phase correlation estimate.

    Parameters
    ----------
    ref, mov        : 2D float images
    ref_mask        : binary mask used to limit ECC to tissue regions
    init_transform  : 2×3 initial transform (from phase correlation)
    max_iterations  : ECC iteration limit
    epsilon         : convergence threshold

    Returns
    -------
    (transform_2x3, ncc_score)
        Returns (identity, 0.0) if ECC fails to converge.
    """
    ref_u8 = apply_clahe(normalize_to_uint8(ref))
    mov_u8 = apply_clahe(normalize_to_uint8(mov))

    ecc_ref  = ref_u8.astype(np.float32) / 255.0
    ecc_mov  = mov_u8.astype(np.float32) / 255.0
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max_iterations,
        epsilon,
    )
    input_mask = ref_mask.astype(np.uint8) * 255

    try:
        _, warp_ecc = cv2.findTransformECC(
            ecc_ref,
            ecc_mov,
            init_transform.copy().astype(np.float32),
            cv2.MOTION_EUCLIDEAN,
            criteria,
            input_mask,
            1,
        )
        ncc = score_transform(ref, mov, ref_mask,
                              np.ones_like(ref_mask), warp_ecc.astype(np.float32))
        return warp_ecc.astype(np.float32), ncc
    except cv2.error as e:
        logger.debug(f"ECC failed: {e}")
        return identity(), 0.0


# ============================================================================
# FEATURE-BASED REGISTRATION
# ============================================================================

class FeatureBasedRegistration:
    """
    SIFT (or ORB fallback) keypoint matching with RANSAC affine estimation.

    Uses Lowe's ratio test (0.75) to filter ambiguous matches.
    Runs detection on ref and mov in parallel via ThreadPoolExecutor.
    """

    def __init__(self, max_features: int = 15000) -> None:
        self.max_features = max_features

    def register(
        self,
        ref: np.ndarray,
        mov: np.ndarray,
        ref_mask: np.ndarray,
        mov_mask: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Detect keypoints, match, estimate affine transform via RANSAC.

        Returns
        -------
        (transform_2x3, ncc_score)
            Returns (identity, 0.0) if not enough matches found.
        """
        ref_u8 = apply_clahe(normalize_to_uint8(ref))
        mov_u8 = apply_clahe(normalize_to_uint8(mov))
        ref_mask_u8 = ref_mask.astype(np.uint8) * 255
        mov_mask_u8 = mov_mask.astype(np.uint8) * 255

        # SIFT with ORB fallback (ORB is available in all OpenCV builds)
        try:
            detector = cv2.SIFT_create(nfeatures=self.max_features)
            use_l2   = True
        except AttributeError:
            detector = cv2.ORB_create(nfeatures=min(self.max_features, 20000))
            use_l2   = False

        def _detect(img, mask):
            return detector.detectAndCompute(img, mask)

        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(_detect, ref_u8, ref_mask_u8)
            f2 = ex.submit(_detect, mov_u8, mov_mask_u8)
            kp1, des1 = f1.result()
            kp2, des2 = f2.result()

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            logger.debug("Not enough keypoints for feature matching")
            return identity(), 0.0

        norm   = cv2.NORM_L2 if use_l2 else cv2.NORM_HAMMING
        matcher = cv2.BFMatcher(norm, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 10:
            logger.debug(f"Too few good matches: {len(good)}")
            return identity(), 0.0

        ref_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        mov_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        try:
            M, _ = cv2.estimateAffinePartial2D(
                mov_pts, ref_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.0,
                maxIters=3000,
                confidence=0.99,
            )
        except cv2.error:
            return identity(), 0.0

        if M is None:
            return identity(), 0.0

        M   = M.astype(np.float32)
        ncc = score_transform(ref, mov, ref_mask, mov_mask, M)
        return M, ncc


# ============================================================================
# MULTI-SCALE RIGID REGISTRAR
# ============================================================================

class RigidRegistrar:
    """
    Orchestrates multi-scale rigid registration across pyramid levels.

    At each scale level:
    1. Warp moving image with accumulated transform so far
    2. Compute baseline NCC (current alignment quality)
    3. Try Phase, ECC, Features — keep the best
    4. Update accumulated transform if best improves on baseline

    After all levels, scale translation back to full resolution.
    """

    def __init__(
        self,
        pyramid_levels: list,
        max_features: int = 15000,
    ) -> None:
        self.pyramid_levels = sorted(pyramid_levels)
        self.feature_reg    = FeatureBasedRegistration(max_features)

    def register(
        self,
        ref: np.ndarray,
        mov: np.ndarray,
        ref_mask: np.ndarray,
        mov_mask: np.ndarray,
    ) -> Tuple[np.ndarray, float, str]:
        """
        Run multi-scale registration.

        Parameters
        ----------
        ref, mov       : full-resolution 2D float images
        ref_mask       : binary tissue mask for ref
        mov_mask       : binary tissue mask for mov

        Returns
        -------
        (transform_2x3, best_ncc, method_description)
            transform_2x3     : full-resolution 2×3 affine matrix
            best_ncc          : NCC score at the final pyramid level
            method_description: e.g. "Phase@0.5" or "Feature@0.25"
        """
        accumulated    = identity()
        current_scale  = 1.0
        best_ncc       = 0.0
        best_method    = "Identity"

        for scale_factor in self.pyramid_levels:
            logger.debug(f"Rigid registration at scale {scale_factor}")

            # downsample to this pyramid level
            ref_s      = resize_image(ref,      scale_factor, is_mask=False)
            mov_s      = resize_image(mov,      scale_factor, is_mask=False)
            ref_mask_s = resize_image(ref_mask.astype(np.uint8), scale_factor, is_mask=True)
            mov_mask_s = resize_image(mov_mask.astype(np.uint8), scale_factor, is_mask=True)

            # bring accumulated transform to this scale
            acc_s     = scale_translation_only(accumulated, scale_factor / current_scale)
            mov_w     = warp_affine(mov_s,      acc_s, ref_s.shape, is_mask=False)
            mov_mask_w = warp_affine(mov_mask_s.astype(np.uint8), acc_s, ref_s.shape, is_mask=True)

            # baseline NCC for this scale
            overlap = ref_mask_s & mov_mask_w
            if np.sum(overlap) > 100:
                w_map        = TissueProcessor.create_weight_map(ref_s, overlap)
                baseline_ncc = compute_weighted_ncc(ref_s, mov_w, w_map)
            else:
                baseline_ncc = 0.0

            best_local     = identity()
            best_local_ncc = baseline_ncc
            best_local_method = "Baseline"

            # --- Phase correlation ---
            try:
                T_phase, phase_ncc = register_phase(ref_s, mov_w, ref_mask_s, mov_mask_w)
                if phase_ncc > best_local_ncc:
                    best_local, best_local_ncc, best_local_method = T_phase, phase_ncc, "Phase"
                    logger.debug(f"  Phase: NCC={phase_ncc:.4f}")
            except Exception as e:
                logger.debug(f"  Phase failed: {e}")

            # --- ECC (initialized with best so far) ---
            try:
                T_ecc, ecc_ncc = register_ecc(
                    ref_s, mov_w, ref_mask_s, best_local
                )
                if ecc_ncc > best_local_ncc:
                    best_local, best_local_ncc, best_local_method = T_ecc, ecc_ncc, "ECC"
                    logger.debug(f"  ECC: NCC={ecc_ncc:.4f}")
            except Exception as e:
                logger.debug(f"  ECC failed: {e}")

            # --- Feature-based ---
            try:
                T_feat, feat_ncc = self.feature_reg.register(
                    ref_s, mov_w, ref_mask_s, mov_mask_w
                )
                if feat_ncc > best_local_ncc:
                    best_local, best_local_ncc, best_local_method = T_feat, feat_ncc, "Feature"
                    logger.debug(f"  Feature: NCC={feat_ncc:.4f}")
            except Exception as e:
                logger.debug(f"  Feature failed: {e}")

            # accept if not worse than baseline by more than 1%
            if best_local_ncc >= baseline_ncc - 0.01:
                accumulated   = compose(best_local, acc_s)
                current_scale = scale_factor
                best_ncc      = best_local_ncc
                best_method   = f"{best_local_method}@{scale_factor}"
                logger.info(f"Scale {scale_factor}: {best_method}, NCC={best_ncc:.4f}")

        # scale translation back to full resolution
        final_transform = upscale_to_full(accumulated, current_scale)
        return final_transform, best_ncc, best_method