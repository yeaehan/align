"""
align/io/writer.py
------------------
Standardized TIFF writing for the alignment pipeline.

All outputs are uint16 with deflate compression (level 1).
Level 1 is fast compression — enough to halve file sizes without
the CPU cost of higher levels.

BigTIFF is automatically enabled when the output would exceed ~3.9 GB,
which is necessary for large z-stacks (33 planes × 16k×22k px ≈ 24 GB).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tifffile

logger = logging.getLogger(__name__)


# ============================================================================
# TIFF WRITING
# ============================================================================

def save_tiff(
    img: np.ndarray,
    output_path: str,
    voxel_spacing_z: Optional[float] = None,
    pixel_size_xy: Optional[float] = None,
    axes: Optional[str] = None,
) -> None:
    """
    Save a numpy array as a compressed TIFF.

    Automatically:
    - Converts float [0,1] to uint16
    - Enables BigTIFF for files > 3.9 GB
    - Adds ImageJ spatial calibration when pixel sizes are provided
    - Uses deflate compression level 1 (fast)

    Parameters
    ----------
    img              : numpy array, any dtype
    output_path      : destination file path (will be created including parents)
    voxel_spacing_z  : Z voxel size in microns, written to ImageJ metadata
                       Set for z-stacks so they open correctly in Fiji/ImageJ
    pixel_size_xy    : XY pixel size in microns
    axes             : axis string e.g. "ZYX", "YX"; inferred if not provided
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # convert float to uint16 — pipeline images are in [0, 1]
    if np.issubdtype(img.dtype, np.floating):
        img_out = (np.clip(img, 0.0, 1.0) * 65535).astype(np.uint16)
    else:
        img_out = img

    use_bigtiff = img_out.nbytes > 3.9e9

    metadata = {}
    if voxel_spacing_z is not None or pixel_size_xy is not None:
        if axes is None:
            axes = "ZYX" if img_out.ndim == 3 else "YX"
        metadata = {
            "axes":    axes,
            "unit":    "um",
        }
        if voxel_spacing_z is not None:
            metadata["spacing"] = voxel_spacing_z

    resolution_kwargs = {}
    if pixel_size_xy is not None:
        if pixel_size_xy <= 0:
            raise ValueError("pixel_size_xy must be greater than zero")
        pixels_per_centimeter = 10000.0 / pixel_size_xy
        resolution_kwargs = {
            "resolution": (pixels_per_centimeter, pixels_per_centimeter),
            "resolutionunit": "CENTIMETER",
        }

    tifffile.imwrite(
        str(output_path),
        img_out,
        bigtiff=use_bigtiff,
        imagej=True,
        metadata=metadata if metadata else None,
        compression="deflate",
        compressionargs={"level": 1},
        **resolution_kwargs,
    )
    logger.debug(f"Saved: {Path(output_path).name} ({img_out.nbytes / 1e9:.2f} GB)")


def save_debug_overlay(
    ref: np.ndarray,
    aligned: np.ndarray,
    output_path: str,
    max_dim: int = 4000,
) -> None:
    """
    Save a green/magenta overlay for visual QC of alignment.

    Green channel  = reference image
    Red channel    = aligned moving image
    Good alignment → yellow/white where tissue overlaps

    Both images are downscaled to max_dim on the longest side before saving
    to keep the QC file small.

    Parameters
    ----------
    ref         : 2D float reference image
    aligned     : 2D float aligned moving image (same shape as ref)
    output_path : destination path
    max_dim     : maximum pixel dimension of the saved overlay
    """
    from align.core.tissue import normalize_to_uint8, resize_image

    debug_scale = min(max_dim / max(ref.shape), 1.0)
    debug_scale = float(max(debug_scale, 1e-4))

    ref_dbg     = resize_image(ref,     debug_scale, is_mask=False)
    aligned_dbg = resize_image(aligned, debug_scale, is_mask=False)

    ref_u8     = normalize_to_uint8(ref_dbg)
    aligned_u8 = normalize_to_uint8(aligned_dbg)

    h, w       = ref_u8.shape
    composite  = np.zeros((h, w, 3), dtype=np.uint8)
    composite[:, :, 0] = aligned_u8   # red   = moving
    composite[:, :, 1] = ref_u8       # green = reference

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output_path), composite, imagej=True)
    logger.debug(f"Debug overlay saved: {Path(output_path).name}")


# ============================================================================
# JSON WRITING
# ============================================================================

def write_json(path: Path, data: Any) -> None:
    """
    Write a Python object to a JSON file.

    Creates parent directories as needed.

    Parameters
    ----------
    path : destination Path
    data : JSON-serializable object (dict, list, etc.)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
