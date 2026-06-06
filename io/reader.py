"""
align/io/reader.py
------------------
Standardized image reading for the alignment pipeline.

All functions return float32 numpy arrays normalized to [0, 1].
This keeps downstream code free of dtype branching (uint8 vs uint16 etc.)

read_tiff            : read a single TIFF, return 2D float (MIP if 3D/4D)
read_zstack          : read a z-stack TIFF, return 3D float array (Z, H, W)
read_2d_as_float     : alias with explicit max-projection control
discover_moving_files: find all non-reference TIFFs in a folder
find_reference_file  : find a specific '_ref' file (for 3D Z-stack pipeline)
select_anchor_file   : pick the best channel to use as registration anchor
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from align.io.lif import LifImageReader

import numpy as np
import tifffile

logger = logging.getLogger(__name__)


# ============================================================================
# DTYPE NORMALIZATION
# ============================================================================

def _to_float32(img: np.ndarray) -> np.ndarray:
    """
    Normalize an image array to float32 in [0, 1] based on its dtype.

    uint16 → divide by 65535
    uint8  → divide by 255
    float  → cast to float32, no rescaling (assumed already in [0,1])
    """
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    else:
        return img.astype(np.float32)


# ============================================================================
# 2D READING
# ============================================================================

def read_2d_as_float(
    path: str,
    use_max_projection: bool = True,
    channel_idx: int = 0, # For LIF files
    scene_idx: int = 0,   # For LIF files
) -> np.ndarray:
    """
    Read any TIFF and return a single 2D float32 image.

    Handles common dimension layouts:
    - 2D (H, W)             → returned as-is
    - 3D (Z, H, W)          → MIP along Z, or middle Z-plane
    - 3D (H, W, C)          → first channel extracted
    - 4D (Z, H, W, C)       → first channel, then MIP along Z

    Parameters
    ----------
    path               : file path
    use_max_projection : if True, take the maximum intensity projection
                         along the Z axis. If False, use the middle Z-plane.

    Returns
    -------
    float32 array of shape (H, W), values in [0, 1]
    """
    img = tifffile.imread(str(path))

    if img.ndim == 2:
        out = img

    elif img.ndim == 3:
        # distinguish (Z, H, W) from (H, W, C)
        if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            # first dim is Z (small relative to spatial dims)
            out = np.max(img, axis=0) if use_max_projection else img[img.shape[0] // 2]
        elif img.shape[2] in (3, 4):
            # last dim is RGB/RGBA channels
            out = img[:, :, 0]
        else:
            # ambiguous — default to MIP
            out = np.max(img, axis=0)

    elif img.ndim == 4:
        # assume (Z, H, W, C) or (Z, C, H, W)
        if img.shape[-1] in (3, 4):
            img = img[..., 0]  # drop colour channel
        out = np.max(img, axis=0) if use_max_projection else img[img.shape[0] // 2]

    else:
        raise ValueError(
            f"Unsupported image shape {img.shape} in {path}. "
            f"Expected 2D, 3D, or 4D array."
        )

    return _to_float32(out)


# ============================================================================
# Z-STACK READING
# ============================================================================

def read_zstack(path: str) -> Tuple[np.ndarray, int]:
    """
    Read a z-stack TIFF and return the full 3D array plus z-plane count.

    Parameters
    ----------
    path : path to the z-stack TIFF

    Returns
    -------
    (zstack, n_z_planes)
        zstack     : float32 array of shape (Z, H, W), values in [0, 1]
        n_z_planes : number of Z planes (same as zstack.shape[0])

    Raises
    ------
    ValueError if the file is not a 3D or 4D array
    """
    raw = tifffile.imread(str(path))
    logger.debug(f"Loaded z-stack {Path(path).name}: shape={raw.shape}, dtype={raw.dtype}")

    if raw.ndim == 2:
        # treat as single-plane z-stack
        zstack = _to_float32(raw)[np.newaxis, ...]
    elif raw.ndim == 3:
        zstack = _to_float32(raw)
    elif raw.ndim == 4:
        if raw.shape[-1] in (3, 4):
            raw = raw[..., 0]
        zstack = _to_float32(raw)
    else:
        raise ValueError(f"Unexpected z-stack shape {raw.shape} in {path}")

    n_z = zstack.shape[0]
    return zstack, n_z


def compute_mip(
    zstack: np.ndarray,
    z_range: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, str]:
    """
    Compute maximum intensity projection along the Z axis.

    Parameters
    ----------
    zstack  : float32 array of shape (Z, H, W)
    z_range : optional (z_start, z_end) to limit the projection range

    Returns
    -------
    (mip_image, description_string)
        mip_image   : 2D float32 array (H, W)
        description : human-readable string describing what was projected
    """
    if zstack.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D array for MIP, got shape {zstack.shape}")

    n_z = zstack.shape[0]

    if z_range is None:
        logger.info(f"Computing MIP from all {n_z} Z-planes")
        return np.max(zstack, axis=0), f"MIP_all_{n_z}_planes"

    z_start, z_end = z_range
    logger.info(f"Computing MIP from Z-planes {z_start} to {z_end - 1}")
    return np.max(zstack[z_start:z_end], axis=0), f"MIP_z{z_start}-{z_end - 1}"


def compute_max_projection(
    zstack: np.ndarray,
    z_range: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, str]:
    return compute_mip(zstack, z_range)


def extract_z_plane(
    zstack: np.ndarray,
    z_plane: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Extract a single Z-plane from a z-stack.

    Parameters
    ----------
    zstack  : (Z, H, W) float32 array
    z_plane : index of the plane to extract; defaults to middle plane

    Returns
    -------
    (plane, z_index) : 2D array and the z index used
    """
    if zstack.ndim not in [3, 4]:
        raise ValueError(f"Expected 3D or 4D array, got shape {zstack.shape}")
    n_z     = zstack.shape[0]
    z_index = n_z // 2 if z_plane is None else z_plane
    logger.info(f"Using Z-plane: {z_index}/{n_z}")
    return zstack[z_index], z_index


def extract_reference_plane(
    zstack: np.ndarray,
    z_plane: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    return extract_z_plane(zstack, z_plane)


def replicate_to_zstack(
    channel_2d: np.ndarray,
    n_z_planes: int,
) -> np.ndarray:
    """
    Replicate a 2D image across multiple Z-planes.
    
    Used to create a z-stack output from a single 2D aligned channel.
    
    Parameters
    ----------
    channel_2d : 2D array (H, W)
        2D image to replicate
    n_z_planes : int
        Number of Z-planes to create
    
    Returns
    -------
    zstack : 3D array (Z, H, W)
        The 2D image replicated n_z_planes times along axis 0
    """
    return np.stack([channel_2d] * n_z_planes, axis=0)


# ============================================================================
# FILE DISCOVERY
# ============================================================================

_TIFF_EXTENSIONS = (".tif", ".tiff")

# Suffixes that identify a file as already processed (skip these)
_SKIP_PATTERNS = ("crop_aligned2dot3d", "debug_ch00_alignment")


def find_reference_candidates(
    folder: Path,
    extensions: tuple = _TIFF_EXTENSIONS,
) -> List[Path]:
    return sorted([
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in extensions
        and "_ref" in p.stem.lower()
        and "crop_aligned2dot3d" not in p.stem.lower()
        and "debug_ch00_alignment" not in p.stem.lower()
    ])


def discover_moving_files(
    reference_path: Path,
    extensions: tuple = _TIFF_EXTENSIONS,
) -> List[str]:
    """
    Discover all non-reference TIFFs in the same folder as the reference file.

    Excludes:
    - The reference file itself
    - Files containing 'crop_aligned2dot3d' (already processed outputs)
    - Files containing 'debug_ch00_alignment' (QC overlays)
    - Files containing '_ref' in the stem (other reference candidates)

    Parameters
    ----------
    reference_path : path to the reference file

    Returns
    -------
    Sorted list of absolute path strings
    """
    ref_path = reference_path.resolve()
    folder   = ref_path.parent

    files = sorted([
        str(p) for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in extensions
        and p.resolve() != ref_path
        and not any(pat in p.stem.lower() for pat in _SKIP_PATTERNS)
        and "_ref" not in p.stem.lower()
    ])
    return files


def select_anchor_file(moving_images: List[str]) -> str:
    """
    Select the best channel to use as the registration anchor.

    Priority order:
    1. ch00 + aligned (already-registered DAPI from a prior 2D run)
    2. ch00 (DAPI channel by naming convention)
    3. dapi (explicit name)
    4. hoechst (nuclear stain alternative)
    5. aligned (any aligned channel)
    6. first file (last resort)

    Parameters
    ----------
    moving_images : list of file paths

    Returns
    -------
    Path string of the selected anchor file
    """
    priorities = [
        lambda n: "ch00" in n and "aligned" in n,
        lambda n: "ch00" in n,
        lambda n: "dapi" in n,
        lambda n: "hoechst" in n,
        lambda n: "aligned" in n,
        lambda n: True,
    ]
    lower_map = [(p, Path(p).name.lower()) for p in moving_images]
    for rule in priorities:
        for p, name in lower_map:
            if rule(name):
                return p
    return moving_images[0]


def find_reference_file(
    folder: Path,
    extensions: tuple = _TIFF_EXTENSIONS,
) -> Tuple[Optional[Path], Optional[str]]:
    """
    Find the reference z-stack file in a folder.

    Expects exactly one file with '_ref' in its stem.

    Returns
    -------
    (ref_path, error_message)
        If found: (Path, None)
        If not found or ambiguous: (None, reason_string)
    """
    refs = find_reference_candidates(folder, extensions=extensions)
    if len(refs) == 0:
        return None, "no _ref file"
    if len(refs) > 1:
        return None, "multiple _ref files"
    return refs[0], None


def find_processable_folders(
    root_folder: str,
    skip_names: Optional[set] = None,
) -> List[Path]:
    """
    Recursively find all subdirectories that may contain data to process.

    Parameters
    ----------
    root_folder : root directory to search
    skip_names  : directory names to skip (case-insensitive)
                  defaults to {'done', 'metadata', '__pycache__'}

    Returns
    -------
    Deduplicated list of Path objects, root first
    """
    root = Path(root_folder)
    return [root] + sorted([p for p in root.rglob("*") if p.is_dir()])


def replicate_to_zstack(channel_2d: np.ndarray, n_z_planes: int) -> np.ndarray:
    """
    Replicate a 2D image N times along a new Z axis.

    Used when projecting aligned 20x images into 63x z-stack space:
    the 20x image has no Z information, so we replicate the same plane
    to fill the z-stack shape.

    Note: the resulting stack has the same 2D content on every Z-plane.
    It is only used for spatial correspondence — do not use for 3D analysis.

    Parameters
    ----------
    channel_2d : (H, W) float32 array
    n_z_planes : number of planes to replicate

    Returns
    -------
    (Z, H, W) float32 array
    """
    return np.stack([channel_2d] * n_z_planes, axis=0)
