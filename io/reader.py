import logging
from pathlib import Path
from typing import Union, Optional
import numpy as np
import tifffile

try:
    from bioio import BioImage
    BIOIO_AVAILABLE = True
except ImportError:
    BIOIO_AVAILABLE = False

logger = logging.getLogger(__name__)

def read_image(file_path: Union[str, Path], scene: int = 0) -> np.ndarray:
    """
    Reads an image file. Supports .tif natively via tifffile, 
    and proprietary formats (.lif, .czi, .nd2) via BioIO.
    
    Args:
        file_path: Path to the image file.
        scene: For multi-scene files (like .lif), which scene to load.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")

    ext = file_path.suffix.lower()

    # Standard TIFF handling
    if ext in ['.tif', '.tiff']:
        image = tifffile.imread(str(file_path))
        # Match pipeline behavior: if 3D (e.g., RGB/RGBA), take first channel
        if image.ndim == 3 and image.shape[-1] <= 4:
            image = image[..., 0]
        return image

    # Proprietary formats (LIF, CZI, ND2)
    if ext in ['.lif', '.czi', '.nd2']:
        if not BIOIO_AVAILABLE:
            raise ImportError(f"Reading {ext} requires 'bioio'. Install with: pip install bioio bioio-lif")
        
        logger.info(f"Loading {ext} via BioIO (Scene {scene})...")
        img_obj = BioImage(file_path)
        img_obj.set_current_scene(scene)
        return img_obj.data.squeeze()  # Squeeze out empty dimensions (T, Z) if 2D

    raise ValueError(f"Unsupported image format: {ext}")