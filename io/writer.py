from pathlib import Path
from typing import Union, Tuple
import numpy as np
import tifffile

def save_aligned_image(
    output_path: Union[str, Path], 
    image: np.ndarray, 
    tile_size: Optional[Tuple[int, int]] = (256, 256)
) -> None:
    """
    Saves an aligned image to disk as an optimized, compressed TIFF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Optimize as 8-bit or 16-bit to save space
    if image.dtype in [np.float32, np.float64]:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
    kwargs = {'compression': 'lzw'}
    if tile_size and max(image.shape) > 1024:
        kwargs['tile'] = tile_size
        
    tifffile.imwrite(str(output_path), image, **kwargs)