"""
align/io/lif.py
---------------
Native Leica LIF file reader for the alignment pipeline.
Dynamically extracts channels to float32 arrays on demand to save memory.
"""

import logging
from pathlib import Path
import numpy as np
from readlif.reader import LifFile
from align.io.reader import _to_float32
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class LifImageReader:
    """
    Manages reading specific channels and scenes from a .lif file.
    Loads data on demand and normalizes to float32.
    """
    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"LIF file not found: {path}")

        try:
            self.lif = LifFile(str(self.path))
        except Exception as e:
            raise ValueError(f"Failed to read LIF file {self.path.name}: {e}") from e
            
        self.num_scenes = self.lif.num_images
        self._scene_infos: List[Dict] = []

        for i in range(self.num_scenes):
            img = self.lif.get_image(i)
            self._scene_infos.append({
                "name": img.name,
                "channels": img.channels,
                "dims": (img.dims.z, img.dims.y, img.dims.x),
                "scene_idx": i
            })
        logger.debug(f"Loaded LIF file '{self.path.name}' with {self.num_scenes} scenes.")

    def get_merged_scene_idx(self) -> int:
        """Find the index of the scene containing 'Merged' in its name."""
        for idx, info in enumerate(self._scene_infos):
            if "merged" in info["name"].lower():
                return idx
        logger.warning(f"No scene containing 'Merged' found in {self.path.name}. Defaulting to scene 0.")
        return 0

    def get_scene_info(self, scene_idx: Optional[int] = None) -> Dict:
        """Return metadata about a specific scene."""
        if scene_idx is None:
            scene_idx = self.get_merged_scene_idx()
        if scene_idx < 0 or scene_idx >= self.num_scenes:
            raise IndexError(f"Scene index {scene_idx} out of range for {self.path.name} (0-{self.num_scenes-1})")
        return self._scene_infos[scene_idx]

    def get_available_scenes(self) -> List[Dict]:
        """Return a list of all scene infos."""
        return self._scene_infos

    def read_channel_as_float(self, channel_idx: int, scene_idx: Optional[int] = None, use_mip: bool = True) -> np.ndarray:
        """
        Extract a single channel from a scene, apply MIP if 3D, and normalize to [0, 1].

        Parameters
        ----------
        channel_idx : The 0-indexed channel to extract.
        scene_idx   : The 0-indexed scene to extract from.
        use_mip     : If True and the image is 3D, performs a Max Intensity Projection.
                      Otherwise, extracts the middle Z-plane.
        """
        if scene_idx is None:
            scene_idx = self.get_merged_scene_idx()
            
        scene_data = self.get_scene_info(scene_idx)
        if channel_idx < 0 or channel_idx >= scene_data['channels']:
            raise IndexError(f"Channel index {channel_idx} out of range for scene {scene_idx} in {self.path.name} (0-{scene_data['channels']-1})")
            
        img = self.lif.get_image(scene_idx) # get the specific image object from readlif
        
        z_stack = np.zeros((img.dims.z, img.dims.y, img.dims.x), dtype=np.float32)
        for z in range(img.dims.z):
            plane = np.array(img.get_frame(z=z, c=channel_idx, t=0))
            z_stack[z] = _to_float32(plane)
            
        if img.dims.z > 1:
            return np.max(z_stack, axis=0) if use_mip else z_stack[img.dims.z // 2]
        return z_stack[0]
