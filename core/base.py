from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseAligner(ABC):
    """
    Abstract base class for all alignment/registration methods.
    This ensures all algorithms share a common interface for easy swapping,
    batch processing, and future GUI integration.
    """

    @abstractmethod
    def compute_transform(
        self, 
        fixed_image: np.ndarray, 
        moving_image: np.ndarray, 
        fixed_mask: Optional[np.ndarray] = None, 
        moving_mask: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Computes the transformation required to align the moving image to the fixed image.
        
        Returns:
            transform: An array representing the transformation (e.g., 2x3 affine matrix).
        """
        pass

    @abstractmethod
    def apply_transform(self, image: np.ndarray, transform: np.ndarray, **kwargs) -> np.ndarray:
        """
        Applies the computed transformation to an image.
        """
        pass