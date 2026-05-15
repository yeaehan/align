import cv2
import numpy as np


def apply_affine_transform(image: np.ndarray, transform_matrix: np.ndarray, use_nearest: bool = False) -> np.ndarray:
    """
    Applies a 2x3 affine transformation matrix to a 2D image.
    """
    h, w = image.shape[:2]
    flags = cv2.INTER_NEAREST if use_nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(image, transform_matrix, (w, h), flags=flags)


def scale_transform(transform_matrix: np.ndarray, scale_ratio: float) -> np.ndarray:
    """
    Scales the translation components of a 2x3 affine matrix.
    Useful for mapping transforms between pyramid levels.
    """
    scaled_matrix = transform_matrix.copy()
    # Scale only the translation (x, y) components
    scaled_matrix[0, 2] *= scale_ratio
    scaled_matrix[1, 2] *= scale_ratio
    return scaled_matrix


def validate_transform(transform: np.ndarray, max_translation: float = 5000.0) -> bool:
    """
    Validates an affine transformation matrix to prevent catastrophic warps.
    """
    if transform.shape != (2, 3):
        return False
        
    tx, ty = transform[0, 2], transform[1, 2]
    if abs(tx) > max_translation or abs(ty) > max_translation:
        return False
        
    det = np.linalg.det(transform[:2, :2])
    return 0.1 < abs(det) < 10.0