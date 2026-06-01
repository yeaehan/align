from __future__ import annotations

import cv2
import numpy as np

from .runtime import GPU_AVAILABLE, cp, cp_map_coordinates


def apply_affine_transform(image: np.ndarray, transform: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.warpAffine(
        image.astype(np.float32, copy=False),
        transform.astype(np.float32, copy=False),
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def apply_affine_transform_gpu(image: np.ndarray, transform: np.ndarray, gpu_id: int = 0) -> np.ndarray:
    if not GPU_AVAILABLE:
        return apply_affine_transform(image, transform)

    with cp.cuda.Device(gpu_id):
        img_gpu = cp.asarray(image.astype(np.float32, copy=False))
        transform_gpu = cp.asarray(transform.astype(np.float32, copy=False))
        h, w = image.shape[:2]
        y_coords, x_coords = cp.mgrid[0:h, 0:w].astype(cp.float32)

        inv_transform = cp.linalg.inv(cp.vstack([transform_gpu, cp.array([0, 0, 1], dtype=cp.float32)]))[:2, :]
        src_x = inv_transform[0, 0] * x_coords + inv_transform[0, 1] * y_coords + inv_transform[0, 2]
        src_y = inv_transform[1, 0] * x_coords + inv_transform[1, 1] * y_coords + inv_transform[1, 2]

        warped = cp_map_coordinates(img_gpu, [src_y, src_x], order=1, mode="constant", cval=0)
        result = cp.asnumpy(warped)

        del img_gpu, transform_gpu, y_coords, x_coords, src_x, src_y, warped
        cp.get_default_memory_pool().free_all_blocks()
        return result
