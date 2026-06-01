from __future__ import annotations

import gc
import logging
import os


os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("nextgen_4i_batch")

try:
    import cupy as cp
    from cupyx.scipy.ndimage import grey_opening, laplace, map_coordinates as cp_map_coordinates, median_filter

    GPU_AVAILABLE = True
    NUM_GPUS = cp.cuda.runtime.getDeviceCount()
    logger.info("%s GPU(s) available", NUM_GPUS)
except ImportError:
    cp = None
    cp_map_coordinates = None
    grey_opening = None
    laplace = None
    median_filter = None
    GPU_AVAILABLE = False
    NUM_GPUS = 0
    logger.warning("CuPy not available, using CPU")


def cleanup_memory() -> None:
    gc.collect()
    if not GPU_AVAILABLE:
        return

    for gpu_id in range(NUM_GPUS):
        with cp.cuda.Device(gpu_id):
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()


def preprocess_dapi_gpu(img, tophat_radius: int = 64, light_background: bool = False):
    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy is not available")

    img = median_filter(img, size=3)
    y, x = cp.ogrid[-tophat_radius : tophat_radius + 1, -tophat_radius : tophat_radius + 1]
    kernel = x**2 + y**2 <= tophat_radius**2
    if light_background:
        img = 1.0 - img

    background = grey_opening(img, structure=kernel)
    img = img - background
    if light_background:
        img = 1.0 - img

    lap = laplace(img)
    lap /= cp.abs(lap).max() + 1e-8
    img = img + 0.3 * lap
    vmin = cp.quantile(img, 0.005)
    vmax = cp.quantile(img, 0.998)
    return cp.clip((img - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
