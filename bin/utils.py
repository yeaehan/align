# ============================================================================
# UTILITIES
# ============================================================================
def safe_tifffile_read(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if not isinstance(img, np.ndarray):
        img = np.asarray(img, dtype=np.float32)
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    return img

def save_aligned_image_optimized(path: Path, data: np.ndarray, tile_size: int = 512):
    tifffile.imwrite(str(path), data, tile=(tile_size, tile_size),
                    compression='deflate', compressionargs={'level': 1})

def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    if not (mask1.any() and mask2.any()):
        return 0.0
    intersection = np.logical_and(mask1, mask2).sum()
    return 2.0 * intersection / (mask1.sum() + mask2.sum())

def to_uint8_for_registration(img: np.ndarray) -> np.ndarray:
    """Convert image to uint8 robustly for registration preprocessing."""
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.0:
            return (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if img.dtype == np.uint16:
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def apply_clahe_uint8(img_uint8: np.ndarray, clip_limit: float = 3.0,
                      tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE to uint8 image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_uint8)


def prepare_registration_image(img: np.ndarray, config: "RegistrationConfig") -> np.ndarray:
    """Prepare image for phase/template/feature registration."""
    img_uint8 = to_uint8_for_registration(img)
    if config.use_clahe_for_registration:
        img_uint8 = apply_clahe_uint8(
            img_uint8,
            clip_limit=config.clahe_clip_limit,
            tile_grid_size=config.clahe_tile_grid_size,
        )
    return img_uint8