# ============================================================================
# TISSUE PROCESSOR
# ============================================================================
class TissueProcessor:
    _mask_cache = {}
   
    @staticmethod
    def create_tissue_mask(img: np.ndarray, percentile: float = 5.0) -> np.ndarray:
        cache_key = f"{img.shape}_{percentile}_{img.mean():.4f}_{img.std():.4f}"
       
        if cache_key in TissueProcessor._mask_cache:
            return TissueProcessor._mask_cache[cache_key] # Return view
       
        logger.info("🎭 Creating tissue mask...")
       
        max_size = 2048
        if max(img.shape) > max_size:
            scale = max_size / max(img.shape)
            img_small = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)),
                                  interpolation=cv2.INTER_AREA)
        else:
            img_small = img
       
        if img_small.dtype == np.float32 and img_small.max() <= 1.0:
            img_uint8 = (np.clip(img_small, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = cv2.normalize(img_small, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
       
        blur = cv2.GaussianBlur(img_uint8, (5, 5), 0)
        _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_coverage = np.sum(thresh_otsu > 0) / thresh_otsu.size
       
        if otsu_coverage < 0.2:
            threshold_val = np.percentile(img_uint8, percentile)
            _, thresh = cv2.threshold(blur, threshold_val, 255, cv2.THRESH_BINARY)
        else:
            thresh = thresh_otsu
       
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        filled = ndimage.binary_fill_holes(cleaned > 0)
        labeled = measure.label(filled)
       
        if labeled.max() == 0:
            mask = filled.astype(bool)
        else:
            regions = measure.regionprops(labeled)
            largest = max(regions, key=lambda x: x.area)
            mask = labeled == largest.label
       
        if max(img.shape) > max_size:
            mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
       
        coverage = np.sum(mask) / mask.size
        logger.info(f"   Coverage: {coverage:.1%}")
       
        TissueProcessor._mask_cache[cache_key] = mask # Store
        return mask

    @staticmethod
    def create_nuclei_weight_map(dapi_img: np.ndarray, base_mask: np.ndarray,
                                  cache: Optional[WeightMapCache] = None) -> np.ndarray:
        if cache:
            cached = cache.get(dapi_img, base_mask)
            if cached is not None:
                logger.info("   ✅ Weight map from cache")
                return cached
       
        logger.info("🎯 Creating weight map...")
       
        # 3. OPTIMIZATION: In-place normalization where possible
        if dapi_img.dtype != np.float32:
            if dapi_img.dtype == np.uint16:
                dapi_norm = dapi_img.astype(np.float32) / 65535.0
            elif dapi_img.dtype == np.uint8:
                dapi_norm = dapi_img.astype(np.float32) / 255.0
            else:
                dapi_norm = dapi_img.astype(np.float32)
        else:
            dapi_norm = dapi_img # View
       
        dapi_norm = np.clip(dapi_norm, 0.0, 1.0)
        laplacian = cv2.Laplacian(dapi_norm, cv2.CV_32F, ksize=3)
        contrast = np.abs(laplacian)
       
        contrast_masked = contrast[base_mask]
        if contrast_masked.size > 0 and contrast_masked.max() > 0:
            contrast = np.clip(contrast, 0, np.percentile(contrast_masked, 99))
            contrast /= contrast.max() # In-place division
        else:
            contrast.fill(0)
       
        weight_map = np.zeros_like(contrast)
        weight_map[base_mask] = 0.2 + 0.8 * contrast[base_mask]
       
        if cache:
            cache.put(dapi_img, base_mask, weight_map)
       
        return weight_map



# ============================================================================
# CACHING - OPTIMIZED
# ============================================================================
class WeightMapCache:
    """LRU cache for weight maps - Optimized to store refs and reduce copy"""
    def __init__(self, max_size: int = 10): # Reduced from 20 to 10 to save RAM
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
   
    def _make_key(self, img_shape: tuple, mask: np.ndarray) -> str:
        return f"{img_shape}_{mask.shape}_{int(mask.sum())}"
   
    def get(self, dapi_img: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        key = self._make_key(dapi_img.shape, mask)
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key] # Return reference, assume caller handles safety
        return None
   
    def put(self, dapi_img: np.ndarray, mask: np.ndarray, weight_map: np.ndarray):
        key = self._make_key(dapi_img.shape, mask)
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
            # Force cleanup when dropping large items
            if len(self.cache) % 5 == 0:
                gc.collect()
        self.cache[key] = weight_map # Store reference, no .copy()
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
   
    def clear(self):
        self.cache.clear()
        self.access_order.clear()
        gc.collect()


