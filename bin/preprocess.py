# ============================================================================
# CONVERT/PAD PROCESSOR
# ============================================================================
class ConvertPadProcessor:
    def __init__(self, config: RegistrationConfig):
        self.config = config
        self.processed_dir = Path(config.output_folder) / "processed_images"
        self.processed_dir.mkdir(exist_ok=True)
       
    def _center_pad_image(self, img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w = img.shape[:2]
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
       
        if img.dtype == np.uint16:
            return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=0)
        else:
            img_uint16 = (img * 65535).astype(np.uint16) if img.max() <= 1.0 else img.astype(np.uint16)
            padded = np.pad(img_uint16, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=0)
            return padded.astype(np.float32) / 65535.0
       
    def _process_single_image(self, file_path: Path, target_h: int, target_w: int, is_dapi: bool = False) -> Path:
        original_path = self.processed_dir / f"{file_path.stem}_original.tif"
        processed_path = self.processed_dir / f"{file_path.stem}_processed.tif"
       
        if is_dapi and processed_path.exists() and original_path.exists():
            return processed_path
        if not is_dapi and original_path.exists():
            return original_path
               
        img = safe_tifffile_read(file_path)
        if img.ndim == 3:
            img = img[..., 0]
           
        if img.dtype != np.float32:
            if img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            elif img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
               
        padded_img = self._center_pad_image(img, target_h, target_w)
       
        if not original_path.exists():
            tifffile.imwrite(str(original_path), padded_img.astype(np.float32),
                           compression='zlib', compressionargs={'level': 1})
           
        if is_dapi and not processed_path.exists() and self.config.apply_advanced_preprocessing:
            if GPU_AVAILABLE:
                try:
                    if padded_img.nbytes > 1.5e9: # Lower threshold
                        scale_factor = 0.5
                        h_small = int(padded_img.shape[0] * scale_factor)
                        w_small = int(padded_img.shape[1] * scale_factor)
                        img_small = cv2.resize(padded_img, (w_small, h_small), interpolation=cv2.INTER_AREA)
                        img_gpu = cp.asarray(img_small.astype(np.float32))
                        img_gpu = preprocess_dapi_gpu(img_gpu, tophat_radius=self.config.preprocessing_tophat_radius,
                                                     light_background=self.config.preprocessing_light_background)
                        result_small = cp.asnumpy(img_gpu)
                        padded_img = cv2.resize(result_small, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        img_gpu = cp.asarray(padded_img.astype(np.float32))
                        img_gpu = preprocess_dapi_gpu(img_gpu, tophat_radius=self.config.preprocessing_tophat_radius,
                                                     light_background=self.config.preprocessing_light_background)
                        padded_img = cp.asnumpy(img_gpu)
                   
                    del img_gpu
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception as e:
                    logger.warning(f"⚠️  GPU preprocess failed: {e}")
                    padded_img = cv2.GaussianBlur(padded_img, (5, 5), 0)
            else:
                padded_img = cv2.GaussianBlur(padded_img, (5, 5), 0)
       
        if is_dapi and not processed_path.exists():
            tifffile.imwrite(str(processed_path), padded_img.astype(np.float32),
                           compression='zlib', compressionargs={'level': 1})
            return processed_path
        return original_path
   
    def process_all(self, raw_files: List[Path]) -> Dict[str, Any]:
        max_h, max_w = 0, 0
        for file in raw_files:
            with tifffile.TiffFile(str(file)) as tif:
                shape = tif.series[0].shape
                h, w = (shape[1], shape[2]) if len(shape) == 3 else (shape[0], shape[1])
                max_h, max_w = max(max_h, h), max(max_w, w)
               
        self.max_height, self.max_width = max_h, max_w
        logger.info(f"📐 Max dimensions: {max_h}×{max_w}")
       
        processed_paths = []
        # 4. OPTIMIZATION: Reduce workers to prevent OOM during loading
        with ThreadPoolExecutor(max_workers=min(3, os.cpu_count() or 3)) as executor:
            futures = {}
            for file_path in raw_files:
                is_dapi = any(re.search(p, file_path.name.lower()) for p in self.config.dapi_patterns)
                future = executor.submit(self._process_single_image, file_path, max_h, max_w, is_dapi)
                futures[future] = file_path
               
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    processed_paths.append(future.result())
                except Exception as e:
                    logger.error(f"Error: {e}")
       
        return {
            "status": "completed",
            "output_dir": str(self.processed_dir),
            "target_shape": (self.max_height, self.max_width),
            "processed_count": len(processed_paths),
        }
