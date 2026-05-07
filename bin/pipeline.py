# ============================================================================
# SINGLE BATCH PIPELINE (Core registration logic)
# ============================================================================
class NextGen4iPipeline:
    def __init__(self, config: RegistrationConfig):
        self.config = config
        self.output_dir = Path(config.output_folder)
        self.aligned_dir = self.output_dir / "aligned"
        self.qc_dir = self.output_dir / "quality_control"
        self.masks_dir = self.output_dir / "masks"
        self.aligned_dir.mkdir(exist_ok=True)
        self.qc_dir.mkdir(exist_ok=True)
        self.masks_dir.mkdir(exist_ok=True)
        self.weight_cache = WeightMapCache(max_size=10)

    def run(self) -> Dict[str, Any]:
        start_time = time.time()
        results = {"status": "started", "processed_rounds": {}, "errors": []}
       
        try:
            logger.info("=== 🚀 NEXT-GEN 4i v4.1 (Optimized) ===")
            ################## Change if you want to exclude certain files ##################
            input_files = [f for f in Path(self.config.input_folder).glob("*.tif") if "Cor" not in f.name]
            ########################################################################
            if not input_files:
                raise ValueError("No .tif files found")
            logger.info(f"📁 Found {len(input_files)} files")
           
            converter = ConvertPadProcessor(self.config)
            proc = converter.process_all(input_files)
            proc_dir = Path(proc["output_dir"])
            input_files = list(proc_dir.glob("*_original.tif")) + list(proc_dir.glob("*_processed.tif"))
            grouped_files = self._group_files_by_round(input_files)
            logger.info(f"📁 Grouped into {len(grouped_files)} rounds")
           
            ref_path = Path(self.config.reference_file)
            padded_ref_path = proc_dir / f"{ref_path.stem}_processed.tif"
            if not padded_ref_path.exists():
                padded_ref_path = converter._process_single_image(
                    ref_path, proc["target_shape"][0], proc["target_shape"][1], is_dapi=True)
           
            ref_full_img = safe_tifffile_read(padded_ref_path)
            if ref_full_img.ndim == 3:
                ref_full_img = ref_full_img[..., 0]
           
            ref_mask = TissueProcessor.create_tissue_mask(ref_full_img, self.config.tissue_mask_percentile)
            self._save_mask(ref_mask, self.masks_dir / "reference_mask.tif")
           
            ref_round_name = ref_path.stem.split("_Merged")[0] if "_Merged" in ref_path.stem else ref_path.stem
           
            for round_name, round_files in tqdm(grouped_files.items(), desc="Rounds"):
                round_start = time.time()
                try:
                    logger.info(f"//n🔄 === {round_name} === ")
                    dapi_file = self._find_dapi_file(round_files)
                    if not dapi_file:
                        continue

                    # Check if done
                    all_done = all(
                        (self.aligned_dir / f"{f.stem.replace('_original', '')}_aligned.tif").exists()
                        for f in round_files if "_processed.tif" not in f.name
                        and not (self.config.skip_non_reference_dapi and round_name != ref_round_name
                                and self._is_dapi_channel(f.name))
                    )
                   
                    if all_done:
                        logger.info("✅ Already aligned")
                        continue

                    is_reference_round = (round_name == ref_round_name)
                   
                    if is_reference_round:
                        logger.info("🎯 Reference round - copying")
                        saved_count = 0
                        for file_path in round_files:
                            if "_processed.tif" in file_path.name:
                                continue
                            channel_name = file_path.stem.replace("_original", "")
                            output_path = self.aligned_dir / f"{channel_name}_aligned.tif"
                            image = safe_tifffile_read(file_path)
                            if image.ndim == 3:
                                image = image[..., 0]
                            img_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                            save_aligned_image_optimized(output_path, img_8bit, tile_size=self.config.tiff_tile_size)
                            saved_count += 1
                            del image, img_8bit
                            gc.collect()
                       
                        results["processed_rounds"][round_name] = {
                            "status": "reference",
                            "files_processed": saved_count,
                            "processing_time": time.time() - round_start
                        }
                        continue

                    # Full registration
                    dapi_full_img = safe_tifffile_read(dapi_file)
                    if dapi_full_img.ndim == 3:
                        dapi_full_img = dapi_full_img[..., 0]
                    dapi_mask = TissueProcessor.create_tissue_mask(dapi_full_img, self.config.tissue_mask_percentile)

                    rigid_full, rigid_ncc, rigid_method = self._multi_scale_rigid_registration(
                        padded_ref_path, dapi_file, ref_mask, dapi_mask
                    )
                   
                    logger.info(f"📊 Rigid: {rigid_method}, NCC={rigid_ncc:.4f}")
                   
                    if not self._validate_transform(rigid_full):
                        rigid_full = np.eye(2, 3, dtype=np.float32)
                        rigid_ncc = 0.0

                    # Flow
                    best_flow_field, best_flow_scale = None, None
                   
                    if self.config.enable_nonrigid and rigid_ncc > 0.25:
                        flow_processor = OpticalFlowProcessor(self.config)
                        accumulated_flow, current_flow_scale = None, None
                       
                        for scale in sorted(self.config.pyramid_levels):
                            logger.info(f"🌊 Flow @ {scale}")
                            ref_s = self._load_pyramid_level(padded_ref_path, scale)
                            mov_s = self._load_pyramid_level(dapi_file, scale)
                            ref_mask_s = cv2.resize(ref_mask.astype(np.uint8), (ref_s.shape[1], ref_s.shape[0]),
                                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                            dapi_mask_s = cv2.resize(dapi_mask.astype(np.uint8), (mov_s.shape[1], mov_s.shape[0]),
                                                     interpolation=cv2.INTER_NEAREST).astype(bool)
                            rigid_scaled = self._scale_transform(rigid_full, scale)
                           
                            if self.config.use_gpu_transforms and GPU_AVAILABLE:
                                mov_rigid_s = apply_affine_transform_gpu(mov_s, rigid_scaled, gpu_id=0)
                            else:
                                mov_rigid_s = apply_affine_transform(mov_s, rigid_scaled)
                           
                            if accumulated_flow is not None:
                                scale_ratio = scale / current_flow_scale
                                flow_upsampled = np.stack([
                                    cv2.resize(accumulated_flow[:, :, 0], (ref_s.shape[1], ref_s.shape[0]),
                                            interpolation=cv2.INTER_LINEAR) * scale_ratio,
                                    cv2.resize(accumulated_flow[:, :, 1], (ref_s.shape[1], ref_s.shape[0]),
                                            interpolation=cv2.INTER_LINEAR) * scale_ratio
                                ], axis=2)
                                mov_rigid_s = flow_processor.apply_nonrigid_transform(mov_rigid_s, flow_upsampled)
                           
                            flow_residual = flow_processor.compute_nonrigid_flow(
                                ref_s, mov_rigid_s, np.eye(2, 3, dtype=np.float32),\
                                ref_mask_s, dapi_mask_s, self.weight_cache
                            )
                           
                            if flow_residual is not None:
                                if accumulated_flow is None:
                                    accumulated_flow = flow_residual
                                else:
                                    accumulated_flow = flow_upsampled + flow_residual
                                current_flow_scale = scale
                                best_flow_field = accumulated_flow
                                best_flow_scale = scale

                    # Save
                    logger.info("💾 Saving...")
                    saved_count = 0
                    for file_path in round_files:
                        if "_processed.tif" in file_path.name:
                            continue
                        channel_name = file_path.stem.replace("_original", "")
                        if self.config.skip_non_reference_dapi and not is_reference_round:
                            if self._is_dapi_channel(file_path.name):
                                continue
                       
                        final_path = self.aligned_dir / f"{channel_name}_aligned.tif"
                        try:
                            image = safe_tifffile_read(file_path)
                            if image.ndim == 3:
                                image = image[..., 0]
                           
                            if self.config.use_gpu_transforms and GPU_AVAILABLE:
                                img_rigid = apply_affine_transform_gpu(image, rigid_full, gpu_id=0)
                            else:
                                img_rigid = apply_affine_transform(image, rigid_full)
                           
                            img_final = img_rigid
                           
                            if best_flow_field is not None and best_flow_scale is not None:
                                scale_factor = 1.0 / best_flow_scale
                                h, w = image.shape
                                flow_full = np.zeros((h, w, 2), dtype=np.float32)
                                flow_full[:, :, 0] = cv2.resize(best_flow_field[:, :, 0], (w, h),
                                                                interpolation=cv2.INTER_LINEAR) * scale_factor
                                flow_full[:, :, 1] = cv2.resize(best_flow_field[:, :, 1], (w, h),
                                                                interpolation=cv2.INTER_LINEAR) * scale_factor
                                max_flow = np.sqrt(flow_full[:,:,0]**2 + flow_full[:,:,1]**2).max()
                                max_allowed = max(image.shape) * 0.3
                                if max_flow > max_allowed:
                                    flow_full *= (max_allowed / max_flow)
                                img_final = flow_processor.apply_nonrigid_transform(img_rigid, flow_full)
                                del flow_full
                           
                            img_final_8bit = (np.clip(img_final, 0, 1) * 255).astype(np.uint8)
                            save_aligned_image_optimized(final_path, img_final_8bit, tile_size=self.config.tiff_tile_size)
                            saved_count += 1
                            del image, img_rigid, img_final, img_final_8bit
                            gc.collect()
                        except Exception as e:
                            logger.error(f"❌ Failed {channel_name}: {e}")
                   
                    logger.info(f"✅ Saved {saved_count} files")
                   
                    results["processed_rounds"][round_name] = {
                        "files_processed": saved_count,
                        "registration_quality": float(rigid_ncc),
                        "rigid_method": rigid_method,
                        "optical_flow_applied": best_flow_field is not None,
                        "processing_time": time.time() - round_start
                    }
                   
                    # 7. OPTIMIZATION: Aggressive per-round cleanup
                    del dapi_full_img, dapi_mask
                    gc.collect()
                    if GPU_AVAILABLE:
                        for gpu_id in range(NUM_GPUS):
                            with cp.cuda.Device(gpu_id):
                                cp.get_default_memory_pool().free_all_blocks()
                                cp.get_default_pinned_memory_pool().free_all_blocks()

                except Exception as e:
                    logger.error(f"❌ Round failed: {e}")

            results["status"] = "completed"
            results["total_time"] = time.time() - start_time
           
            if self.config.cleanup_preprocessed:
                self._cleanup_preprocessed(proc_dir)
           
            with open(self.output_dir / "results.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
           
            logger.info(f"\\n✅ COMPLETED in {results['total_time']:.1f}s")
            self.weight_cache.clear()
            return results

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _multi_scale_rigid_registration(self, ref_path: Path, mov_path: Path,\
                                        ref_mask: np.ndarray, mov_mask: np.ndarray) -> Tuple[np.ndarray, float, str]:
        logger.info(f"🔍 Multi-scale registration: {self.config.pyramid_levels}")
        accumulated_transform = np.eye(2, 3, dtype=np.float32)
        current_scale = 1.0
        best_ncc, best_method = 0.0, "Identity"

        for scale in self.config.pyramid_levels:
            logger.info(f"   📍 Scale {scale}")
            ref_s = self._load_pyramid_level(ref_path, scale)
            mov_s = self._load_pyramid_level(mov_path, scale)
            ref_mask_s = cv2.resize(ref_mask.astype(np.uint8), (ref_s.shape[1], ref_s.shape[0]),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
            mov_mask_s = cv2.resize(mov_mask.astype(np.uint8), (mov_s.shape[1], mov_s.shape[0]),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)

            scale_ratio = scale / current_scale
            accumulated_at_scale = self._scale_transform(accumulated_transform, scale_ratio)

            if self.config.use_gpu_transforms and GPU_AVAILABLE:
                mov_warped = apply_affine_transform_gpu(mov_s, accumulated_at_scale, gpu_id=0)
            else:
                mov_warped = apply_affine_transform(mov_s, accumulated_at_scale)

            combined_mask = ref_mask_s & mov_mask_s

            if np.sum(combined_mask) > 100:
                weight_map = TissueProcessor.create_nuclei_weight_map(ref_s, combined_mask, self.weight_cache)
                baseline_ncc = _compute_weighted_ncc(ref_s, mov_warped, weight_map)
            else:
                baseline_ncc = 0.0

            logger.info(f"      Baseline: {baseline_ncc:.4f}")

            best_T = np.eye(2, 3, dtype=np.float32)
            best_ncc_scale, method = baseline_ncc, "Baseline"

            # CLAHE-enhanced registration images
            ref_reg_8bit = prepare_registration_image(ref_s, self.config)
            mov_reg_8bit = prepare_registration_image(mov_warped, self.config)

            # Template matching
            try:
                coords = cv2.findNonZero(ref_mask_s.astype(np.uint8))
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    margin = int(min(w, h) * 0.05)
                    x_start = max(0, x - margin)
                    y_start = max(0, y - margin)
                    x_end = min(ref_reg_8bit.shape[1], x + w + margin)
                    y_end = min(ref_reg_8bit.shape[0], y + h + margin)

                    ref_crop = ref_reg_8bit[y_start:y_end, x_start:x_end]

                    if ref_crop.shape[0] <= mov_reg_8bit.shape[0] and ref_crop.shape[1] <= mov_reg_8bit.shape[1]:
                        res = cv2.matchTemplate(mov_reg_8bit, ref_crop, cv2.TM_CCOEFF_NORMED)
                        _, _, _, max_loc = cv2.minMaxLoc(res)

                        T_template = np.eye(2, 3, dtype=np.float32)
                        T_template[0, 2] = x_start - max_loc[0]
                        T_template[1, 2] = y_start - max_loc[1]

                        if self.config.use_gpu_transforms and GPU_AVAILABLE:
                            mov_template = apply_affine_transform_gpu(mov_warped, T_template, gpu_id=0)
                        else:
                            mov_template = apply_affine_transform(mov_warped, T_template)

                        if np.sum(combined_mask) > 100:
                            weight_map_template = TissueProcessor.create_nuclei_weight_map(ref_s, combined_mask, self.weight_cache)
                            template_ncc = _compute_weighted_ncc(ref_s, mov_template, weight_map_template)
                            logger.info(f"      Template: {template_ncc:.4f}")
                            if template_ncc > best_ncc_scale:
                                best_T, best_ncc_scale, method = T_template.copy(), template_ncc, "Template"
            except Exception as e:
                logger.warning(f"      Template failed: {e}")

            # Phase
            try:
                shift, _, _ = phase_cross_correlation(ref_reg_8bit, mov_reg_8bit, upsample_factor=10)
                T_phase = np.eye(2, 3, dtype=np.float32)
                T_phase[0, 2], T_phase[1, 2] = shift[1], shift[0]

                if self.config.use_gpu_transforms and GPU_AVAILABLE:
                    mov_phase = apply_affine_transform_gpu(mov_warped, T_phase, gpu_id=0)
                else:
                    mov_phase = apply_affine_transform(mov_warped, T_phase)

                if np.sum(combined_mask) > 100:
                    weight_map_phase = TissueProcessor.create_nuclei_weight_map(ref_s, combined_mask, self.weight_cache)
                    phase_ncc = _compute_weighted_ncc(ref_s, mov_phase, weight_map_phase)
                    logger.info(f"      Phase: {phase_ncc:.4f} (shift=[{shift[1]:.1f}, {shift[0]:.1f}])")
                    if phase_ncc > best_ncc_scale:
                        best_T, best_ncc_scale, method = T_phase.copy(), phase_ncc, "Phase"
            except Exception as e:
                logger.warning(f"      Phase failed: {e}")

            # Features
            should_try_features = (
                best_ncc_scale > baseline_ncc + 0.01 or
                baseline_ncc < 0.85 or
                scale == self.config.pyramid_levels[0]
            )

            if should_try_features:
                logger.info(f"      🔧 Trying features...")
                if self.config.use_gpu_transforms and GPU_AVAILABLE:
                    mov_for_feat = apply_affine_transform_gpu(mov_warped, best_T, gpu_id=0)
                else:
                    mov_for_feat = apply_affine_transform(mov_warped, best_T)

                feat_reg = FeatureBasedRegistration(self.config)
                T_feat, ncc_feat = feat_reg.register(ref_s, mov_for_feat, ref_mask_s, mov_mask_s, self.weight_cache)

                if ncc_feat > best_ncc_scale + 0.005:
                    T_feat_3x3 = np.vstack([T_feat, [0, 0, 1]])
                    best_3x3 = np.vstack([best_T, [0, 0, 1]])
                    composed = T_feat_3x3 @ best_3x3
                    best_T, best_ncc_scale, method = composed[:2, :], ncc_feat, method + "+Feature"
                    logger.info(f"      ✅ Features improved to {ncc_feat:.4f}")
                else:
                    logger.info(f"      ⚠️  Features didn't help (NCC={ncc_feat:.4f})")
            else:
                logger.info(f"      ⏭️  Skipping features (NCC already good: {baseline_ncc:.4f})")

            if best_ncc_scale >= baseline_ncc - 0.01:
                T_corr_3x3 = np.vstack([best_T, [0, 0, 1]])
                acc_3x3 = np.vstack([accumulated_at_scale, [0, 0, 1]])
                composed = T_corr_3x3 @ acc_3x3
                accumulated_transform = composed[:2, :]
                current_scale = scale
                best_ncc = best_ncc_scale
                best_method = f"{method}@{scale}"
                logger.info(f"      📊 Final for scale {scale}: {method}, NCC={best_ncc_scale:.4f}")

        accumulated_transform_full = self._scale_transform(accumulated_transform, 1.0 / current_scale)
        return accumulated_transform_full, best_ncc, best_method

    def _scale_transform(self, T: np.ndarray, ratio: float) -> np.ndarray:
        T_scaled = T.copy()
        T_scaled[0, 2] *= ratio
        T_scaled[1, 2] *= ratio
        return T_scaled

    def _validate_transform(self, transform: np.ndarray, max_translation: float = 5000.0) -> bool:
        if transform.shape != (2, 3):
            return False
        tx, ty = transform[0, 2], transform[1, 2]
        if abs(tx) > max_translation or abs(ty) > max_translation:
            return False
        det = np.linalg.det(transform[:2, :2])
        return 0.1 < abs(det) < 10.0

    def _load_pyramid_level(self, path: Path, scale: float) -> np.ndarray:
        # 8. OPTIMIZATION: Avoid copies on load
        img = safe_tifffile_read(path)
        if img.ndim == 3:
            img = img[..., 0]
       
        # Optimize resize
        if scale < 1.0:
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA).astype(np.float32, copy=False)
       
        return img.astype(np.float32, copy=False)

    def _save_mask(self, mask: np.ndarray, output_path: Path):
        mask_vis = (mask > 0).astype(np.uint8) * 255
        tifffile.imwrite(str(output_path), mask_vis, compression='lzw')

    def _group_files_by_round(self, files: List[Path]) -> Dict[str, List[Path]]:
        rounds = {}
        for file in files:
            round_id = file.stem.split("_Merged")[0] if "_Merged" in file.stem else file.stem
            rounds.setdefault(round_id, []).append(file)
        return {k: sorted(v) for k, v in rounds.items()}

    def _find_dapi_file(self, round_files: List[Path]) -> Optional[Path]:
        for file in round_files:
            if any(re.search(p, file.name.lower()) for p in self.config.dapi_patterns):
                if "_processed.tif" in file.name.lower():
                    return file
        for file in round_files:
            if any(re.search(p, file.name.lower()) for p in self.config.dapi_patterns):
                return file
        return round_files[0] if round_files else None

    def _is_dapi_channel(self, filename: str) -> bool:
        return any(re.search(p, filename.lower()) for p in self.config.dapi_patterns)

    def _cleanup_preprocessed(self, proc_dir: Path):
        if not self.config.cleanup_preprocessed:
            return
        try:
            deleted_size = sum(f.stat().st_size for f in proc_dir.glob("*") if f.is_file())
            for file in proc_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            if not any(proc_dir.iterdir()):
                proc_dir.rmdir()
            logger.info(f"🗑️  Cleaned {deleted_size / 1e9:.2f} GB")
        except Exception as e:
            logger.warning(f"⚠️  Cleanup failed: {e}")


# ============================================================================
# BATCH PROCESSOR - Automated multi-folder processing
# ============================================================================
class BatchProcessor:
    """Process multiple folders automatically"""
   
    
    def __init__(self, root_folder: str, output_root: str):
        self.root_folder = Path(root_folder)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
       
    def find_batches(self) -> List[Dict[str, Any]]:
        """Auto-detect batches in ALL subfolders (recursive)"""
        batches = []
       
        # ✅ Use rglob to recursively search all subfolders
        for subfolder in self.root_folder.rglob("*"):
            if not subfolder.is_dir():
                continue
           
            # Look for .tif files directly in this folder
             ################## Change if you want to exclude certain files ##################
            tif_files = [f for f in subfolder.glob("*.tif") if "Cor" not in f.name]
            ###################################################################################
            if not tif_files:
                continue
           
            # Try to find reference file
            ref_file = self._auto_detect_reference(tif_files)
            if not ref_file:
                logger.warning(f"⚠️  No reference found in {subfolder.relative_to(self.root_folder)}, skipping")
                continue
           
            # Use relative path for better names
            relative_name = str(subfolder.relative_to(self.root_folder)).replace(os.sep, "_")
           
            batches.append({
                "name": relative_name,
                "input_folder": str(subfolder),
                "output_folder": str(self.output_root / relative_name),
                "reference_file": str(ref_file),
                "num_files": len(tif_files)
            })
       
        return batches
   
    def _auto_detect_reference(self, tif_files: List[Path]) -> Optional[Path]:
        """Auto-detect reference file (p16, p21, ch00, etc.)"""
        ref_patterns = [r'p16', r'p21', r'ch00']
       
        for pattern in ref_patterns:
            for file in tif_files:
                if re.search(pattern, file.name, re.IGNORECASE):
                    return file
       
        # Fallback: first file with ch00 or dapi
        for file in tif_files:
            if re.search(r'ch00|dapi', file.name, re.IGNORECASE):
                return file
       
        return None
   
    def run(self, pyramid_levels: List[float] = [0.25, 0.5],
            enable_nonrigid: bool = True,
            enable_pass3: bool = True,
            cleanup: bool = True) -> Dict[str, Any]:
        """Run batch processing"""
       
        overall_start = time.time()
        batches = self.find_batches()
       
        if not batches:
            logger.error("❌ No batches found!")
            return {"status": "failed", "error": "No batches found"}
       
        logger.info(f"\\n{'='*80}")
        logger.info(f"🎯 BATCH PROCESSING: {len(batches)} batches found")
        logger.info(f"{'='*80}\\n")
       
        results = {
            "status": "started",
            "total_batches": len(batches),
            "batches": {}
        }
       
        for i, batch_info in enumerate(batches, 1):
            batch_name = batch_info["name"]
            logger.info(f"\\n{'='*80}")
            logger.info(f"📦 BATCH {i}/{len(batches)}: {batch_name}")
            logger.info(f"   Files: {batch_info['num_files']}")
            logger.info(f"   Reference: {Path(batch_info['reference_file']).name}")
            logger.info(f"{'='*80}\\n")
           
            try:
                config = RegistrationConfig(
                    input_folder=batch_info["input_folder"],
                    output_folder=batch_info["output_folder"],
                    reference_file=batch_info["reference_file"],
                    pyramid_levels=pyramid_levels,
                    use_gpu=True,
                    enable_nonrigid=enable_nonrigid,
                    enable_pass3_refinement=enable_pass3,
                    apply_advanced_preprocessing=True,
                    tissue_mask_percentile=1.0,
                    cleanup_preprocessed=cleanup,
                    skip_non_reference_dapi=True,
                    use_gpu_transforms=True,
                )
               
                pipeline = NextGen4iPipeline(config)
                batch_result = pipeline.run()
               
                results["batches"][batch_name] = batch_result
               
                # Cleanup between batches
                logger.info(f"\\n🧹 Cleaning up between batches...")
                gc.collect()
                if GPU_AVAILABLE:
                    for gpu_id in range(NUM_GPUS):
                        with cp.cuda.Device(gpu_id):
                            cp.get_default_memory_pool().free_all_blocks()
                            cp.get_default_pinned_memory_pool().free_all_blocks()
               
            except Exception as e:
                logger.error(f"❌ Batch {batch_name} failed: {e}")
                results["batches"][batch_name] = {
                    "status": "failed",
                    "error": str(e)
                }
       
        results["status"] = "completed"
        results["total_time"] = time.time() - overall_start
       
        # Save overall results
        overall_results_file = self.output_root / "batch_results.json"
        with open(overall_results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
       
        logger.info(f"\\n{'='*80}")
        logger.info(f"✅ BATCH PROCESSING COMPLETED")
        logger.info(f"   Total time: {results['total_time']:.1f}s")
        logger.info(f"   Results: {overall_results_file}")
        logger.info(f"{'='*80}\\n")
       
        return results