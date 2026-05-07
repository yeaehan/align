# ============================================================================
# FEATURE REGISTRATION
# ============================================================================
class FeatureBasedRegistration:
    def __init__(self, config: RegistrationConfig):
        self.config = config

    def register(self, ref: np.ndarray, mov: np.ndarray, ref_mask: np.ndarray, mov_mask: np.ndarray,
                 weight_cache: Optional[WeightMapCache] = None) -> Tuple[np.ndarray, float]:
       
        if ref_mask.shape != ref.shape:
            ref_mask = cv2.resize(ref_mask.astype(np.uint8), (ref.shape[1], ref.shape[0]),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
        if mov_mask.shape != mov.shape:
            mov_mask = cv2.resize(mov_mask.astype(np.uint8), (mov.shape[1], mov.shape[0]),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
       
        min_h, min_w = min(ref.shape[0], mov.shape[0]), min(ref.shape[1], mov.shape[1])
        ref_safe, mov_safe = ref[:min_h, :min_w], mov[:min_h, :min_w]
        ref_mask_safe, mov_mask_safe = ref_mask[:min_h, :min_w], mov_mask[:min_h, :min_w]
       
        ref_8bit = cv2.normalize(ref_safe, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        mov_8bit = cv2.normalize(mov_safe, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        combined_mask = (ref_mask_safe & mov_mask_safe).astype(np.uint8) * 255
       
        try:
            detector = cv2.SIFT_create(nfeatures=self.config.max_features)
        except AttributeError:
            detector = cv2.xfeatures2d.SIFT_create(nfeatures=self.config.max_features)
       
        # 5. OPTIMIZATION: Remove ThreadPool to reduce peak memory usage
        # Sequential is safer for memory
        kp1, des1 = detector.detectAndCompute(ref_8bit, combined_mask)
        kp2, des2 = detector.detectAndCompute(mov_8bit, combined_mask)
       
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return np.eye(2, 3), 0.0
       
        # Adaptive matcher
        if len(des1) < 1000:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            matcher = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=5),
                dict(checks=50)
            )
        matches = matcher.knnMatch(des1, des2, k=2)
       
        good_matches = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
       
        if len(good_matches) < 10:
            return np.eye(2, 3), 0.0
       
        ref_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        mov_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
       
        try:
            transform_matrix, _ = cv2.estimateAffinePartial2D(
                mov_pts, ref_pts, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                maxIters=3000, confidence=0.99
            )
            if transform_matrix is None:
                return np.eye(2, 3, dtype=np.float32), 0.0
        except:
            return np.eye(2, 3, dtype=np.float32), 0.0
       
        if self.config.use_gpu_transforms and GPU_AVAILABLE:
            mov_transformed = apply_affine_transform_gpu(mov_safe, transform_matrix, gpu_id=0)
        else:
            mov_transformed = apply_affine_transform(mov_safe, transform_matrix)
       
        valid_mask = ref_mask_safe & mov_mask_safe
        if np.sum(valid_mask) < 100:
            return transform_matrix, 0.0
       
        weight_map = TissueProcessor.create_nuclei_weight_map(ref_safe, valid_mask, weight_cache)
        ncc_quality = _compute_weighted_ncc(ref_safe, mov_transformed, weight_map)
       
        return transform_matrix, ncc_quality


# ============================================================================
# OPTICAL FLOW PROCESSOR
# ============================================================================
class OpticalFlowProcessor:
    def __init__(self, config: RegistrationConfig):
        self.config = config

    def compute_nonrigid_flow(self, ref_img: np.ndarray, mov_img: np.ndarray,
                              rigid_transform: np.ndarray, ref_mask: np.ndarray, mov_mask: np.ndarray,
                              weight_cache: Optional[WeightMapCache] = None):
        try:
            logger.info("🌊 Optical flow...")
           
            ref = ref_img.astype(np.float32)
            mov = mov_img.astype(np.float32)
           
            if self.config.use_gpu_transforms and GPU_AVAILABLE:
                mov_rigid = apply_affine_transform_gpu(mov, rigid_transform, gpu_id=0)
            else:
                mov_rigid = apply_affine_transform(mov, rigid_transform)
           
            combined_mask = ref_mask & mov_mask
           
            if combined_mask.sum() / combined_mask.size < 0.01:
                return None
           
            weight_map = TissueProcessor.create_nuclei_weight_map(ref, combined_mask, weight_cache)
            baseline_ncc = _compute_weighted_ncc(ref, mov_rigid, weight_map)
           
            logger.info(f"   Baseline: {baseline_ncc:.4f}")
           
            if baseline_ncc < 0.25:
                return None

            ref8 = (np.clip(ref, 0, 1) * 255).astype(np.uint8)
            mov8_rigid = (np.clip(mov_rigid, 0, 1) * 255).astype(np.uint8)
            if combined_mask is not None:
                ref8 *= combined_mask.astype(np.uint8)
                mov8_rigid *= combined_mask.astype(np.uint8)

            def create_dis():
                inst = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                inst.setVariationalRefinementAlpha(25.0)
                inst.setVariationalRefinementDelta(5.0)
                inst.setVariationalRefinementGamma(15.0)
                inst.setVariationalRefinementIterations(45)
                return inst

            # Pass 1
            dis1 = create_dis()
            flow1 = dis1.calc(ref8, mov8_rigid, None)
            if combined_mask is not None:
                flow1 *= combined_mask[..., None].astype(np.float32)
           
            mov1 = self._apply_flow(mov_rigid, flow1[..., 1], flow1[..., 0])
            ncc1 = _compute_weighted_ncc(ref, mov1, weight_map)
            logger.info(f"   Pass 1: {ncc1:.4f}")
            best_flow, best_ncc = flow1, ncc1

            # Pass 2
            if ncc1 > baseline_ncc + 0.001:
                mov1_8 = (np.clip(mov1, 0, 1) * 255).astype(np.uint8)
                if combined_mask is not None:
                    mov1_8 *= combined_mask.astype(np.uint8)
               
                dis2 = create_dis()
                flow2 = dis2.calc(ref8, mov1_8, None)
                if combined_mask is not None:
                    flow2 *= combined_mask[..., None].astype(np.float32)
               
                flow_total = flow1 + flow2
                mov2 = self._apply_flow(mov_rigid, flow_total[..., 1], flow_total[..., 0])
                ncc2 = _compute_weighted_ncc(ref, mov2, weight_map)
                logger.info(f"   Pass 2: {ncc2:.4f}")
               
                if ncc2 > best_ncc:
                    best_flow, best_ncc = flow_total, ncc2
                   
                    # Pass 3
                    if self.config.enable_pass3_refinement and ncc2 > baseline_ncc + 0.02 and ncc2 < 0.94:
                        logger.info("   🎯 Pass 3...")
                        flow_pass3 = self._fast_selective_pass3(ref, mov_rigid, flow_total, combined_mask, weight_map)
                        if flow_pass3 is not None:
                            mov3 = self._apply_flow(mov_rigid, flow_pass3[..., 1], flow_pass3[..., 0])
                            ncc3 = _compute_weighted_ncc(ref, mov3, weight_map)
                            if ncc3 > best_ncc + 0.002:
                                best_flow, best_ncc = flow_pass3, ncc3

            return best_flow

        except Exception as e:
            logger.error(f"Flow error: {e}")
            return None

    def _fast_selective_pass3(self, ref: np.ndarray, mov: np.ndarray,
                              current_flow: np.ndarray, mask: np.ndarray,
                              weight_map: np.ndarray) -> Optional[np.ndarray]:
        try:
            h, w = ref.shape
           
            # 6. OPTIMIZATION: Tiled processing for CPU transform on large images
            if h * w > 4000 * 4000: # 16MP threshold
                mov_warped = np.empty_like(ref)
                tile_size = 2048
                y_grid, x_grid = np.mgrid[0:tile_size, 0:tile_size].astype(np.float32)
               
                for i in range(0, h, tile_size):
                    for j in range(0, w, tile_size):
                        i_end, j_end = min(i + tile_size, h), min(j + tile_size, w)
                        curr_h, curr_w = i_end - i, j_end - j
                       
                        # Generate tile coords on fly
                        y_tile = y_grid[:curr_h, :curr_w] + i + current_flow[i:i_end, j:j_end, 1]
                        x_tile = x_grid[:curr_h, :curr_w] + j + current_flow[i:i_end, j:j_end, 0]
                       
                        mov_warped[i:i_end, j:j_end] = map_coordinates(
                            mov, [y_tile, x_tile], order=1, mode='constant', cval=0
                        )
            else:
                y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
                mov_warped = map_coordinates(mov,
                                              [y_coords + current_flow[:,:,1],
                                               x_coords + current_flow[:,:,0]],
                                              order=1, mode='constant', cval=0)
                del y_coords, x_coords # Explicit cleanup
           
            block_size = 8
            error_map = np.abs(ref - mov_warped)
            h_blocks, w_blocks = h // block_size, w // block_size
            h_trim, w_trim = h_blocks * block_size, w_blocks * block_size
           
            error_trimmed = error_map[:h_trim, :w_trim]
            error_blocks = error_trimmed.reshape(h_blocks, block_size, w_blocks, block_size).mean(axis=(1, 3))
           
            threshold = np.percentile(error_blocks[error_blocks > 0], 80)
            bad_blocks = error_blocks > threshold
           
            if bad_blocks.sum() < 10:
                logger.info("      ⏭️  Pass 3 skipped (alignment already good)")
                return None
           
            bad_mask = np.kron(bad_blocks, np.ones((block_size, block_size), dtype=bool))
            bad_mask_full = np.zeros((h, w), dtype=bool)
            bad_mask_full[:h_trim, :w_trim] = bad_mask
           
            kernel = np.ones((32, 32), np.uint8)
            refine_mask = cv2.dilate(bad_mask_full.astype(np.uint8), kernel).astype(bool) & mask
           
            rows, cols = np.any(refine_mask, axis=1), np.any(refine_mask, axis=0)
            if not rows.any() or not cols.any():
                return None
           
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
           
            pad = 128
            y_min, y_max = max(0, y_min - pad), min(h, y_max + pad)
            x_min, x_max = max(0, x_min - pad), min(w, x_max + pad)
           
            ref_roi = ref[y_min:y_max, x_min:x_max]
            mov_roi = mov[y_min:y_max, x_min:x_max]
            flow_roi = current_flow[y_min:y_max, x_min:x_max, :]
            mask_roi = refine_mask[y_min:y_max, x_min:x_max]
           
            y_roi, x_roi = np.mgrid[0:ref_roi.shape[0], 0:ref_roi.shape[1]].astype(np.float32)
            mov_warped_roi = map_coordinates(mov_roi,
                                              [y_roi + flow_roi[:,:,1],
                                               x_roi + flow_roi[:,:,0]],
                                              order=1, mode='constant', cval=0)
           
            ref_roi_8 = (np.clip(ref_roi, 0, 1) * 255).astype(np.uint8)
            mov_roi_8 = (np.clip(mov_warped_roi, 0, 1) * 255).astype(np.uint8)
           
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            dis.setVariationalRefinementAlpha(30.0)
            dis.setVariationalRefinementDelta(7.0)
            dis.setVariationalRefinementGamma(20.0)
            dis.setVariationalRefinementIterations(60)
           
            flow_residual = dis.calc(ref_roi_8, mov_roi_8, None)
            flow_residual[:, :, 0] *= mask_roi
            flow_residual[:, :, 1] *= mask_roi
            flow_refined_roi = flow_roi + flow_residual
           
            refined_flow = current_flow.copy()
            feather_mask = cv2.GaussianBlur(mask_roi.astype(np.float32), (51, 51), 0)
           
            refined_flow[y_min:y_max, x_min:x_max, 0] = (
                current_flow[y_min:y_max, x_min:x_max, 0] * (1 - feather_mask) +
                flow_refined_roi[:, :, 0] * feather_mask
            )
            refined_flow[y_min:y_max, x_min:x_max, 1] = (
                current_flow[y_min:y_max, x_min:x_max, 1] * (1 - feather_mask) +
                flow_refined_roi[:, :, 1] * feather_mask
            )
           
            refined_flow[:, :, 0] *= mask
            refined_flow[:, :, 1] *= mask
           
            # Use tiled check for final
            if h * w > 4000 * 4000:
                mov_final = np.empty_like(mov)
                # (Reuse tiling logic)
                for i in range(0, h, tile_size):
                    for j in range(0, w, tile_size):
                        i_end, j_end = min(i + tile_size, h), min(j + tile_size, w)
                        y_tile = y_grid[:i_end-i, :j_end-j] + i + refined_flow[i:i_end, j:j_end, 1]
                        x_tile = x_grid[:i_end-i, :j_end-j] + j + refined_flow[i:i_end, j:j_end, 0]
                        mov_final[i:i_end, j:j_end] = map_coordinates(
                            mov, [y_tile, x_tile], order=1, mode='constant', cval=0
                        )
            else:
                mov_final = map_coordinates(mov,
                                         [y_coords + refined_flow[:,:,1],
                                          x_coords + refined_flow[:,:,0]],
                                         order=1, mode='constant', cval=0)
           
            final_ncc = _compute_weighted_ncc(ref, mov_final, weight_map)
            current_ncc = _compute_weighted_ncc(ref, mov_warped, weight_map)
           
            logger.info(f"      Pass 3: {current_ncc:.4f} → {final_ncc:.4f} (Δ={final_ncc - current_ncc:+.4f})")
           
            if final_ncc > current_ncc + 0.002:
                logger.info("      ✅ Pass 3 improved!")
                return refined_flow
            else:
                logger.info("      ⏭️  Pass 3 no improvement")
                return None
           
        except Exception as e:
            logger.warning(f"      ⚠️  Pass 3 failed: {e}")
            return None

    def _apply_flow(self, image, flow_y, flow_x):
        H, W = image.shape
        y_coords, x_coords = np.mgrid[0:H, 0:W].astype(np.float32)
        return map_coordinates(image, [y_coords + flow_y, x_coords + flow_x],
                             order=1, mode='constant', cval=0, prefilter=False)

    def apply_nonrigid_transform(self, image, flow_field):
        if flow_field is None:
            return image
        h, w = image.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        return map_coordinates(image, [y_coords + flow_field[:, :, 1], x_coords + flow_field[:, :, 0]],
                               order=1, mode='constant', cval=0, prefilter=False)
