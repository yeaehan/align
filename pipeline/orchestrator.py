import logging
from pathlib import Path
import subprocess
import re

import numpy as np
import cv2

from align.config import RegistrationConfig
from align.io.reader import read_2d_as_float, discover_moving_files
from align.io.writer import save_tiff, save_debug_overlay
from align.core.tissue import TissueProcessor
from align.core.preprocessing import preprocess_dapi
from align.core.transforms import warp_affine, identity
from align.registration.rigid import RigidRegistrar
from align.registration.nonrigid import OpticalFlowRegistrar

logger = logging.getLogger(__name__)

# --- HARDWARE PROFILER ---
def log_hardware_usage(step_name="Hardware Status"):
    """Logs current CPU, System RAM, and GPU VRAM utilization."""
    try:
        import psutil
        cpu_util = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024**3)
        ram_total_gb = ram.total / (1024**3)
        logger.info(f"[{step_name}] CPU: {cpu_util}% | System RAM: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram.percent}%)")
    except ImportError:
        pass
        
    try:
        # Query nvidia-smi for GPU stats natively
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        for line in result.stdout.strip().split('\n'):
            if line:
                idx, gpu_util, mem_used, mem_total = line.split(', ')
                logger.info(f"[{step_name}] GPU {idx}: Util {gpu_util} | VRAM {mem_used} / {mem_total}")
    except Exception:
        pass  # nvidia-smi not found or failed

# --- GPU MEMORY CLEARING ---
def clear_gpu_memory():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except ImportError:
        pass

class AlignmentPipeline:
    def __init__(self, config: RegistrationConfig):
        self.config = config
        cv2.setUseOptimized(True)
        cv2.setNumThreads(max(1, config.n_workers))
        self.rigid_registrar = RigidRegistrar(
            pyramid_levels=config.pyramid_levels,
            max_features=config.max_features
        )
        
        if config.enable_nonrigid:
            self.nonrigid_registrar = OpticalFlowRegistrar(
                pyramid_levels=config.pyramid_levels,
                enable_pass3=config.enable_pass3_refinement,
                use_gpu=config.use_gpu,
            )
        else:
            self.nonrigid_registrar = None

    def _warp_tiled(
        self,
        mov_img: np.ndarray,
        transform: np.ndarray,
        output_shape: tuple,
        tile_size: int = 25000,
        is_mask: bool = False,
    ) -> np.ndarray:
        """Warp image in destination tiles to avoid OpenCV's 32767-pixel limit."""
        h_out, w_out = output_shape
        h_src, w_src = mov_img.shape
        max_cv_dim = 32766
        tile_size = min(tile_size, 16000)
        
        logger.debug(f"_warp_tiled: src={mov_img.shape}, out={output_shape}, tile_size={tile_size}")
        
        # If both source and output fit, use direct warp
        if (h_src < max_cv_dim and w_src < max_cv_dim and
            h_out < max_cv_dim and w_out < max_cv_dim):
            return warp_affine(mov_img, transform, (h_out, w_out), is_mask=is_mask)
        
        # Initialize output
        result = np.zeros(output_shape, dtype=np.float32)
        
        # Map each destination tile back to the source, crop that source patch,
        # and adjust the affine so OpenCV never sees an oversized src or dst.
        T = np.vstack([transform.astype(np.float32), [0, 0, 1]])
        try:
            T_inv = np.linalg.inv(T)[:2, :]
        except np.linalg.LinAlgError:
            logger.warning("Singular transform matrix, returning blank warped image")
            return result

        margin = 4
        tile_count = 0
        skipped_count = 0
        for out_y0 in range(0, h_out, tile_size):
            out_y1 = min(out_y0 + tile_size, h_out)
            for out_x0 in range(0, w_out, tile_size):
                out_x1 = min(out_x0 + tile_size, w_out)
                tile_h = out_y1 - out_y0
                tile_w = out_x1 - out_x0

                out_corners = np.array([
                    [out_x0, out_y0],
                    [out_x1, out_y0],
                    [out_x1, out_y1],
                    [out_x0, out_y1],
                ], dtype=np.float32).reshape(-1, 1, 2)
                src_corners = cv2.transform(out_corners, T_inv).reshape(-1, 2)

                src_x0 = max(0, int(np.floor(src_corners[:, 0].min())) - margin)
                src_x1 = min(w_src, int(np.ceil(src_corners[:, 0].max())) + margin)
                src_y0 = max(0, int(np.floor(src_corners[:, 1].min())) - margin)
                src_y1 = min(h_src, int(np.ceil(src_corners[:, 1].max())) + margin)

                if src_x0 >= src_x1 or src_y0 >= src_y1:
                    skipped_count += 1
                    continue

                src_patch = mov_img[src_y0:src_y1, src_x0:src_x1]
                src_h, src_w = src_patch.shape
                if src_h >= max_cv_dim or src_w >= max_cv_dim:
                    logger.warning(
                        f"Skipping tile with oversized source patch {src_h}x{src_w}; "
                        f"reduce tile_size below {tile_size}"
                    )
                    skipped_count += 1
                    continue

                tile_transform = transform.copy().astype(np.float32)
                tile_transform[:, 2] += transform[:, :2] @ np.array([src_x0, src_y0], dtype=np.float32)
                tile_transform[0, 2] -= out_x0
                tile_transform[1, 2] -= out_y0

                try:
                    warped_tile = warp_affine(
                        src_patch,
                        tile_transform,
                        (tile_h, tile_w),
                        is_mask=is_mask,
                    )
                except cv2.error as e:
                    logger.warning(f"Warp failed for output tile ({out_y0}:{out_y1}, {out_x0}:{out_x1}): {e}")
                    skipped_count += 1
                    continue

                result[out_y0:out_y1, out_x0:out_x1] = warped_tile
                tile_count += 1

        logger.debug(f"Tiled warp wrote {tile_count} tiles, skipped {skipped_count} tiles")
        
        return result

    def run(self):
        logger.info("=== 🚀 NEXT-GEN 4i v4.1 (Optimized) ===")
        
        ref_file_str = self.config.reference_file
        if "::" in ref_file_str:
            ref_real_path, ref_ch = ref_file_str.split("::")
            ref_path = Path(ref_real_path)
            ref_name = f"{ref_path.stem}_{ref_ch}.tif"
        else:
            ref_path = Path(ref_file_str)
            ref_name = ref_path.name
        
        if not ref_path.exists():
            logger.error(f"Reference file not found: {ref_path}")
            return

        # 1. Load and prepare reference image
        logger.info(f"Loading reference: {ref_name}")
        logger.info("Reading reference image (Network I/O)...")
        ref_img = read_2d_as_float(ref_file_str)
        
        logger.info("Creating reference tissue mask...")
        ref_mask = TissueProcessor.create_tissue_mask(ref_img, self.config.tissue_mask_percentile)
        clear_gpu_memory()

        # 2. Find all moving files in the folder
        if hasattr(self.config, 'moving_files') and self.config.moving_files is not None:
            moving_files = self.config.moving_files
        else:
            moving_files = discover_moving_files(ref_file_str)
        logger.info(f"Found {len(moving_files)} moving files to align.")
        
        # Sort files to ensure anchor channels (ch00) are processed first
        moving_files = sorted(
            moving_files, 
            key=lambda x: (0 if any(p in x.lower() for p in self.config.dapi_patterns) else 1, x)
        )

        out_dir = Path(self.config.output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Helper to group files by round (strips off '_ch01.tif', etc.)
        def get_group_key(filename):
            match = re.search(r'[_\\-]ch\d+', filename, re.IGNORECASE)
            return filename[:match.start()] if match else filename
            
        ref_group_key = get_group_key(ref_name)
        round_transform_cache = {}

        # 3. Align each file to the reference
        for mov_file in moving_files:
            if "::" in mov_file:
                mov_real_path, mov_ch = mov_file.split("::")
                mov_path = Path(mov_real_path)
                mov_name = f"{mov_path.stem}_{mov_ch}.tif"
            else:
                mov_path = Path(mov_file)
                mov_name = mov_path.name
                
            logger.info(f"--- Aligning {mov_name} ---")
            log_hardware_usage("Pre-Registration")
            
            group_key = get_group_key(mov_name)
            is_anchor = any(p in mov_name.lower() for p in self.config.dapi_patterns)
            
            if group_key == ref_group_key:
                logger.info("🎯 Reference round - copying")
                transform = identity()
                flow_steps = []
                skip_registration = True
            elif is_anchor or group_key not in round_transform_cache:
                flow_steps = []
                skip_registration = False
            else:
                logger.info(f"Using cached DAPI transforms for round: {group_key}")
                cached_transforms = round_transform_cache[group_key]
                transform = cached_transforms["affine"]
                flow_steps = cached_transforms["flow_steps"]
                skip_registration = True
            
            logger.info("Reading moving image (Network I/O)...")
            mov_img = read_2d_as_float(mov_file)
            
            if not skip_registration:
                logger.info("Creating moving tissue mask...")
                mov_mask = TissueProcessor.create_tissue_mask(mov_img, self.config.tissue_mask_percentile)
                clear_gpu_memory()

                # Rigid Registration
                logger.info("🔍 Multi-scale registration...")
                
                # --- PROXY SCALING (From Notebook Optimization) ---
                # The notebook limits registration to `refine_max_dim=4096`.
                # We cap at 8192 to guarantee it runs in seconds while keeping high accuracy.
                h_mov, w_mov = mov_img.shape
                h_ref, w_ref = ref_img.shape
                max_dim = max(h_mov, w_mov, h_ref, w_ref)
                max_rigid_dim = 8192.0
                
                if max_dim > max_rigid_dim:
                    proxy_scale = max_rigid_dim / max_dim
                    logger.info(f"   📐 Image is {max_dim}px. Downscaling to {max_rigid_dim}px proxy for instant rigid registration...")
                    
                    # Resize the RAW images first before applying the heavy morphological filters!
                    ref_proxy_raw = cv2.resize(ref_img, (0,0), fx=proxy_scale, fy=proxy_scale, interpolation=cv2.INTER_AREA)
                    mov_proxy_raw = cv2.resize(mov_img, (0,0), fx=proxy_scale, fy=proxy_scale, interpolation=cv2.INTER_AREA)
                    
                    logger.info("   ⚡ Applying morphological filters to proxy images (instant)...")
                    ref_proxy = preprocess_dapi(ref_proxy_raw, tophat_radius=self.config.preprocessing_tophat_radius, use_gpu=self.config.use_gpu)
                    mov_proxy = preprocess_dapi(mov_proxy_raw, tophat_radius=self.config.preprocessing_tophat_radius, use_gpu=self.config.use_gpu)
                    
                    ref_mask_p = cv2.resize(ref_mask.astype(np.uint8), (0,0), fx=proxy_scale, fy=proxy_scale, interpolation=cv2.INTER_NEAREST).astype(bool)
                    mov_mask_p = cv2.resize(mov_mask.astype(np.uint8), (0,0), fx=proxy_scale, fy=proxy_scale, interpolation=cv2.INTER_NEAREST).astype(bool)
                    
                    proxy_transform, rigid_ncc, method = self.rigid_registrar.register(ref_proxy, mov_proxy, ref_mask_p, mov_mask_p)
                    transform = proxy_transform.copy()
                    transform[0, 2] /= proxy_scale
                    transform[1, 2] /= proxy_scale
                else:
                    logger.info("   ⚡ Applying morphological filters to images...")
                    ref_prep = preprocess_dapi(ref_img, tophat_radius=self.config.preprocessing_tophat_radius, use_gpu=self.config.use_gpu)
                    mov_prep = preprocess_dapi(mov_img, tophat_radius=self.config.preprocessing_tophat_radius, use_gpu=self.config.use_gpu)
                    transform, rigid_ncc, method = self.rigid_registrar.register(ref_prep, mov_prep, ref_mask, mov_mask)
                log_hardware_usage("Post-Rigid Registration")
                logger.info(f"Rigid alignment complete (Method: {method}, NCC: {rigid_ncc:.4f})")
            
            # --- FIXED: UNIFORM CANVAS SIZES ---
            # Output everything to the exact shape of the reference image
            # so all rounds and channels perfectly stack on top of each other.
            h_ref, w_ref = ref_img.shape
            logger.info(f"   📐 Warping to reference canvas size: ({h_ref}, {w_ref})...")
            
            # Warp using tiled approach to handle large images (bypasses OpenCV limits)
            aligned_img = self._warp_tiled(mov_img, transform, (h_ref, w_ref))

            # Non-Rigid Registration (Optical flow)
            if self.config.enable_nonrigid and not skip_registration and is_anchor:
                logger.info("Running Non-Rigid Registration (Optical Flow)...")
                # For optical flow, also need to warp the mask
                mov_mask_w = self._warp_tiled(
                    mov_mask, transform, (h_ref, w_ref), is_mask=True
                )
                mov_mask_w = np.squeeze(mov_mask_w)
                mov_mask_w = (mov_mask_w > 0.5).astype(bool)  # binarize
                
                # Use the raw reference image directly
                ref_raw = np.squeeze(ref_img).astype(np.float32)
                ref_mask_sq = np.squeeze(ref_mask).astype(bool)
                aligned_img = np.squeeze(aligned_img).astype(np.float32)
                
                logger.debug(
                    f"Non-rigid shapes: ref={ref_raw.shape}, aligned={aligned_img.shape}, "
                    f"ref_mask={ref_mask_sq.shape}, mov_mask={mov_mask_w.shape}"
                )
                aligned_img, base_ncc, final_ncc, flow_steps = self.nonrigid_registrar.refine(
                    ref_raw, aligned_img, ref_mask_sq, mov_mask_w
                )
                logger.info(f"Non-rigid alignment complete (NCC: {base_ncc:.4f} -> {final_ncc:.4f})")
                log_hardware_usage("Post-Optical Flow")
            elif (
                self.config.enable_nonrigid
                and skip_registration
                and group_key != ref_group_key
                and flow_steps
            ):
                logger.info(
                    f"Applying {len(flow_steps)} cached DAPI optical-flow step(s) "
                    f"to channel: {mov_name}"
                )
                aligned_img = self.nonrigid_registrar.apply_flow_steps(
                    aligned_img,
                    flow_steps,
                )

            if not skip_registration:
                round_transform_cache[group_key] = {
                    "affine": transform,
                    "flow_steps": flow_steps,
                }

            # Output Generation
            logger.info("💾 Saving...")

            # Readers normalize to [0, 1]; interpolation can overshoot slightly.
            aligned_img = np.clip(aligned_img, 0.0, 1.0).astype(np.float32)

            save_tiff(aligned_img, str(out_dir / f"aligned_{mov_name}"))
            
            if not skip_registration:
                save_debug_overlay(ref_img, aligned_img, str(out_dir / f"qc_{mov_name}"))
                
            # Force Garbage Collection to prevent VRAM accumulation
            del mov_img, aligned_img
            if not skip_registration:
                del mov_mask
            import gc
            gc.collect()
            clear_gpu_memory()
            logger.info("🗑️  Cleaned up memory")
            
        logger.info("\n✅ COMPLETED pipeline")
