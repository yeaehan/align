from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from align.config import ZStackConfig
from align.core.tissue import TissueProcessor, normalize_to_uint8, resize_image
from align.core.transforms import warp_affine
from align.io.reader import (
    compute_max_projection,
    discover_moving_files,
    extract_reference_plane,
    find_processable_folders,
    find_reference_file,
    read_2d_as_float,
    read_zstack,
    replicate_to_zstack,
    select_anchor_file,
)
from align.io.writer import save_tiff, write_json
from align.registration.coarse import (
    build_coarse_proxy_images_from_orig,
    build_refine_registration_images_from_orig,
    masked_template_localization_topk,
)
from align.registration.refine import (
    evaluate_registration_candidate,
    full_transform_from_refine_registration,
)
from align.registration.rigid import RigidRegistrar

logger = logging.getLogger(__name__)


class ZStackAlignmentPipeline:
    def __init__(self, config: ZStackConfig):
        self.config = config
        self.registrar = RigidRegistrar(config.pyramid_levels, config.max_features)

    def _warp_affine_large_safe(
        self,
        img: np.ndarray,
        transform: np.ndarray,
        output_shape: tuple[int, int],
        tile_size: int = 16000,
    ) -> np.ndarray:
        h_out, w_out = output_shape
        h_src, w_src = img.shape
        max_cv_dim = 32766

        if (h_src < max_cv_dim and w_src < max_cv_dim and
            h_out < max_cv_dim and w_out < max_cv_dim):
            return warp_affine(img, transform, output_shape, is_mask=False)

        result = np.zeros(output_shape, dtype=np.float32)
        T = np.vstack([transform.astype(np.float32), [0, 0, 1]])
        try:
            T_inv = np.linalg.inv(T)[:2, :]
        except np.linalg.LinAlgError:
            logger.warning("Singular transform matrix during large-image warp")
            return result

        margin = 4
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
                    continue

                src_patch = img[src_y0:src_y1, src_x0:src_x1]
                if src_patch.shape[0] >= max_cv_dim or src_patch.shape[1] >= max_cv_dim:
                    logger.warning(
                        f"Skipping oversized source patch {src_patch.shape} for "
                        f"output tile [{out_y0}:{out_y1}, {out_x0}:{out_x1}]"
                    )
                    continue

                tile_transform = transform.copy().astype(np.float32)
                tile_transform[:, 2] += transform[:, :2] @ np.array([src_x0, src_y0], dtype=np.float32)
                tile_transform[0, 2] -= out_x0
                tile_transform[1, 2] -= out_y0

                result[out_y0:out_y1, out_x0:out_x1] = warp_affine(
                    src_patch,
                    tile_transform,
                    (tile_h, tile_w),
                    is_mask=False,
                )

        return result

    def _get_reference_2d_and_zcount(self):
        zstack, n_z_planes = read_zstack(self.config.reference_zstack_63x)
        if self.config.use_max_projection:
            ref_2d, ref_description = compute_max_projection(
                zstack, self.config.max_projection_range
            )
        else:
            ref_2d, z_index = extract_reference_plane(
                zstack, self.config.reference_z_plane
            )
            ref_description = f"Z_plane_{z_index}"
        return ref_2d.astype(np.float32), n_z_planes, ref_description

    def _save_debug_overlay(
        self,
        ref_full: np.ndarray,
        anchor_orig: np.ndarray,
        transform_orig_to_ref: np.ndarray,
        output_path: Path,
    ):
        try:
            max_dim = self.config.debug_max_dim
            debug_scale = min(max_dim / max(ref_full.shape), 1.0)
            debug_scale = float(max(debug_scale, 1e-4))

            ref_dbg = resize_image(ref_full, debug_scale, is_mask=False)
            anchor_dbg = resize_image(anchor_orig, debug_scale, is_mask=False)

            M_dbg = transform_orig_to_ref.copy().astype(np.float32)
            M_dbg[0, 2] *= debug_scale
            M_dbg[1, 2] *= debug_scale
            M_dbg[0, 0] *= debug_scale
            M_dbg[0, 1] *= debug_scale
            M_dbg[1, 0] *= debug_scale
            M_dbg[1, 1] *= debug_scale

            aligned_dbg = warp_affine(anchor_dbg, M_dbg, ref_dbg.shape, is_mask=False)

            ref_u8 = normalize_to_uint8(ref_dbg)
            mov_u8 = normalize_to_uint8(aligned_dbg)

            h, w = ref_u8.shape
            composite = np.zeros((h, w, 3), dtype=np.uint8)
            composite[:, :, 0] = mov_u8
            composite[:, :, 1] = ref_u8

            import tifffile

            tifffile.imwrite(str(output_path), composite, imagej=True)
            logger.info(f"Debug saved: {output_path.name}")
        except Exception as e:
            logger.warning(f"Could not save debug overlay: {e}")

    def _compute_overlap_crop_box(
        self,
        ref_full: np.ndarray,
        ref_mask_full: np.ndarray,
        anchor_orig: np.ndarray,
        transform_orig_to_ref: np.ndarray,
    ):
        if self.config.final_crop_mode == "full_reference":
            return 0, ref_full.shape[0], 0, ref_full.shape[1]

        h, w = anchor_orig.shape
        corners = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ], dtype=np.float32).reshape(-1, 1, 2)

        poly = cv2.transform(corners, transform_orig_to_ref).reshape(-1, 2)
        poly_i = np.round(poly).astype(np.int32)

        valid_ref = np.zeros(ref_full.shape, dtype=np.uint8)
        try:
            cv2.fillConvexPoly(valid_ref, poly_i, 1)
        except Exception:
            logger.warning("Polygon fill failed, falling back to full reference extent")
            return 0, ref_full.shape[0], 0, ref_full.shape[1]

        overlap = ref_mask_full.astype(bool) & valid_ref.astype(bool)
        if np.sum(overlap) < 100:
            logger.warning("Overlap too small, falling back to full reference extent")
            return 0, ref_full.shape[0], 0, ref_full.shape[1]

        ys, xs = np.where(overlap)
        return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1

    def run(self) -> Dict[str, Any]:
        start = time.time()
        results: Dict[str, Any] = {"status": "started", "saved_files": []}

        try:
            logger.info("=" * 80)
            logger.info("Z-STACK ALIGNMENT v7.1.1 - TEMPLATE FIRST ROI LARGE-IMAGE SAFE")
            logger.info("=" * 80)

            logger.info("Step 1: Loading reference z-stack...")
            ref_full, n_z_planes, ref_description = self._get_reference_2d_and_zcount()
            logger.info(f"Reference: {ref_description} | Shape: {ref_full.shape}")

            logger.info("Step 2: Creating reference tissue mask...")
            ref_mask_full = TissueProcessor.create_tissue_mask(
                ref_full, self.config.tissue_mask_percentile
            )

            logger.info("Step 3: Selecting anchor channel...")
            anchor_path = select_anchor_file(self.config.moving_images_20x)
            logger.info(f"Anchor file: {Path(anchor_path).name}")
            anchor_orig = read_2d_as_float(anchor_path, use_max_projection=True)

            logger.info("Step 4: Building coarse proxies directly from original anchor...")
            ref_proxy, mov_proxy, proxy_scale = build_coarse_proxy_images_from_orig(
                ref_full, anchor_orig, self.config
            )
            logger.info(f"Reference proxy shape: {ref_proxy.shape}")
            logger.info(f"Moving proxy shape: {mov_proxy.shape}")
            logger.info(f"Proxy scale: {proxy_scale:.6f}")

            ref_mask_proxy = TissueProcessor.create_tissue_mask(
                ref_proxy, self.config.tissue_mask_percentile
            )

            logger.info("Step 5: Template-first coarse localization (top-k)...")
            coarse_candidates = masked_template_localization_topk(
                ref_proxy,
                mov_proxy,
                ref_mask_proxy,
                roi_margin_factor=self.config.roi_margin_factor,
                top_k=self.config.coarse_top_k,
                min_peak_distance=self.config.coarse_min_peak_distance,
                min_score=self.config.coarse_min_score,
            )
            if len(coarse_candidates) == 0:
                raise RuntimeError("No coarse candidates passed the minimum score threshold.")

            logger.info("Step 6: Refining top-k coarse candidates...")
            candidate_objects = []
            candidate_summaries = []
            for coarse in coarse_candidates:
                logger.info(
                    f"Refining candidate {coarse['rank']}/{len(coarse_candidates)} "
                    f"(score={coarse['score']:.4f})..."
                )
                ref_reg, mov_reg, refine_meta = build_refine_registration_images_from_orig(
                    ref_full,
                    anchor_orig,
                    coarse["roi_bbox_mov_proxy"],
                    proxy_scale,
                    self.config,
                )
                ref_mask_reg = TissueProcessor.create_tissue_mask(
                    ref_reg, self.config.tissue_mask_percentile
                )
                mov_mask_reg = TissueProcessor.create_tissue_mask(
                    mov_reg, self.config.tissue_mask_percentile
                )
                M_refine, final_ncc, method = self.registrar.register(
                    ref_reg, mov_reg, ref_mask_reg, mov_mask_reg
                )
                eval_info = evaluate_registration_candidate(
                    ref_reg,
                    mov_reg,
                    ref_mask_reg,
                    mov_mask_reg,
                    M_refine,
                    final_ncc,
                    coarse["score"],
                    coarse["touches_border"],
                )
                M_orig_to_ref = full_transform_from_refine_registration(
                    M_refine, refine_meta, self.config.scale_factor
                )

                candidate_objects.append({
                    "candidate_index": int(coarse["rank"]),
                    "coarse": coarse,
                    "refine_meta": refine_meta,
                    "M_refine": M_refine,
                    "M_orig_to_ref": M_orig_to_ref,
                    "final_ncc": float(final_ncc),
                    "method": method,
                    "evaluation": eval_info,
                })
                candidate_summaries.append({
                    "candidate_index": int(coarse["rank"]),
                    "coarse_score": float(coarse["score"]),
                    "touches_border": bool(coarse["touches_border"]),
                    "template_bbox_ref_proxy": [int(v) for v in coarse["template_bbox_ref_proxy"]],
                    "hit_bbox_mov_proxy": [int(v) for v in coarse["hit_bbox_mov_proxy"]],
                    "roi_bbox_mov_proxy": [int(v) for v in coarse["roi_bbox_mov_proxy"]],
                    "refine_scale": float(refine_meta["refine_scale"]),
                    "mov_roi_orig_bbox": [int(v) for v in refine_meta["mov_roi_orig_bbox"]],
                    "registration_method": method,
                    "registration_ncc": float(final_ncc),
                    "selection_score": float(eval_info["selection_score"]),
                    "overlap_pixels": int(eval_info["overlap_pixels"]),
                    "overlap_ratio_ref": float(eval_info["overlap_ratio_ref"]),
                    "overlap_ratio_mov": float(eval_info["overlap_ratio_mov"]),
                    "determinant": float(eval_info["determinant"]),
                    "scale_x": float(eval_info["scale_x"]),
                    "scale_y": float(eval_info["scale_y"]),
                    "M_refine": M_refine.tolist(),
                    "M_orig_to_ref": M_orig_to_ref.tolist(),
                })

            best = max(candidate_objects, key=lambda x: x["evaluation"]["selection_score"])
            coarse = best["coarse"]
            refine_meta = best["refine_meta"]
            M_orig_to_ref = best["M_orig_to_ref"]
            final_ncc = best["final_ncc"]
            method = best["method"]

            results["coarse_candidates"] = candidate_summaries
            results["chosen_candidate_index"] = int(best["candidate_index"])
            results["coarse_localization"] = {
                "candidate_index": int(best["candidate_index"]),
                "coarse_score": float(coarse["score"]),
                "touches_border": bool(coarse["touches_border"]),
                "template_bbox_ref_proxy": [int(v) for v in coarse["template_bbox_ref_proxy"]],
                "hit_bbox_mov_proxy": [int(v) for v in coarse["hit_bbox_mov_proxy"]],
                "roi_bbox_mov_proxy": [int(v) for v in coarse["roi_bbox_mov_proxy"]],
            }

            logger.info("Step 8: Computing final crop in reference space...")
            y0, y1, x0, x1 = self._compute_overlap_crop_box(
                ref_full, ref_mask_full, anchor_orig, M_orig_to_ref
            )
            crop_h = y1 - y0
            crop_w = x1 - x0
            results["shared_crop_box"] = [int(y0), int(y1), int(x0), int(x1)]
            logger.info(f"Shared crop box: [{y0}:{y1}, {x0}:{x1}] -> ({crop_h}, {crop_w})")

            if self.config.save_debug:
                self._save_debug_overlay(
                    ref_full,
                    anchor_orig,
                    M_orig_to_ref,
                    Path(self.config.output_folder) / "debug_ch00_alignment.tif",
                )

            logger.info("Step 9: Warping/saving all channels in reference space...")
            M_crop = M_orig_to_ref.copy().astype(np.float32)
            M_crop[0, 2] -= x0
            M_crop[1, 2] -= y0

            for i, moving_path in enumerate(self.config.moving_images_20x, start=1):
                channel_name = Path(moving_path).stem
                logger.info(f"Channel {i}/{len(self.config.moving_images_20x)}: {channel_name}")

                mov_orig = read_2d_as_float(moving_path, use_max_projection=True)
                mov_cropped = self._warp_affine_large_safe(mov_orig, M_crop, (crop_h, crop_w))
                channel_zstack = replicate_to_zstack(mov_cropped, n_z_planes)
                output_path = Path(self.config.output_folder) / f"{channel_name}_Crop_aligned2dot3d.tif"
                save_tiff(
                    channel_zstack,
                    str(output_path),
                    voxel_spacing_z=self.config.voxel_size_63x[2],
                    pixel_size_xy=self.config.voxel_size_63x[0],
                    axes="ZYX",
                )
                logger.info(f"Saved: {output_path.name}")
                results["saved_files"].append(str(output_path))

                del mov_orig, mov_cropped, channel_zstack
                gc.collect()

            elapsed = time.time() - start
            results["status"] = "completed"
            results["elapsed_seconds"] = round(elapsed, 2)
            results["registration_ncc"] = float(final_ncc)
            results["registration_method"] = method
            results["reference_shape"] = [int(ref_full.shape[0]), int(ref_full.shape[1])]
            results["crop_shape"] = [int(crop_h), int(crop_w)]
            results["coarse_localization"] = coarse
            results["refine_meta"] = refine_meta

            logger.info(f"COMPLETED in {elapsed:.1f}s")
            return results

        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}


def process_one_folder(folder: Path) -> Dict[str, Any]:
    ref_file, ref_issue = find_reference_file(folder)
    if ref_file is None:
        result = {"status": "skipped", "folder": str(folder), "reason": ref_issue}
        logger.warning(f"Skipping {folder}: {ref_issue}")
        return result

    moving_images = discover_moving_files(ref_file)
    if len(moving_images) == 0:
        result = {
            "status": "skipped",
            "folder": str(folder),
            "reference": str(ref_file),
            "reason": "no moving images",
        }
        logger.warning(f"Skipping {folder}: no moving images found")
        return result

    output_folder = folder / "Done"
    config = ZStackConfig(
        reference_zstack_63x=str(ref_file),
        moving_images_20x=moving_images,
        output_folder=str(output_folder),
        use_max_projection=True,
        pyramid_levels=[0.25, 0.5],
        coarse_max_dim=4096,
        refine_max_dim=4096,
        roi_margin_factor=1.4,
        final_crop_mode="overlap",
        save_debug=True,
        debug_max_dim=4000,
    )

    results = ZStackAlignmentPipeline(config).run()
    folder_summary = {
        "folder": str(folder),
        "reference": str(ref_file),
        "moving_images": moving_images,
        "result": results,
    }
    write_json(output_folder / "batch_results.json", folder_summary)
    results["folder"] = str(folder)
    results["reference"] = str(ref_file)
    results["moving_images"] = moving_images
    return results


def run_batch(root_folder: str):
    cv2.setUseOptimized(True)
    cv2.setNumThreads(8)

    root_path = Path(root_folder)
    folders = find_processable_folders(root_folder)
    logger.info(f"Scanning {len(folders)} folder(s) under: {root_folder}")

    all_results = []
    for folder in folders:
        try:
            result = process_one_folder(folder)
        except Exception as e:
            logger.exception(f"Failed folder: {folder}")
            result = {"status": "failed", "folder": str(folder), "error": str(e)}
        all_results.append(result)

    batch_summary = {
        "root_folder": str(root_path),
        "completed": sum(r.get("status") == "completed" for r in all_results),
        "skipped": sum(r.get("status") == "skipped" for r in all_results),
        "failed": sum(r.get("status") == "failed" for r in all_results),
        "results": all_results,
    }
    write_json(root_path / "batch_results.json", batch_summary)
    return all_results


def main(root_folder: str):
    return run_batch(root_folder)
