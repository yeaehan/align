from __future__ import annotations

import numpy as np


def compute_weighted_ncc(ref: np.ndarray, mov: np.ndarray, weight: np.ndarray) -> float:
    w_sum = np.sum(weight)
    if w_sum < 100:
        return 0.0

    ref_mean = np.sum(ref * weight) / w_sum
    mov_mean = np.sum(mov * weight) / w_sum
    ref_centered = ref - ref_mean
    mov_centered = mov - mov_mean

    numerator = np.sum(weight * ref_centered * mov_centered)
    ref_var = np.sum(weight * ref_centered**2)
    mov_var = np.sum(weight * mov_centered**2)
    if ref_var == 0 or mov_var == 0:
        return 0.0

    ncc = numerator / np.sqrt(ref_var * mov_var)
    return max(0.0, min(1.0, float(ncc)))


_compute_weighted_ncc = compute_weighted_ncc
