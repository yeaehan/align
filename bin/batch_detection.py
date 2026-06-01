from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional


def is_3d_file(filename: str) -> bool:
    return "3d" in Path(filename).stem.lower()


def get_sample_name(filename: str) -> str:
    return Path(filename).stem.split("_")[0]


def detect_batches(file_list: Iterable[str]) -> Dict[str, List[str]]:
    batches: Dict[str, List[str]] = {}
    for filename in file_list:
        sample = get_sample_name(filename)
        batches.setdefault(sample, []).append(filename)
    return batches


def get_location_tag(filename: str) -> Optional[str]:
    parts = Path(filename).stem.split("_")
    if len(parts) > 2 and parts[-1].lower().startswith("location"):
        return parts[-1]
    return None


def find_corresponding_2d(sample_name: str, markers: str, all_files: Iterable[str]) -> Optional[str]:
    prefix = f"{sample_name}_{markers}"
    for filename in all_files:
        stem = Path(filename).stem
        if stem.startswith(prefix) and "3d" not in stem.lower() and "location" not in stem.lower():
            return filename
    return None
