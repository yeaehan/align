import argparse
from pathlib import Path
import numpy as np
import tifffile

from align.io.lif import LifImageReader

def inspect_and_extract(lif_path: str, output_dir: str):
    lif_file = Path(lif_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Loading LIF file: {lif_file.name} ---")
    try:
        # Use the pipeline's native LIF reader class
        reader = LifImageReader(str(lif_file))
    except Exception as e:
        print(f"Error reading LIF file: {e}")
        return

    print(f"Total Scenes (Images) Found: {reader.num_scenes}\n")

    # Auto-detect the "Merged" scene
    merged_idx = reader.get_merged_scene_idx()
    scene_info = reader.get_scene_info(merged_idx)

    print(f"Targeting Scene {merged_idx + 1}/{reader.num_scenes}: '{scene_info['name']}'")
    dims = scene_info['dims']
    print(f"  Dimensions: X={dims[2]}, Y={dims[1]}, Z={dims[0]}, Channels={scene_info['channels']}")
    print(f"  -> Extracting all {scene_info['channels']} channels to float32 & applying MIP")

    # Loop through every channel in the target scene
    for c in range(scene_info['channels']):
        # Utilize lif.py's robust extraction and float32 normalization
        mip = reader.read_channel_as_float(channel_idx=c, scene_idx=merged_idx, use_mip=True)

        # Determine suffix based on whether it was a z-stack
        suffix = f"_ch{c:02d}_MIP" if dims[0] > 1 else f"_ch{c:02d}"

        # Clean up the scene name so it is safe to use as a file name
        safe_name = "".join([char if char.isalnum() or char in " _-" else "_" for char in scene_info['name']])
        out_file = out_dir / f"{lif_file.stem}_{safe_name}{suffix}.tif"

        # Convert back to uint16 for saving
        out_uint16 = (mip * 65535.0).astype(np.uint16)
        tifffile.imwrite(out_file, out_uint16)
        print(f"    -> Saved channel {c} to: {out_file.name}")
    print("\nExtraction complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment with LIF files for the alignment pipeline")
    parser.add_argument("--lif_file", required=True, help="Path to the .lif file")
    parser.add_argument("--out", default="./lif_experiments", help="Output directory for experimental TIFFs")
    args = parser.parse_args()
    
    inspect_and_extract(args.lif_file, args.out)
