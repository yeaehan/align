import argparse
from pathlib import Path
import numpy as np
import tifffile

from align.io.lif import LifImageReader

def inspect_and_extract(lif_path: str, output_dir: str, extract_3d: bool = False):
    lif_file = Path(lif_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Loading LIF file: {lif_file.name} ---")
    try:
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

    if extract_3d and dims[0] > 1:
        print(f"  -> Extracting all {scene_info['channels']} channels as full 3D Z-stacks")
    else:
        print(f"  -> Extracting all {scene_info['channels']} channels to float32 & applying MIP")

    # Loop through every channel in the target scene
    for c in range(scene_info['channels']):
        # If it's a Z-stack AND the user wants the full 3D volume
        if extract_3d and dims[0] > 1:
            img = reader.lif.get_image(scene_info['scene_idx'])
            z_stack = np.zeros((img.dims.z, img.dims.y, img.dims.x), dtype=np.float32)
            for z in range(img.dims.z):
                plane = np.array(img.get_frame(z=z, c=c, t=0))
                if plane.dtype == np.uint16:
                    z_stack[z] = plane.astype(np.float32) / 65535.0
                elif plane.dtype == np.uint8:
                    z_stack[z] = plane.astype(np.float32) / 255.0
                else:
                    z_stack[z] = plane.astype(np.float32)
            extracted = z_stack
            suffix = f"_ch{c:02d}"
        # Otherwise (default behavior), get the MIP for any Z-stack, or the 2D plane if it's already 2D
        else:
            extracted = reader.read_channel_as_float(channel_idx=c, scene_idx=merged_idx, use_mip=True)
            suffix = f"_ch{c:02d}_MIP" if dims[0] > 1 else f"_ch{c:02d}"

        # Clean up the scene name so it is safe to use as a file name
        safe_name = "".join([char if char.isalnum() or char in " _-" else "_" for char in scene_info['name']])
        out_file = out_dir / f"{lif_file.stem}_{safe_name}{suffix}.tif"

        # Convert back to uint16 for saving
        tifffile.imwrite(out_file, (extracted * 65535.0).astype(np.uint16))
        print(f"    -> Saved channel {c} to: {out_file.name}")
    print("\nExtraction complete!")

def process_path(input_path: str, output_dir: str, extract_3d: bool = False):
    p = Path(input_path)
    if p.is_dir():
        lif_files = list(p.glob("*.lif"))
        if not lif_files:
            print(f"No .lif files found in directory: {p}")
            return
        print(f"Found {len(lif_files)} LIF files in directory. Processing all...")
        for lif_file in lif_files:
            inspect_and_extract(str(lif_file), output_dir, extract_3d)
    else:
        inspect_and_extract(input_path, output_dir, extract_3d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment with LIF files for the alignment pipeline")
    parser.add_argument("--lif_path", required=True, help="Path to a .lif file or a directory containing .lif files")
    parser.add_argument("--out", default="./lif_experiments", help="Output directory for experimental TIFFs")
    parser.add_argument("--3d", dest="extract_3d", action="store_true", help="Extract the full 3D Z-stack instead of the default 2D Maximum Intensity Projection (MIP).")
    args = parser.parse_args()
    
    process_path(args.lif_path, args.out, args.extract_3d)
