import argparse
from pathlib import Path
import numpy as np
import tifffile

from align.io.lif import LifImageReader

def inspect_and_extract(lif_path: str, output_dir: str, z_plane: int = None):
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
    
    use_mip = (z_plane is None)
    if use_mip:
        print(f"  -> Extracting all {scene_info['channels']} channels to float32 & applying MIP")
    else:
        print(f"  -> Extracting all {scene_info['channels']} channels to float32 at Z-plane {z_plane}")

    # Loop through every channel in the target scene
    for c in range(scene_info['channels']):
        if use_mip or dims[0] == 1:
            extracted = reader.read_channel_as_float(channel_idx=c, scene_idx=merged_idx, use_mip=True)
            suffix = f"_ch{c:02d}_MIP" if dims[0] > 1 else f"_ch{c:02d}"
        else:
            # Extract specific Z-plane from the raw LIF data
            img = reader.lif.get_image(scene_info['scene_idx'])
            valid_z = max(0, min(z_plane, img.dims.z - 1))
            plane = np.array(img.get_plane(z=valid_z, c=c, t=0))
            
            if plane.dtype == np.uint16:
                extracted = plane.astype(np.float32) / 65535.0
            elif plane.dtype == np.uint8:
                extracted = plane.astype(np.float32) / 255.0
            else:
                extracted = plane.astype(np.float32)
                
            suffix = f"_ch{c:02d}_z{valid_z}"

        # Clean up the scene name so it is safe to use as a file name
        safe_name = "".join([char if char.isalnum() or char in " _-" else "_" for char in scene_info['name']])
        out_file = out_dir / f"{lif_file.stem}_{safe_name}{suffix}.tif"

        # Convert back to uint16 for saving
        out_uint16 = (extracted * 65535.0).astype(np.uint16)
        tifffile.imwrite(out_file, out_uint16)
        print(f"    -> Saved channel {c} to: {out_file.name}")
    print("\nExtraction complete!")

def process_path(input_path: str, output_dir: str, z_plane: int = None):
    p = Path(input_path)
    if p.is_dir():
        lif_files = list(p.glob("*.lif"))
        if not lif_files:
            print(f"No .lif files found in directory: {p}")
            return
        print(f"Found {len(lif_files)} LIF files in directory. Processing all...")
        for lif_file in lif_files:
            inspect_and_extract(str(lif_file), output_dir, z_plane)
    else:
        inspect_and_extract(input_path, output_dir, z_plane)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment with LIF files for the alignment pipeline")
    parser.add_argument("--lif_path", required=True, help="Path to the .lif file or a directory containing .lif files")
    parser.add_argument("--out", default="./lif_experiments", help="Output directory for experimental TIFFs")
    parser.add_argument("--z_plane", type=int, default=None, help="Specific Z-plane to extract (0-indexed). Defaults to Maximum Intensity Projection (MIP).")
    args = parser.parse_args()
    
    process_path(args.lif_path, args.out, args.z_plane)
