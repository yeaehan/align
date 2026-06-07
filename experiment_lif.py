import argparse
from pathlib import Path
import numpy as np
import tifffile
from readlif.reader import LifFile

def inspect_and_extract(lif_path: str, output_dir: str):
    lif_file = Path(lif_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Loading LIF file: {lif_file.name} ---")
    try:
        reader = LifFile(str(lif_file))
    except Exception as e:
        print(f"Error reading LIF file: {e}")
        return
        
    print(f"Total Scenes (Images) Found: {reader.num_images}\n")
    
    # Auto-detect the "Merged" scene
    merged_idx = 0
    for i in range(reader.num_images):
        if "merged" in reader.get_image(i).name.lower():
            merged_idx = i
            break
            
    img = reader.get_image(merged_idx)
    print(f"Targeting Scene {merged_idx+1}/{reader.num_images}: '{img.name}'")
    print(f"  Dimensions: X={img.dims.x}, Y={img.dims.y}, Z={img.dims.z}, Channels={img.channels}")
    
    print(f"  -> Extracting all {img.channels} channels to float32 & applying MIP")
    
    # Loop through every channel in the target scene
    for c in range(img.channels):
        z_stack = np.zeros((img.dims.z, img.dims.y, img.dims.x), dtype=np.float32)
        
        for z in range(img.dims.z):
            # readlif returns a PIL image, convert it to numpy
            plane = np.array(img.get_plane(z=z, c=c, t=0))
            
            # Normalize to float32 [0, 1] exactly like align.io.reader does
            if plane.dtype == np.uint16:
                plane = plane.astype(np.float32) / 65535.0
            elif plane.dtype == np.uint8:
                plane = plane.astype(np.float32) / 255.0
                
            z_stack[z, :, :] = plane
            
        # Perform Max Intensity Projection if there are multiple Z planes
        if img.dims.z > 1:
            mip = np.max(z_stack, axis=0)
            suffix = f"_ch{c:02d}_MIP"
        else:
            mip = z_stack[0]
            suffix = f"_ch{c:02d}"
            
        # Clean up the scene name so it is safe to use as a file name
        safe_name = "".join([char if char.isalnum() or char in " _-" else "_" for char in img.name])
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
