import argparse
from pathlib import Path
import numpy as np
import tifffile
from readlif.reader import LifFile

def convert_lif_to_tiff(lif_path: str, output_dir: str):
    lif_file = Path(lif_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading LIF file: {lif_file.name}")
    try:
        reader = LifFile(str(lif_file))
    except Exception as e:
        print(f"Error reading LIF file: {e}")
        return

    print(f"Found {reader.num_images} image(s) in the LIF file.")
    
    for i in range(reader.num_images):
        img = reader.get_image(i)
        print(f"\nProcessing Image {i+1}/{reader.num_images}: '{img.name}'")
        print(f"  Dimensions: X={img.dims.x}, Y={img.dims.y}, Z={img.dims.z}, C={img.channels}, T={img.dims.t}")
        
        # Create an empty numpy array: shape (C, Z, Y, X)
        # (Assuming T=1 for standard multi-channel z-stacks or 2D images)
        shape = (img.channels, img.dims.z, img.dims.y, img.dims.x)
        
        # LIF pixels are typically 8-bit or 16-bit. We use uint16 to be safe.
        data = np.zeros(shape, dtype=np.uint16) 
        
        # readlif extracts images plane by plane
        for c in range(img.channels):
            for z in range(img.dims.z):
                # get_plane returns a PIL image object, so we cast it to a numpy array
                plane = img.get_plane(z=z, c=c, t=0)
                data[c, z, :, :] = np.array(plane)
                
        # Squeeze out dimensions of size 1 (e.g., if Z=1, the array becomes CYX instead of CZYX)
        data = np.squeeze(data)
        
        # Clean up the scene name so it is safe for the filesystem
        safe_name = "".join([char if char.isalnum() or char in " _-" else "_" for char in img.name])
        out_path = out_dir / f"{lif_file.stem}_{safe_name}.tif"
        
        # Save using tifffile with ImageJ metadata so ImageJ/Fiji reads the channels properly
        print(f"  Saving to: {out_path}")
        
        # Best-effort guess at the axes for ImageJ
        axes_str = 'CZYX'[-data.ndim:] 
        tifffile.imwrite(out_path, data, imagej=True, metadata={'axes': axes_str})
        
    print("\nConversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone LIF to TIFF converter")
    parser.add_argument("lif_file", help="Path to the .lif file")
    parser.add_argument("--out", default="./lif_converted", help="Output directory for TIFFs")
    args = parser.parse_args()
    
    convert_lif_to_tiff(args.lif_file, args.out)