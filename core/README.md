# Multi-Round Image Alignment Pipeline

This is a robust and highly optimized workspace designed for large-scale, multiplexed image alignment (specifically tailored for techniques like 4i - Iterative Indirect Immunofluorescence Imaging).

## Setup Environment

Ensure you have the required dependencies installed. You can install them via pip:

```bash
pip install numpy scipy opencv-python-headless scikit-image tifffile cupy-cuda11x
```
*(Note: If you do not have a GPU, the pipeline will fall back to CPU via scipy gracefully. Omit `cupy-cuda11x` if running CPU-only).*

## Running 2D Alignment (Command Line)

You can execute the 2D multi-round alignment pipeline using the `run_pipeline.py` script.

### Usage

Navigate to the root of your workspace:

```bash
cd /Users/hanyeaeun/Desktop/workspace/
```

Run the alignment script by providing the input folder containing the TIFFs, an output destination, and the specific reference image (usually your Round 1 DAPI channel).

```bash
python align/core/run_pipeline.py \
    --input_folder ./data/raw_sample_01 \
    --output_folder ./data/aligned_sample_01 \
    --reference_file ./data/raw_sample_01/R1_DAPI.tif
```

### Optional Flags

- `--n_workers 8`: Increase thread count for SIFT feature detection.
- `--no_gpu`: Force CPU processing (useful for testing on laptops).
- `--disable_nonrigid`: Skip optical flow deformation corrections (much faster, but only applies affine transformations).

### Example CPU-Only Test

```bash
python align/core/run_pipeline.py \
    --input_folder ./data/raw \
    --output_folder ./data/aligned \
    --reference_file ./data/raw/ref_dapi.tif \
    --no_gpu
```