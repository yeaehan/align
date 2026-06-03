# Multi-Round Image Alignment Pipeline

This is a robust and highly optimized workspace designed for large-scale, multiplexed image alignment (specifically tailored for techniques like 4i - Iterative Indirect Immunofluorescence Imaging).

## Setup Environment

The project is now packaged using `pyproject.toml`, which makes installation very easy and creates a global command-line shortcut for you.

Open your terminal, navigate to the `align` directory, and run:
Open your terminal, navigate to the root of the workspace, and run:

```bash
cd /Users/hanyeaeun/Desktop/workspace/align
cd /Users/hanyeaeun/Desktop/workspace/
pip install -e .
```
*(Note: If you do not have a GPU, the pipeline will fall back to CPU via scipy gracefully).*

If you are on a machine with an NVIDIA GPU and want maximum performance, install the optional GPU dependencies instead:
```bash
pip install -e .[gpu]
```

## Running the Pipeline (Command Line)

Installing the package exposes a new terminal command: `align-pipeline`.

### 1. Single Sample Alignment (2D)
Provide the input folder containing the TIFFs, an output destination, and the channel to use as the reference (default is `ch00`).

```bash
align-pipeline \
    --input_folder ../data/raw_sample_01 \
    --output_folder ../data/aligned_sample_01 \
    --input_folder ./data/raw_sample_01 \
    --output_folder ./data/aligned_sample_01 \
    --ref ch00
```

### 2. Batch Processing (2D or 3D)
Use the `--batch` flag to auto-detect multiple samples in a directory. It will group files that share the same prefix (separated by `_`) or process nested subfolders automatically.

```bash
align-pipeline \
    --input_folder ../data/all_raw_samples \
    --output_folder ../data/all_aligned_outputs \
    --batch
```

### Optional Flags

- `--3d`: Run the 3D Z-stack alignment logic instead of the 2D multi-round logic.
- `--n_workers 8`: Increase thread count for SIFT feature detection.
- `--no_gpu`: Force CPU processing (useful for testing on laptops).
- `--disable_nonrigid`: Skip optical flow deformation corrections (much faster, but only applies affine transformations).

### Example CPU-Only Test

```bash
align-pipeline \
    --input_folder ../data/raw \
    --output_folder ../data/aligned \
    --input_folder ./data/raw \
    --output_folder ./data/aligned \
    --batch \
    --no_gpu \
    --disable_nonrigid
```