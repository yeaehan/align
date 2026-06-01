#!/usr/bin/env python
"""
Entry point for the alignment pipeline.
Run from the align directory with:
    python run_alignment.py --input_folder ... --output_folder ...
"""

import sys
from pathlib import Path

# Add the parent directory to Python path so 'align' package can be found
sys.path.insert(0, str(Path(__file__).parent.parent))

from align.core.run_pipeline import main

if __name__ == "__main__":
    main()
