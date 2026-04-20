#!/usr/bin/env python3
"""
Jacquard Dataset Download Helper

Prints instructions for downloading and extracting the Jacquard dataset.
Direct automated download is not available because the dataset requires
manual registration at https://jacquard.liris.cnrs.fr/.

Usage::

    python scripts/download_dataset.py
"""

INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════════╗
║              Jacquard Dataset – Download Instructions                    ║
╚══════════════════════════════════════════════════════════════════════════╝

1. Register at https://jacquard.liris.cnrs.fr/ and download the archive.

2. Extract the archive so the layout looks like:

       jacquard/
           <category_01>/
               <scene_id>/
                   <scene_id>_grasps.txt
                   <scene_id>_RGB.png
                   <scene_id>_perfect_depth.tiff
           <category_02>/
               ...

3. Set the GGCNN2_DATASET_PATH environment variable (optional):

       export GGCNN2_DATASET_PATH=/path/to/jacquard   # Linux / macOS
       set GGCNN2_DATASET_PATH=C:\\path\\to\\jacquard  # Windows cmd

4. Or pass --dataset-path to the training script:

       python scripts/train.py --dataset-path /path/to/jacquard
"""

if __name__ == "__main__":
    print(INSTRUCTIONS)
