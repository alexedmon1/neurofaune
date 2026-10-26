#!/usr/bin/env python3
"""
Batch single-voxel spectroscopy processing (Bruker PRESS -> FSL-MRS).

Kept for existing callers. The code is now part of the installed package,
neurofaune.preprocess.workflows.mrs_batch, so a study that pins neurofaune pins
it too; prefer the console command, which takes the same arguments:

    neurofaune mrs BRUKER_ROOT MRS_ROOT --config config.yaml [--derivatives ...] [--dry-run]

Run from a checkout, this wrapper imports neurofaune from that checkout (as it
always has), so the code that runs is the checkout's, not an installed pin's.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurofaune.preprocess.workflows.mrs_batch import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
