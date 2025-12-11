#!/usr/bin/env python3
"""Direct test of Atropos to debug output issues."""

from pathlib import Path
from nipype.interfaces.ants import Atropos
import os

# Paths
work_dir = Path('/mnt/arborea/bpa-rat/test/work/sub-Rat207/ses-p60/anat_preproc')
input_file = work_dir / 'sub-Rat207_ses-p60_T2w_n4.nii.gz'
mask_file = work_dir / 'SIGMA_brain_mask.nii.gz'

print(f"Input file: {input_file}")
print(f"Input exists: {input_file.exists()}")
print(f"Mask file: {mask_file}")
print(f"Mask exists: {mask_file.exists()}")
print(f"Working directory: {work_dir}")
print()

# Change to work directory
os.chdir(work_dir)
print(f"Changed to: {os.getcwd()}")
print()

# Setup Atropos
atropos = Atropos()
atropos.inputs.dimension = 3
atropos.inputs.intensity_images = [str(input_file)]
atropos.inputs.mask_image = str(mask_file)
atropos.inputs.number_of_tissue_classes = 5
atropos.inputs.n_iterations = 5
atropos.inputs.convergence_threshold = 0.0
atropos.inputs.mrf_smoothing_factor = 0.1
atropos.inputs.mrf_radius = [1, 1, 1]
atropos.inputs.initialization = 'KMeans'
atropos.inputs.save_posteriors = True

print("Atropos configuration:")
print(f"  intensity_images: {atropos.inputs.intensity_images}")
print(f"  mask_image: {atropos.inputs.mask_image}")
print(f"  number_of_tissue_classes: {atropos.inputs.number_of_tissue_classes}")
print(f"  n_iterations: {atropos.inputs.n_iterations}")
print(f"  initialization: {atropos.inputs.initialization}")
print()

# Generate command line
cmdline = atropos.cmdline
print("Command line:")
print(cmdline)
print()

# Run it
print("Running Atropos...")
try:
    result = atropos.run()
    print("Success!")
    print(f"Outputs: {result.outputs}")
    print(f"  classified_image: {result.outputs.classified_image}")
    print(f"  posteriors: {result.outputs.posteriors}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
