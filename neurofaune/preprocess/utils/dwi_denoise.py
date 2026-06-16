"""Pre-eddy DWI denoising and Gibbs-ringing removal (dipy-based).

Standard modern DWI pre-processing applied to the raw 4-D series *before* eddy /
motion correction, in this order:

  1. MP-PCA denoising (Marchenko-Pastur PCA, ``dipy.denoise.localpca.mppca``) —
     removes thermal noise while preserving signal; especially valuable for the
     high-b shells feeding DKI / NODDI.
  2. Gibbs-ringing removal (``dipy.denoise.gibbs.gibbs_removal``) — suppresses the
     truncation ripples near sharp boundaries.

Both operate on the 4-D array directly (no gradient table needed) and are pure
in/out NIfTI transforms, so they slot in ahead of intensity normalisation,
masking and eddy. dipy is already a dependency (used for tensor / DKI fitting),
so no external tools (MRtrix) are required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np


def denoise_dwi_mppca(
    in_file: Path,
    out_file: Path,
    patch_radius: int = 2,
    mask_file: Optional[Path] = None,
) -> Path:
    """MP-PCA denoise a 4-D DWI series. Returns ``out_file``."""
    from dipy.denoise.localpca import mppca

    img = nib.load(str(in_file))
    data = np.asarray(img.dataobj, dtype=np.float32)
    mask = None
    if mask_file is not None:
        mask = np.asarray(nib.load(str(mask_file)).dataobj) > 0
    den = mppca(data, mask=mask, patch_radius=patch_radius)
    den = np.asarray(den, dtype=np.float32)
    np.clip(den, 0, None, out=den)  # denoising can produce small negatives
    nib.save(nib.Nifti1Image(den, img.affine, img.header), str(out_file))
    return Path(out_file)


def degibbs_dwi(
    in_file: Path,
    out_file: Path,
    num_processes: int = 1,
) -> Path:
    """Remove Gibbs ringing from a 3-D/4-D image (slices along axis 2). Returns ``out_file``."""
    from dipy.denoise.gibbs import gibbs_removal

    img = nib.load(str(in_file))
    data = np.asarray(img.dataobj, dtype=np.float32)
    dg = gibbs_removal(data, slice_axis=2, inplace=False, num_processes=num_processes)
    dg = np.asarray(dg, dtype=np.float32)
    np.clip(dg, 0, None, out=dg)
    nib.save(nib.Nifti1Image(dg, img.affine, img.header), str(out_file))
    return Path(out_file)
