"""Functional network layer for the mouse hub: RABIES group-ICA priors in Antwerp space.

RABIES (Desrosiers-Gregoire et al. 2024, Nat Commun) ships 30 group-ICA
components (``melodic_IC.nii.gz``, Zenodo 10.5281/zenodo.5118030, CC-BY-4.0) on
its EPI template, used as dual-regression priors. Its defaults mark
components 5, 12 and 19 (0-based) as BOLD networks and 16 others as
confounds; the remainder are unassigned.

The template is EPI contrast and covers the cerebrum but truncates the
olfactory bulb and caudal cerebellum, so registration onto the Antwerp T2w
uses mutual information throughout and masks on both sides. Components are
then described by where they sit in our own labels -- allen26 / allen32 and
the skull frame -- so each network is named against the parcels the EEG uses
rather than by its index alone.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np

from neurofaune.atlas.label_transfer import apply_transforms, forward_transforms, register_templates

#: RABIES default ``--prior_bold_idx`` / ``--prior_confound_idx`` (rabies/parser.py).
RABIES_BOLD_IDX = (5, 12, 19)
RABIES_CONFOUND_IDX = (0, 2, 6, 7, 8, 9, 10, 11, 13, 14, 21, 22, 24, 26, 28, 29)


def rabies_role(ic: int) -> str:
    if ic in RABIES_BOLD_IDX:
        return "bold"
    if ic in RABIES_CONFOUND_IDX:
        return "confound"
    return "unassigned"


def transfer_rabies_priors(rabies_dir: Path, hub_dir: Path, work_dir: Path,
                           n_threads: int = 4) -> Dict[str, Any]:
    """Register the RABIES EPI template to Antwerp and warp its IC maps onto both hub grids."""
    rabies_dir, hub_dir, work_dir = Path(rabies_dir), Path(hub_dir), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = hub_dir / "functional"
    out.mkdir(parents=True, exist_ok=True)

    tpl = nib.load(str(rabies_dir / "EPI_template.nii.gz"))
    mask = nib.load(str(rabies_dir / "EPI_brain_mask.nii.gz"))
    m = mask.get_fdata() > 0
    moving = work_dir / "rabies_template_brain.nii.gz"
    nib.save(nib.Nifti1Image((tpl.get_fdata() * m).astype(np.float32), tpl.affine), str(moving))
    moving_mask = work_dir / "rabies_brain_mask.nii.gz"
    nib.save(nib.Nifti1Image(m.astype(np.uint8), tpl.affine), str(moving_mask))

    reg = register_templates(moving, hub_dir / "brain.nii.gz", work_dir / "rabies_to_antwerp_",
                             fixed_mask=hub_dir / "brain_mask.nii.gz", moving_mask=moving_mask,
                             n_threads=n_threads, syn_metric="MI", syn_radius=32)
    tf = forward_transforms(reg)
    tdir = hub_dir / "transforms"
    import shutil
    for key in ("affine", "warp", "inverse_warp"):
        shutil.copy2(reg[key], tdir / f"rabies_to_antwerp_{reg[key].name.split('_')[-1]}")

    fixed = nib.load(str(hub_dir / "brain.nii.gz")).get_fdata()
    fm = nib.load(str(hub_dir / "brain_mask.nii.gz")).get_fdata() > 0
    warped = nib.load(str(reg["warped"])).get_fdata()
    paths = {}
    for grid, ref in (("native", hub_dir / "brain.nii.gz"), ("hub100um", hub_dir / "hub100um" / "brain_mask.nii.gz")):
        sub = out if grid == "native" else hub_dir / "hub100um" / "functional"
        paths[f"ic_{grid}"] = apply_transforms(rabies_dir / "melodic_IC.nii.gz", ref, tf,
                                               sub / "rabies_melodic_IC.nii.gz", "Linear", n_threads, time_series=True)
        paths[f"coverage_{grid}"] = apply_transforms(moving_mask, ref, tf, sub / "rabies_coverage_mask.nii.gz",
                                                     "NearestNeighbor", n_threads)
    cover = nib.load(str(paths["coverage_native"])).get_fdata() > 0.5
    both = fm & cover & (warped != 0)
    return {
        "registration": {"moving": "RABIES EPI_template (masked)", "fixed": "Antwerp brain",
                         "metric": "MI rigid/affine/SyN, masks on both images",
                         "t2w_vs_warped_epi_correlation_in_overlap": float(np.corrcoef(fixed[both], warped[both])[0, 1]),
                         "antwerp_brain_covered_fraction": float((fm & cover).sum() / fm.sum())},
        "paths": {k: str(v) for k, v in paths.items()},
    }


def describe_components(ic_img: nib.Nifti1Image, parcels: nib.Nifti1Image, names: Dict[int, str],
                        coverage: Optional[np.ndarray] = None, stereo: Optional[Dict[str, np.ndarray]] = None,
                        z_thresh: float = 3.0, top: int = 3) -> List[Dict[str, Any]]:
    """Where each IC lives: top parcels by share of its suprathreshold volume, peak, symmetry.

    ``stereo`` = {"ml": ..., "ap": ..., "dv": ...} skull-frame maps on the same grid.
    Hemisphere symmetry is the correlation between the map and its left-right
    mirror about the parcels' centre of mass (1 = bilateral network).
    """
    data = ic_img.get_fdata()
    lab = np.asanyarray(parcels.dataobj)
    cov = np.ones(lab.shape, bool) if coverage is None else coverage
    brain = (lab > 0) & cov
    mid = int(round(np.argwhere(lab > 0)[:, 0].mean()))
    rows = []
    for ic in range(data.shape[3]):
        z = np.where(cov, data[..., ic], 0.0)
        supra = (z > z_thresh) & brain
        row: Dict[str, Any] = {"ic": ic, "rabies_role": rabies_role(ic),
                               "supra_voxels": int(supra.sum()), "max_z": float(z.max())}
        if supra.any():
            ids, counts = np.unique(lab[supra], return_counts=True)
            order = np.argsort(-counts)[:top]
            row["top_parcels"] = "; ".join(f"{names.get(int(ids[i]), ids[i])} ({counts[i] / supra.sum():.0%})"
                                           for i in order)
            peak = np.unravel_index(np.argmax(np.where(brain, z, -np.inf)), z.shape)
            row["peak_parcel"] = names.get(int(lab[peak]), str(lab[peak]))
            if stereo is not None:
                row.update({f"peak_{k}_mm": float(v[peak]) for k, v in stereo.items()})
        # symmetry about the midline column
        half = min(mid, lab.shape[0] - mid - 1)
        left = z[mid - half:mid]
        right = z[mid + 1:mid + 1 + half][::-1]
        sel = brain[mid - half:mid] & brain[mid + 1:mid + 1 + half][::-1]
        row["lr_symmetry_r"] = float(np.corrcoef(left[sel], right[sel])[0, 1]) if sel.sum() > 10 else None
        rows.append(row)
    return rows


def write_table(rows: Sequence[Dict[str, Any]], path: Path) -> Path:
    cols: List[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    return Path(path)
