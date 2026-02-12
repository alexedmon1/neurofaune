"""
Automated Bruker session processing.

Inventories a raw Bruker session directory, identifies all scan types that
have a preprocessing pipeline (T2w, DTI, fMRI, MSME), converts to NIfTI,
and runs each applicable pipeline.

Usage (programmatic)::

    from neurofaune.preprocess.workflows.bruker_session import process_bruker_session

    results = process_bruker_session(
        session_dir=Path('/path/to/bruker_session'),
        output_dir=Path('/path/to/study'),
        subject='sub-PRE10',
        session='ses-01',
    )

See also ``scripts/process_bruker_session.py`` for the CLI wrapper.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np

from neurofaune.utils.bruker_convert import (
    convert_bruker_to_nifti,
    extract_bids_metadata,
    extract_bvec_bval,
    inventory_session,
    select_best_dwi_from_inventory,
    select_best_func_from_inventory,
    select_best_msme_from_inventory,
    select_best_t2w_from_inventory,
)

logger = logging.getLogger(__name__)


def _build_minimal_config() -> Dict[str, Any]:
    """Build a minimal config dict without atlas section.

    Suitable for data where no SIGMA atlas is available (e.g. mouse data,
    quick processing without atlas registration).
    """
    return {
        "anatomical": {
            "bet": {"frac": 0.3},
        },
        "diffusion": {
            "intensity_normalization": {
                "enabled": True,
                "target_max": 10000.0,
            },
            "eddy": {
                "slice_padding": 2,
            },
        },
    }


def _convert_scan(
    session_dir: Path,
    scan_num: str,
    output_nii: Path,
    modality: str,
    force: bool,
) -> Path:
    """Convert a single Bruker scan to NIfTI with JSON sidecar."""
    scan_dir = session_dir / scan_num

    if not force and output_nii.exists():
        print(f"  NIfTI already exists: {output_nii.name}")
    else:
        print(f"  Converting scan {scan_num} ({modality})...")
        output_nii.parent.mkdir(parents=True, exist_ok=True)
        if not convert_bruker_to_nifti(scan_dir, output_nii):
            raise RuntimeError(f"Conversion failed for scan {scan_num}")
        img = nib.load(output_nii)
        print(f"    Shape: {img.shape}, voxel sizes: {img.header.get_zooms()[:3]}")

    # JSON sidecar
    json_file = output_nii.with_suffix("").with_suffix(".json")
    if not force and json_file.exists():
        print(f"  JSON already exists: {json_file.name}")
    else:
        meta = extract_bids_metadata(scan_dir, modality)
        with open(json_file, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    return output_nii


def _convert_dwi_gradients(
    session_dir: Path,
    dwi_scan_num: str,
    dwi_nii: Path,
    force: bool,
) -> tuple:
    """Extract bval/bvec, handle 4D/5D mismatch, return (bval_file, bvec_file)."""
    dwi_scan_dir = session_dir / dwi_scan_num
    bvec_file = dwi_nii.with_suffix("").with_suffix(".bvec")
    bval_file = dwi_nii.with_suffix("").with_suffix(".bval")

    if not force and bvec_file.exists() and bval_file.exists():
        print(f"  Gradients already exist: {bval_file.name}, {bvec_file.name}")
        bvals = np.loadtxt(bval_file).flatten()
    else:
        gradient_info = extract_bvec_bval(dwi_scan_dir)
        if gradient_info is None:
            raise RuntimeError("Could not extract gradient table from DWI scan")

        bvals, bvecs = gradient_info
        print(
            f"  Gradient table: {len(bvals)} entries, "
            f"unique b-values: {sorted(set(np.round(bvals, -1).astype(int)))}"
        )

        bvecs_t = bvecs.T if bvecs.ndim > 1 else bvecs.reshape(-1, 1).T
        np.savetxt(bvec_file, bvecs_t, fmt="%.6f", delimiter=" ")
        np.savetxt(bval_file, bvals.reshape(1, -1), fmt="%.1f", delimiter=" ")

    # Handle 4D/5D mismatch with bval count
    dwi_img = nib.load(dwi_nii)
    dwi_data = dwi_img.get_fdata()
    n_bvals = len(bvals)

    if len(dwi_data.shape) == 5:
        print(f"  5D data detected: {dwi_data.shape}")
        print("  Pipeline will average across 5th dimension (reps)")
    elif len(dwi_data.shape) == 4:
        n_vols = dwi_data.shape[3]
        if n_vols == n_bvals:
            print(f"  4D data: {n_vols} volumes matches {n_bvals} bval entries")
        elif n_vols == 2 * n_bvals:
            print(f"  4D data: {n_vols} volumes = 2x {n_bvals} bval entries")
            print("  Averaging pairs (2 repetitions interleaved)...")
            vol1 = dwi_data[:, :, :, :n_bvals]
            vol2 = dwi_data[:, :, :, n_bvals:]
            averaged = (vol1 + vol2) / 2.0
            avg_img = nib.Nifti1Image(averaged, dwi_img.affine, dwi_img.header)
            nib.save(avg_img, dwi_nii)
            print(f"  Saved averaged 4D: {averaged.shape}")
        elif n_vols != n_bvals:
            print(f"  WARNING: {n_vols} volumes vs {n_bvals} bval entries -- mismatch!")
            print("  Proceeding anyway; pipeline will attempt gradient validation.")

    return bval_file, bvec_file


def process_bruker_session(
    session_dir: Path,
    output_dir: Path,
    subject: str,
    session: str,
    config_path: Optional[Path] = None,
    use_gpu: bool = True,
    skip_registration: bool = True,
    force: bool = False,
    inventory_only: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end processing of a single Bruker session.

    Automatically discovers which modalities are present and runs every
    applicable preprocessing pipeline:

    - **anat** (Bruker:RARE / T2w) -- always required as the anatomical anchor
    - **dwi** (Bruker:DtiEpi) -- eddy correction, DTI fitting, QC
    - **func** (Bruker:EPI / BOLD) -- motion correction, nuisance regression, QC
    - **msme** (Bruker:MSME) -- T2 mapping via multi-echo fitting

    Parameters
    ----------
    session_dir : Path
        Bruker session directory containing numbered scan sub-directories.
    output_dir : Path
        Study root directory (creates raw/, derivatives/, qc/, work/, transforms/).
    subject : str
        Subject identifier (e.g. ``'sub-PRE10'``).
    session : str
        Session identifier (e.g. ``'ses-01'``).
    config_path : Path, optional
        YAML config file.  If *None*, a minimal config is generated.
    use_gpu : bool
        Use ``eddy_cuda`` if available (default ``True``).
    skip_registration : bool
        Skip cross-modal registration to template (default ``True``).
    force : bool
        Re-run even if derivatives already exist (default ``False``).
    inventory_only : bool
        Only inventory scans and write CSV; do not process (default ``False``).

    Returns
    -------
    dict
        Processing results keyed by modality, plus ``inventory`` and
        ``inventory_csv``.
    """
    session_dir = Path(session_dir)
    output_dir = Path(output_dir)

    print("=" * 80)
    print(f"Bruker Session Processing: {subject} / {session}")
    print(f"Session dir: {session_dir.name}")
    print(f"Output dir:  {output_dir}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Step 1: Inventory
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Step 1: Scan Inventory")
    print("=" * 80)

    qc_session_dir = output_dir / "qc" / subject / session
    qc_session_dir.mkdir(parents=True, exist_ok=True)
    inventory_csv = qc_session_dir / "scan_inventory.csv"

    inventory = inventory_session(session_dir, output_csv=inventory_csv)

    if not inventory:
        raise RuntimeError(f"No valid scans found in {session_dir}")

    print(f"\nFound {len(inventory)} classified scans:")
    for rec in inventory:
        extras: List[str] = []
        if rec.get("max_bvalue"):
            extras.append(f"bmax={rec['max_bvalue']:.0f}")
        if rec.get("n_echoes"):
            extras.append(f"echoes={rec['n_echoes']}")
        if rec.get("n_repetitions") and rec["modality"] == "func":
            extras.append(f"reps={rec['n_repetitions']}")
        extra_str = "  " + ", ".join(extras) if extras else ""
        print(
            f"  Scan {rec['scan_number']:>3d}: {rec['method']:25s} "
            f"({rec['modality']}/{rec['suffix']})"
            f"  slices={rec.get('n_slices', '?')}{extra_str}"
        )

    print(f"\nInventory CSV: {inventory_csv}")

    if inventory_only:
        return {"inventory": inventory, "inventory_csv": inventory_csv}

    # ------------------------------------------------------------------
    # Step 2: Select best scan for each modality
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Step 2: Scan Selection")
    print("=" * 80)

    best_t2w = select_best_t2w_from_inventory(inventory)
    best_dwi = select_best_dwi_from_inventory(inventory)
    best_func = select_best_func_from_inventory(inventory)
    best_msme = select_best_msme_from_inventory(inventory)

    if best_t2w is None:
        raise RuntimeError("No T2w (RARE) scans found in session -- cannot proceed")

    t2w_scan_num = str(best_t2w["scan_number"])
    print(f"  T2w:  scan {t2w_scan_num} ({best_t2w.get('n_slices', '?')} slices)")

    if best_dwi:
        dwi_scan_num = str(best_dwi["scan_number"])
        print(f"  DWI:  scan {dwi_scan_num} (bmax={best_dwi.get('max_bvalue', '?')})")
    else:
        print("  DWI:  none found -- skipping DWI preprocessing")

    if best_func:
        func_scan_num = str(best_func["scan_number"])
        print(f"  func: scan {func_scan_num} ({best_func.get('n_repetitions', '?')} reps)")
    else:
        print("  func: none found -- skipping functional preprocessing")

    if best_msme:
        msme_scan_num = str(best_msme["scan_number"])
        print(f"  MSME: scan {msme_scan_num} ({best_msme.get('n_echoes', '?')} echoes)")
    else:
        print("  MSME: none found -- skipping MSME preprocessing")

    # ------------------------------------------------------------------
    # Step 3: Configuration
    # ------------------------------------------------------------------
    if config_path is not None:
        from neurofaune.config import load_config

        config = load_config(config_path)
    else:
        config = _build_minimal_config()

    from neurofaune.utils.transforms import TransformRegistry

    registry = TransformRegistry(
        transforms_dir=output_dir / "transforms",
        subject=subject,
        session=session,
    )

    results: Dict[str, Any] = {
        "inventory": inventory,
        "inventory_csv": inventory_csv,
    }

    # ------------------------------------------------------------------
    # Step 4: Convert + preprocess T2w (always)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Step 4: Bruker -> NIfTI Conversion (T2w)")
    print("=" * 80)

    bids_anat = output_dir / "raw" / "bids" / subject / session / "anat"
    t2w_nii = bids_anat / f"{subject}_{session}_run-{t2w_scan_num}_T2w.nii.gz"
    _convert_scan(session_dir, t2w_scan_num, t2w_nii, "anat", force)

    print("\n" + "=" * 80)
    print("Step 4b: Anatomical Preprocessing (T2w)")
    print("=" * 80)

    from neurofaune.preprocess.workflows.anat_preprocess import (
        run_anatomical_preprocessing,
    )

    anat_results = run_anatomical_preprocessing(
        config=config,
        subject=subject,
        session=session,
        t2w_file=t2w_nii,
        output_dir=output_dir,
        transform_registry=registry,
    )
    results["anat"] = anat_results
    results["t2w_scan"] = t2w_scan_num

    # ------------------------------------------------------------------
    # Step 5: Convert + preprocess DWI (if present)
    # ------------------------------------------------------------------
    if best_dwi:
        print("\n" + "=" * 80)
        print("Step 5: Bruker -> NIfTI Conversion (DWI)")
        print("=" * 80)

        bids_dwi_dir = output_dir / "raw" / "bids" / subject / session / "dwi"
        dwi_nii = bids_dwi_dir / f"{subject}_{session}_run-{dwi_scan_num}_dwi.nii.gz"
        _convert_scan(session_dir, dwi_scan_num, dwi_nii, "dwi", force)

        bval_file, bvec_file = _convert_dwi_gradients(
            session_dir, dwi_scan_num, dwi_nii, force
        )

        print("\n" + "=" * 80)
        print("Step 5b: DWI Preprocessing")
        print("=" * 80)

        from neurofaune.preprocess.workflows.dwi_preprocess import (
            run_dwi_preprocessing,
        )

        dwi_results = run_dwi_preprocessing(
            config=config,
            subject=subject,
            session=session,
            dwi_file=dwi_nii,
            bval_file=bval_file,
            bvec_file=bvec_file,
            output_dir=output_dir,
            transform_registry=registry,
            use_gpu=use_gpu,
            run_registration=not skip_registration,
        )
        results["dwi"] = dwi_results
        results["dwi_scan"] = dwi_scan_num

    # ------------------------------------------------------------------
    # Step 6: Convert + preprocess functional (if present)
    # ------------------------------------------------------------------
    if best_func:
        print("\n" + "=" * 80)
        print("Step 6: Bruker -> NIfTI Conversion (func)")
        print("=" * 80)

        bids_func_dir = output_dir / "raw" / "bids" / subject / session / "func"
        func_nii = bids_func_dir / f"{subject}_{session}_run-{func_scan_num}_bold.nii.gz"
        _convert_scan(session_dir, func_scan_num, func_nii, "func", force)

        print("\n" + "=" * 80)
        print("Step 6b: Functional Preprocessing")
        print("=" * 80)

        from neurofaune.preprocess.workflows.func_preprocess import (
            run_functional_preprocessing,
        )

        func_results = run_functional_preprocessing(
            config=config,
            subject=subject,
            session=session,
            bold_file=func_nii,
            output_dir=output_dir,
            transform_registry=registry,
            run_registration=not skip_registration,
        )
        results["func"] = func_results
        results["func_scan"] = func_scan_num

    # ------------------------------------------------------------------
    # Step 7: Convert + preprocess MSME (if present)
    # ------------------------------------------------------------------
    if best_msme:
        print("\n" + "=" * 80)
        print("Step 7: Bruker -> NIfTI Conversion (MSME)")
        print("=" * 80)

        bids_msme_dir = output_dir / "raw" / "bids" / subject / session / "msme"
        msme_nii = bids_msme_dir / f"{subject}_{session}_run-{msme_scan_num}_MSME.nii.gz"
        _convert_scan(session_dir, msme_scan_num, msme_nii, "msme", force)

        # Extract TE values from inventory metadata (if available)
        te_values = None
        echo_times = best_msme.get("echo_times")
        if echo_times:
            te_values = np.array(echo_times)
            print(f"  TE values from Bruker: {len(te_values)} echoes, "
                  f"{te_values[0]:.1f}-{te_values[-1]:.1f} ms")

        print("\n" + "=" * 80)
        print("Step 7b: MSME Preprocessing (T2 mapping)")
        print("=" * 80)

        from neurofaune.preprocess.workflows.msme_preprocess import (
            run_msme_preprocessing,
        )

        msme_results = run_msme_preprocessing(
            config=config,
            subject=subject,
            session=session,
            msme_file=msme_nii,
            output_dir=output_dir,
            transform_registry=registry,
            te_values=te_values,
            run_registration=not skip_registration,
        )
        results["msme"] = msme_results
        results["msme_scan"] = msme_scan_num

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)

    _print_modality_outputs("Anatomical", results.get("anat"),
                            ["brain", "mask", "segmentation"])

    if "dwi" in results:
        _print_modality_outputs("DWI", results["dwi"],
                                ["dwi_preproc", "dwi_mask", "fa", "md", "ad", "rd"])

    if "func" in results:
        _print_modality_outputs("Functional", results["func"],
                                ["bold_preproc", "bold_mask", "confounds"])

    if "msme" in results:
        _print_modality_outputs("MSME", results["msme"],
                                ["t2_map", "mwf_map"])

    print(f"\nInventory CSV: {inventory_csv}")
    print(f"QC reports:    {output_dir / 'qc' / subject / session}")
    print(f"Transforms:    {output_dir / 'transforms' / subject / session}")
    print("=" * 80)

    return results


def _print_modality_outputs(
    label: str,
    result_dict: Optional[Dict[str, Any]],
    keys: List[str],
) -> None:
    """Print a summary block for one modality."""
    if result_dict is None:
        return
    print(f"\n{label} outputs:")
    for key in keys:
        p = result_dict.get(key)
        if p and Path(p).exists():
            img = nib.load(p)
            print(f"  {key:15s}: {Path(p).name}  {img.shape}")
