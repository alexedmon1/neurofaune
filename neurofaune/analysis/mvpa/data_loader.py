"""
MVPA data loading utilities.

Discovers individual SIGMA-space NIfTIs from the derivatives tree,
stacks them into 4D volumes, and loads NeuroAider design matrices.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# Cohorts with templates (ses-unknown is excluded)
VALID_COHORTS = {"p30", "p60", "p90"}


def discover_sigma_images(
    derivatives_root: Path,
    metric: str,
    session_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Discover SIGMA-space NIfTIs for a given metric.

    Scans derivatives_root/sub-*/ses-*/dwi/ for files matching
    ``sub-{subject}_{session}_space-SIGMA_{metric}.nii.gz``.
    Skips ``ses-unknown`` sessions.

    Args:
        derivatives_root: Path to derivatives directory.
        metric: DTI metric name (e.g. 'FA', 'MD').
        session_filter: If set, only include this session (e.g. 'ses-p60').

    Returns:
        Sorted list of dicts with keys: subject, session, cohort, image_path.
    """
    derivatives_root = Path(derivatives_root)
    results = []

    pattern = f"*_space-SIGMA_{metric}.nii.gz"

    for subject_dir in sorted(derivatives_root.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("sub-"):
            continue
        subject = subject_dir.name

        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir() or not session_dir.name.startswith("ses-"):
                continue
            session = session_dir.name
            cohort = session.replace("ses-", "")

            # Skip unknown cohort
            if cohort not in VALID_COHORTS:
                continue

            # Apply session filter
            if session_filter and session != session_filter:
                continue

            dwi_dir = session_dir / "dwi"
            if not dwi_dir.is_dir():
                continue

            # Look for the SIGMA-space metric file
            sigma_file = dwi_dir / f"{subject}_{session}_space-SIGMA_{metric}.nii.gz"
            if sigma_file.exists():
                results.append({
                    "subject": subject,
                    "session": session,
                    "cohort": cohort,
                    "image_path": sigma_file,
                })

    logger.info(
        "Discovered %d SIGMA-space %s images in %s",
        len(results), metric, derivatives_root,
    )
    return results


def load_mvpa_data(
    image_info_list: List[Dict[str, Any]],
    mask_img=None,
) -> Dict[str, Any]:
    """Load and stack SIGMA-space images into a 4D volume.

    Args:
        image_info_list: List of dicts from discover_sigma_images().
        mask_img: Optional Nifti1Image mask. If None, creates a brain
            mask from the mean image (voxels > 0).

    Returns:
        Dict with keys: images_4d (Nifti1Image), mask_img (Nifti1Image),
        subjects (list), sessions (list), cohorts (list).
    """
    from nilearn.image import concat_imgs

    if not image_info_list:
        raise ValueError("No images to load")

    image_paths = [info["image_path"] for info in image_info_list]
    subjects = [info["subject"] for info in image_info_list]
    sessions = [info["session"] for info in image_info_list]
    cohorts = [info["cohort"] for info in image_info_list]

    logger.info("Stacking %d images into 4D volume...", len(image_paths))
    images_4d = concat_imgs(image_paths)

    if mask_img is None:
        logger.info("Creating brain mask from mean image (voxels > 0)")
        mean_data = np.mean(images_4d.get_fdata(), axis=-1)
        mask_data = (mean_data > 0).astype(np.uint8)
        mask_img = nib.Nifti1Image(mask_data, images_4d.affine)

    return {
        "images_4d": images_4d,
        "mask_img": mask_img,
        "subjects": subjects,
        "sessions": sessions,
        "cohorts": cohorts,
    }


def load_design(design_dir: Path) -> Dict[str, Any]:
    """Load a NeuroAider design from its summary JSON.

    Reads design_summary.json and subject_order.txt to reconstruct
    the subject ordering and group labels.

    Args:
        design_dir: Path to design directory containing
            design_summary.json and subject_order.txt.

    Returns:
        Dict with keys: labels (list of str or float), groups (list),
        design_type (str), subject_order (list of str),
        subject_key_to_label (dict).
    """
    design_dir = Path(design_dir)

    summary_path = design_dir / "design_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Design summary not found: {summary_path}")

    with open(summary_path) as f:
        summary = json.load(f)

    # Load subject order
    subject_order_path = design_dir / "subject_order.txt"
    if not subject_order_path.exists():
        raise FileNotFoundError(f"Subject order not found: {subject_order_path}")

    with open(subject_order_path) as f:
        subject_order = [line.strip() for line in f if line.strip()]

    # Determine design type from column names or design structure
    columns = summary.get("columns", [])
    design_type = "categorical"

    # Check if this is a dose-response design (has dose_numeric column)
    for col in columns:
        if "dose_numeric" in col.get("name", ""):
            design_type = "dose_response"
            break

    # Extract group labels from design matrix structure
    groups = summary.get("groups", {})
    subject_key_to_label = {}

    if design_type == "categorical":
        # For categorical designs, reconstruct group membership from
        # the design matrix. The design_summary includes group info.
        design_matrix = summary.get("design_matrix", [])
        dose_columns = [
            c for c in columns
            if c.get("name", "").startswith("dose_")
            and "numeric" not in c.get("name", "")
        ]

        for i, subj_key in enumerate(subject_order):
            if i < len(design_matrix):
                row = design_matrix[i]
                # Find which dose group: if all dose dummies are 0, it's reference (C)
                label = "C"
                for col_info in dose_columns:
                    col_idx = col_info.get("index", -1)
                    if col_idx >= 0 and col_idx < len(row) and row[col_idx] == 1.0:
                        label = col_info["name"].replace("dose_", "")
                        break
                subject_key_to_label[subj_key] = label
            else:
                subject_key_to_label[subj_key] = "C"

    elif design_type == "dose_response":
        # For dose-response, labels are ordinal numeric values
        dose_col = None
        for col in columns:
            if col.get("name", "") == "dose_numeric":
                dose_col = col
                break

        design_matrix = summary.get("design_matrix", [])
        center = dose_col.get("mean", 1.5) if dose_col else 1.5

        for i, subj_key in enumerate(subject_order):
            if i < len(design_matrix):
                row = design_matrix[i]
                col_idx = dose_col.get("index", -1) if dose_col else -1
                if col_idx >= 0 and col_idx < len(row):
                    # Undo mean-centering to get ordinal value
                    subject_key_to_label[subj_key] = round(row[col_idx] + center)
                else:
                    subject_key_to_label[subj_key] = 0
            else:
                subject_key_to_label[subj_key] = 0

    labels = [subject_key_to_label.get(s, "C") for s in subject_order]

    logger.info(
        "Loaded design: %s, %d subjects, type=%s",
        design_dir.name, len(subject_order), design_type,
    )

    return {
        "labels": labels,
        "groups": list(set(labels)),
        "design_type": design_type,
        "subject_order": subject_order,
        "subject_key_to_label": subject_key_to_label,
    }


def align_data_to_design(
    image_info: List[Dict[str, Any]],
    design: Dict[str, Any],
) -> Dict[str, Any]:
    """Align image data to design matrix subject ordering.

    Filters and reorders image_info to match the design's subject_order.
    Subjects present in the design but missing from image_info are dropped,
    and vice versa.

    Args:
        image_info: List of dicts from discover_sigma_images().
        design: Dict from load_design().

    Returns:
        Dict with keys: image_info (filtered/reordered list),
        labels (matched list), n_matched, n_design_only, n_data_only.
    """
    subject_order = design["subject_order"]
    subject_key_to_label = design["subject_key_to_label"]

    # Build lookup from subject_key to image_info entry
    # subject_key format: "sub-Rat1_ses-p60"
    info_by_key = {}
    for info in image_info:
        key = f"{info['subject']}_{info['session']}"
        info_by_key[key] = info

    design_keys = set(subject_order)
    data_keys = set(info_by_key.keys())
    matched_keys = design_keys & data_keys
    design_only = design_keys - data_keys
    data_only = data_keys - design_keys

    if design_only:
        logger.warning(
            "%d subjects in design but not in data: %s",
            len(design_only), sorted(design_only)[:5],
        )
    if data_only:
        logger.info(
            "%d subjects in data but not in design (excluded from analysis)",
            len(data_only),
        )

    # Reorder to match design subject_order, keeping only matched
    aligned_info = []
    aligned_labels = []
    for key in subject_order:
        if key in info_by_key:
            aligned_info.append(info_by_key[key])
            aligned_labels.append(subject_key_to_label.get(key))

    logger.info(
        "Aligned: %d matched, %d design-only, %d data-only",
        len(aligned_info), len(design_only), len(data_only),
    )

    return {
        "image_info": aligned_info,
        "labels": aligned_labels,
        "n_matched": len(aligned_info),
        "n_design_only": len(design_only),
        "n_data_only": len(data_only),
    }
