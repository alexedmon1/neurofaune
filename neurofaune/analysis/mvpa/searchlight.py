"""
Searchlight MVPA analysis using nilearn SearchLight.

Maps local discriminability across the brain by running a classifier
in a sliding sphere. Optional max-statistic FWER correction via
label permutations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import nibabel as nib
import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def run_searchlight(
    images_4d,
    labels: Union[List[str], List[float], np.ndarray],
    mask_img,
    analysis_type: str = "classification",
    radius: float = 2.0,
    cv_folds: int = 3,
    n_jobs: int = 1,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    n_perm_fwer: int = 100,
) -> Dict[str, Any]:
    """Run searchlight analysis with optional FWER correction.

    Args:
        images_4d: 4D Nifti1Image (subjects along 4th dimension).
        labels: Group labels (str for classification, numeric for regression).
        mask_img: Brain mask Nifti1Image.
        analysis_type: 'classification' or 'dose_response'.
        radius: Searchlight sphere radius in mm (default: 2.0 for scaled space).
        cv_folds: Number of cross-validation folds.
        n_jobs: Number of parallel jobs for SearchLight.
        seed: Random seed.
        output_dir: If set, save maps and results.json.
        n_perm_fwer: Number of label permutations for FWER correction.

    Returns:
        Dict with: searchlight_img, threshold_fwer, n_significant_voxels,
        mean_score, analysis_type.
    """
    from nilearn.decoding import SearchLight

    labels_arr = np.array(labels)
    n_samples = len(labels_arr)

    if analysis_type == "classification":
        from sklearn.svm import LinearSVC
        estimator = LinearSVC(dual=False, max_iter=10000, random_state=seed)
        scoring = "accuracy"
        stratify_labels = labels_arr
    else:
        from sklearn.linear_model import Ridge
        estimator = Ridge(alpha=1.0)
        scoring = "r2"
        labels_arr = labels_arr.astype(float)
        bins = np.digitize(labels_arr, np.percentile(labels_arr, [25, 50, 75]))
        stratify_labels = bins

    cv = StratifiedKFold(n_splits=min(cv_folds, n_samples), shuffle=True,
                         random_state=seed)

    logger.info(
        "Searchlight: %s, n=%d, radius=%.1fmm, cv=%d-fold, n_jobs=%d",
        analysis_type, n_samples, radius, cv_folds, n_jobs,
    )

    # Run searchlight
    searchlight = SearchLight(
        mask_img=mask_img,
        radius=radius,
        estimator=estimator,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=0,
    )
    searchlight.fit(images_4d, labels_arr)

    # Build searchlight score image
    searchlight_scores = searchlight.scores_
    searchlight_img = nib.Nifti1Image(
        searchlight_scores.astype(np.float32), mask_img.affine
    )

    mask_data = mask_img.get_fdata() > 0
    scores_in_mask = searchlight_scores[mask_data]
    mean_score = float(np.mean(scores_in_mask))
    logger.info("Searchlight mean %s in mask: %.3f", scoring, mean_score)

    # Max-statistic FWER correction via label permutations
    threshold_fwer = None
    n_significant = 0
    thresholded_img = None

    if n_perm_fwer > 0:
        logger.info("Running %d permutations for FWER correction...", n_perm_fwer)
        rng = np.random.RandomState(seed)
        max_stats = []

        for i_perm in range(n_perm_fwer):
            perm_labels = rng.permutation(labels_arr)
            perm_sl = SearchLight(
                mask_img=mask_img,
                radius=radius,
                estimator=estimator,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                verbose=0,
            )
            perm_sl.fit(images_4d, perm_labels)
            perm_scores = perm_sl.scores_
            max_stats.append(float(np.max(perm_scores[mask_data])))

            if (i_perm + 1) % 10 == 0:
                logger.info("  FWER permutation %d/%d", i_perm + 1, n_perm_fwer)

        max_stats = np.array(max_stats)
        threshold_fwer = float(np.percentile(max_stats, 95))
        n_significant = int(np.sum(scores_in_mask > threshold_fwer))

        logger.info(
            "FWER threshold (p<0.05): %.3f, %d significant voxels",
            threshold_fwer, n_significant,
        )

        # Create thresholded map
        thresholded_data = searchlight_scores.copy()
        thresholded_data[thresholded_data <= threshold_fwer] = 0
        thresholded_img = nib.Nifti1Image(
            thresholded_data.astype(np.float32), mask_img.affine
        )

    results = {
        "analysis_type": analysis_type,
        "searchlight_img": searchlight_img,
        "thresholded_img": thresholded_img,
        "threshold_fwer": threshold_fwer,
        "n_significant_voxels": n_significant,
        "mean_score": mean_score,
        "radius": radius,
        "cv_folds": cv_folds,
        "n_samples": n_samples,
        "n_perm_fwer": n_perm_fwer,
        "score_label": scoring,
    }

    # Save outputs
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        nib.save(searchlight_img, output_dir / "searchlight_map.nii.gz")

        if thresholded_img is not None:
            nib.save(thresholded_img, output_dir / "searchlight_thresholded.nii.gz")

        results_json = {
            k: v for k, v in results.items()
            if k not in ("searchlight_img", "thresholded_img")
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(results_json, f, indent=2)

        logger.info("Saved searchlight results to %s", output_dir)

    return results
