"""
Whole-brain MVPA decoding using nilearn Decoder.

Wraps nilearn.decoding.Decoder for classification (SVM) and
dose-response regression (Ridge), with ANOVA feature screening
and permutation testing for empirical p-values.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import nibabel as nib
import numpy as np
from sklearn.model_selection import StratifiedKFold, permutation_test_score

logger = logging.getLogger(__name__)


def run_whole_brain_decoding(
    images_4d,
    labels: Union[List[str], List[float], np.ndarray],
    mask_img,
    analysis_type: str = "classification",
    n_permutations: int = 1000,
    cv_folds: int = 5,
    screening_percentile: int = 20,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run whole-brain decoding with permutation testing.

    Args:
        images_4d: 4D Nifti1Image (subjects along 4th dimension).
        labels: Group labels (str for classification, numeric for regression).
        mask_img: Brain mask Nifti1Image.
        analysis_type: 'classification' or 'dose_response'.
        n_permutations: Number of permutations for empirical p-value.
        cv_folds: Number of cross-validation folds.
        screening_percentile: ANOVA feature screening percentile.
        seed: Random seed.
        output_dir: If set, save weight_map.nii.gz and results.json.

    Returns:
        Dict with: mean_score, std_score, fold_scores, permutation_p,
        null_distribution, weight_img, analysis_type.
    """
    from nilearn.decoding import Decoder

    labels_arr = np.array(labels)
    n_samples = len(labels_arr)

    if analysis_type == "classification":
        estimator = "svc_l1"
        scoring = "accuracy"
        score_label = "accuracy"
        # StratifiedKFold for balanced folds
        cv = StratifiedKFold(n_splits=min(cv_folds, n_samples), shuffle=True,
                             random_state=seed)
        stratify_labels = labels_arr
    else:
        estimator = "ridge_regressor"
        scoring = "r2"
        score_label = "r2"
        labels_arr = labels_arr.astype(float)
        # For regression, bin labels for stratification
        bins = np.digitize(labels_arr, np.percentile(labels_arr, [25, 50, 75]))
        cv = StratifiedKFold(n_splits=min(cv_folds, n_samples), shuffle=True,
                             random_state=seed)
        stratify_labels = bins

    logger.info(
        "Whole-brain decoding: %s, n=%d, cv=%d-fold, screening=%d%%",
        analysis_type, n_samples, cv_folds, screening_percentile,
    )

    # Build decoder
    decoder = Decoder(
        estimator=estimator,
        mask=mask_img,
        screening_percentile=screening_percentile,
        scoring=scoring,
        cv=cv,
        standardize=True,
    )

    # Fit decoder
    decoder.fit(images_4d, labels_arr)

    # Extract fold scores from decoder
    fold_scores = list(decoder.cv_scores_.values())[0]
    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    logger.info(
        "Decoding %s: %.3f +/- %.3f",
        score_label, mean_score, std_score,
    )

    # Extract weight map
    weight_img = decoder.coef_img_[list(decoder.coef_img_.keys())[0]]

    # Permutation test for empirical p-value
    logger.info("Running %d permutations for p-value...", n_permutations)

    # Use sklearn permutation_test_score with the decoder's estimator
    from nilearn.maskers import NiftiMasker
    masker = NiftiMasker(mask_img=mask_img, standardize=True)
    X_masked = masker.fit_transform(images_4d)

    # Feature screening (ANOVA)
    from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
    if analysis_type == "classification":
        selector = SelectPercentile(f_classif, percentile=screening_percentile)
    else:
        selector = SelectPercentile(f_regression, percentile=screening_percentile)
    X_screened = selector.fit_transform(X_masked, labels_arr)

    # Build sklearn estimator matching the decoder
    if analysis_type == "classification":
        from sklearn.svm import LinearSVC
        sklearn_estimator = LinearSVC(penalty="l1", dual=False, max_iter=10000,
                                      random_state=seed)
    else:
        from sklearn.linear_model import Ridge
        sklearn_estimator = Ridge(alpha=1.0)

    observed_score, null_dist, perm_p = permutation_test_score(
        sklearn_estimator,
        X_screened,
        labels_arr,
        scoring=scoring,
        cv=cv,
        n_permutations=n_permutations,
        random_state=seed,
        n_jobs=1,
    )

    logger.info(
        "Permutation p-value: %.4f (observed=%.3f)",
        perm_p, observed_score,
    )

    results = {
        "analysis_type": analysis_type,
        "mean_score": mean_score,
        "std_score": std_score,
        "fold_scores": [float(s) for s in fold_scores],
        "permutation_p": float(perm_p),
        "observed_permutation_score": float(observed_score),
        "null_distribution": null_dist,
        "weight_img": weight_img,
        "n_samples": n_samples,
        "cv_folds": cv_folds,
        "screening_percentile": screening_percentile,
        "n_permutations": n_permutations,
        "score_label": score_label,
    }

    # Save outputs
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        nib.save(weight_img, output_dir / "weight_map.nii.gz")

        results_json = {
            k: v for k, v in results.items()
            if k not in ("null_distribution", "weight_img")
        }
        results_json["null_distribution_summary"] = {
            "mean": float(np.mean(null_dist)),
            "std": float(np.std(null_dist)),
            "min": float(np.min(null_dist)),
            "max": float(np.max(null_dist)),
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(results_json, f, indent=2)

        logger.info("Saved whole-brain results to %s", output_dir)

    return results
