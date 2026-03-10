#!/usr/bin/env python3
"""
Prepare FSL randomise designs for nonlinear dose-response testing.

Uses the same categorical design matrix as prepare_tbss_designs.py
(Intercept + dose_L + dose_M + dose_H + sex_M) but with orthogonal
polynomial contrasts and F-test files to test:

  F1: Omnibus dose effect (3 df: linear + quadratic + cubic)
  F2: Deviation from linearity (2 df: quadratic + cubic)

Plus directional t-contrasts for post-hoc characterization:
  t1/t2: Linear trend (positive/negative)
  t3/t4: Quadratic component (positive = U-shaped / negative = inverted-U)
  t5/t6: Cubic component (positive/negative)

Polynomial contrasts are computed from the actual log-transformed BPA
doses (0, 2.5, 25, 250 µg/kg/day) via QR decomposition, ensuring proper
orthogonality for the log-spaced dose levels.

Usage:
    uv run python scripts/prepare_tbss_nonlinear_designs.py \
        --study-tracker $STUDY_ROOT/study_tracker_combined_250916.csv \
        --tbss-dir $STUDY_ROOT/analysis/tbss/template/msme/p90 \
        --output-dir $STUDY_ROOT/analysis/tbss/template/msme/p90/designs
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# BPA dose levels (µg/kg/day)
DOSE_VALUES = {'C': 0, 'L': 2.5, 'M': 25, 'H': 250}


def compute_orthogonal_polynomial_contrasts(dose_values: dict) -> dict:
    """
    Compute orthogonal polynomial contrasts for the given dose values.

    Uses log10(1 + dose) transform then QR decomposition to get
    properly orthogonal linear, quadratic, and cubic contrasts.

    Returns dict mapping contrast name to weights on group means [C, L, M, H].
    """
    levels = ['C', 'L', 'M', 'H']
    raw = np.array([dose_values[l] for l in levels], dtype=float)
    x = np.log10(1 + raw)

    # Vandermonde matrix centered at mean
    x_c = x - x.mean()
    V = np.column_stack([x_c**0, x_c**1, x_c**2, x_c**3])

    Q, _ = np.linalg.qr(V)

    # Verify orthogonality
    ortho_check = Q.T @ Q
    if not np.allclose(ortho_check, np.eye(4), atol=1e-10):
        logger.warning("Polynomial contrasts are not perfectly orthogonal")

    return {
        'linear': Q[:, 1],
        'quadratic': Q[:, 2],
        'cubic': Q[:, 3],
    }


def contrast_on_group_means_to_dummy(
    contrast_weights: np.ndarray,
    design_columns: list,
) -> list:
    """
    Convert contrast weights on group means [C, L, M, H] to weights
    on dummy-coded design columns.

    With dummy coding (reference = C):
        mu_C = b0, mu_L = b0+bL, mu_M = b0+bM, mu_H = b0+bH

    So contrast c on means = c[0]*b0 + c[1]*(b0+bL) + c[2]*(b0+bM) + c[3]*(b0+bH)
                            = sum(c)*b0 + c[1]*bL + c[2]*bM + c[3]*bH
    Since orthogonal polynomials sum to 0: intercept coef = 0.

    The mapping from group means [C, L, M, H] to dummy betas is:
        dose_L coef = c[1] (weight on L mean)
        dose_M coef = c[2] (weight on M mean)
        dose_H coef = c[3] (weight on H mean)

    design_columns provides the actual column order from DesignHelper.
    """
    # Map group mean weights to the actual column positions
    mean_to_dummy = {'dose_L': 1, 'dose_M': 2, 'dose_H': 3}
    result = [0.0] * len(design_columns)
    for col_name, mean_idx in mean_to_dummy.items():
        if col_name in design_columns:
            col_idx = design_columns.index(col_name)
            result[col_idx] = float(contrast_weights[mean_idx])
    return result


def write_fsl_con(contrasts: dict, n_waves: int, output_path: Path):
    """Write FSL .con file with named contrasts."""
    n_contrasts = len(contrasts)

    lines = [
        f'/ContrastName1\t{list(contrasts.keys())[0]}',
    ]
    for i, name in enumerate(contrasts.keys()):
        lines.append(f'/ContrastName{i+1}\t{name}')

    # Overwrite the placeholder first line
    lines = []
    for i, name in enumerate(contrasts.keys()):
        lines.append(f'/ContrastName{i+1}\t{name}')
    lines.append(f'/NumWaves\t{n_waves}')
    lines.append(f'/NumContrasts\t{n_contrasts}')
    lines.append('')
    lines.append('/Matrix')
    for weights in contrasts.values():
        lines.append(' '.join(f'{w:.6f}' for w in weights))

    output_path.write_text('\n'.join(lines) + '\n')


def write_fsl_fts(ftest_specs: dict, n_contrasts: int, output_path: Path):
    """
    Write FSL .fts file.

    ftest_specs: dict mapping F-test name to list of 1-based contrast indices.
    """
    n_ftests = len(ftest_specs)

    lines = []
    for i, name in enumerate(ftest_specs.keys()):
        lines.append(f'/FtestName{i+1}\t{name}')
    lines.append(f'/NumWaves\t{n_contrasts}')
    lines.append(f'/NumContrasts\t{n_ftests}')
    lines.append('')
    lines.append('/Matrix')
    for included_contrasts in ftest_specs.values():
        row = [0] * n_contrasts
        for idx in included_contrasts:
            row[idx - 1] = 1  # Convert 1-based to 0-based
        lines.append(' '.join(str(v) for v in row))

    output_path.write_text('\n'.join(lines) + '\n')


def load_and_merge_data(study_tracker_path, tbss_dir):
    """Load study tracker and merge with TBSS subject list."""
    tracker = pd.read_csv(study_tracker_path)
    valid = tracker[tracker['irc.ID'].notna()].copy()
    valid['bids_id'] = 'sub-' + valid['irc.ID']

    subject_list_path = tbss_dir / 'subject_list.txt'
    if not subject_list_path.exists():
        raise FileNotFoundError(f"TBSS subject list not found: {subject_list_path}")

    with open(subject_list_path) as f:
        tbss_subjects = [line.strip() for line in f if line.strip()]

    tbss_parsed = []
    for s in tbss_subjects:
        m = re.match(r'(sub-\w+)_(ses-\w+)', s)
        if m:
            tbss_parsed.append({
                'subject_key': s,
                'bids_id': m.group(1),
                'session': m.group(2),
            })
    tbss_df = pd.DataFrame(tbss_parsed)
    tbss_df['PND'] = tbss_df['session'].str.replace('ses-', '').str.upper()

    merged = tbss_df.merge(
        valid[['bids_id', 'sex', 'dose.level']],
        on='bids_id', how='inner'
    ).rename(columns={'dose.level': 'dose'})

    unmatched = len(tbss_df) - len(merged)
    if unmatched > 0:
        missing_ids = set(tbss_df['bids_id']) - set(valid['bids_id'])
        logger.warning(f"{unmatched} subjects not in tracker: {sorted(missing_ids)}")

    logger.info(f"Merged dataset: {len(merged)} subjects")
    return merged


def write_provenance(design_dir, tbss_dir, n_subjects, design_type):
    """Write provenance.json."""
    subject_list_path = tbss_dir / 'subject_list.txt'
    h = hashlib.sha256()
    with open(subject_list_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)

    provenance = {
        'tbss_dir': str(tbss_dir),
        'subject_list_sha256': h.hexdigest(),
        'n_subjects_in_design': n_subjects,
        'design_type': design_type,
        'date_created': datetime.now().isoformat(),
        'dose_values': DOSE_VALUES,
        'dose_transform': 'log10(1 + dose)',
    }
    with open(design_dir / 'provenance.json', 'w') as f:
        json.dump(provenance, f, indent=2)


def create_nonlinear_design(
    data: pd.DataFrame,
    pnd: str,
    output_dir: Path,
    tbss_dir: Optional[Path] = None,
):
    """
    Create nonlinear dose-response design for a single PND.

    Uses the same design matrix as the categorical per-PND design
    (Intercept + dose dummies + sex), but with orthogonal polynomial
    contrasts and F-test files.
    """
    subset = data[data['PND'] == pnd].copy().reset_index(drop=True)
    n = len(subset)
    if n == 0:
        logger.warning(f"No subjects for {pnd}, skipping")
        return

    # Check that all 4 dose levels are present
    dose_levels_present = set(subset['dose'].unique())
    if dose_levels_present != {'C', 'L', 'M', 'H'}:
        logger.warning(
            f"Not all dose levels present for {pnd}: {dose_levels_present}. "
            "Polynomial contrasts require all 4 levels. Skipping."
        )
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Nonlinear dose-response design: {pnd} (n={n})")
    logger.info(f"{'='*60}")

    # Build design matrix: [Intercept, dose_L, dose_M, dose_H, sex_M]
    # Using same dummy coding as prepare_tbss_designs.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'neuroaider'))
    from neuroaider import DesignHelper

    design_df = pd.DataFrame({
        'participant_id': subset['subject_key'].values,
        'dose': subset['dose'].values,
        'sex': subset['sex'].values,
    })

    helper = DesignHelper(design_df, subject_column='participant_id')
    helper.add_categorical('dose', coding='dummy', reference='C')
    helper.add_categorical('sex', coding='effect', reference='F')
    helper.build_design_matrix()

    cols = helper.design_column_names
    n_waves = len(cols)
    logger.info(f"Design columns: {cols}")

    # Compute orthogonal polynomial contrasts
    poly = compute_orthogonal_polynomial_contrasts(DOSE_VALUES)

    logger.info("\nOrthogonal polynomial contrasts on group means [C, L, M, H]:")
    for name, weights in poly.items():
        logger.info(f"  {name}: [{', '.join(f'{w:.4f}' for w in weights)}]")

    # Convert to dummy-coded design contrasts
    contrasts = {}
    for name, weights in poly.items():
        dummy_weights = contrast_on_group_means_to_dummy(weights, cols)
        contrasts[f'{name}_pos'] = dummy_weights
        contrasts[f'{name}_neg'] = [-w for w in dummy_weights]

    logger.info("\nContrasts in dummy-coded design:")
    for name, weights in contrasts.items():
        logger.info(f"  {name}: [{', '.join(f'{w:.4f}' for w in weights)}]")

    # F-test specifications (1-based contrast indices)
    # Contrasts are: 1=linear_pos, 2=linear_neg, 3=quad_pos, 4=quad_neg, 5=cubic_pos, 6=cubic_neg
    ftest_specs = {
        'omnibus_dose': [1, 2, 3, 4, 5, 6],       # All 3 df
        'deviation_from_linearity': [3, 4, 5, 6],   # Quadratic + cubic (2 df)
        'linear_only': [1, 2],                       # Linear (1 df)
    }

    # Save design files
    design_dir = output_dir / f'nonlinear_{pnd.lower()}'
    design_dir.mkdir(parents=True, exist_ok=True)

    # Save .mat file directly (DesignHelper.save requires contrasts)
    mat = helper.design_matrix
    mat_path = design_dir / 'design.mat'
    with open(mat_path, 'w') as f:
        f.write(f"/NumWaves {mat.shape[1]}\n")
        f.write(f"/NumPoints {mat.shape[0]}\n")
        f.write("/PPheights 1\n")
        f.write("\n/Matrix\n")
        for row in mat:
            f.write(' '.join(f'{v:.6f}' for v in row) + '\n')
    logger.info(f"Wrote design.mat ({mat.shape[0]} x {mat.shape[1]})")

    # Write our polynomial .con file
    write_fsl_con(contrasts, n_waves, design_dir / 'design.con')
    logger.info(f"\nWrote design.con with {len(contrasts)} contrasts")

    # Write .fts file
    write_fsl_fts(ftest_specs, len(contrasts), design_dir / 'design.fts')
    logger.info(f"Wrote design.fts with {len(ftest_specs)} F-tests")

    # Save subject order
    subset[['subject_key']].to_csv(
        design_dir / 'subject_order.txt', index=False, header=False
    )

    # Save design description
    desc_lines = [
        f"Nonlinear dose-response design: {pnd}",
        f"Date: {datetime.now().isoformat()}",
        f"Subjects: {n}",
        f"",
        f"Design matrix: {n_waves} columns",
        f"  {cols}",
        f"",
        f"BPA doses: {DOSE_VALUES}",
        f"Dose transform: log10(1 + dose)",
        f"",
        f"Orthogonal polynomial contrasts (on group means [C, L, M, H]):",
    ]
    for name, weights in poly.items():
        desc_lines.append(f"  {name}: [{', '.join(f'{w:.4f}' for w in weights)}]")
    desc_lines.extend([
        f"",
        f"T-contrasts ({len(contrasts)}):",
    ])
    for name, weights in contrasts.items():
        desc_lines.append(f"  {name}: [{', '.join(f'{w:.4f}' for w in weights)}]")
    desc_lines.extend([
        f"",
        f"F-tests ({len(ftest_specs)}):",
        f"  F1 omnibus_dose: all polynomial components (3 df)",
        f"  F2 deviation_from_linearity: quadratic + cubic (2 df)",
        f"  F3 linear_only: linear component (1 df)",
        f"",
        f"Interpretation of F2 (deviation_from_linearity):",
        f"  Significant F2 = dose-response is non-monotonic/non-linear",
        f"  Post-hoc: examine quadratic_pos/neg t-stats to determine shape",
        f"    quadratic_pos significant → U-shaped response",
        f"    quadratic_neg significant → inverted-U response",
    ])
    (design_dir / 'design_description.txt').write_text('\n'.join(desc_lines) + '\n')

    # Write provenance
    if tbss_dir is not None:
        write_provenance(design_dir, tbss_dir, n, f'nonlinear_{pnd.lower()}')

    logger.info(f"Saved to {design_dir}")

    # Save summary JSON for downstream processing
    summary = {
        'design_type': 'nonlinear_dose_response',
        'pnd': pnd,
        'n_subjects': n,
        'dose_values': DOSE_VALUES,
        'dose_transform': 'log10(1 + dose)',
        'contrasts': list(contrasts.keys()),
        'ftests': {name: indices for name, indices in ftest_specs.items()},
        'polynomial_weights_on_means': {
            name: [float(w) for w in weights]
            for name, weights in poly.items()
        },
    }
    with open(design_dir / 'design_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare nonlinear dose-response designs for TBSS'
    )
    parser.add_argument(
        '--study-tracker', type=Path, required=True,
        help='Path to study_tracker_combined CSV'
    )
    parser.add_argument(
        '--tbss-dir', type=Path, required=True,
        help='Path to TBSS directory (must contain subject_list.txt)'
    )
    parser.add_argument(
        '--output-dir', type=Path, required=True,
        help='Output directory for design files'
    )
    parser.add_argument(
        '--skip-subset', action='store_true',
        help='Skip pre-creating 4D subsets'
    )
    args = parser.parse_args()

    data = load_and_merge_data(args.study_tracker, args.tbss_dir)

    logger.info("\nData distribution:")
    for pnd in ['P30', 'P60', 'P90']:
        subset = data[data['PND'] == pnd]
        if len(subset) > 0:
            dose_dist = dict(subset['dose'].value_counts().sort_index())
            logger.info(f"  {pnd}: n={len(subset)}, dose={dose_dist}")

    # Create per-PND nonlinear designs
    for pnd in ['P30', 'P60', 'P90']:
        create_nonlinear_design(data, pnd, args.output_dir, tbss_dir=args.tbss_dir)

    logger.info(f"\nAll nonlinear designs saved to {args.output_dir}")

    # Pre-create 4D subsets if requested
    if not args.skip_subset:
        from prepare_tbss_designs import pre_subset_4d_volumes
        design_dirs = sorted(args.output_dir.glob('nonlinear_*'))
        design_dirs = [d for d in design_dirs if d.is_dir()]
        if design_dirs:
            pre_subset_4d_volumes(args.tbss_dir, design_dirs)
    else:
        logger.info("\nSkipping 4D subset creation (--skip-subset)")


if __name__ == '__main__':
    main()
