# TBSS Implementation Plan for Neurofaune

## Overview

Implement Tract-Based Spatial Statistics (TBSS) for rodent DTI analysis in neurofaune, leveraging:
- Our new template-based registration pipeline
- Slice correspondence system for partial-coverage modalities
- SIGMA atlas in study-space orientation
- Improved WM masking to remove spurious exterior "WM" signal
- Integration with neuroaider for design matrix creation
- **Slice-level QC with bad slice exclusion** (new)

## Background Analysis

### Existing Approaches Reviewed

**Old BPA-rat TBSS (`/mnt/arborea/Preclinical/BPA-rat/code/dwi`):**
- 2D FLIRT for FA-to-T2w (partial brain constraint)
- FNIRT for T2w-to-SIGMA
- Standard FSL tbss_* commands for skeleton creation
- FA threshold = 0.4
- Manual design matrix scripts

**neurovrai TBSS:**
- Human-focused (FMRIB58_FA template)
- Modular: prepare once, analyze many times
- Supports 11 metrics (DTI, DKI, NODDI)
- Atlas-based cluster reporting with JHU atlas
- Separate design matrix module

### Key Improvements for Neurofaune

| Issue | Old Approach | New Approach |
|-------|--------------|--------------|
| Registration | 2D FLIRT + FNIRT chain | Template-based with slice correspondence |
| WM mask | Simple brain mask | Tissue probability + edge erosion |
| Exterior WM | Not addressed | Explicit removal using cortical boundary |
| Design matrix | Custom Python scripts | neuroaider integration |
| Atlas labels | Manual | SIGMA parcellation labels |

---

## Architecture

### Module Structure

```
neurofaune/
└── analysis/
    ├── __init__.py
    ├── tbss/
    │   ├── __init__.py
    │   ├── prepare.py          # Data preparation and skeleton creation
    │   ├── skeleton.py         # WM skeleton extraction (rodent-optimized)
    │   ├── project.py          # Project metrics onto skeleton
    │   ├── stats.py            # FSL randomise wrapper
    │   └── reporting.py        # Cluster extraction with SIGMA labels
    └── utils/
        ├── __init__.py
        └── subject_discovery.py  # Find subjects with required outputs
```

### Output Directory Structure

```
{study_root}/analysis/tbss/
├── FA/
│   ├── native/                 # Original FA in subject space
│   │   └── sub-*_FA.nii.gz
│   ├── template/               # FA warped to cohort template
│   │   └── sub-*_FA_template.nii.gz
│   └── sigma/                  # FA warped to SIGMA space
│       └── sub-*_FA_sigma.nii.gz
├── stats/
│   ├── mean_FA.nii.gz
│   ├── mean_FA_skeleton.nii.gz
│   ├── mean_FA_skeleton_mask.nii.gz
│   ├── all_FA_skeletonised.nii.gz
│   ├── all_MD_skeletonised.nii.gz
│   ├── all_AD_skeletonised.nii.gz
│   └── all_RD_skeletonised.nii.gz
├── wm_mask/
│   ├── wm_probability_sigma.nii.gz
│   ├── wm_skeleton_mask.nii.gz      # Interior WM only
│   └── exterior_wm_mask.nii.gz      # Removed regions
├── randomise/
│   └── {analysis_name}/
│       ├── design.mat
│       ├── design.con
│       └── tfce_corrp_tstat*.nii.gz
├── reports/
│   └── cluster_report_*.html
├── subject_list.txt
└── manifest.json
```

---

## Implementation Phases

### Phase 1: Data Preparation Module

**File: `neurofaune/analysis/tbss/prepare.py`**

#### 1.1 Subject Discovery

```python
def discover_tbss_subjects(
    derivatives_dir: Path,
    cohorts: List[str] = ['p30', 'p60', 'p90'],
    required_metrics: List[str] = ['FA', 'MD', 'AD', 'RD']
) -> Dict[str, SubjectInfo]:
    """
    Find all subjects with completed DTI preprocessing.

    Returns dict mapping subject_id -> SubjectInfo with:
    - subject, session, cohort
    - paths to FA, MD, AD, RD
    - path to T2w (for registration chain)
    - QC status
    """
```

#### 1.2 Registration to SIGMA Space

Leverage existing registration chain:

```
Subject FA → Subject T2w → Cohort Template → SIGMA (study-space)
```

```python
def register_fa_to_sigma(
    fa_file: Path,
    subject: str,
    session: str,
    config: Dict,
    transform_registry: TransformRegistry
) -> Path:
    """
    Apply full transform chain to warp FA to SIGMA space.

    Uses:
    1. FA → T2w transform (from DTI preprocessing)
    2. T2w → Template transform (subject-to-template)
    3. Template → SIGMA transform (template-to-atlas)

    Returns path to FA in SIGMA space.
    """
```

#### 1.3 Batch Registration

```python
def prepare_tbss_data(
    config: Dict,
    output_dir: Path,
    subjects: Optional[List[str]] = None,
    cohorts: List[str] = ['p30', 'p60', 'p90'],
    n_workers: int = 4
) -> Dict:
    """
    Prepare all subjects for TBSS analysis.

    Steps:
    1. Discover subjects with completed DTI
    2. Register all FA maps to SIGMA space
    3. Register MD, AD, RD using same transforms
    4. Generate manifest with included/excluded subjects

    Returns summary dict with paths and statistics.
    """
```

---

### Phase 2: WM Skeleton with Exterior Removal

**File: `neurofaune/analysis/tbss/skeleton.py`**

#### 2.1 The Exterior WM Problem

Rodent DTI often shows high FA at brain edges due to:
- Partial volume effects with skull/meninges
- Susceptibility artifacts
- Registration imperfections

Standard TBSS includes these as "white matter" which contaminates results.

#### 2.2 Solution: Tissue-Informed Skeleton Masking

```python
def create_interior_wm_mask(
    mean_fa: Path,
    wm_probability: Path,
    gm_probability: Path,
    brain_mask: Path,
    output_dir: Path,
    fa_threshold: float = 0.3,        # Lower for rodent (vs 0.4 human)
    wm_prob_threshold: float = 0.5,   # Must be >50% WM probability
    erosion_mm: float = 1.0           # Erode edge by 1mm (10 voxels at 0.1mm)
) -> Tuple[Path, Path]:
    """
    Create WM skeleton mask excluding exterior "WM".

    Algorithm:
    1. Start with tissue probability WM map from SIGMA
    2. Threshold at wm_prob_threshold
    3. Erode from cortical surface by erosion_mm
    4. AND with FA > fa_threshold
    5. Remove small isolated clusters

    Returns:
    - interior_wm_mask: Valid WM for skeleton
    - exterior_wm_removed: Regions excluded (for QC visualization)
    """
```

#### 2.3 Skeleton Creation

```python
def create_rodent_skeleton(
    all_fa_4d: Path,
    wm_mask: Path,
    output_dir: Path,
    skeleton_threshold: float = 0.3
) -> Dict[str, Path]:
    """
    Create mean FA skeleton optimized for rodent brain.

    Steps:
    1. Compute mean FA across subjects
    2. Apply interior WM mask
    3. Run FSL tbss_skeleton with modified threshold
    4. Compute distance map for projection

    Returns dict with paths to:
    - mean_FA, mean_FA_masked, mean_FA_skeleton
    - skeleton_mask, distance_map
    """
```

#### 2.4 Rodent-Specific Parameters

| Parameter | Human TBSS | Rodent TBSS | Rationale |
|-----------|------------|-------------|-----------|
| FA threshold | 0.2-0.4 | 0.2-0.3 | Lower FA in rodent WM |
| Skeleton threshold | 0.4 | 0.3 | Preserve more tracts |
| WM probability | N/A | 0.5 | Tissue-informed masking |
| Edge erosion | N/A | 1.0 mm | Remove boundary artifacts |

---

### Phase 3: Metric Projection

**File: `neurofaune/analysis/tbss/project.py`**

```python
def project_metric_to_skeleton(
    metric_4d: Path,
    mean_fa: Path,
    skeleton_mask: Path,
    distance_map: Path,
    output_file: Path,
    search_rule: str = 'perpendicular'
) -> Path:
    """
    Project diffusion metric onto FA skeleton.

    Uses FSL tbss_skeleton with:
    - alt_metric: The non-FA metric to project
    - search_rule: How to find skeleton values

    Returns path to skeletonised metric (4D: voxels × subjects).
    """

def project_all_metrics(
    tbss_dir: Path,
    metrics: List[str] = ['FA', 'MD', 'AD', 'RD']
) -> Dict[str, Path]:
    """
    Project all DTI metrics onto the FA skeleton.

    FA is used to create the skeleton.
    MD, AD, RD are projected using the FA skeleton structure.
    """
```

---

### Phase 4: Statistical Analysis

**File: `neurofaune/analysis/tbss/stats.py`**

#### 4.1 Design Matrix Integration

Design matrices will be created externally using **neuroaider**. This module just validates and runs the analysis.

```python
def validate_design_files(
    design_mat: Path,
    design_con: Path,
    n_subjects: int
) -> bool:
    """
    Validate FSL design matrix and contrast files.

    Checks:
    - Matrix dimensions match n_subjects
    - Contrast dimensions match design columns
    - Files are valid FSL format
    """
```

#### 4.2 Randomise Wrapper

```python
def run_randomise(
    skeletonised_data: Path,
    skeleton_mask: Path,
    design_mat: Path,
    design_con: Path,
    output_dir: Path,
    n_permutations: int = 5000,
    tfce: bool = True,
    seed: int = None
) -> Dict[str, Path]:
    """
    Run FSL randomise for voxel-wise statistics.

    Parameters:
    - tfce: Use Threshold-Free Cluster Enhancement (recommended)
    - n_permutations: Number of permutations (5000 standard)

    Returns dict with paths to:
    - tstat maps for each contrast
    - tfce_corrp maps (TFCE-corrected p-values)
    """

def run_tbss_stats(
    tbss_dir: Path,
    design_dir: Path,
    analysis_name: str,
    metrics: List[str] = ['FA', 'MD', 'AD', 'RD'],
    n_permutations: int = 5000,
    n_workers: int = 4
) -> Dict:
    """
    Run statistical analysis on all metrics.

    Runs randomise in parallel for each metric.
    """
```

---

### Phase 5: Reporting

**File: `neurofaune/analysis/tbss/reporting.py`**

#### 5.1 Cluster Extraction

```python
def extract_clusters(
    tfce_corrp_file: Path,
    p_threshold: float = 0.05,
    min_cluster_size: int = 10
) -> pd.DataFrame:
    """
    Extract significant clusters from TFCE-corrected p-value map.

    Returns DataFrame with:
    - cluster_id, size, peak_x, peak_y, peak_z
    - peak_value (1-p), mean_value
    """
```

#### 5.2 SIGMA Atlas Labels

```python
def add_sigma_labels(
    clusters: pd.DataFrame,
    sigma_parcellation: Path,
    sigma_labels: Path
) -> pd.DataFrame:
    """
    Add anatomical labels from SIGMA atlas.

    For each cluster:
    - Find overlapping SIGMA regions
    - Report percentage overlap with each region
    - Identify primary anatomical location
    """
```

#### 5.3 HTML Report Generation

```python
def generate_tbss_report(
    analysis_name: str,
    tbss_dir: Path,
    randomise_dir: Path,
    output_file: Path
) -> Path:
    """
    Generate comprehensive HTML report.

    Includes:
    - Analysis parameters
    - Subject summary
    - Significant clusters for each metric/contrast
    - Slice visualizations of significant regions
    - SIGMA atlas overlay
    """
```

---

### Phase 6: Slice QC and Bad Slice Handling

**File: `neurofaune/analysis/tbss/slice_qc.py`**

This is a critical addition to handle the common rodent DTI problem of bad slices (artifacts, signal dropout, motion).

#### 6.1 The Problem

Rodent DTI often has slice-specific issues:
- **Signal dropout**: Complete or partial loss in specific slices
- **Motion artifacts**: Blurring or ghosting in some slices
- **Susceptibility artifacts**: Distortion near air-tissue interfaces
- **Partial coverage**: Different subjects may have different valid slice ranges

Standard randomise requires all subjects to have identical dimensions - it can't handle per-subject missing data natively.

#### 6.2 Automatic Slice QC Metrics

```python
def compute_slice_qc_metrics(
    fa_file: Path,
    md_file: Path,
    mask_file: Path
) -> pd.DataFrame:
    """
    Compute QC metrics for each slice.

    Metrics per slice:
    - mean_fa: Mean FA within mask
    - std_fa: FA standard deviation (high = artifacts)
    - mean_md: Mean MD within mask
    - snr: Signal-to-noise ratio estimate
    - coverage: % of mask voxels with signal
    - outlier_voxels: % voxels with FA > 1.0 or < 0
    - edge_artifact_score: High FA at brain boundary

    Returns DataFrame: slice_index × metrics
    """

def flag_bad_slices(
    slice_metrics: pd.DataFrame,
    thresholds: Dict[str, float] = None
) -> Tuple[List[int], List[int], pd.DataFrame]:
    """
    Automatically flag slices as good/bad.

    Default thresholds:
    - snr_min: 5.0 (slices with SNR < 5 are bad)
    - coverage_min: 0.7 (slices with <70% coverage are bad)
    - outlier_max: 0.05 (slices with >5% outlier voxels are bad)
    - fa_std_max: 0.35 (excessive variance suggests artifacts)

    Returns:
    - good_slices: List of valid slice indices
    - bad_slices: List of flagged slice indices
    - flags_df: DataFrame with flag reasons per slice
    """
```

#### 6.3 Manual QC Review Interface

```python
def generate_slice_qc_mosaic(
    fa_file: Path,
    slice_flags: pd.DataFrame,
    output_file: Path
) -> Path:
    """
    Generate visual QC mosaic for manual review.

    Shows each slice with:
    - FA image
    - Automatic QC flag (green=good, red=bad, yellow=borderline)
    - Key metrics overlay

    Allows quick visual verification of automatic flags.
    """

def load_manual_qc_overrides(
    qc_file: Path
) -> Dict[str, List[int]]:
    """
    Load manual QC overrides from CSV/JSON.

    Format:
    subject,session,bad_slices
    sub-Rat102,ses-p60,"3,4,5"
    sub-Rat108,ses-p30,"0,1"

    Returns dict: {subject_session: [bad_slice_indices]}
    """
```

#### 6.4 Validity Mask Creation

```python
def create_subject_validity_mask(
    fa_file: Path,
    good_slices: List[int],
    output_file: Path
) -> Path:
    """
    Create 3D binary mask where valid slices = 1.

    Bad slices are set to 0 (will be excluded from analysis).
    """

def create_group_validity_map(
    subjects: List[SubjectInfo],
    validity_masks: Dict[str, Path],
    output_dir: Path
) -> Tuple[Path, Path, pd.DataFrame]:
    """
    Create group-level validity information.

    Returns:
    - valid_subject_count.nii.gz: Per-voxel count of valid subjects
    - min_threshold_mask.nii.gz: Mask where >= threshold subjects valid
    - coverage_df: Per-voxel coverage statistics
    """
```

#### 6.5 Analysis Strategy for Missing Slices

> **RECOMMENDED: Strategy B (Threshold-Based)** - Preserves maximum data while handling
> bad slices. See "Batch QC Integration" section below for implementation details.

**Strategy A: Conservative (Intersection)** - *Use only for high-stakes analyses*
```python
def create_intersection_mask(
    validity_masks: List[Path],
    output_file: Path
) -> Path:
    """
    Create mask where ALL subjects have valid data.

    Most conservative - no missing data issues.
    But may exclude large regions if even one subject has a bad slice.
    NOT RECOMMENDED: Wastes valid data from subjects with isolated bad slices.
    """
```

**Strategy B: Threshold-Based (RECOMMENDED)**
```python
def create_threshold_mask(
    valid_subject_count: Path,
    n_subjects: int,
    min_fraction: float = 0.8,
    output_file: Path
) -> Path:
    """
    Create mask where >= min_fraction of subjects have valid data.

    RECOMMENDED APPROACH: Preserves data from subjects with isolated bad slices.

    E.g., with 100 subjects and min_fraction=0.8:
    Include voxels where >= 80 subjects have valid data.

    For voxels with fewer valid subjects, those subjects are
    effectively excluded from analysis at that voxel (imputed with group mean).

    Benefits:
    - Subject with 1-2 bad slices still contributes 9-10 valid slices
    - Only truly problematic regions are excluded
    - Maximizes statistical power
    """
```

**Strategy C: Tiered Analysis**
```python
def run_tiered_analysis(
    skeletonised_data: Path,
    valid_subject_count: Path,
    design_mat: Path,
    design_con: Path,
    output_dir: Path,
    thresholds: List[float] = [1.0, 0.9, 0.8]
) -> Dict[float, Path]:
    """
    Run multiple analyses at different validity thresholds.

    Returns results for each threshold:
    - threshold=1.0: Only voxels with 100% valid subjects
    - threshold=0.9: Voxels with >= 90% valid subjects
    - threshold=0.8: Voxels with >= 80% valid subjects

    Report shows which regions gain/lose significance at each level.
    """
```

#### 6.6 Handling Missing Data in Randomise

Since FSL randomise doesn't support per-voxel subject exclusion, we use a **data imputation + masking** approach:

```python
def prepare_data_with_missing(
    skeletonised_4d: Path,
    validity_masks: List[Path],
    min_valid_fraction: float,
    output_dir: Path
) -> Tuple[Path, Path]:
    """
    Prepare 4D data for randomise with missing slice handling.

    Algorithm:
    1. For each voxel, count valid subjects
    2. Create analysis mask: voxels with >= min_valid_fraction subjects
    3. For subjects with invalid data at included voxels:
       - Option A: Set to 0 (they contribute nothing at that voxel)
       - Option B: Set to group mean (neutral imputation)
       - Option C: Set to NaN and use weighted analysis
    4. Run randomise with analysis mask

    The key insight: randomise permutes the ENTIRE design matrix.
    So subjects with imputed values will be permuted, but their
    contribution at imputed voxels is neutral (mean) or zero.

    Returns:
    - prepared_4d: Data ready for randomise
    - analysis_mask: Voxels to analyze
    """
```

#### 6.7 QC Report with Validity Summary

```python
def generate_validity_report(
    subjects: List[SubjectInfo],
    slice_qc_results: Dict[str, pd.DataFrame],
    group_validity_map: Path,
    output_file: Path
) -> Path:
    """
    Generate comprehensive validity report.

    Includes:
    - Per-subject slice validity summary
    - Heatmap: subjects × slices (good/bad/borderline)
    - Group coverage map visualization
    - Recommendations for analysis thresholds
    - List of subjects to consider excluding entirely
    """
```

#### 6.8 Example Workflow

```python
# 1. Run automatic slice QC
for subject in subjects:
    metrics = compute_slice_qc_metrics(subject.fa, subject.md, subject.mask)
    good, bad, flags = flag_bad_slices(metrics)
    generate_slice_qc_mosaic(subject.fa, flags, qc_dir / f"{subject.id}_qc.png")

# 2. Manual review and override
# User reviews mosaics, creates overrides.csv if needed
overrides = load_manual_qc_overrides(qc_dir / "manual_overrides.csv")

# 3. Create validity masks
for subject in subjects:
    good_slices = get_final_good_slices(subject, overrides)
    create_subject_validity_mask(subject.fa_sigma, good_slices,
                                  validity_dir / f"{subject.id}_valid.nii.gz")

# 4. Create group validity map
valid_count, threshold_mask, coverage = create_group_validity_map(
    subjects, validity_masks, analysis_dir
)

# 5. Prepare data and run analysis
prepared_data, analysis_mask = prepare_data_with_missing(
    all_fa_skeletonised, validity_masks, min_valid_fraction=0.8, analysis_dir
)

# 6. Run randomise with validity-aware mask
run_randomise(prepared_data, analysis_mask, design_mat, design_con, ...)

# 7. Generate report showing coverage
generate_validity_report(subjects, slice_qc_results, valid_count, report_file)
```

---

## CLI Interface

### Scripts

**`scripts/run_tbss_slice_qc.py`:**
```bash
# Run automatic slice QC on all subjects
python scripts/run_tbss_slice_qc.py \
    --tbss-dir /study/analysis/tbss \
    --output-dir /study/analysis/tbss/qc \
    --generate-mosaics
```

**`scripts/run_tbss_prepare.py`:**
```bash
python scripts/run_tbss_prepare.py \
    --config configs/default.yaml \
    --output-dir /study/analysis/tbss \
    --cohorts p30 p60 p90 \
    --n-workers 8
```

**`scripts/run_tbss_skeleton.py`:**
```bash
python scripts/run_tbss_skeleton.py \
    --tbss-dir /study/analysis/tbss \
    --fa-threshold 0.3 \
    --remove-exterior-wm \
    --erosion-mm 1.0
```

**`scripts/run_tbss_stats.py`:**
```bash
python scripts/run_tbss_stats.py \
    --tbss-dir /study/analysis/tbss \
    --design-dir /study/designs/dose_response \
    --analysis-name dose_p60 \
    --n-permutations 5000
```

---

## Integration Points

### 1. Transform Registry

Use existing `TransformRegistry` to retrieve and chain transforms:

```python
from neurofaune.utils.transforms import TransformRegistry

registry = TransformRegistry(transforms_dir, subject, cohort)

# Get transforms for FA → SIGMA chain
fa_to_t2w = registry.get_transform('FA', 'T2w')
t2w_to_template = registry.get_transform('T2w', 'template')
template_to_sigma = registry.get_transform('template', 'SIGMA')
```

### 2. Slice Correspondence

For partial-coverage DTI, use slice correspondence to determine valid atlas regions:

```python
from neurofaune.registration import find_slice_correspondence

correspondence = find_slice_correspondence(
    partial_image=fa_file,
    full_image=t2w_file,
    modality='dwi'
)
# Use correspondence.start_slice, correspondence.end_slice
# to mask SIGMA atlas to valid FOV
```

### 3. Tissue Templates

Use cohort-specific tissue probability templates:

```python
# Templates built in Phase 8
wm_prob = templates_dir / f'{cohort}' / 'tpl-BPARat_{cohort}_WM.nii.gz'
gm_prob = templates_dir / f'{cohort}' / 'tpl-BPARat_{cohort}_GM.nii.gz'
```

### 4. neuroaider Integration

Design matrices created externally:

```bash
# In neuroaider
neuroaider design create \
    --participants /study/participants.tsv \
    --formula "dose + age + sex" \
    --contrasts dose_linear dose_high_vs_control \
    --output /study/designs/dose_response/
```

Expected outputs:
- `design.mat` (FSL format)
- `design.con` (FSL format)
- `design.json` (metadata)

### 5. Batch QC Integration (RECOMMENDED APPROACH)

**Use slice-level masking instead of excluding entire subjects** to preserve maximum data.

The batch QC system (`neurofaune.preprocess.qc.batch_summary`) generates exclusion lists at both subject and slice levels:

```
qc/dwi_batch_summary/
├── exclude_subjects.txt          # Subjects with severe issues (exclude entirely)
├── include_subjects.txt          # Good subjects
├── exclusions_by_reason.json     # Categorized: high_motion, extreme_diffusion, low_snr
├── by_cohort/                    # Cohort-specific lists
│   ├── include_p60.txt
│   └── exclude_p60.txt
└── slice_qc/
    ├── bad_slices.tsv            # subject, session, slice_idx
    ├── slice_exclusions.json     # Detailed per-slice flags
    └── slice_quality_heatmap.png # Visual summary
```

**Recommended TBSS workflow with slice masking:**

```python
from neurofaune.preprocess.qc import generate_slice_qc_summary
import pandas as pd

# Step 1: Load slice exclusions from batch QC
slice_exclusions = json.load(open('qc/dwi_batch_summary/slice_qc/slice_exclusions.json'))
bad_slices_df = pd.read_csv('qc/dwi_batch_summary/slice_qc/bad_slices.tsv', sep='\t')

# Step 2: Create per-subject validity masks (3D: 1=valid, 0=invalid slice)
def create_validity_mask_from_batch_qc(
    fa_file: Path,
    subject: str,
    session: str,
    slice_exclusions: dict,
    output_file: Path
) -> Path:
    """Create validity mask using batch QC results."""
    import nibabel as nib
    import numpy as np

    fa_img = nib.load(fa_file)
    mask = np.ones(fa_img.shape[:3], dtype=np.uint8)

    key = f"{subject}_{session}"
    if key in slice_exclusions['exclusions']:
        bad_slices = slice_exclusions['exclusions'][key]['bad_slices']
        for s in bad_slices:
            mask[:, :, s] = 0

    nib.save(nib.Nifti1Image(mask, fa_img.affine), output_file)
    return output_file

# Step 3: Use threshold-based analysis (Strategy B)
# Include voxels where >= 80% of subjects have valid data
# For those voxels, impute missing values with group mean
def prepare_tbss_with_slice_masking(
    all_fa_skeletonised: Path,
    validity_masks: List[Path],
    min_valid_fraction: float = 0.8
) -> Tuple[Path, Path]:
    """
    Prepare TBSS data preserving subjects but masking bad slices.

    - Voxels with < min_valid_fraction valid subjects are excluded from analysis
    - For included voxels, subjects with bad slices get imputed with group mean
    - This preserves data from subjects with isolated bad slices
    """
    # ... implementation in neurofaune/analysis/tbss/prepare.py
```

**Key insight:** A subject with 1-2 bad slices out of 11 can still contribute valid data from the other 9-10 slices. Only exclude entire subjects if:
- Mean FD > threshold (severe global motion)
- Multiple bad slices (>50% of volume)
- Systematic artifacts affecting whole brain

**Overlap analysis (BPA-rat dataset):**
- 18 subjects flagged for high motion
- 26 subjects have ≥1 bad slice
- Only 7 subjects have BOTH issues
- 19 subjects have bad slices but acceptable motion → preserve with slice masking

---

## Testing Strategy

### Unit Tests

```
tests/unit/analysis/
├── test_tbss_prepare.py
├── test_tbss_skeleton.py
├── test_tbss_project.py
└── test_tbss_stats.py
```

### Integration Tests

```
tests/integration/
└── test_tbss_pipeline.py  # End-to-end with synthetic data
```

### Test Data

Create synthetic 3D FA/MD volumes with known structure for testing:
- Known "WM" regions with FA > 0.4
- Known "exterior" regions to verify removal
- Small dataset (5 subjects) for fast testing

---

## Dependencies

### Python Packages
- nipype (FSL interface)
- nibabel (NIfTI I/O)
- numpy, scipy (array operations)
- pandas (cluster tables)
- jinja2 (HTML reports)

### External Tools
- FSL 6.0+ (tbss_*, randomise, cluster)
- ANTs (for registration, already used)

---

## Timeline Estimate

| Phase | Description | Complexity |
|-------|-------------|------------|
| Phase 1 | Data preparation & registration | Medium |
| Phase 2 | WM skeleton with exterior removal | Medium-High |
| Phase 3 | Metric projection | Low |
| Phase 4 | Statistical analysis | Medium |
| Phase 5 | Reporting | Medium |
| Testing | Unit + integration tests | Medium |

---

## Open Questions

1. **FA threshold**: Should we use 0.3 or let it be configurable per-cohort?
2. **Age pooling**: Should p30/p60/p90 have separate skeletons or one pooled?
3. **DKI/NODDI**: Add support for advanced diffusion metrics later?
4. **Tractography**: Integrate with tract-specific analysis (beyond TBSS)?

---

## References

1. Smith et al. (2006). Tract-based spatial statistics. NeuroImage.
2. FSL TBSS documentation: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TBSS
3. SIGMA rat brain atlas documentation
4. neurovrai TBSS implementation (internal)
