# Anatomical Preprocessing Workflow Plan

## Overview

Redesign the anatomical preprocessing workflow to integrate template building and registration in a cohesive pipeline. The workflow supports two modes:

1. **Template-based (recommended)**: Build study-specific templates from a subset of subjects, then register all subjects to template → SIGMA
2. **Direct-to-atlas (optional)**: Skip template building and register directly to SIGMA (less accurate but simpler)

## Prerequisites

### Study-Space Atlas

Before running the pipeline, ensure the SIGMA atlas has been reoriented to match your study's acquisition orientation:

```bash
# This creates {study_root}/atlas/SIGMA_study_space/ with reoriented atlas files
python -c "from neurofaune.templates.slice_registration import setup_study_atlas; setup_study_atlas(...)"
```

The study-space atlas is created once per study and contains:
- `SIGMA_InVivo_Brain_Template.nii.gz` - T2w template (reoriented)
- `SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz` - Parcellation labels (reoriented)
- `SIGMA_InVivo_GM/WM/CSF.nii.gz` - Tissue probability maps (reoriented)
- `atlas_metadata.json` - Reorientation parameters

**IMPORTANT**: Always use the study-space atlas for registration and atlas propagation. The original SIGMA atlas has a different orientation and will produce incorrect results.

## Current State

| Component | Status | Location |
|-----------|--------|----------|
| T2w preprocessing (skull strip, bias, tissue seg) | ✅ Complete | `anat_preprocess.py` |
| Template building | ✅ Complete | `templates/builder.py` |
| Template → SIGMA registration | ✅ Complete | `templates/builder.py` |
| Subject → Template registration | ✅ Complete | `templates/anat_registration.py` |
| Atlas propagation to T2w | ✅ Complete | `templates/anat_registration.py` |
| Registration QC (Dice, correlation, overlays) | ✅ Complete | `templates/registration_qc.py` |
| Template manifest tracking | ✅ Complete | `templates/manifest.py` |
| Batch processing script | ✅ Complete | `scripts/batch_preprocess_anat.py` |

## Proposed Workflow

### Phase 1: Template Building (One-time per cohort)

```
┌─────────────────────────────────────────────────────────────────┐
│  SELECT TEMPLATE SUBJECTS (configurable %, default 20%)         │
│  - Random selection or quality-based                            │
│  - Stratified by cohort (p30, p60, p90)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PREPROCESS TEMPLATE SUBJECTS                                   │
│  1. Bias field correction (N4)                                  │
│  2. Skull stripping (Atropos + BET)                            │
│  3. Tissue segmentation (GM/WM/CSF)                            │
│  4. Intensity normalization                                     │
│  5. Save to derivatives/ + record in manifest                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  BUILD COHORT TEMPLATE                                          │
│  - ANTs buildtemplateparallel.sh                               │
│  - Creates: tpl-{study}_{cohort}_T2w.nii.gz                    │
│  - Saves subject→template transforms                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  REGISTER TEMPLATE → SIGMA                                      │
│  - ANTs SyN registration                                        │
│  - Saves: tpl-to-SIGMA_0GenericAffine.mat                      │
│  - Saves: tpl-to-SIGMA_1Warp.nii.gz                            │
│  - Saves: tpl-to-SIGMA_1InverseWarp.nii.gz                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  GENERATE TEMPLATE MANIFEST                                     │
│  - List of subjects used                                        │
│  - Preprocessing parameters                                     │
│  - Transform locations                                          │
│  - Timestamp                                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 2: Full Subject Processing (All subjects)

```
┌─────────────────────────────────────────────────────────────────┐
│  FOR EACH SUBJECT                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  CHECK: Was subject used in template building?                  │
│  - Read template manifest                                       │
│  - If YES: Skip to Step 5 (registration)                       │
│  - If NO: Continue to Step 1                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            │ NOT in template                   │ IN template
            ▼                                   ▼
┌───────────────────────────┐     ┌───────────────────────────┐
│  1. Bias field correction │     │  Skip preprocessing       │
│  2. Skull stripping       │     │  (already done)           │
│  3. Tissue segmentation   │     │                           │
│  4. Intensity norm        │     │                           │
└───────────────────────────┘     └───────────────────────────┘
            │                                   │
            └─────────────────┬─────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. REGISTER T2w → COHORT TEMPLATE                              │
│     - ANTs SyN registration                                     │
│     - Save: T2w_to_template_0GenericAffine.mat                 │
│     - Save: T2w_to_template_1Warp.nii.gz                       │
│     - Save: T2w_to_template_1InverseWarp.nii.gz                │
│                                                                 │
│     NOTE: Template subjects already have these from building!   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. PROPAGATE SIGMA LABELS → T2w                                │
│     - Use STUDY-SPACE atlas (already reoriented to match study) │
│     - Located at: {study_root}/atlas/SIGMA_study_space/         │
│     - Use inverse transforms                                    │
│     - Chain: SIGMA (study-space) → Template → Subject T2w      │
│     - Save: sub-*_space-T2w_atlas-SIGMA.nii.gz                 │
│                                                                 │
│     IMPORTANT: Do NOT use original SIGMA atlas - use the        │
│     study-space version created by setup_study_atlas()          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. WARP T2w → SIGMA SPACE (optional)                          │
│     - For group analysis in standard space                     │
│     - Save: sub-*_space-SIGMA_T2w.nii.gz                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  8. GENERATE QC                                                 │
│     - Registration overlay figures                              │
│     - Tissue segmentation QC                                    │
│     - Metrics JSON                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Alternative: Direct-to-SIGMA Mode (No Template)

```
┌─────────────────────────────────────────────────────────────────┐
│  FOR EACH SUBJECT (direct mode)                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1-4. Standard preprocessing (same as above)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. REGISTER T2w → SIGMA DIRECTLY                               │
│     - Skip template, register each subject to SIGMA             │
│     - Less accurate (no study-specific intermediate)            │
│     - Faster (no template building needed)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6-8. Same as template mode                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Template Builder Enhancements

**File**: `scripts/build_anat_templates.py` (enhance existing)

#### 1.1 Subject Selection Module

```python
def select_template_subjects(
    bids_root: Path,
    cohort: str,
    fraction: float = 0.20,
    min_subjects: int = 8,
    max_subjects: int = 15,
    selection_method: str = 'random',  # or 'quality'
    seed: int = 42
) -> List[str]:
    """
    Select subjects for template building.

    Parameters
    ----------
    fraction : float
        Fraction of total subjects to use (default 20%)
    min_subjects : int
        Minimum subjects regardless of fraction
    max_subjects : int
        Maximum subjects (template quality plateaus)
    selection_method : str
        'random' or 'quality' (based on QC metrics)
    """
```

#### 1.2 Template Manifest

```python
@dataclass
class TemplateManifest:
    """Track template building metadata."""
    cohort: str
    subjects_used: List[str]
    sessions_used: List[str]
    n_subjects: int
    fraction_used: float
    template_path: Path
    sigma_transforms: Dict[str, Path]
    subject_transforms: Dict[str, Dict[str, Path]]
    preprocessing_params: Dict[str, Any]
    created_at: str
    neurofaune_version: str
```

Save as: `templates/anat/{cohort}/template_manifest.json`

#### 1.3 Enhanced Template Builder

```python
def build_cohort_template(
    config: Dict,
    cohort: str,
    output_dir: Path,
    subject_fraction: float = 0.20,
    preprocess_subjects: bool = True,
    register_to_sigma: bool = True,
    n_cores: int = 8
) -> TemplateManifest:
    """
    Build cohort template with integrated preprocessing.

    1. Select subjects
    2. Preprocess each (if needed)
    3. Build template
    4. Register to SIGMA
    5. Save manifest
    """
```

### Phase 2: Anatomical Registration Module

**File**: `neurofaune/templates/anat_registration.py` (new)

#### 2.1 Subject-to-Template Registration

```python
def register_anat_to_template(
    t2w_file: Path,
    template_file: Path,
    output_dir: Path,
    subject: str,
    session: str,
    mask_file: Optional[Path] = None,
    n_cores: int = 4
) -> Dict[str, Path]:
    """
    Register preprocessed T2w to cohort template.

    Returns dict with:
    - composite_transform
    - inverse_composite_transform
    - warped_t2w
    """
```

#### 2.2 Atlas Propagation to T2w

```python
def propagate_atlas_to_anat(
    atlas_path: Path,
    t2w_reference: Path,
    transforms_dir: Path,
    templates_dir: Path,
    subject: str,
    session: str,
    output_path: Path
) -> Path:
    """
    Propagate SIGMA atlas to T2w space.

    Transform chain (inverse):
        SIGMA → Template → Subject T2w

    Uses:
        - tpl-to-SIGMA_1InverseWarp.nii.gz
        - [tpl-to-SIGMA_0GenericAffine.mat, 1]
        - T2w_to_template_1InverseWarp.nii.gz
        - [T2w_to_template_0GenericAffine.mat, 1]
    """
```

#### 2.3 Direct-to-SIGMA Registration (Optional Mode)

```python
def register_anat_to_sigma_direct(
    t2w_file: Path,
    sigma_template: Path,
    output_dir: Path,
    subject: str,
    session: str,
    n_cores: int = 4
) -> Dict[str, Path]:
    """
    Register T2w directly to SIGMA (no study template).

    WARNING: Less accurate than template-based registration.
    Use only when template building is not feasible.
    """
```

### Phase 3: Integrated Preprocessing Workflow

**File**: `neurofaune/preprocess/workflows/anat_preprocess.py` (modify)

#### 3.1 Add Registration Step

```python
def run_anatomical_preprocessing(
    config: Dict[str, Any],
    subject: str,
    session: str,
    output_dir: Path,
    transform_registry: TransformRegistry,
    t2w_file: Optional[Path] = None,
    # NEW PARAMETERS
    register_to_template: bool = True,
    template_manifest: Optional[Path] = None,
    direct_to_sigma: bool = False,
    propagate_atlas: bool = True,
    warp_to_sigma: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Enhanced workflow with registration.

    Parameters
    ----------
    register_to_template : bool
        Register to cohort template (default True)
    template_manifest : Path
        Path to template manifest (auto-detected if None)
    direct_to_sigma : bool
        Skip template, register directly to SIGMA
    propagate_atlas : bool
        Propagate SIGMA labels to T2w space
    warp_to_sigma : bool
        Warp T2w to SIGMA space for group analysis
    """
```

#### 3.2 Smart Resume for Template Subjects

```python
def check_template_subject(
    subject: str,
    session: str,
    template_manifest: TemplateManifest
) -> Tuple[bool, Optional[Dict[str, Path]]]:
    """
    Check if subject was used in template building.

    Returns
    -------
    (is_template_subject, existing_transforms)

    If True, returns paths to existing:
    - Preprocessed T2w
    - Brain mask
    - Tissue segmentations
    - Subject→Template transforms
    """
```

### Phase 4: Batch Processing Script

**File**: `scripts/batch_preprocess_anat.py` (new)

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)

    # Template options
    parser.add_argument('--build-template', action='store_true',
                        help='Build template before processing')
    parser.add_argument('--template-fraction', type=float, default=0.20,
                        help='Fraction of subjects for template (default: 0.20)')
    parser.add_argument('--no-template', action='store_true',
                        help='Skip template, register directly to SIGMA')

    # Processing options
    parser.add_argument('--subjects', nargs='+', help='Specific subjects')
    parser.add_argument('--cohorts', nargs='+', default=['p30', 'p60', 'p90'])
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--n-cores', type=int, default=4)
```

---

## Output Structure

### With Template (Recommended)

```
{study_root}/
├── templates/anat/{cohort}/
│   ├── tpl-{study}_{cohort}_T2w.nii.gz           # Cohort template
│   ├── template_manifest.json                     # Subjects used, params
│   ├── transforms/
│   │   ├── tpl-to-SIGMA_0GenericAffine.mat
│   │   ├── tpl-to-SIGMA_1Warp.nii.gz
│   │   └── tpl-to-SIGMA_1InverseWarp.nii.gz
│   └── qc/
│       └── template_qc.png
│
├── transforms/sub-{subject}/{session}/
│   ├── T2w_to_template_0GenericAffine.mat
│   ├── T2w_to_template_1Warp.nii.gz
│   └── T2w_to_template_1InverseWarp.nii.gz
│
└── derivatives/sub-{subject}/{session}/anat/
    ├── sub-*_desc-preproc_T2w.nii.gz             # Preprocessed T2w
    ├── sub-*_desc-brain_mask.nii.gz              # Brain mask
    ├── sub-*_dseg.nii.gz                         # Tissue segmentation
    ├── sub-*_label-{GM,WM,CSF}_probseg.nii.gz    # Tissue probabilities
    ├── sub-*_space-template_T2w.nii.gz           # T2w in template space
    ├── sub-*_space-SIGMA_T2w.nii.gz              # T2w in SIGMA space (optional)
    └── sub-*_space-T2w_atlas-SIGMA_dseg.nii.gz   # SIGMA labels in T2w space
```

### Without Template (Direct Mode)

```
{study_root}/
├── transforms/sub-{subject}/{session}/
│   ├── T2w_to_SIGMA_0GenericAffine.mat           # Direct registration
│   ├── T2w_to_SIGMA_1Warp.nii.gz
│   └── T2w_to_SIGMA_1InverseWarp.nii.gz
│
└── derivatives/sub-{subject}/{session}/anat/
    ├── (same as above, minus template-space outputs)
    └── sub-*_space-T2w_atlas-SIGMA_dseg.nii.gz   # SIGMA labels in T2w space
```

---

## Configuration

Add to `configs/default.yaml`:

```yaml
anatomical:
  # Preprocessing
  bias_correction:
    enabled: true
    n_iterations: [50, 50, 30, 20]
    shrink_factor: 3

  skull_stripping:
    method: 'atropos'  # 'atropos', 'bet', or 'ants'
    bet_frac: 0.3

  # Template building
  template:
    enabled: true
    subject_fraction: 0.20
    min_subjects: 8
    max_subjects: 15
    selection_method: 'random'  # or 'quality'

  # Registration
  registration:
    use_template: true           # false = direct to SIGMA
    transform_type: 'SyN'
    n_cores: 4
    propagate_atlas: true
    warp_to_sigma: false         # optional, for group analysis
```

---

## Implementation Order

### Week 1: Template Builder Enhancements
1. [ ] Create `TemplateManifest` dataclass
2. [ ] Implement `select_template_subjects()`
3. [ ] Enhance `build_cohort_template()` with manifest generation
4. [ ] Test with existing templates

### Week 2: Registration Module
1. [ ] Create `neurofaune/templates/anat_registration.py`
2. [ ] Implement `register_anat_to_template()`
3. [ ] Implement `propagate_atlas_to_anat()`
4. [ ] Implement `register_anat_to_sigma_direct()` (optional mode)
5. [ ] Add unit tests

### Week 3: Workflow Integration
1. [ ] Add registration to `anat_preprocess.py`
2. [ ] Implement `check_template_subject()` for smart resume
3. [ ] Add configuration options
4. [ ] Test end-to-end workflow

### Week 4: Batch Processing & QC
1. [ ] Create `scripts/batch_preprocess_anat.py`
2. [ ] Add registration QC visualizations
3. [ ] Test on full dataset
4. [ ] Update documentation

---

## Open Questions

1. **Template subject selection**: Random vs. quality-based? Need QC metrics first.

2. **Existing preprocessed data**: ~119 T2w already preprocessed. Should we:
   - Use existing preprocessing and just add registration?
   - Reprocess with consistent parameters?

3. **Transform registry integration**: Currently transforms saved to `transforms/` directory. Should we integrate with `TransformRegistry` class?

4. **QC thresholds**: What metrics indicate failed registration? Dice with template mask? Correlation?

---

## References

- Current template building: `scripts/build_templates.py`
- Registration utilities: `neurofaune/templates/registration.py`
- Transform registry: `neurofaune/utils/transforms.py`
- DTI atlas propagation: `propagate_atlas_to_dwi()` in `registration.py`
