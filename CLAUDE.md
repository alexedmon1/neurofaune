# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goals & Current Status

### Overall Objectives
1. **Rodent-specific MRI preprocessing pipeline** for T2w anatomical, DTI/DWI, and resting-state fMRI data
2. **Config-driven architecture** using YAML for all processing parameters
3. **Standardized directory hierarchy** (neurovrai-compatible): `{study_root}/derivatives/{subject}/{modality}/`
4. **SIGMA atlas integration** with slice-specific registration capabilities
5. **Transform registry** to avoid redundant computation across workflows
6. **Comprehensive quality control** with automated QC for all modalities

### Current Implementation Status

**✅ Phase 1 Completed (Foundation)**:
- **Project structure**: Complete directory hierarchy with modular organization
- **Configuration system**: YAML-based config with variable substitution (`neurofaune/config.py`)
- **Transform registry**: ANTs-based registry with slice-specific metadata (`neurofaune/utils/transforms.py`)
- **Default configuration**: Rodent-optimized parameters (`configs/default.yaml`)
- **Documentation**: README, CLAUDE.md, ARCHITECTURE.md
- **Package setup**: `pyproject.toml` with all dependencies

**🔄 In Progress**:
- **Phase 2**: Atlas management module (SIGMA integration, slice extraction)

**📋 Planned**:
- **Phase 3**: Anatomical preprocessing (T2w)
- **Phase 4**: Diffusion preprocessing (DWI/DTI)
- **Phase 5**: Functional preprocessing (fMRI)
- **Phase 6**: Advanced modalities (MSME, MTR, MRS)
- **Phase 7**: CLI and batch processing
- **Phase 8**: Testing and validation
- **Phase 9**: Comprehensive documentation

---

## Project Overview

This repository contains a rodent-specific MRI preprocessing pipeline built with Nipype and ANTs. The project processes multiple MRI modalities (T2w anatomical, diffusion DWI, resting-state fMRI, spectroscopy) from Bruker 7T scanner to analysis-ready formats.

**Key Innovation**: Slice-specific atlas registration to improve accuracy for modalities that only cover specific brain regions (e.g., 11 hippocampal slices for DTI).

---

## Development Environment

**Python Version**: 3.10+
**Package Manager**: pip (standard installation)
**Virtual Environment**: Recommended (venv or conda)

### Key Dependencies
- **nipype** (1.10.0+): Workflow engine for FSL/ANTs interfaces
- **nibabel** (5.3.2+): NIfTI file I/O
- **pydicom** (3.0.1+): DICOM reading/parsing
- **ANTs** (2.3+): Required system dependency for registration and segmentation
- **FSL** (6.0+): Required for BET, eddy, ICA-AROMA
- **FSL-MRS** (optional): Spectroscopy processing
- **pandas**, **numpy**, **scipy**: Data analysis and numerical operations

### Installation
```bash
# Clone repository
git clone <repo_url>
cd neurofaune

# Install in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[all]"
```

---

## Architecture

The codebase is organized by function, with clear separation of concerns:

### Module Structure

- **`neurofaune/`**: Main package
  - **`config.py`**: YAML configuration loading with variable substitution
  - **`preprocess/`**: Preprocessing workflows and utilities
    - `workflows/`: Main preprocessing pipelines (anat, dwi, func, etc.)
    - `utils/`: Helper functions (registration, normalization, file discovery)
    - `qc/`: Quality control modules (organized by modality)
  - **`atlas/`**: Atlas management (SIGMA integration, slice extraction)
  - **`utils/`**: Shared utilities
    - **`transforms.py`**: Transform registry for spatial transformations

- **`docs/`**: Documentation
  - Implementation guides, API documentation, tutorials

- **`examples/`**: Example scripts demonstrating usage

- **`configs/`**: Configuration templates
  - `default.yaml`: Default rodent-optimized parameters

- **`tests/`**: Unit and integration tests
  - `unit/`: Component-level tests
  - `integration/`: End-to-end workflow tests

### Workflow Pattern

**Modern workflows** use functional pattern with standardized directory structure:

```python
from pathlib import Path
from neurofaune.config import load_config
from neurofaune.utils.transforms import create_transform_registry

# Load configuration
config = load_config(Path('config.yaml'))

# Create transform registry
registry = create_transform_registry(config, subject='sub-001', cohort='p60')

# Run workflow
from neurofaune.preprocess.workflows.anat_preprocess import run_anatomical_preprocessing

results = run_anatomical_preprocessing(
    config=config,
    subject='sub-001',
    t2w_file=Path('T2w.nii.gz'),
    output_dir=Path('/study/derivatives'),
    transform_registry=registry
)
```

**Key characteristics**:
- Function-based workflows (not class-based)
- Configuration-driven (YAML)
- Transform registry integration from the start
- Immediate QC integration (QC runs during workflow, not after)
- Use `pathlib.Path` for all paths
- Type hints and comprehensive docstrings

---

## Configuration

The pipeline uses a YAML-based configuration system as documented in `README.md`.

### Configuration File Format

Study-specific configuration (`config.yaml`):

```yaml
# Study information
study:
  name: "Rodent MRI Study"
  code: "STUDY001"
  species: "rat"
  strain: "Sprague-Dawley"

# Directory structure (neurovrai-compatible)
paths:
  study_root: "/data/study"
  derivatives: "${paths.study_root}/derivatives"
  transforms: "${paths.study_root}/transforms"
  qc: "${paths.study_root}/qc"
  work: "${paths.study_root}/work"

# Atlas configuration
atlas:
  name: "SIGMA"
  base_path: "/data/atlases/SIGMA"
  slice_definitions:
    dwi:
      start: 15
      end: 25  # 11 hippocampal slices
    func:
      start: 10
      end: 35  # Broader cortical coverage
    anat:
      start: 0
      end: -1  # Full atlas

# Age cohorts
cohorts:
  p30:
    age_days: 30
  p60:
    age_days: 60
  p90:
    age_days: 90

# Execution
execution:
  plugin: "MultiProc"
  n_procs: 6

# Modality-specific parameters
anatomical:
  bet:
    frac: 0.3  # Rodent-optimized
  registration:
    method: "ants"

diffusion:
  eddy:
    use_cuda: true
  normalization:
    target_space: "SIGMA"

functional:
  smoothing:
    fwhm: 0.5  # Smaller for rodent brains
```

### Variable Substitution

The config loader supports:
- **Config references**: `${paths.study_root}` → other config values
- **Environment variables**: `${HOME}` → environment variables

### Using Configuration

```python
from neurofaune.config import load_config, get_config_value
from pathlib import Path

# Load config
config = load_config(Path('config.yaml'))

# Access values
study_root = Path(config['paths']['study_root'])
n_procs = config['execution']['n_procs']

# Use dot notation for nested values
bet_frac = get_config_value(config, 'anatomical.bet.frac', default=0.3)
```

---

## Directory Structure

All preprocessing workflows use a standardized directory hierarchy **compatible with neurovrai**:

```
{study_root}/
├── raw/
│   ├── bruker/                    # Raw Bruker data
│   └── bids/                      # Converted BIDS data
│       └── sub-{subject}/
│           ├── anat/
│           ├── dwi/
│           └── func/
├── derivatives/                   # Preprocessed outputs
│   └── sub-{subject}/
│       ├── anat/
│       │   ├── sub-{subject}_desc-preproc_T2w.nii.gz
│       │   ├── sub-{subject}_desc-brain_mask.nii.gz
│       │   ├── sub-{subject}_label-GM_probseg.nii.gz
│       │   ├── sub-{subject}_label-WM_probseg.nii.gz
│       │   ├── sub-{subject}_label-CSF_probseg.nii.gz
│       │   └── sub-{subject}_space-SIGMA_T2w.nii.gz
│       ├── dwi/
│       │   ├── sub-{subject}_desc-preproc_dwi.nii.gz
│       │   ├── sub-{subject}_FA.nii.gz
│       │   ├── sub-{subject}_MD.nii.gz
│       │   └── sub-{subject}_space-SIGMA_FA.nii.gz
│       └── func/
│           ├── sub-{subject}_desc-preproc_bold.nii.gz
│           ├── sub-{subject}_desc-confounds_timeseries.tsv
│           └── sub-{subject}_space-SIGMA_bold.nii.gz
├── transforms/                    # Transform registry
│   └── sub-{subject}/
│       ├── T2w_to_SIGMA_composite.h5
│       ├── FA_to_SIGMA_dwi_composite.h5
│       └── transforms.json        # Registry metadata
├── qc/                            # Quality control reports
│   └── sub-{subject}/
│       ├── anat/
│       │   ├── sub-{subject}_desc-skullstrip_qc.html
│       │   └── figures/
│       ├── dwi/
│       │   ├── sub-{subject}_desc-motion_qc.html
│       │   └── figures/
│       └── func/
│           ├── sub-{subject}_desc-motion_qc.html
│           └── figures/
└── work/                          # Temporary Nipype files (can be deleted)
    └── sub-{subject}/{workflow}/
```

**Key principles:**
- All workflows accept `output_dir` as the **study root** (not derivatives directory)
- Workflows create `derivatives/sub-{subject}/{modality}/` structure
- Transform registry is centralized in `transforms/`
- QC reports follow same hierarchy as derivatives
- Work directory is temporary and can be deleted after successful completion

---

## Transform Registry

The transform registry is a **critical component** that avoids redundant computation:

### Key Features
- **Centralized storage**: All transforms in `{study_root}/transforms/sub-{subject}/`
- **ANTs format**: Uses ANTs composite transforms (.h5) as primary format
- **Slice metadata**: Tracks slice ranges for slice-specific registrations
- **Reuse across workflows**: Functional workflow reuses anatomical→atlas transform
- **Modality-specific**: Supports separate transforms per modality (e.g., DTI 11-slice atlas)

### Usage Pattern

```python
from neurofaune.utils.transforms import create_transform_registry

# Create registry
registry = create_transform_registry(config, subject='sub-001', cohort='p60')

# Save transform after computing
registry.save_ants_composite_transform(
    composite_file=Path('ants_Composite.h5'),
    source_space='T2w',
    target_space='SIGMA',
    reference=Path('SIGMA_template.nii.gz'),
    source_image=Path('T2w_brain.nii.gz')
)

# Save slice-specific transform
registry.save_ants_composite_transform(
    composite_file=Path('fa_to_sigma_Composite.h5'),
    source_space='FA',
    target_space='SIGMA',
    modality='dwi',  # Modality-specific
    slice_range=(15, 25)  # 11 hippocampal slices
)

# Retrieve for reuse
transform = registry.get_ants_composite_transform('T2w', 'SIGMA')
dwi_transform = registry.get_ants_composite_transform('FA', 'SIGMA', modality='dwi')

# Check if transform exists before computing
if not registry.has_transform('T2w', 'SIGMA'):
    # Compute transform
    pass
else:
    # Reuse existing
    transform = registry.get_ants_composite_transform('T2w', 'SIGMA')
```

### Transform Naming Convention

Transforms follow BIDS-like naming:
- `{source}_to_{target}_composite.h5` (e.g., `T2w_to_SIGMA_composite.h5`)
- `{source}_to_{target}_{modality}_composite.h5` (e.g., `FA_to_SIGMA_dwi_composite.h5`)
- Metadata stored in `transforms.json`

---

## Quality Control Integration

**Critical**: QC must be developed **in parallel** with each workflow, not after.

### QC Development Pattern

```python
# neurofaune/preprocess/qc/anat/skull_strip_qc.py
def generate_skull_strip_qc(
    subject: str,
    t2w_file: Path,
    brain_file: Path,
    mask_file: Path,
    output_dir: Path
) -> Path:
    """
    Generate skull stripping QC report.

    Returns
    -------
    Path
        Path to HTML report
    """
    # Generate visualizations
    # Create HTML report with matplotlib/seaborn
    # Return report path
    pass

# Integrate into workflow
from neurofaune.preprocess.qc.anat import skull_strip_qc

# After skull stripping step
qc_report = skull_strip_qc.generate_skull_strip_qc(
    subject=subject,
    t2w_file=t2w_file,
    brain_file=brain_file,
    mask_file=mask_file,
    output_dir=qc_dir
)
```

### QC Module Structure

```
neurofaune/preprocess/qc/
├── anat/
│   ├── __init__.py
│   ├── skull_strip_qc.py
│   ├── segmentation_qc.py
│   └── registration_qc.py
├── dwi/
│   ├── __init__.py
│   ├── motion_qc.py
│   ├── eddy_qc.py
│   ├── dti_qc.py
│   └── registration_qc.py
└── func/
    ├── __init__.py
    ├── motion_qc.py
    ├── confounds_qc.py
    └── carpet_qc.py
```

---

## Key Design Decisions

### 1. ANTs for All Registrations
- **Rationale**: ANTs provides better registration quality for rodent brains than FSL
- **Implementation**: All workflows use ANTs composite transforms (.h5 format)
- **Consistency**: Same registration framework across all modalities

### 2. Slice-Specific Atlas Registration
- **Challenge**: DTI often only covers 11 hippocampal slices, not full brain
- **Solution**: Extract relevant atlas slices before registration
- **Benefits**: Improved registration accuracy, reduced interference from distant anatomy
- **Flexibility**: Different slice ranges per modality (DTI: 11 slices, fMRI: broader)

### 3. T2w as Primary Anatomical Modality
- **Rationale**: T2w provides better contrast for rodent brain anatomy than T1w
- **Impact**: All registration workflows use T2w as anatomical reference
- **Atlas**: SIGMA rat brain atlas is T2w-based

### 4. Transform Reuse Strategy
- **Principle**: Compute each transform only once, reuse everywhere
- **Example**: Functional workflow reuses T2w→SIGMA transform from anatomical preprocessing
- **Storage**: Centralized transform registry with metadata tracking

### 5. Age Cohort Support
- **Motivation**: Many rodent studies are developmental (p30, p60, p90)
- **Implementation**: Cohort metadata tracked in transform registry and config
- **Future**: Age-specific atlases if needed

---

## Development Workflow

### Adding a New Preprocessing Workflow

1. **Create workflow function** in `neurofaune/preprocess/workflows/`
2. **Implement QC modules** in `neurofaune/preprocess/qc/{modality}/`
3. **Integrate transform registry** (save and reuse transforms)
4. **Add utilities** to `neurofaune/preprocess/utils/` if needed
5. **Write unit tests** in `tests/unit/`
6. **Write integration test** in `tests/integration/`
7. **Update CLI** in `neurofaune/preprocess/cli.py`
8. **Document** in `docs/` and update README

### Code Style

- **Type hints**: Use for all function parameters and return values
- **Docstrings**: Google style with Parameters, Returns, Examples sections
- **Pathlib**: Use `Path` objects, not strings
- **Configuration-driven**: All parameters from config, not hardcoded
- **Error handling**: Comprehensive try-except with informative messages
- **Logging**: Use Python logging module, not print statements

### Example Workflow Structure

```python
def run_anatomical_preprocessing(
    config: Dict[str, Any],
    subject: str,
    t2w_file: Path,
    output_dir: Path,
    transform_registry: TransformRegistry,
    work_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run anatomical T2w preprocessing workflow.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    subject : str
        Subject identifier
    t2w_file : Path
        Input T2w image
    output_dir : Path
        Study root directory (will create derivatives/sub-{subject}/anat/)
    transform_registry : TransformRegistry
        Transform registry for saving spatial transforms
    work_dir : Path, optional
        Working directory (defaults to output_dir/work/sub-{subject}/anat_preproc)

    Returns
    -------
    dict
        Dictionary with output file paths and QC reports

    Examples
    --------
    >>> from neurofaune.config import load_config
    >>> from neurofaune.utils.transforms import create_transform_registry
    >>>
    >>> config = load_config(Path('config.yaml'))
    >>> registry = create_transform_registry(config, 'sub-001', cohort='p60')
    >>>
    >>> results = run_anatomical_preprocessing(
    ...     config=config,
    ...     subject='sub-001',
    ...     t2w_file=Path('T2w.nii.gz'),
    ...     output_dir=Path('/study'),
    ...     transform_registry=registry
    ... )
    """
    # Setup directories
    derivatives_dir = output_dir / 'derivatives' / f'sub-{subject}' / 'anat'
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    qc_dir = output_dir / 'qc' / f'sub-{subject}' / 'anat'
    qc_dir.mkdir(parents=True, exist_ok=True)

    if work_dir is None:
        work_dir = output_dir / 'work' / f'sub-{subject}' / 'anat_preproc'
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build Nipype workflow
    # ... workflow implementation ...

    # Execute workflow
    wf.run()

    # Save transforms to registry
    transform_registry.save_ants_composite_transform(
        composite_file=...,
        source_space='T2w',
        target_space='SIGMA',
        reference=...,
        source_image=t2w_file
    )

    # Generate QC reports
    from neurofaune.preprocess.qc.anat import skull_strip_qc
    qc_report = skull_strip_qc.generate_skull_strip_qc(...)

    # Return results
    return {
        'brain': brain_file,
        'mask': mask_file,
        'segmentation': seg_dir,
        'transforms': {...},
        'qc_reports': [qc_report, ...]
    }
```

---

## Testing

### Unit Tests
Test individual components in isolation:
- Configuration loading
- Transform registry operations
- Utility functions
- QC report generation

### Integration Tests
Test complete workflows end-to-end:
- Full anatomical preprocessing pipeline
- DTI preprocessing with slice-specific atlas
- Transform reuse across modalities

### Test Data
- Small test datasets in `tests/data/`
- Synthetic data for unit tests
- Real (anonymized) rodent data for integration tests

---

## Documentation

### Documentation Structure

```
docs/
├── README.md                      # Documentation index
├── ARCHITECTURE.md                # Four-part architecture overview
├── ATLAS_GUIDE.md                 # SIGMA atlas usage and slice extraction
├── TRANSFORMS_GUIDE.md            # Transform registry patterns
├── configuration.md               # Configuration file reference
├── workflows.md                   # Workflow documentation
├── cli.md                         # CLI reference
├── implementation/                # Technical implementation details
└── archive/                       # Outdated documentation
```

### Updating Documentation

When implementing features:
1. **User-facing**: Update README.md with new capabilities
2. **Developer**: Update CLAUDE.md with implementation details
3. **Technical**: Add detailed docs to `docs/implementation/`
4. **API**: Comprehensive docstrings in code

---

## Common Issues and Solutions

### Issue: Transform not found
**Cause**: Transform registry not passed to workflow, or transform not saved
**Solution**: Always pass `transform_registry` to workflows and save transforms immediately after computation

### Issue: Slice-specific registration fails
**Cause**: Atlas slice range doesn't match subject data
**Solution**: Implement slice interpolation in atlas module to handle mismatched geometries

### Issue: QC reports empty
**Cause**: QC not integrated into workflow
**Solution**: Call QC functions immediately after each preprocessing step

### Issue: Wrong directory structure
**Cause**: Creating derivatives directly instead of using config paths
**Solution**: Always use `config['paths']['derivatives']` and follow neurovrai hierarchy

---

## Future Enhancements

### Short-term (Next 3 months)
- [ ] Complete Phase 2 (Atlas module)
- [ ] Implement anatomical preprocessing (Phase 3)
- [ ] Implement diffusion preprocessing (Phase 4)
- [ ] Develop comprehensive test suite

### Medium-term (6 months)
- [ ] Functional preprocessing (Phase 5)
- [ ] Advanced modalities (MSME, MTR, MRS)
- [ ] CLI implementation
- [ ] Batch processing with HPC integration

### Long-term (1 year+)
- [ ] Group-level analysis module
- [ ] Connectome analysis module
- [ ] Web-based QC interface
- [ ] Containerization (Docker/Singularity)
- [ ] BIDS compliance validation

---

## Project Organization

Keep the repository clean and organized:

**Documentation**:
- Root: README.md, CLAUDE.md only
- `docs/`: All technical documentation

**Code**:
- `neurofaune/`: All production code
- No scripts in root except `setup.py`

**Data** (not in repository):
- Study data goes to `{study_root}/` (configured per study)
- Test data in `tests/data/` (small, anonymized)

**Logs**:
- Not committed to repository
- Generated in `work/` directories

---

## Contact

For questions during development:
1. Check this file (CLAUDE.md) first
2. Review README.md for user-facing information
3. Check relevant documentation in `docs/`
4. Review neurovrai implementation for reference patterns
