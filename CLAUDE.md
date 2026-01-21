# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

**Package manager**: Always use `uv` for Python operations, never `pip` directly.

```bash
# Install in development mode
uv pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run single test file
uv run pytest tests/unit/test_atlas.py -v

# Run specific test
uv run pytest tests/unit/test_atlas.py::TestAtlasManager::test_initialization -v

# Run tests with markers
uv run pytest -m "not slow"          # Skip slow tests
uv run pytest -m integration          # Integration tests only

# Linting and formatting
uv run ruff check .                   # Check for issues
uv run ruff check --fix .             # Auto-fix issues
uv run black .                        # Format code

# Type checking
uv run mypy neurofaune/
```

**Batch processing scripts** (in `scripts/`):
```bash
# Phase 1: Initialize (one-time per study)
uv run python scripts/init_study.py /path/to/study --name "Study" --bids-root /path/to/bids
uv run python scripts/batch_preprocess_for_templates.py /path/to/bids /path/to/output
uv run python scripts/build_templates.py --config config.yaml --cohort p60

# Phase 2: Preprocessing (all subjects)
uv run python scripts/batch_preprocess_anat.py --config config.yaml
uv run python scripts/batch_preprocess_dwi.py --bids-root /path/to/bids --output-root /path/to/output
uv run python scripts/batch_preprocess_func.py /path/to/bids /path/to/output
```

## Architecture Overview

Neurofaune is a rodent-specific MRI preprocessing pipeline built on Nipype/ANTs.

### Module Structure
```
neurofaune/
├── config.py                 # YAML config loader with variable substitution
├── atlas/                    # SIGMA atlas management
│   ├── manager.py            # AtlasManager class for atlas access
│   └── slice_extraction.py   # Modality-specific slice extraction
├── preprocess/
│   ├── workflows/            # Main preprocessing pipelines
│   │   ├── anat_preprocess.py   # T2w anatomical
│   │   ├── dwi_preprocess.py    # DTI/DWI diffusion
│   │   ├── func_preprocess.py   # fMRI functional
│   │   └── msme_preprocess.py   # Multi-echo T2 mapping
│   ├── qc/                   # Quality control per modality
│   │   ├── anat/             # Anatomical QC
│   │   ├── dwi/              # DTI QC (eddy, tensor metrics)
│   │   ├── func/             # Functional QC (motion, confounds)
│   │   └── msme/             # MSME T2 mapping QC
│   └── utils/                # Preprocessing utilities
│       └── func/             # Functional-specific utils (ICA, aCompCor, skull stripping)
├── registration/             # Cross-modal registration utilities
│   ├── slice_correspondence.py  # Partial-to-full volume slice matching
│   └── qc_visualization.py      # Registration QC figures (checkerboard, edge overlay)
├── templates/                # Template building and registration
│   ├── builder.py            # ANTs template construction
│   ├── registration.py       # Subject-to-template, atlas propagation
│   └── slice_registration.py # Study-space atlas setup
└── utils/
    ├── transforms.py         # Transform registry system
    ├── exclusion.py          # Subject exclusion tracking
    ├── orientation.py        # Image orientation utilities
    └── select_anatomical.py  # Automatic T2w scan selection
```

Key architectural decisions:

### Transform Registry Pattern
All workflows share transforms through a centralized registry to avoid redundant computation:
- Location: `{study_root}/transforms/sub-{subject}/`
- Format: ANTs composite transforms (.h5)
- Supports slice-specific metadata for partial-brain acquisitions (e.g., 11 hippocampal slices for DTI)

```python
from neurofaune.utils.transforms import create_transform_registry
registry = create_transform_registry(config, subject='sub-001', cohort='p60')

# Check before computing
if not registry.has_transform('T2w', 'SIGMA'):
    # compute and save
    registry.save_ants_composite_transform(...)
else:
    transform = registry.get_ants_composite_transform('T2w', 'SIGMA')
```

### Workflow Pattern
Workflows are function-based (not class-based), accepting `output_dir` as the **study root**:

```python
from neurofaune.preprocess.workflows.anat_preprocess import run_anatomical_preprocessing

results = run_anatomical_preprocessing(
    config=config,
    subject='sub-001',
    t2w_file=Path('T2w.nii.gz'),
    output_dir=Path('/study'),  # Creates derivatives/sub-001/anat/ automatically
    transform_registry=registry
)
```

### Directory Hierarchy (neurovrai-compatible)
```
{study_root}/
├── raw/bids/sub-{subject}/          # Input BIDS data
├── derivatives/sub-{subject}/       # Preprocessed outputs by modality
├── templates/{modality}/{cohort}/   # Age-specific templates
├── transforms/sub-{subject}/        # Transform registry
├── qc/sub-{subject}/                # Quality control reports
└── work/                            # Temporary Nipype files (deletable)
```

### Slice Correspondence for Partial-Coverage Modalities
DWI (11 slices) and fMRI (9 slices) don't cover the full brain (41 T2w slices). The slice correspondence system determines which T2w slices correspond to partial volumes:

```python
from neurofaune.registration import find_slice_correspondence

result = find_slice_correspondence(
    partial_image='sub-001_dwi_b0.nii.gz',
    full_image='sub-001_T2w.nii.gz',
    modality='dwi'
)
# result.start_slice, result.end_slice - T2w slice indices
# result.combined_confidence - matching confidence score
# result.physical_offset - offset in mm from T2w start
```

Uses dual-approach matching: intensity correlation + ventricle landmark detection.

### Study-Space Atlas Pattern
SIGMA atlas must be reoriented to match study acquisition orientation before registration:

```python
from neurofaune.templates.slice_registration import setup_study_atlas

# Run once per study - creates atlas/SIGMA_study_space/
setup_study_atlas(
    config_path=Path('config.yaml'),
    sigma_base_path=Path('/path/to/SIGMA_scaled')
)
```

This avoids resampling every image to atlas orientation.

## Key Design Constraints

1. **T2w is primary anatomical modality** (not T1w) - better rodent brain contrast
2. **ANTs for all registrations** - better quality than FSL for rodent brains
3. **QC must be integrated into workflows** - runs during preprocessing, not after
4. **Age cohorts** (p30, p60, p90) tracked in transform registry and config
5. **SIGMA atlas** - standard rat brain atlas, scaled 10x for FSL/ANTs compatibility
6. **Voxel scaling** - rodent MRI has sub-mm voxels; scale 10x for FSL/ANTs compatibility

## System Dependencies

- **FSL 6.0+**: BET, eddy, MCFLIRT, MELODIC
- **ANTs 2.3+**: Registration, N4 bias correction, Atropos segmentation
- **CUDA** (optional): GPU-accelerated eddy correction

## Configuration System

Configs use YAML with variable substitution (`${paths.study_root}`, `${HOME}`):
- `configs/default.yaml` - base defaults
- `configs/bpa_rat_example.yaml` - study-specific overrides

```python
from neurofaune.config import load_config, get_config_value
config = load_config(Path('config.yaml'))
bet_frac = get_config_value(config, 'anatomical.bet.frac', default=0.3)
```

## Testing Patterns

```bash
# Run all unit tests
uv run pytest tests/unit/ -v

# Run with coverage
uv run pytest --cov=neurofaune --cov-report=term-missing

# Test specific module
uv run pytest tests/unit/test_slice_correspondence.py -v
```

Tests use synthetic data generation - no external data files required. Integration tests marked with `@pytest.mark.integration` require FSL/ANTs installed.

## Development Scripts

Development scripts in `scripts/dev_registration/` follow numbered naming for workflow order:
- `001_explore_geometry.py` - Investigate image geometries
- `003_register_dwi_to_t2w.py` - FA→T2w registration
- `007_register_subject_to_template.py` - Subject→template registration
- `008_register_template_to_sigma.py` - Template→SIGMA registration

Run with: `uv run python scripts/dev_registration/003_register_dwi_to_t2w.py`

## Current Status

**Completed**: Phases 1-8 (Foundation, Atlas, Anatomical, DTI, Templates, MSME, fMRI, Template-Based Registration)

See **README.md** for comprehensive workflow documentation including:
- Two-phase workflow (Initialize → Preprocessing)
- Study-space atlas setup
- Batch processing commands
- Output directory structure

See **STATUS.md** for detailed current state and **ROADMAP.md** for full project plan.

**Important**: Update STATUS.md after completing significant milestones or before ending a session.
