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
# Build age-specific templates
uv run python scripts/build_templates.py --config config.yaml --cohort p60 --modality anat

# Batch anatomical preprocessing
uv run python scripts/batch_preprocess_for_templates.py /path/to/bids /path/to/output

# Batch functional preprocessing
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
│   └── utils/                # Preprocessing utilities
│       └── func/             # Functional-specific utils (ICA, aCompCor, skull stripping)
├── templates/                # Template building and registration
│   ├── builder.py            # ANTs template construction
│   └── registration.py       # Subject-to-template registration
└── utils/
    ├── transforms.py         # Transform registry system
    ├── exclusion.py          # Subject exclusion tracking
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

### Slice-Specific Atlas Registration
For modalities that don't cover the full brain, extract relevant atlas slices before registration:

```python
from neurofaune.atlas import AtlasManager
atlas = AtlasManager(config)
dwi_template = atlas.get_template(modality='dwi')  # Uses config slice_definitions
```

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

## Project Status Tracking

**STATUS.md** contains the current project state. **Update it:**
- After completing significant milestones (template builds, workflow changes, etc.)
- Before ending a session
- When discovering issues or blockers

Key sections to update:
- Template/registration status tables
- Workflow integration status
- Recent changes log
- Known issues

## Current Status

**Completed**: Phases 1-7 (Foundation, Atlas, Anatomical, DTI, Templates, MSME, fMRI)
**In Progress**: Phase 8 (Template-Based Registration) - all templates built, integration pending

See STATUS.md for detailed current state and ROADMAP.md for full project plan.
