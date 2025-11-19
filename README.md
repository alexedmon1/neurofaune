# Neurofaune

**Rodent-specific MRI preprocessing, analysis, and connectivity package**

Neurofaune is a comprehensive neuroimaging pipeline designed specifically for rodent (rat and mouse) brain MRI data. Built on the architecture of [neurovrai](https://github.com/yourusername/neurovrai), it provides Bruker data conversion and advanced preprocessing for multiple MRI modalities.

**Current test study**: BPA-Rat (Bisphenol A rat cohort study) - 141 subjects, 189 sessions across 7 cohorts

---

## Features

### Current Status

**✅ Phase 1 (Foundation) - Complete**
- Configuration System: YAML-based configuration with variable substitution
- Transform Registry: Centralized storage for spatial transformations with slice-specific support
- Neurovrai-compatible Output: Same directory hierarchy (derivatives/, transforms/, qc/)
- Rodent-optimized Parameters: Atlas slice extraction, age cohort support

**✅ Phase 2 (Atlas Management) - Complete**
- AtlasManager: Unified interface for SIGMA rat brain atlas
- Slice Extraction: Modality-specific slice extraction utilities
- Template Access: InVivo/ExVivo templates, tissue masks, parcellations
- ROI Operations: Label loading, ROI mask creation, metadata access

**✅ Bruker to BIDS Conversion - Complete**
- 141 subjects converted from 7 cohorts (Cohorts 1-5, 7-8)
- 189 sessions (54 p30, 50 p60, 79 p90, 6 unknown)
- 2,256 NIfTI files with JSON metadata sidecars
- Modalities: T2w anatomical, DTI/DWI, fMRI, MSME, MTR, FLASH, field maps
- See [`docs/BRUKER_CONVERSION_SUMMARY.md`](docs/BRUKER_CONVERSION_SUMMARY.md) for details

### Planned Preprocessing Capabilities

#### Core Modalities
- **Anatomical (T2w)**: N4 bias correction, skull stripping, ANTs registration to SIGMA atlas, tissue segmentation
- **Diffusion (DWI/DTI)**: Eddy correction, DTI fitting, advanced models (DKI, NODDI), spatial normalization
- **Functional (fMRI)**: ICA-AROMA denoising, ACompCor, motion correction, GLM confound regression
- **Spectroscopy (MRS)**: FSL-MRS integration, metabolite quantification, regional analysis

#### Advanced Features
- **Slice-specific Atlas Registration**: Register to relevant anatomical slices only (e.g., 11 hippocampal slices for DTI)
- **SIGMA Atlas Integration**: Rat brain atlas with modality-specific slice extraction
- **Age Cohort Support**: Developmental studies (p30, p60, p90)
- **Transform Reuse**: Compute once, reuse across modalities
- **Comprehensive QC**: Automated quality control reports for all preprocessing steps

---

## BPA-Rat Study Data

### Study Organization
The BPA-Rat study folder serves as a test dataset for neurofaune development:

- **Study root**: `/mnt/arborea/bpa-rat/`
- **Source Bruker data**: `/mnt/arborea/bruker/` (permanent archive)
- **Converted BIDS data**: `/mnt/arborea/bpa-rat/raw/bids/`
  - 141 subjects: `sub-Rat001` through `sub-Rat298`
  - Sessions: `ses-p30`, `ses-p60`, `ses-p90`
  - Modalities: `anat/`, `dwi/`, `func/`, `msme/`, `mtr/`, `flash/`, `fmap/`

### Atlas Location
- **SIGMA rat brain atlas**: `/mnt/arborea/atlases/SIGMA/`

**Note**: Bruker data in temporary Cohort folders has been deleted after successful conversion. The original data is preserved in `/mnt/arborea/bruker/`.

See [`docs/BRUKER_CONVERSION_SUMMARY.md`](docs/BRUKER_CONVERSION_SUMMARY.md) for complete conversion details.

---

## Installation

### Prerequisites

- Python 3.10+
- FSL 6.0+
- ANTs 2.3+
- (Optional) CUDA for GPU acceleration

### Install Neurofaune

```bash
# Clone repository
git clone https://github.com/yourusername/neurofaune.git
cd neurofaune

# Install with pip
pip install -e .

# Or install with optional dependencies
pip install -e ".[all]"  # Includes FSL-MRS, Bruker conversion tools
```

---

## Quick Start

### 1. Create Configuration

Create a `config.yaml` file for your study:

```yaml
# Study information
study:
  name: "My Rodent Study"
  code: "STUDY001"
  species: "rat"

# Directory structure
paths:
  study_root: "/path/to/study"
  derivatives: "${paths.study_root}/derivatives"
  transforms: "${paths.study_root}/transforms"
  qc: "${paths.study_root}/qc"

# Atlas configuration
atlas:
  name: "SIGMA"
  base_path: "/path/to/SIGMA"
  slice_definitions:
    dwi:
      start: 15
      end: 25  # 11 hippocampal slices

# Processing parameters
execution:
  n_procs: 6

anatomical:
  bet:
    frac: 0.3  # Rodent-optimized

diffusion:
  eddy:
    use_cuda: true
```

### 2. Run Preprocessing

```bash
# Anatomical preprocessing
neurofaune run anatomical --config config.yaml --subject sub-001

# Diffusion preprocessing
neurofaune run dwi --config config.yaml --subject sub-001

# Functional preprocessing
neurofaune run functional --config config.yaml --subject sub-001

# All modalities
neurofaune run all --config config.yaml --subject sub-001
```

### 3. Quality Control

```bash
# Generate QC reports
neurofaune qc generate --config config.yaml --subject sub-001

# View QC in browser
open qc/sub-001/anat/skull_strip_qc.html
```

---

## Directory Structure

Neurofaune uses a standardized directory hierarchy compatible with neurovrai:

```
study_root/
├── raw/
│   ├── bruker/              # Raw Bruker data
│   └── bids/                # Converted BIDS data
│       └── sub-001/
│           ├── anat/
│           ├── dwi/
│           └── func/
├── derivatives/             # Preprocessed outputs
│   └── sub-001/
│       ├── anat/
│       │   ├── sub-001_desc-preproc_T2w.nii.gz
│       │   ├── sub-001_desc-brain_mask.nii.gz
│       │   └── sub-001_space-SIGMA_T2w.nii.gz
│       ├── dwi/
│       │   ├── sub-001_desc-preproc_dwi.nii.gz
│       │   ├── sub-001_FA.nii.gz
│       │   └── sub-001_space-SIGMA_FA.nii.gz
│       └── func/
│           └── sub-001_desc-preproc_bold.nii.gz
├── transforms/              # Transform registry
│   └── sub-001/
│       ├── T2w_to_SIGMA_composite.h5
│       ├── FA_to_SIGMA_dwi_composite.h5
│       └── transforms.json
├── qc/                      # Quality control reports
│   └── sub-001/
│       ├── anat/
│       ├── dwi/
│       └── func/
└── work/                    # Temporary files
    └── sub-001/
```

---

## Architecture

Neurofaune follows a modular, four-part architecture:

### 1. Preprocessing (`neurofaune.preprocess`)
- Subject-level preprocessing for all modalities
- Workflows: `anatomical`, `diffusion`, `functional`, `spectroscopy`
- Utilities: Registration, normalization, file discovery
- QC: Automated quality control for each modality

### 2. Atlas Management (`neurofaune.atlas`)
- SIGMA rat brain atlas integration
- Slice-specific extraction for different modalities
- Atlas-aware registration with ANTs
- Registration quality metrics

### 3. Analysis (Planned)
- Group-level statistical analyses
- ROI-based analyses
- Connectivity metrics

### 4. Connectome (Planned)
- Network neuroscience
- Structural and functional connectivity
- Graph theory metrics

---

## Key Differences from Neurovrai

| Feature | Neurovrai (Human) | Neurofaune (Rodent) |
|---------|-------------------|---------------------|
| **Primary modality** | T1w | T2w |
| **Atlas** | MNI152 | SIGMA (rat brain) |
| **Atlas strategy** | Full template | Slice-specific extraction |
| **Registration** | ANTs | ANTs (slice-aware) |
| **Skull stripping** | BET (default) | BET (rodent-optimized) |
| **Input format** | DICOM | Bruker / DICOM |
| **Age cohorts** | Single | Developmental (p30/p60/p90) |
| **Spectroscopy** | Not included | FSL-MRS integration |

---

## Configuration

Neurofaune uses YAML configuration files with:
- **Variable substitution**: `${paths.study_root}` references other config values
- **Environment variables**: `${HOME}` expands to environment variables
- **Nested parameters**: Hierarchical organization of preprocessing parameters
- **Default values**: Sensible defaults for rodent brains

See `configs/default.yaml` for all available parameters.

---

## Transform Registry

The transform registry avoids redundant computation by storing all spatial transformations:

```python
from neurofaune.utils.transforms import create_transform_registry
from neurofaune.config import load_config

config = load_config("config.yaml")
registry = create_transform_registry(config, "sub-001", cohort="p60")

# Save transform after computing
registry.save_ants_composite_transform(
    composite_file=Path("T2w_to_SIGMA_Composite.h5"),
    source_space="T2w",
    target_space="SIGMA"
)

# Reuse in other workflows
transform = registry.get_ants_composite_transform("T2w", "SIGMA")
if transform:
    # Apply to functional data without recomputing
    pass
```

---

## Contributing

Contributions are welcome! Please see `CLAUDE.md` for development guidelines.

---

## Citation

If you use Neurofaune in your research, please cite:

```
[Citation to be added]
```

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

- Built on the neurovrai architecture
- Preprocessing patterns adapted from rat-mri-preprocess
- SIGMA rat brain atlas ([Barrière et al., 2019](https://doi.org/10.1016/j.neuroimage.2019.06.063))
- FSL-MRS for spectroscopy ([Clarke et al., 2021](https://doi.org/10.1002/mrm.28630))

---

## Contact

For questions or issues, please open an issue on GitHub.
