# Neurofaune Architecture

This document describes the overall architecture of Neurofaune, a rodent-specific MRI preprocessing, analysis, and connectivity package.

---

## Four-Part Architecture

Neurofaune follows a modular, four-part architecture inspired by neurovrai:

```
Neurofaune
├── Part 1: Preprocessing (neurofaune.preprocess)
├── Part 2: Atlas Management (neurofaune.atlas)
├── Part 3: Analysis (neurofaune.analysis) [Planned]
└── Part 4: Connectome (neurofaune.connectome) [Planned]
```

---

## Part 1: Preprocessing

**Status**: Implementation in progress

**Purpose**: Subject-level preprocessing for all MRI modalities

### Modalities

#### 1. Anatomical (T2w)
- N4 bias correction
- Skull stripping (BET with rodent parameters)
- ANTs registration to SIGMA atlas
- Tissue segmentation (Atropos)
- Output: Brain-extracted T2w, tissue probability maps, transforms

#### 2. Diffusion (DWI/DTI)
- Eddy current and motion correction (GPU accelerated)
- DTI fitting (FA, MD, AD, RD)
- Advanced models (DKI, NODDI via DIPY or AMICO)
- Registration to slice-specific SIGMA atlas
- Output: DTI metrics, normalized maps, transforms

#### 3. Functional (fMRI)
- Motion correction
- ICA-AROMA denoising
- ACompCor nuisance regression
- GLM confound regression
- Temporal filtering
- Registration to SIGMA atlas (reusing anatomical transforms)
- Output: Denoised BOLD, confounds, normalized timeseries

#### 4. Spectroscopy (MRS)
- FSL-MRS preprocessing
- Metabolite quantification
- Regional analysis with SIGMA parcellation
- Output: Metabolite concentrations, spectra, QC reports

#### 5. Advanced Modalities
- **MSME**: T2 relaxometry mapping
- **MTR**: Magnetization transfer ratio

### Workflow Structure

```python
# Example: Anatomical preprocessing
from neurofaune.preprocess.workflows.anat_preprocess import run_anatomical_preprocessing

results = run_anatomical_preprocessing(
    config=config,
    subject='sub-001',
    t2w_file=Path('T2w.nii.gz'),
    output_dir=Path('/study'),
    transform_registry=registry
)
```

### Key Features
- **Transform registry integration**: Save and reuse spatial transforms
- **Quality control**: Immediate QC generation during workflow
- **Configuration-driven**: All parameters from YAML config
- **Modular utilities**: Reusable helper functions

---

## Part 2: Atlas Management

**Status**: Next phase of development

**Purpose**: SIGMA rat brain atlas integration with slice-specific extraction

### Components

#### 1. Atlas Registry (`neurofaune.atlas.atlas_manager`)
- SIGMA atlas loading and validation
- Metadata parsing (labels, coordinates)
- Multi-resolution support

#### 2. Slice Extraction (`neurofaune.atlas.slice_extractor`)
- Modality-specific slice range extraction
- **Key innovation**: Extract only relevant slices before registration
- Example use cases:
  - DTI: 11 hippocampal slices (15-25)
  - fMRI: Broader cortical coverage (10-35)
  - Anatomical: Full atlas (0 to end)

#### 3. Atlas-Aware Registration (`neurofaune.atlas.registration`)
- ANTs registration optimized for slice-specific atlases
- Multi-resolution strategies
- Quality metrics (MI, CC)
- Transform registry integration

#### 4. Quality Control (`neurofaune.atlas.qc`)
- Registration overlay visualization
- Slice extraction verification
- Boundary alignment metrics

### Slice-Specific Registration Rationale

**Problem**: In rodent neuroimaging, different modalities often cover different brain regions:
- DTI acquisitions may only include 11 slices around the hippocampus
- Whole-brain atlas registration includes irrelevant anatomy (e.g., cerebellum)
- Irrelevant anatomy can interfere with registration accuracy

**Solution**: Extract only the relevant atlas slices before registration:
1. Determine slice range for modality (from config)
2. Extract corresponding atlas slices
3. Register subject data to slice-specific atlas
4. Save slice range metadata with transform

**Benefits**:
- Improved registration accuracy
- Faster computation
- Better handling of partial brain coverage

### Example Usage

```python
from neurofaune.atlas.slice_extractor import extract_atlas_slices

# Extract 11 hippocampal slices for DTI
atlas_slices = extract_atlas_slices(
    atlas_path=Path('/data/SIGMA/SIGMA_template.nii.gz'),
    modality='dwi',
    config=config
)
# Returns: Slices 15-25 (11 slices)

# Register FA map to slice-specific atlas
from neurofaune.atlas.registration import register_to_sigma_sliced

transform = register_to_sigma_sliced(
    subject_img=fa_map,
    atlas_slices=atlas_slices,
    modality='dwi',
    config=config,
    transform_registry=registry
)
```

---

## Part 3: Analysis (Planned)

**Status**: Future development

**Purpose**: Group-level statistical analyses

### Planned Features

#### 1. ROI-Based Analysis
- Extract mean values from SIGMA parcellation
- Tissue-specific measurements (GM, WM, CSF)
- Age cohort comparisons (p30 vs p60 vs p90)

#### 2. Voxel-Based Analysis
- Voxel-wise statistical comparisons
- Multiple comparison correction (FWE, FDR)
- Group differences in DTI metrics, CBF, BOLD

#### 3. Tract-Based Analysis
- Tract-specific DTI metrics
- TBSS-like analyses for rodent brains

#### 4. Functional Connectivity
- Seed-based connectivity
- ROI-to-ROI connectivity matrices
- Network metrics (ReHo, ALFF, fALFF)

---

## Part 4: Connectome (Planned)

**Status**: Future development

**Purpose**: Network neuroscience and connectivity analysis

### Planned Features

#### 1. Structural Connectivity
- Probabilistic tractography
- Connectivity matrices
- Graph theory metrics

#### 2. Functional Connectivity
- Correlation-based connectivity
- Partial correlation
- Dynamic functional connectivity

#### 3. Multi-Modal Integration
- Structure-function coupling
- Combining DTI and fMRI connectivity

#### 4. Network Visualization
- Connectome visualization
- Community detection
- Hub identification

---

## Shared Infrastructure

### Configuration System (`neurofaune.config`)

All parts share the same configuration system:
- YAML-based configuration
- Variable substitution (${paths.study_root})
- Validation and defaults
- Rodent-specific parameters

### Transform Registry (`neurofaune.utils.transforms`)

Centralized spatial transformation management:
- ANTs composite transforms (.h5)
- Slice-specific metadata tracking
- Reuse across workflows and parts
- Validation and quality metrics

### Directory Hierarchy

Standard neurovrai-compatible structure:
```
{study_root}/
├── raw/                  # Raw data (Bruker, BIDS)
├── derivatives/          # All preprocessed outputs
│   └── sub-{subject}/
│       ├── anat/
│       ├── dwi/
│       └── func/
├── transforms/           # Transform registry
│   └── sub-{subject}/
├── qc/                   # Quality control
│   └── sub-{subject}/
├── analysis/             # Group-level analyses (Part 3)
└── connectome/           # Network analyses (Part 4)
```

### Quality Control Framework

Consistent QC across all parts:
- HTML reports with interactive visualizations
- Automated pass/fail criteria
- Visual inspection tools
- Outlier detection

---

## Data Flow

### Subject-Level Processing (Parts 1-2)

```
Raw Bruker Data
    ↓
[Conversion to BIDS]
    ↓
raw/bids/sub-{subject}/
    ↓
[Anatomical Preprocessing]
    ↓
derivatives/sub-{subject}/anat/
transforms/sub-{subject}/  (T2w → SIGMA)
qc/sub-{subject}/anat/
    ↓
[Diffusion Preprocessing]
    ↓
derivatives/sub-{subject}/dwi/
transforms/sub-{subject}/  (FA → SIGMA, reuse T2w→SIGMA)
qc/sub-{subject}/dwi/
    ↓
[Functional Preprocessing]
    ↓
derivatives/sub-{subject}/func/
transforms/sub-{subject}/  (reuse T2w→SIGMA)
qc/sub-{subject}/func/
```

### Group-Level Processing (Part 3)

```
derivatives/sub-*/anat/
derivatives/sub-*/dwi/
derivatives/sub-*/func/
    ↓
[Group Analysis]
    ↓
analysis/
├── roi_analysis/
├── voxel_analysis/
└── tract_analysis/
```

### Network Analysis (Part 4)

```
derivatives/sub-*/dwi/
derivatives/sub-*/func/
analysis/roi_analysis/
    ↓
[Connectome Analysis]
    ↓
connectome/
├── structural/
├── functional/
└── multimodal/
```

---

## Design Principles

### 1. Compute Once, Reuse Everywhere

**Transform Registry**: Every spatial transformation is computed once and reused:
- Anatomical workflow computes T2w → SIGMA
- Functional workflow reuses T2w → SIGMA (no recomputation)
- DTI workflow computes modality-specific FA → SIGMA (hippocampal slices)

### 2. Configuration-Driven

**No Hardcoded Parameters**: All processing parameters come from YAML config:
- Atlas paths and slice definitions
- Registration parameters (ANTs)
- Preprocessing thresholds (BET frac)
- Execution settings (n_procs)

### 3. Slice-Aware Processing

**Intelligent Atlas Handling**: Different modalities get different atlas slices:
- DTI: Hippocampal region only (11 slices)
- fMRI: Broader cortical + subcortical
- Anatomical: Full atlas

### 4. Immediate QC Integration

**QC During Processing**: Quality control is generated during workflows, not after:
- Skull stripping QC right after BET
- Registration QC right after normalization
- Integrated into workflow returns

### 5. Modularity

**Reusable Components**:
- Workflows are functions, not classes
- Utilities are modality-agnostic where possible
- QC modules are standalone
- Atlas management is separate from preprocessing

### 6. Neurovrai Compatibility

**Shared Structure**: Same directory hierarchy enables:
- Easy migration between human and rodent pipelines
- Shared analysis tools
- Consistent documentation
- Future integration

---

## Technology Stack

### Core Dependencies
- **Python 3.10+**: Modern Python with type hints
- **Nipype 1.10.0+**: Workflow engine
- **ANTs 2.3+**: Registration and segmentation
- **FSL 6.0+**: Preprocessing tools (BET, eddy, ICA-AROMA)
- **Nibabel 5.3.2+**: NIfTI I/O

### Rodent-Specific
- **SIGMA Atlas**: Rat brain template and parcellation
- **FSL-MRS**: Spectroscopy processing
- **Bruker conversion**: Bruker file format handling

### Optional
- **CUDA**: GPU acceleration for eddy, BEDPOSTX
- **DIPY**: Advanced diffusion models
- **AMICO**: Fast microstructure modeling

---

## Development Roadmap

### Phase 1: Foundation ✅ (Completed)
- [x] Project structure
- [x] Configuration system
- [x] Transform registry
- [x] Documentation framework

### Phase 2: Atlas Management 🔄 (In Progress)
- [ ] SIGMA atlas integration
- [ ] Slice extraction engine
- [ ] Atlas-aware registration
- [ ] Atlas QC module

### Phase 3: Anatomical Preprocessing 📋
- [ ] T2w workflow
- [ ] N4 bias correction
- [ ] ANTs registration
- [ ] Tissue segmentation
- [ ] Anatomical QC

### Phase 4: Diffusion Preprocessing 📋
- [ ] Eddy correction
- [ ] DTI fitting
- [ ] Slice-specific atlas registration
- [ ] Advanced models (DKI, NODDI)
- [ ] Diffusion QC

### Phase 5: Functional Preprocessing 📋
- [ ] Motion correction
- [ ] ICA-AROMA
- [ ] ACompCor
- [ ] Transform reuse
- [ ] Functional QC

### Phase 6: Advanced Modalities 📋
- [ ] MSME preprocessing
- [ ] MTR preprocessing
- [ ] FSL-MRS integration
- [ ] Spectroscopy QC

### Phase 7: CLI & Batch Processing 📋
- [ ] Command-line interface
- [ ] Batch processing scripts
- [ ] Configuration generator
- [ ] HPC integration

### Phase 8: Testing & Validation 📋
- [ ] Unit tests (atlas, config, transforms)
- [ ] Integration tests (workflows)
- [ ] Real data validation
- [ ] Performance benchmarking

### Phase 9: Documentation 📋
- [ ] Complete user guide
- [ ] API documentation
- [ ] Tutorials and examples
- [ ] Troubleshooting guide

### Future: Analysis & Connectome
- [ ] Part 3: Group-level analysis
- [ ] Part 4: Network connectivity
- [ ] Web-based QC interface
- [ ] BIDS validation

---

## Summary

Neurofaune's architecture is designed for:
1. **Rodent-specific processing** with slice-aware atlas registration
2. **Efficiency** through transform reuse and GPU acceleration
3. **Quality** through integrated QC and validation
4. **Extensibility** through modular design
5. **Compatibility** with neurovrai for shared analysis tools

The four-part architecture (Preprocessing, Atlas, Analysis, Connectome) provides a clear path from raw data to publishable results, with each part building on the infrastructure established in earlier parts.
