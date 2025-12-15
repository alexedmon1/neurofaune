# Neurofaune Development Roadmap

**Last Updated:** December 15, 2024
**Current Status:** Phase 8 In Progress (Template-Based Registration)

This document outlines the implementation plan for neurofaune, a rodent-specific MRI preprocessing pipeline.

---

## 📊 Current Status

### 🚧 **Active Work: Phase 8 - Template Building**

Template building is currently running for T2w anatomical templates:
- **Directory structure updated:** `templates/{modality}/{cohort}/` (e.g., `templates/anat/p60/`)
- **p30:** Building (38 subjects available, using top 10)
- **p60:** Pending (34 subjects available)
- **p90:** Pending (47 subjects available)

Each template build includes:
- T2w template via ANTs multivariate template construction (4 iterations)
- Tissue probability templates (GM, WM, CSF)
- SIGMA atlas registration with composite transforms

### ✅ **Completed Phases**

**Phase 1: Foundation** ✓
- YAML-based configuration system with variable substitution
- Transform registry for centralized spatial transformation storage
- Neurovrai-compatible directory structure
- Atlas management system (SIGMA rat brain atlas)

**Phase 2: Atlas Management** ✓
- AtlasManager class for unified SIGMA atlas interface
- Slice extraction utilities for modality-specific registration
- Template access (InVivo/ExVivo, tissue masks, parcellations)
- ROI operations (label loading, mask creation)

**Phase 3: Anatomical T2w Preprocessing** ✓
- Image validation (voxel size, orientation, dimensions)
- Adaptive skull stripping (Atropos + BET with CNR-based frac calculation)
- N4 bias field correction
- Tissue segmentation (GM, WM, CSF probability maps)
- Batch processing support
- **Validated:** 189 subject/session combinations on BPA-Rat dataset

**Phase 4: DTI/DWI Preprocessing** ✓
- 5D→4D conversion for Bruker multi-average acquisitions
- GPU-accelerated eddy correction (with CPU fallback)
- DTI fitting (FA, MD, AD, RD maps)
- Comprehensive QC (motion, eddy, tensor metrics)

**Phase 5: Multi-Modal Template Architecture** ✓
- Template building infrastructure (`scripts/build_templates.py`)
- Age-cohort aware architecture (p30, p60, p90)
- ANTs-based template construction framework

**Phase 6: MSME T2 Mapping** ✓
- Multi-echo T2 mapping workflow
- Myelin Water Fraction (MWF) calculation via NNLS
- T2 compartment analysis (myelin, intra/extra-cellular, CSF)
- QC with decay curves and NNLS spectra

**Phase 7: Functional fMRI Preprocessing** ✓ **(NEWLY COMPLETED)**
- **Adaptive skull stripping** (per-slice BET with -R flag, 0.8% protocol variability)
- **Motion correction** (MCFLIRT with middle volume reference)
- **ICA denoising** (MELODIC with automatic dimensionality estimation)
  - Rodent-specific component classification (motion, edge, CSF, frequency)
  - Typical signal retention: 75-77%
- **Spatial smoothing** (6mm FWHM, optimized for rodents)
- **Temporal filtering** (0.01-0.1 Hz bandpass for resting-state)
- **Confound extraction** (24 extended motion regressors + aCompCor)
- **Comprehensive QC** (motion FD/DVARS, ICA classification, confounds)
- **Batch processing** (`scripts/batch_preprocess_func.py`)
- **Validated:** BPA-Rat dataset (294 BOLD scans across 6 acquisition protocols)
- **Note:** Slice timing correction disabled (acquisition artifacts present)

---

## 🎯 Next Steps

### **Phase 8: Template-Based Registration** ⭐ **IN PROGRESS**

**Status:** Template building actively running
**Started:** December 15, 2024
**Blocking:** ROI extraction, group analysis, cross-subject comparisons

#### 8.1 Build Age-Specific Templates 🚧 **ACTIVE**
**Goal:** Create p30, p60, p90 cohort templates from preprocessed anatomical data

**Directory Structure (Updated December 15, 2024):**
```
templates/
├── anat/                    # T2w anatomical templates
│   ├── p30/
│   │   ├── tpl-BPARat_p30_T2w.nii.gz
│   │   ├── tpl-BPARat_p30_space-SIGMA_T2w.nii.gz
│   │   ├── tpl-BPARat_p30_label-GM_probseg.nii.gz
│   │   ├── tpl-BPARat_p30_label-WM_probseg.nii.gz
│   │   ├── tpl-BPARat_p30_label-CSF_probseg.nii.gz
│   │   └── transforms/
│   │       ├── tpl-to-SIGMA_Composite.h5
│   │       └── SIGMA-to-tpl_Composite.h5
│   ├── p60/
│   └── p90/
├── dwi/                     # FA templates (future)
│   └── ...
└── func/                    # BOLD templates (future)
    └── ...
```

**Implementation:**
1. ✅ Updated `scripts/build_templates.py` for new directory structure
2. 🚧 Building T2w templates with ANTs (4 iterations, 10 subjects per cohort)
3. ⏳ Tissue probability templates (GM, WM, CSF)
4. ⏳ SIGMA atlas registration
5. ⏳ QC: Visual inspection, sharpness metrics

**Input:** Preprocessed T2w from Phase 3
- p30: 38 subjects available
- p60: 34 subjects available
- p90: 47 subjects available

**Validation:** Template sharpness, tissue boundary clarity, cohort representativeness

---

#### 8.2 Register Templates to SIGMA Atlas
**Goal:** Create template→SIGMA transforms for each cohort

**Implementation:**
1. For each cohort template:
   - Run ANTs SyN registration to SIGMA InVivo template
   - Use mutual information similarity metric
   - Generate composite transform (.h5)
2. Store transforms in centralized registry
3. Create inverse transforms for atlas→template propagation

**Input:** Cohort templates + SIGMA atlas
**Output:**
- `transforms/template_p30_to_SIGMA_composite.h5`
- `transforms/SIGMA_to_template_p30_composite.h5`
- Similar for p60, p90

**Validation:** Overlay registered templates on SIGMA, check anatomical alignment

---

#### 8.3 Integrate Registration into Workflows
**Goal:** Add individual→template→SIGMA registration to all preprocessing pipelines

**A. Anatomical Workflow Updates:**
```python
# neurofaune/preprocess/workflows/anat_preprocess.py

# After tissue segmentation:
# 1. Register subject T2w → cohort template
subject_to_template_transform = register_to_template(
    moving=preprocessed_t2w,
    template=cohort_template,
    output_dir=work_dir
)

# 2. Compose with template→SIGMA transform
subject_to_atlas_transform = compose_transforms(
    subject_to_template_transform,
    template_to_sigma_transform
)

# 3. Apply to T2w and tissue maps
warp_to_atlas(preprocessed_t2w, subject_to_atlas_transform, sigma_space)
warp_to_atlas(tissue_probsegs, subject_to_atlas_transform, sigma_space)

# 4. Save to transform registry
registry.save_ants_composite_transform(
    subject_to_atlas_transform,
    source_space='T2w',
    target_space='SIGMA',
    cohort=cohort
)
```

**B. Functional Workflow Updates:**
```python
# neurofaune/preprocess/workflows/func_preprocess.py

# Reuse anatomical→atlas transform:
anat_to_atlas_transform = registry.get_ants_composite_transform(
    source='T2w',
    target='SIGMA',
    cohort=cohort
)

# Chain: BOLD→T2w (boundary-based registration) → template → SIGMA
bold_to_atlas = compose_transforms(
    bold_to_t2w_transform,
    anat_to_atlas_transform
)

# Apply to preprocessed BOLD
warp_to_atlas(preprocessed_bold, bold_to_atlas, sigma_space)
```

**C. DTI Workflow Updates:**
- Slice-specific template creation (11-slice hippocampus)
- Register DTI metrics (FA, MD) to corresponding atlas slices
- Handle geometry mismatches via slice interpolation

**Output per subject:**
- `derivatives/sub-*/anat/sub-*_space-SIGMA_T2w.nii.gz`
- `derivatives/sub-*/func/sub-*_space-SIGMA_bold.nii.gz`
- `derivatives/sub-*/dwi/sub-*_space-SIGMA_FA.nii.gz`

**Validation:**
- Visual QC: Overlay subject data on SIGMA atlas
- Quantitative: Dice coefficient for tissue overlap
- Functional: Extract ROI timeseries from SIGMA parcellations

---

#### 8.4 ROI Extraction Module
**Goal:** Extract timeseries/metrics from SIGMA parcellations

**Implementation:**
```python
# neurofaune/analysis/roi_extraction.py

def extract_roi_timeseries(
    bold_atlas_space: Path,
    sigma_parcellation: Path,
    roi_labels: List[int]
) -> pd.DataFrame:
    """Extract mean timeseries per ROI."""
    # Load data and parcellation
    # For each ROI: compute mean timeseries
    # Return DataFrame: columns=ROI_names, rows=timepoints
    pass
```

**Deliverables:**
- `derivatives/sub-*/func/sub-*_desc-roitimeseries.tsv`
- `derivatives/sub-*/dwi/sub-*_desc-roimetrics.tsv`

---

### **Phase 9: Additional Modalities**

**Timeline:** 1-2 months

#### 9.1 MTR (Magnetization Transfer Ratio) Preprocessing
**Status:** Not started
**Priority:** Medium

**Implementation:**
1. Multi-echo T1 processing workflow
2. MTR map calculation: `MTR = (M0 - MT) / M0 × 100`
3. Registration to template/atlas
4. QC with MTR histograms and overlays

**Deliverables:**
- `neurofaune/preprocess/workflows/mtr_preprocess.py`
- `derivatives/sub-*/mtr/sub-*_MTR.nii.gz`

---

#### 9.2 Spectroscopy (MRS) Preprocessing
**Status:** Not started
**Priority:** Low (requires FSL-MRS conda installation)

**Implementation:**
1. Bruker spectroscopy format conversion
2. Water suppression and frequency correction
3. Metabolite quantification (NAA, Glu, Cr, Cho)
4. QC with spectra plots

**Dependencies:** FSL-MRS (conda-only, not pip-installable)

---

### **Phase 10: CLI & Production Tools**

**Timeline:** 1-2 months

#### 10.1 Unified CLI Interface
**Status:** Not started
**Priority:** High (usability)

**Goal:** Single command-line interface for all modalities

**Implementation:**
```bash
# Proposed interface
neurofaune preprocess \
    --modality anat \
    --bids-root /data/bids \
    --output-root /data/derivatives \
    --config config.yaml \
    --subjects sub-001 sub-002 \
    --sessions ses-p60 \
    --n-workers 6

# Or for batch processing
neurofaune batch \
    --modality func \
    --bids-root /data/bids \
    --config config.yaml \
    --resume batch_results.json  # Resume from checkpoint
```

**Features:**
- Modality selection (anat, dwi, func, mtr, all)
- Subject/session filtering
- Resume/checkpoint support
- Parallel execution
- Real-time progress monitoring

**Deliverables:**
- `neurofaune/cli.py` (main CLI entry point)
- `scripts/neurofaune` (executable wrapper)

---

#### 10.2 HPC Integration
**Status:** Not started
**Priority:** Medium

**Goal:** SLURM/SGE job submission support

**Implementation:**
```bash
neurofaune submit \
    --scheduler slurm \
    --partition compute \
    --mem 16G \
    --time 4:00:00 \
    --array 1-100  # Process 100 subjects
```

---

#### 10.3 Progress Monitoring Dashboard
**Status:** Not started
**Priority:** Low

**Goal:** Web-based progress visualization

**Features:**
- Real-time batch processing status
- QC report browser
- Error log viewer
- Resource usage plots

---

### **Phase 11: Testing & Validation**

**Timeline:** 1-2 months
**Priority:** High (code quality)

#### 11.1 Unit Tests
**Status:** Minimal coverage
**Target:** 80% code coverage

**Implementation:**
```python
# tests/unit/test_skull_strip.py
# tests/unit/test_registration.py
# tests/unit/test_roi_extraction.py
```

**Framework:** pytest + pytest-cov

---

#### 11.2 Integration Tests
**Status:** Manual testing only
**Target:** Automated end-to-end tests

**Implementation:**
```python
# tests/integration/test_anat_workflow.py
# tests/integration/test_func_workflow.py
```

**Test data:** Small synthetic datasets (10MB)

---

#### 11.3 Continuous Integration
**Status:** Not configured
**Target:** GitHub Actions CI/CD

**Pipeline:**
1. Lint (ruff, mypy)
2. Unit tests
3. Integration tests (on small dataset)
4. Build documentation
5. Deploy to PyPI (on release tags)

---

#### 11.4 External Validation
**Status:** Validated on BPA-Rat only
**Target:** 2-3 external rodent datasets

**Datasets:**
- Public rat fMRI dataset
- Mouse anatomical dataset
- Multi-site DTI dataset

---

### **Phase 12: Documentation**

**Timeline:** 1-2 months
**Priority:** High (usability)

#### 12.1 API Documentation
**Status:** Docstrings exist, not compiled
**Target:** Sphinx-generated HTML docs

**Implementation:**
```bash
# Build docs
cd docs/
sphinx-apidoc -o api/ ../neurofaune/
make html
```

**Host:** GitHub Pages or ReadTheDocs

---

#### 12.2 User Tutorials
**Status:** None
**Target:** 5-10 Jupyter notebooks

**Tutorials:**
1. Getting Started (installation, BIDS conversion)
2. Anatomical Preprocessing (T2w workflow, QC)
3. Functional Preprocessing (BOLD workflow, ICA denoising)
4. DTI Preprocessing (eddy correction, tensor fitting)
5. ROI Analysis (extraction, connectivity matrices)
6. Template Building (custom cohort templates)
7. Advanced Configuration (YAML customization)

---

#### 12.3 Developer Guide
**Status:** CLAUDE.md exists
**Target:** Comprehensive developer documentation

**Sections:**
1. Architecture overview
2. Adding new workflows
3. Extending QC modules
4. Contributing guidelines
5. Release process

---

#### 12.4 Methods Description
**Status:** Not started
**Target:** Publication-ready methods section

**Content:**
- Preprocessing algorithms (skull stripping, registration, denoising)
- Software versions and parameters
- Validation results (accuracy, reproducibility)
- Comparison to existing tools (fMRIPrep, FSL, SPM)

**Output:** LaTeX document for publication

---

## 🚀 Recommended Implementation Order

### **Immediate (In Progress)**
1. 🚧 **Build age-specific templates** (Phase 8.1) - ACTIVELY RUNNING
2. ⭐ **Register templates to SIGMA** (Phase 8.2) - Next
3. ⭐ **Integrate registration into anatomical workflow** (Phase 8.3)

**Rationale:** Registration is the critical bottleneck preventing downstream analysis

---

### **Short-Term (Next 1-2 Months)**
4. Add functional registration (Phase 8.3B)
5. Implement ROI extraction (Phase 8.4)
6. DTI slice-specific registration (Phase 8.3C)
7. MTR preprocessing workflow (Phase 9.1)

**Rationale:** Completes core preprocessing, enables group analysis

---

### **Medium-Term (Next 3-6 Months)**
8. Unified CLI interface (Phase 10.1)
9. Unit + integration testing (Phase 11.1, 11.2)
10. API documentation (Phase 12.1)
11. User tutorials (Phase 12.2)

**Rationale:** Improves usability, code quality, and accessibility

---

### **Long-Term (Next 6-12 Months)**
12. HPC integration (Phase 10.2)
13. Progress dashboard (Phase 10.3)
14. External validation (Phase 11.4)
15. Methods paper (Phase 12.4)
16. Spectroscopy support (Phase 9.2)

**Rationale:** Advanced features, publication, community adoption

---

## 💡 Key Decisions & Open Questions

### **1. Template Strategy**
- **Decision:** Age-specific templates (p30, p60, p90)
- **Rationale:** Developmental changes in rat brain anatomy
- **Alternative:** Unified template across ages (rejected due to morphological differences)

### **2. Registration Target**
- **Decision:** Register to SIGMA atlas (primary) + optional study-specific template
- **Rationale:** SIGMA is standard for rat neuroimaging, enables cross-study comparisons
- **Future:** Support custom atlas registration

### **3. Slice Timing Correction**
- **Decision:** Disabled in current functional workflow
- **Rationale:** Acquisition artifacts (zebra stripes) worsened by slice timing
- **Open Question:** Re-enable with improved artifact handling? Alternative algorithms?

### **4. ICA Component Selection**
- **Decision:** Automatic dimensionality estimation (MELODIC default)
- **Rationale:** Prevents signal truncation, typical 75-77% signal retention
- **Validation:** Component classification QC shows appropriate noise removal

### **5. Development Priority**
- **Decision:** Registration (Phase 8) before MTR (Phase 9)
- **Rationale:** Registration unlocks analysis of existing data, MTR adds new modality
- **Community Input:** Survey users for priority features

---

## 📈 Success Metrics

### **Technical Metrics**
- ✅ Code coverage: >80% (target)
- ✅ Documentation coverage: 100% public API
- ✅ Test suite runtime: <10 minutes
- ✅ CI/CD: Automated testing on every commit

### **Scientific Metrics**
- ✅ Validation: Tested on ≥3 external datasets
- ✅ Reproducibility: ICC >0.95 for test-retest
- ✅ Accuracy: Dice >0.90 for atlas registration
- ✅ Publication: Methods paper accepted in peer-reviewed journal

### **Usability Metrics**
- ✅ Installation: <10 minutes with `uv pip install neurofaune`
- ✅ Tutorial completion: Users can preprocess sample data in <1 hour
- ✅ Error handling: Informative error messages with suggested fixes
- ✅ Community: ≥10 GitHub stars, ≥3 external contributors

---

## 📝 Version History

**v0.8.0-dev (December 2024)** - Phase 8 In Progress
- Template building actively running for T2w (p30, p60, p90)
- Updated directory structure: `templates/{modality}/{cohort}/`
- 119 preprocessed subjects available (38 p30, 34 p60, 47 p90)
- Improved build_templates.py with modality-first organization

**v0.7.0 (December 2024)** - Phase 7 Complete
- Functional fMRI preprocessing with adaptive skull stripping
- ICA denoising with automatic dimensionality
- Batch processing support for 294 BOLD scans
- Validated on BPA-Rat dataset across 6 acquisition protocols

**v0.6.0 (November 2024)** - Phase 6 Complete
- MSME T2 mapping with MWF calculation
- Comprehensive QC for T2 compartment analysis

**v0.5.0 (November 2024)** - Phase 5 Complete
- Multi-modal template building architecture
- ANTs-based template construction framework

**v0.4.0 (November 2024)** - Phase 4 Complete
- DTI preprocessing with GPU-accelerated eddy
- Tensor fitting and comprehensive QC

**v0.3.0 (November 2024)** - Phase 3 Complete
- Anatomical T2w preprocessing with adaptive skull stripping
- Tissue segmentation and batch processing

**v0.2.0 (November 2024)** - Phase 2 Complete
- SIGMA atlas management system
- Slice extraction and ROI operations

**v0.1.0 (November 2024)** - Phase 1 Complete
- Foundation: Configuration, transform registry, directory structure

---

## 🤝 Contributing

Neurofaune is open-source! Contributions welcome:
- **Bug reports:** GitHub Issues
- **Feature requests:** GitHub Discussions
- **Code contributions:** Pull requests (see Developer Guide)
- **Documentation:** Tutorials, examples, typo fixes

---

## 📧 Contact

- **Maintainer:** [Your Name]
- **Repository:** https://github.com/alexedmon1/neurofaune
- **Documentation:** (TBD - Phase 12)

---

**Last Updated:** December 11, 2024 by Claude Code
