# Rat MRI Preprocessing Pipeline - Exploration Report

## Overview

The `rat-mri-preprocess` project is a specialized neuroimaging preprocessing pipeline designed for multimodal rat brain MRI analysis from a Bruker 7T scanner. This document summarizes the architecture, preprocessing approaches, and reusable patterns that inform neurofaune's preprocessing design.

**Source**: `/home/edm9fd/sandbox/rat-mri-preprocess/`

---

## 1. Project Structure

### Directory Organization
```
rat-mri-preprocess/
├── anat/          # T2-weighted anatomical preprocessing (18 scripts)
├── dwi/           # Diffusion imaging & tractography (22 scripts)  
├── rest/          # Resting state fMRI preprocessing (10 scripts)
├── msme/          # Multi-echo myelin water fraction (9 scripts)
├── mtr/           # Magnetization transfer ratio (6 scripts)
├── spec/          # MR Spectroscopy (7 scripts)
├── tbss/          # Tract-based spatial statistics (1 module)
├── vbm/           # Voxel-based morphometry (experimental)
├── file_management/ # Data conversion utilities (8 scripts)
├── analysis/      # ROI extraction & clustering (3 scripts)
├── hist/          # Histopathology processing (1 script)
└── archive/       # Legacy implementations (~25 deprecated versions)
```

### Key Characteristics
- **Modular by Modality**: Each imaging type has dedicated processing directory
- **Multi-Cohort Support**: Handles 5 cohorts with multiple developmental timepoints (p30, p60, p90)
- **Iterative Development**: Archive folder shows evolution with 3+ versions of each pipeline
- **Script-Driven**: Designed for direct execution with hardcoded parameters
- **Nipype-Based**: Uses Nipype workflow engine for FSL integration

### Python Configuration
```toml
# pyproject.toml
requires-python = ">=3.13"
dependencies:
  - brukerapi>=0.1.9          # Bruker raw data access
  - nibabel>=5.3.2           # NIfTI file I/O
  - nipype>=1.10.0           # Workflow orchestration
  - numpy, pandas, scipy     # Data processing
  - scikit-learn>=1.7.2      # ACompCor implementation
  - natsort>=8.4.0           # Natural filename sorting
  - matplotlib>=3.10.7       # Visualization
```

---

## 2. Preprocessing Pipelines by Modality

### 2.1 Anatomical (T2-weighted) - `anat/`

**Purpose**: Generate reference anatomical images registered to SIGMA atlas

**Key Steps**:
1. **Slice Extraction**: Select z-range (z_min=15, z_size=14)
2. **Skull Stripping**: `bet4animal` (FSL variant for animal brains)
   - Parameters vary by cohort/timepoint
   - e.g., p30: center=[135,100,7], rad=100
3. **Intensity Normalization**: Multiply by 1000
4. **Linear Registration**: FLIRT 2D rigid (4 DOF) to modality-specific atlas
5. **Non-linear Registration**: FNIRT (B-spline) for final alignment
6. **Tissue Segmentation**: FAST for GM/WM/CSF probability maps

**Modality-Specific Atlases**:
- REST → `SIGMA_InVivo_Brain_Template_REST.nii.gz`
- DTI → `SIGMA_InVivo_Brain_Template_DTI.nii.gz`  
- MTR → `SIGMA_InVivo_Brain_Template_MTR.nii.gz`
- MSME → `SIGMA_InVivo_Brain_Template_MSME.nii.gz`

**File Naming**: `{RatID}_{Sequence}_{ProcessingStage}.nii.gz`
- Example: `Rat68_2d_anat_extract_brain_SIGMA.nii.gz`

### 2.2 Diffusion Imaging (DTI/Multi-Shell) - `dwi/`

**Purpose**: Quantify water diffusion and white matter characteristics

**Processing Pipeline**:

1. **DWI Pairing**: Combine 14 directions into 7 pairs
   ```
   pairs = [[0,7], [1,8], [2,9], [3,10], [4,11], [5,12], [6,13]]
   # Extract + average each pair
   ```

2. **B0 Extraction**: Extract reference volume (t_min=4, t_size=1)

3. **Skull Stripping**: BET with DWI-optimized parameters
   ```
   center=[80,70,3], rad=100
   ```

4. **Eddy Current Correction**: 
   - Uses FSL Eddy with CUDA acceleration
   - Requires: acqp.txt (acquisition parameters), index.txt (volume mapping)

5. **DTI Tensor Fitting**:
   - Outputs: FA, MD, AD, RD
   - Optional: BEDPOSTX (burn_in=200, n_jumps=5000, n_fibres=3)

6. **Atlas Registration**: Same 3-step as anatomical
   - Linear: FLIRT 2D to T2w
   - Non-linear: FNIRT to SIGMA DTI atlas

7. **TBSS Setup**: Register all subjects to common skeleton for group analysis

**Output Metrics**:
- FA, MD, AD, RD (standard DTI)
- Fiber orientation maps (if BEDPOSTX run)

### 2.3 Resting State fMRI - `rest/`

**Purpose**: Functional connectivity analysis with comprehensive denoising

**11-Step Preprocessing Pipeline**:

| Step | Operation | Details |
|------|-----------|---------|
| 1 | **Image Setup** | Create headers (fslcreatehd), fix orientation codes |
| 2 | **Skull Strip** | bet4animal (center=[40,25,5], rad=125) |
| 3-5 | **Volume Split/Mask/Merge** | Split 360 volumes, apply mask, recombine |
| 6 | **Motion Correction** | MCFLIRT (6 DOF) → generates .par motion file |
| 7 | **Smoothing** | Gaussian kernel (FWHM=6mm) |
| 8 | **Intensity Norm** | Scale to mean=1000 |
| 9 | **ICA-AROMA** | MELODIC → identify motion-correlated components |
| 10 | **ACompCor** | Extract tissue confounds (CSF+WM masks) |
| 11 | **Coregistration** | fMRI→T2w→SIGMA atlas (apply individually per volume) |

**Advanced Confound Handling**:

1. **ICA-AROMA Component Selection**:
   - Correlate ICA components with motion parameters
   - Flag components with correlation > 0.5
   - Remove via `fsl_regfilt`

2. **ACompCor Extraction**:
   - Segment T2w with FAST (3 tissue classes)
   - Threshold at PVE=2 (isolate CSF+WM)
   - Erode to prevent GM contamination
   - Extract 6 principal components

3. **GLM Denoising**:
   - Combine ACompCor components + motion parameters
   - Regress via GLM to create residuals

4. **Bandpass Filtering**: 0.01-0.08 Hz (AFNI 3dBandpass)

**Final Output**: `{RatID}_rest_final_sigma.nii.gz` (atlas space, fully denoised)

### 2.4 Multi-Echo T2 Relaxometry (MSME) - `msme/`

**Purpose**: Quantify myelin water fraction (MWF)

**Approach**:
1. Load multi-echo MSME data from Bruker dataset
2. Fit 3-exponential decay model:
   ```
   Signal = m1*exp(-t1*x) + m2*exp(-t2*x) + m3*exp(-t3*x) + baseline
   ```
3. Component identification:
   - m1, t1 (fast T2): Myelin water
   - m2, t2 (intermediate): Intra/extracellular water
   - m3, t3 (slow): CSF
4. Calculate MWF = m1/(m1+m2+m3)
5. Register to atlas (standard 3-step approach)

**Implementation**:
- Uses scipy.optimize.curve_fit for exponential fitting
- Multi-voxel curve fitting with initial guesses

### 2.5 Magnetization Transfer Ratio (MTR) - `mtr/`

**Purpose**: Quantify macromolecule content (myelin proxy)

**Calculation**:
```
MTR = (Reference - MT_Saturated) / Reference * 100
```

**Source**: MT-FLASH with 1500 Hz and 6000 Hz saturation pulses

**Processing**: Same atlas registration as anatomical

### 2.6 MR Spectroscopy (MRS) - `spec/`

**Purpose**: Chemical composition and metabolite quantification

**Libraries**:
- `brukerapi.Dataset`: Load Bruker FID files
- `suspect`: MRS processing pipeline

**Workflow**:
1. Load FID from rawdata.job0
2. Load reference scan (fid.refscan)
3. Frequency/phase correction via suspect
4. FFT for frequency domain analysis
5. Metabolite fitting
6. LCModel integration for quantification

### 2.7 Tract-Based Spatial Statistics (TBSS) - `tbss/`

**Purpose**: Group-level white matter analysis

**Pipeline**:
1. Register FA to anatomical (FLIRT 2D rigid)
2. Combine transformations:
   - FA→T2w transform
   - T2w→SIGMA atlas transform
   - Concatenate via ConvertXFM
3. Apply combined transform to atlas space
4. Transform MD, AD, RD with same matrix
5. Create group skeleton
6. Run voxel-wise statistics

---

## 3. Core Functions and Reusable Patterns

### 3.1 Registration Pipeline (Universal Pattern)

**Used Across All Modalities**: Anatomical, DTI, MTR, MSME, TBSS

**3-Step Pattern**:
```python
# Step 1: Linear (FLIRT 2D rigid)
flirt.inputs.searchr_x = [-20, 20]
flirt.inputs.searchr_y = [-20, 20]
flirt.inputs.args = "-schedule $FSLDIR/etc/flirtsch/sch2D_4dof"
flirt.inputs.rigid2D = True
flirt.run()

# Step 2: Non-linear (FNIRT)
fnirt.inputs.affine_file = linear_matrix  # Initialize from step 1
fnirt.inputs.in_fwhm = [8, 4, 2, 2]
fnirt.inputs.ref_fwhm = [4, 2, 1, 1]
fnirt.run()

# Step 3: Transform other modalities with same matrix
for metric in [FA, MD, AD, RD]:
    applyxfm(metric, combined_matrix, atlas_template)
```

**Key Parameters**:
- Search range: ±20mm in each direction
- 2D rigid (4 DOF) for initial alignment
- B-spline non-linear refinement

### 3.2 Skull Stripping (Animal-Specific BET)

**Key Function**: `bet4animal()`

```python
def bet4animal(infile, outfile, center, rad):
    """
    FSL BET variant for animal brains
    center: (x, y, z) brain center coordinates
    rad: brain radius in mm
    """
    command = ["-f", "0.3", "-c", center, "-r", rad,
               "-x", "1,1,2", "-w", "2.5", "-m"]
    subprocess.run(["bet4animal", infile, outfile] + command)
```

**Cohort-Specific Parameters**:
- General: center=[135,70,6], rad=125
- P30: center=[135,100,7], rad=100  
- Anatomical: center=[135,70,6], rad=125
- fMRI: center=[40,25,5], rad=125

### 3.3 Image Manipulation Functions

```python
def extract_roi(f, zmin, zsize):
    """Extract z-range from 3D image"""
    roi = fsl.ExtractROI(in_file=f, z_min=zmin, z_size=zsize)
    roi.run()

def merge(files):
    """Stack 2D slices into 3D volume"""
    merged = fsl.Merge(in_files=files, dimension='z')
    merged.run()

def fslmaths(f, cmd):
    """Generic FSL math operations"""
    subprocess.run(['fslmaths', f] + cmd + [f])

def setsformcode(f):
    """Set spatial form code = 2 (scanner coordinates)"""
    subprocess.run(["fslorient", "-setsformcode", "2", f])
```

### 3.4 Confound Extraction (fMRI-Specific)

**ICA-AROMA Component Selection**:
```python
melodic_mix = pd.read_table('aroma/melodic_mix')
motion_params = pd.read_table(rat+'_rest360_brain_mcf.nii.gz.par')

# Correlate components with motion
for i in range(melodic_mix.shape[1]):
    for j in range(6):
        coef, p = stats.pearsonr(melodic_mix[i], motion_params[j])
        if coef > 0.5:  # Flag as motion-related
            remove_IC(...)
```

**ACompCor Component Extraction**:
```python
acompcor = confounds.ACompCor(
    realigned_file=fmri_data,
    mask_files=csf_wm_mask,
    num_components=6,
    repetition_time=2)
```

### 3.5 ROI-Based Statistics

**Using Nilearn**:
```python
def extract_mean_from_SIGMA_rois(label_img, label_dict, target_img):
    """Extract mean values from labeled atlas regions"""
    masker = NiftiLabelsMasker(labels_img=label_img,
                               labels=label_dict,
                               strategy='mean')
    masker.fit(target_img)
    means = masker.transform(target_img)
    df = pd.DataFrame([masker.labels_, means[0]]).T
    return df.merge(label_dict, on='index')
```

**Applied To**:
- DTI metrics (FA, MD, AD, RD)
- Rest connectivity measures
- Any atlas-labeled image

---

## 4. Input/Output Formats

### 4.1 Data Sources

**Bruker Raw Data**:
- Format: `{experiment_number}/pdata/1/2dseq`
- Access: `brukerapi.Dataset(path)`
- Data shape: 4D (x, y, z, echoes/volumes)

**Sequence Mapping**:
```python
modalities = {
    '<T2_Axial_750um>': '2d_anat',
    '<T2map_MSME>': 'msme',
    '<DTI_EPI_8SEG>': 'dwi',
    '<mt_Array_FLASH>': 'mtr',
    '<EPI_TR500_Reps360>': 'rest360',
    '<PRESS_1H>': 'PRESS'
}
```

### 4.2 Output Directory Structure

```
/mnt/arborea/Preclinical/BPA-rat/
├── nifti/              # Raw converted NIfTI
│   └── cohort{1-5}/
│       └── Rat{ID}/{pnd}/{modality}/
│           └── Rat{ID}_{modality}.nii.gz [+ .bvec/.bval]
│
├── preproc/            # Preprocessed outputs
│   ├── anat/{cohort}/{pnd}/{rat}/
│   │   ├── *_extract.nii.gz
│   │   ├── *_brain.nii.gz
│   │   ├── *_flirt.nii.gz
│   │   └── *_SIGMA.nii.gz
│   │
│   ├── dwi/{cohort}/{pnd}/{rat}/
│   │   ├── *_dwi_avg.nii.gz
│   │   ├── *_FA.nii.gz, *_MD.nii.gz, etc.
│   │   └── SIGMA/{metric}/ (*_to_target.nii.gz)
│   │
│   └── rest/{cohort}/{pnd}/{rat}/
│       ├── *_rest_final.nii.gz
│       └── *_rest_final_sigma.nii.gz
│
├── data/roi/           # Extracted ROI statistics
│   └── dwi/*.csv       # FA, MD, AD, RD by region
│
└── rat_atlas/
    └── SIGMA_Atlas_for_BPA/
        ├── SIGMA_InVivo_Brain_Template_*.nii.gz
        └── SIGMA_InVivo_Brain_Labels_*.nii.gz
```

### 4.3 File Naming Patterns

```
{RatID}_{Sequence}_{ProcessingStage}.nii.gz

Examples:
- Rat68_2d_anat_extract.nii.gz              # Anatomical slices extracted
- Rat68_2d_anat_extract_brain.nii.gz        # Skull-stripped
- Rat68_dwi_avg.nii.gz                      # Averaged DWI
- Rat68_FA_to_target.nii.gz                 # FA in atlas space
- Rat68_rest360_brain_mcf.nii.gz            # Motion-corrected fMRI
- Rat68_rest_final_sigma.nii.gz             # Preprocessed fMRI in atlas
```

### 4.4 Support Files

**DTI Configuration**:
- `acqp.txt`: FSL acquisition parameters (phase encoding direction)
- `index.txt`: Maps each volume to acquisition parameters

**Atlas Labels**:
- `SIGMA_InVivo_Anatomical_Brain_Atlas_Labels.csv`
- Columns: Labels, Region of interest, [Hemisphere]

---

## 5. Configuration Approach

### Current Implementation (Hardcoded Parameters)

**Issue**: Parameters scattered throughout scripts

```python
# From 1-anat-sigma.py
z_min = 15
z_size = 14
center = "135 70 6"
rad = "125"
searchr_x = [-20, 20]
rest_atlas = '/mnt/arborea/Preclinical/BPA-rat/rat_atlas/SIGMA_Atlas_for_BPA/SIGMA_InVivo_Brain_Template_REST.nii.gz'
```

**Challenges**:
- Difficult to adapt for different protocols
- Inconsistencies across similar pipelines
- Hard to document parameter sources
- Difficult to track parameter changes

### Recommended Configuration (YAML-Based)

Similar to neurovrai's approach:

```yaml
# config.yaml (study-specific)
project_dir: /mnt/arborea/Preclinical/BPA-rat
nifti_dir: ${project_dir}/nifti
preproc_dir: ${project_dir}/preproc
work_dir: ${project_dir}/work

execution:
  plugin: MultiProc
  n_procs: 6
  
templates:
  sigma_rest: ${project_dir}/rat_atlas/SIGMA_Atlas_for_BPA/SIGMA_InVivo_Brain_Template_REST.nii.gz
  sigma_dti: ${project_dir}/rat_atlas/SIGMA_Atlas_for_BPA/SIGMA_InVivo_Brain_Template_DTI.nii.gz
  sigma_mtr: ${project_dir}/rat_atlas/SIGMA_Atlas_for_BPA/SIGMA_InVivo_Brain_Template_MTR.nii.gz
  sigma_msme: ${project_dir}/rat_atlas/SIGMA_Atlas_for_BPA/SIGMA_InVivo_Brain_Template_MSME.nii.gz

anatomical:
  slice_extraction:
    z_min: 15
    z_size: 14
  bet:
    general:
      center: [135, 70, 6]
      rad: 125
    p30:
      center: [135, 100, 7]
      rad: 100
  registration:
    flirt_2d:
      searchr_x: [-20, 20]
      searchr_y: [-20, 20]
      searchr_z: [-20, 20]
    fnirt:
      in_fwhm: [8, 4, 2, 2]
      ref_fwhm: [4, 2, 1, 1]
  run_qc: true

dwi:
  pairing: [[0, 7], [1, 8], [2, 9], [3, 10], [4, 11], [5, 12], [6, 13]]
  b0_extraction:
    t_min: 4
    t_size: 1
  bet:
    center: [80, 70, 3]
    rad: 100
  eddy:
    use_cuda: true
  bedpostx:
    enabled: true
    burn_in: 200
    n_jumps: 5000
    n_fibres: 3
    use_gpu: true

functional:
  motion_correction:
    dof: 6
  smoothing:
    fwhm: 6
  normalization:
    scale_factor: 1000
  temporal_filtering:
    highpass: 0.01  # Hz
    lowpass: 0.08   # Hz
  iaca_aroma:
    enabled: true
    motion_threshold: 0.5
  acompcor:
    num_components: 6
    variance_threshold: 0.5
  bet:
    center: [40, 25, 5]
    rad: 125
```

---

## 6. Dependencies and Technologies

### Python Libraries
- **brukerapi**: Bruker data access
- **nibabel**: NIfTI I/O
- **nipype**: FSL/FreeSurfer workflow orchestration
- **numpy, scipy, pandas**: Numerical/data operations
- **scikit-learn**: ACompCor implementation
- **matplotlib**: Visualization
- **suspect**: MRS processing (not in dependencies, but imported)
- **nilearn**: ROI extraction (not in dependencies, but imported)

### External Tools (Called via subprocess/Nipype)
- **FSL**: FLIRT, FNIRT, BET, MCFLIRT, FAST, Eddy, DTIFit, BEDPOSTX5, MELODIC, GLM, etc.
- **AFNI**: 3dBandpass for temporal filtering
- **Bruker converters**: bruker2nii (legacy)

---

## 7. Reusable Components for Neurofaune

### High-Value Patterns

#### 1. Universal 3-Step Registration
- **Generic pattern** applicable to any modality
- Parametrize by: atlas path, search ranges, BET parameters
- Proven effective for animal brains

#### 2. Confound Modeling Pipeline  
- ICA-AROMA component selection logic
- ACompCor tissue-based extraction
- GLM-based final denoising
- Directly applicable to mouse/other species fMRI

#### 3. Modality-Agnostic ROI Extraction
- Nilearn-based framework (atlas-agnostic)
- Standardized DataFrame output
- Reusable across all modalities

#### 4. Batch Processing Structure
- Subject iteration with path parsing
- Status file tracking for resumable processing
- Parallel execution via Nipype MultiProc

#### 5. FSL Command Wrappers
- Encapsulate tool-specific parameters
- Enable easy tool swapping (FLIRT↔ANTs)
- Cleaner code organization

### Patterns to Adapt

1. **File Conversion** (convert_bruker_files.py)
   - Adapt brukerapi pattern for neurofaune data
   - Keep modality naming convention

2. **Registration Pipeline** (1-anat-sigma.py)
   - Generalize for different atlases
   - Maintain FLIRT→FNIRT sequence
   - Parametrize BET settings

3. **Confound Handling** (rest_preproc.py)
   - ICA-AROMA component selection
   - ACompCor implementation
   - GLM-based denoising

4. **DTI/TBSS Workflow** (dwi/ modules)
   - DWI pairing/averaging logic
   - Eddy correction setup
   - Tensor fitting pipeline
   - Multi-metric registration with single matrix

5. **Spectroscopy Processing** (spec/mrs.py)
   - suspect library integration
   - Frequency correction workflow
   - Metabolite fitting approach

---

## 8. Architectural Recommendations for Neurofaune

### Design Principles

1. **Configuration-Driven**: Move parameters to config.yaml
2. **Abstraction**: Create base classes for modality-specific pipelines
3. **Modularity**: Separate concerns (registration, denoising, etc.)
4. **Error Handling**: Add validation and logging
5. **Testing**: Include unit and integration tests
6. **Documentation**: Comprehensive docstrings and guides

### Suggested Structure

```
neurofaune/
├── config.yaml                        # Study-specific configuration
├── preprocessing/
│   ├── core/
│   │   ├── registration.py           # Universal registration pipeline
│   │   ├── skull_stripping.py        # BET wrapper + variants
│   │   ├── confounds.py              # ICA-AROMA, ACompCor, GLM
│   │   ├── io.py                     # Data I/O patterns
│   │   └── utils.py                  # FSL command wrappers
│   ├── anatomical/
│   │   ├── preprocess.py             # Main anatomical workflow
│   │   └── segmentation.py           # Tissue segmentation
│   ├── dwi/
│   │   ├── preprocess.py             # DWI preprocessing
│   │   ├── eddy_correction.py        # Eddy correction setup
│   │   ├── tensor_fitting.py         # DTI metrics
│   │   └── fiber_tracking.py         # Tractography
│   ├── fmri/
│   │   ├── preprocess.py             # fMRI preprocessing
│   │   ├── motion_correction.py      # MCFLIRT wrapper
│   │   ├── denoising.py              # ICA-AROMA, ACompCor, GLM
│   │   └── filtering.py              # Temporal filtering
│   └── spectroscopy/
│       ├── preprocess.py
│       └── mrs_processing.py
├── analysis/
│   ├── roi_extraction.py             # Atlas-based ROI extraction
│   └── statistics.py                 # Group analysis
├── workflows/
│   ├── anatomical_workflow.py
│   ├── dwi_workflow.py
│   ├── fmri_workflow.py
│   └── full_pipeline.py
└── utils/
    └── config.py                      # Configuration loader
```

### Key Improvements Over rat-mri-preprocess

| Aspect | rat-mri-preprocess | Recommended | Benefit |
|--------|-------------------|-------------|---------|
| **Configuration** | Hardcoded throughout | YAML config files | Easy parameter adjustment |
| **Abstraction** | Repetitive functions | Base classes + parametrization | Reduced code duplication |
| **Error Handling** | Minimal | Try-except + logging | Better debugging |
| **Documentation** | None | Comprehensive docstrings | Easier to maintain |
| **Testing** | None | Unit + integration tests | Catch regressions |
| **Paths** | Hardcoded absolute | Relative + environment variables | Cross-platform compatible |
| **Modality Variation** | Copy-paste & modify | Single parametrized function | Consistency guaranteed |

---

## 9. Key Takeaways

### What rat-mri-preprocess Does Well
1. **Proven FSL integration patterns**: FLIRT, FNIRT, BET, MCFLIRT, etc.
2. **Complex confound handling**: ICA-AROMA + ACompCor + GLM combination
3. **Multi-modality preprocessing**: Anatomical, DTI, fMRI, MTR, MSME all integrated
4. **Robust registration**: 3-step approach (linear → non-linear)
5. **Batch processing**: Subject iteration with error tracking

### What to Adapt for Neurofaune
1. **3-step registration pattern** (FLIRT 2D + FNIRT)
2. **Confound modeling approach** (ICA-AROMA + ACompCor + GLM)
3. **ROI extraction framework** (Nilearn-based)
4. **FSL command wrappers** (reduce subprocess boilerplate)
5. **Batch processing structure** (subject iteration with status tracking)

### What to Improve
1. **Configuration management**: Move hardcoded parameters to YAML
2. **Code abstraction**: Parametrize repeated patterns
3. **Error handling**: Add validation and logging
4. **Documentation**: Comprehensive guides and docstrings
5. **Testing**: Unit and integration test coverage
6. **Dependency declarations**: Include all imported libraries

---

## 10. References

- **rat-mri-preprocess Location**: `/home/edm9fd/sandbox/rat-mri-preprocess/`
- **Key Modules**:
  - Anatomical: `/anat/1-anat-sigma.py`
  - DTI: `/dwi/2-dwi-preproc.py`
  - fMRI: `/rest/rest_preproc.py`
  - ROI extraction: `/analysis/roi_extraction.py`
  - File management: `/file_management/convert_bruker_files.py`

