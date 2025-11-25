# Preprocessing Workflows Implementation Summary

## Completed Implementation (2025-11-25)

### 1. DWI/DTI Preprocessing ✅

**Location**: `neurofaune/preprocess/workflows/dwi_preprocess.py`

**Features**:
- 5D→4D conversion for Bruker data
- Gradient table validation
- GPU-accelerated eddy correction (eddy_cuda)
- Brain masking with BET
- **DTI fitting using FSL dtifit** (FA, MD, AD, RD)
- Comprehensive QC with eddy/motion and DTI metrics reports

**QC Modules** (`neurofaune/preprocess/qc/dwi/`):
- `eddy_qc.py`: Motion parameters, framewise displacement, signal intensity
- `dti_qc.py`: DTI metrics histograms, slice montages, scatter plots

**Key Updates**:
- Replaced dipy DTI fitting with FSL dtifit
- Integrated comprehensive QC generation
- Added HTML reports for motion and DTI metrics

---

### 2. MSME T2 Mapping ✅

**Location**: `neurofaune/preprocess/workflows/msme_preprocess.py`

**Features**:
- Multi-echo T2-weighted data processing
- Skull stripping with BET
- **Myelin Water Fraction (MWF) calculation using NNLS**
- Intra/extra-cellular water fraction (IWF)
- CSF fraction calculation
- Mono-exponential T2 fitting

**QC Module** (`neurofaune/preprocess/qc/msme/`):
- `msme_qc.py`: MWF/IWF/CSF histograms, T2 maps, slice montages

**Implementation Details**:
- Non-Negative Least Squares (NNLS) with regularization
- T2 compartments: Myelin (10-25ms), IW (25-40ms), CSF (41-2000ms)
- Log-spaced T2 distribution (10-2000ms, 120 bins)

---

### 3. Resting-State fMRI (To Complete)

**Planned Location**: `neurofaune/preprocess/workflows/func_preprocess.py`

**Required Steps** (based on old implementation):
1. **Skull stripping**: BET with rodent-optimized parameters
2. **Motion correction**: MCFLIRT (6 DOF rigid)
3. **Spatial smoothing**: Small FWHM (0.5mm for rodents)
4. **ICA-AROMA**: Automated artifact removal
   - Run MELODIC ICA
   - Correlate ICs with motion parameters
   - Remove motion-related components (r > 0.5)
5. **Tissue segmentation**: Reuse from anatomical preprocessing or run FAST
6. **Coregistration**:
   - BOLD → T2w (rigid, 2D)
   - T2w → SIGMA (reuse anatomical transform)
7. **Bandpass filtering**: 0.01-0.08 Hz
8. **aCompCor**: Extract 5-6 components from WM/CSF mask
9. **Confound regression**: GLM with aCompCor + motion parameters

**QC Requirements** (`neurofaune/preprocess/qc/func/`):
- `motion_qc.py`: Framewise displacement, motion parameters
- `confounds_qc.py`: Confound timeseries, correlations
- `carpet_qc.py`: Carpet plots before/after denoising
- `registration_qc.py`: Overlay plots for coregistration

---

## Configuration Updates

All three workflows are configured in `configs/default.yaml`:

```yaml
# DWI parameters
diffusion:
  eddy:
    use_cuda: true
    repol: true
  dti:
    fit_method: "WLS"  # Now uses FSL dtifit
  run_qc: true

# MSME parameters
msme:
  fitting:
    method: "nonlinear"
    bounds: [10, 500]
  nnls:
    lambda_reg: 0.5
    t2_range: [10, 2000]
    n_bins: 120
  run_qc: true

# Functional parameters
functional:
  tr: 2.0
  motion_correction:
    method: "mcflirt"
  smoothing:
    fwhm: 0.5  # Rodent-optimized
  filtering:
    highpass: 0.01
    lowpass: 0.08
  denoising:
    ica_aroma:
      enabled: true
    acompcor:
      enabled: true
      num_components: 6
    glm:
      enabled: true
  run_qc: true
```

---

## Directory Structure

```
neurofaune/
├── preprocess/
│   ├── workflows/
│   │   ├── dwi_preprocess.py    ✅
│   │   ├── msme_preprocess.py   ✅
│   │   └── func_preprocess.py   ⚠️ TO IMPLEMENT
│   ├── qc/
│   │   ├── dwi/
│   │   │   ├── eddy_qc.py       ✅
│   │   │   └── dti_qc.py        ✅
│   │   ├── msme/
│   │   │   └── msme_qc.py       ✅
│   │   └── func/                ⚠️ TO IMPLEMENT
│   │       ├── motion_qc.py
│   │       ├── confounds_qc.py
│   │       ├── carpet_qc.py
│   │       └── registration_qc.py
│   └── utils/
│       ├── dwi_utils.py
│       ├── validation.py
│       └── orientation.py
```

---

## Usage Examples

### DWI Preprocessing

```python
from neurofaune.config import load_config
from neurofaune.utils.transforms import create_transform_registry
from neurofaune.preprocess.workflows.dwi_preprocess import run_dwi_preprocessing
from pathlib import Path

config = load_config(Path('config.yaml'))
registry = create_transform_registry(config, 'sub-Rat207', cohort='p60')

results = run_dwi_preprocessing(
    config=config,
    subject='sub-Rat207',
    session='ses-p60',
    dwi_file=Path('dwi.nii.gz'),
    bval_file=Path('dwi.bval'),
    bvec_file=Path('dwi.bvec'),
    output_dir=Path('/study'),
    transform_registry=registry,
    use_gpu=True
)

# Results include:
# - results['fa'], results['md'], results['ad'], results['rd']
# - results['qc_results']['eddy_qc']['html_report']
# - results['qc_results']['dti_qc']['html_report']
```

### MSME Preprocessing

```python
from neurofaune.preprocess.workflows.msme_preprocess import run_msme_preprocessing

results = run_msme_preprocessing(
    config=config,
    subject='sub-Rat207',
    session='ses-p60',
    msme_file=Path('msme.nii.gz'),
    output_dir=Path('/study'),
    transform_registry=registry,
    te_values=np.arange(10, 330, 10)  # 32 echoes from 10-320ms
)

# Results include:
# - results['mwf'], results['iwf'], results['csf'], results['t2']
# - results['qc_results']['html_report']
```

---

## Testing Plan

1. **DWI**: Test on BPA-Rat DTI data (6 directions, single-shell)
2. **MSME**: Test on BPA-Rat T2 data (32 echoes)
3. **fMRI**: Test on BPA-Rat resting-state data (360 volumes, TR=2s)

All workflows should produce:
- Preprocessed derivatives in `derivatives/{subject}/{session}/{modality}/`
- QC HTML reports in `qc/{subject}/{session}/{modality}/`
- Transform registry entries in `transforms/{subject}/`

---

## Next Steps

1. **Complete fMRI workflow implementation** with all denoising steps
2. **Implement fMRI QC modules** (motion, confounds, carpet plots)
3. **Test all workflows** on real BPA-Rat data
4. **Add multi-shell support** to DWI workflow for future 30+ direction data
5. **Integrate with batch processing** system

---

## Notes

- All workflows use **neurovrai-compatible** directory structure
- **FSL dtifit** now used for DTI fitting (not dipy)
- **Comprehensive QC** integrated into all workflows
- **Transform registry** ready for multi-modal integration
- **Config-driven** parameters for all processing steps
