# Bruker to BIDS Conversion Summary

**Date**: November 18, 2025
**Project**: Neurofaune (rodent MRI preprocessing pipeline)
**Study**: BPA-Rat (Bisphenol A rat cohort study)
**Conversion Tool**: `scripts/convert_bruker_to_bids.py`

## Source Data Locations

- **Original Bruker data**: `/mnt/arborea/bruker/` (permanent storage)
- **Study folder**: `/mnt/arborea/bpa-rat/`
- **Converted BIDS data**: `/mnt/arborea/bpa-rat/raw/bids/`

**Note**: Bruker data in temporary Cohort folders has been deleted after successful conversion. The canonical Bruker data is preserved in `/mnt/arborea/bruker/`.

## Conversion Statistics

### Overall
- **141 unique subjects** (Rat001 - Rat298)
- **189 sessions** across all timepoints
- **2,256 NIfTI files** converted
- **2,256 JSON metadata sidecars** generated
- **193 DTI scans** with bvec/bval gradient files (386 files total)

### Sessions by Timepoint
| Timepoint | Sessions | Description |
|-----------|----------|-------------|
| p30 | 54 | Postnatal day 30 |
| p60 | 50 | Postnatal day 60 |
| p90 | 79 | Postnatal day 90 |
| unknown | 6 | Could not parse timepoint from directory name |

### Scans by Modality
| Modality | Count | Description | BIDS Folder |
|----------|-------|-------------|-------------|
| **anat** | 960 | T2-weighted anatomical (RARE) | `anat/` |
| **func** | 294 | Resting-state fMRI (EPI/BOLD) | `func/` |
| **flash** | 275 | FLASH sequences | `flash/` |
| **fmap** | 194 | Field maps for distortion correction | `fmap/` |
| **dwi** | 193 | Diffusion-weighted imaging (DtiEpi) | `dwi/` |
| **msme** | 189 | Multi-slice multi-echo (T2 mapping) | `msme/` |
| **mtr** | 151 | Magnetization transfer ratio | `mtr/` |

**Spectroscopy scans**: Skipped during conversion (will be processed with fsl-mrs in Phase 6)

## Directory Structure

```
/mnt/arborea/bpa-rat/raw/bids/
├── dataset_description.json
└── sub-Rat###/
    ├── ses-p30/
    │   ├── anat/
    │   │   ├── sub-Rat###_ses-p30_run-#_T2w.nii.gz
    │   │   └── sub-Rat###_ses-p30_run-#_T2w.json
    │   ├── dwi/
    │   │   ├── sub-Rat###_ses-p30_run-#_dwi.nii.gz
    │   │   ├── sub-Rat###_ses-p30_run-#_dwi.json
    │   │   ├── sub-Rat###_ses-p30_run-#_dwi.bvec
    │   │   └── sub-Rat###_ses-p30_run-#_dwi.bval
    │   ├── func/
    │   │   ├── sub-Rat###_ses-p30_run-#_bold.nii.gz
    │   │   └── sub-Rat###_ses-p30_run-#_bold.json
    │   ├── msme/
    │   │   ├── sub-Rat###_ses-p30_run-#_MSME.nii.gz
    │   │   └── sub-Rat###_ses-p30_run-#_MSME.json
    │   ├── mtr/
    │   │   ├── sub-Rat###_ses-p30_run-#_MTR.nii.gz
    │   │   └── sub-Rat###_ses-p30_run-#_MTR.json
    │   ├── flash/
    │   │   ├── sub-Rat###_ses-p30_run-#_FLASH.nii.gz
    │   │   └── sub-Rat###_ses-p30_run-#_FLASH.json
    │   └── fmap/
    │       ├── sub-Rat###_ses-p30_run-#_fieldmap.nii.gz
    │       └── sub-Rat###_ses-p30_run-#_fieldmap.json
    ├── ses-p60/
    │   └── [same structure]
    └── ses-p90/
        └── [same structure]
```

## Conversion Features

### Enhanced Metadata (JSON Sidecars)
Each NIfTI file has an accompanying JSON sidecar with BIDS-compliant metadata:

**Common fields**:
- `Manufacturer`: "Bruker"
- `MagneticFieldStrength`: 7.0 Tesla
- `ScanName`: Descriptive scan name from Bruker protocol (e.g., "T2_RARE_2D_AXIAL (E3)")
- `ProtocolName`: Bruker protocol name (e.g., "05_T2_TurboRARE_3D")
- `RepetitionTime`: TR in seconds
- `EchoTime`: TE in seconds
- `AcquisitionDateTime`: Scan acquisition timestamp

**Modality-specific fields**:
- **DTI/DWI**: `MaxBValue`, `NumberOfDirections`, `NumberOfVolumes`
- **fMRI**: `TaskName` ("rest"), `NumberOfVolumes`
- **All**: `BrukerMethod` (original Bruker method name)

### DTI Gradient Files
All diffusion scans include FSL-format gradient files:
- `.bvec`: Gradient directions (3 × N format)
- `.bval`: b-values (1 × N format)

### Separate Modality Folders
Unlike traditional BIDS which groups many scan types in `anat/`, this organization uses separate folders for each modality type to improve clarity and organization.

## Bruker Method Classification

The conversion script classified Bruker scans using the following mapping:

| Bruker Method | BIDS Modality | Suffix |
|---------------|---------------|--------|
| `Bruker:RARE` | anat | T2w |
| `Bruker:DtiEpi` | dwi | dwi |
| `Bruker:EPI` | func | bold |
| `Bruker:MSME` | msme | MSME |
| `User:mt_Array_RARE` | mtr | MTR |
| `User:mt_Array_FLASH` | mtr | MTR |
| `Bruker:FLASH` | flash | FLASH |
| `Bruker:FieldMap` | fmap | fieldmap |
| `Bruker:PRESS` | spec | svs (skipped) |

## Cohort Information

### Cohort 1
- **Sessions**: 32 (8 p30, 8 p60, 16 p90)
- **Directory format**: Old format with timepoint grouping
  - `p30_202210/`, `p60_202211/`, `p90_202212/`
  - Session dirs: `YYYYMMDD_HHMMSS_IRC###_Rat###_1_#`

### Cohort 2
- **Sessions**: 32 (9 p30, 8 p60, 15 p90)
- **Directory format**: Old format with cohort label
  - `p30/`, `p60/`, `p90/`
  - Session dirs: `YYYYMMDD_HHMMSS_IRC###_Cohort2_Rat###_1_#`

### Cohort 3
- **Sessions**: 32 (8 p30, 8 p60, 16 p90)
- **Directory format**: Same as Cohort 2
  - Session dirs: `YYYYMMDD_HHMMSS_IRC###_Cohort3_Rat###_1_#`

### Cohort 4
- **Sessions**: ~25
- **Directory format**: Same as Cohort 2/3
- **Note**: Has "P60" (capital P) in addition to lowercase timepoint folders

### Cohort 5
- **Sessions**: ~25
- **Directory format**: Same as Cohort 2/3/4

### Cohort 7
- **Sessions**: 16 (6 p30, 6 p60, 4 p90)
- **Directory format**: New format with cohort/rat in name
  - Session dirs: `IRC####_Cohort7_Rat###_1__Rat###__p##_1_1_YYYYMMDD_HHMMSS`

### Cohort 8
- **Sessions**: 29 (9 p30, 10 p60, 10 p90)
- **Directory format**: Same as Cohort 7

## Parser Implementation

The conversion script (`neurofaune/utils/bruker_convert.py`) handles three directory naming formats:

1. **Old format**: `YYYYMMDD_HHMMSS_IRC###_Rat###_1_#`
2. **Old format with cohort**: `YYYYMMDD_HHMMSS_IRC###_Cohort#_Rat###_1_#`
3. **New format**: `IRC####_Cohort#_Rat###_1__Rat###__p##_1_1_YYYYMMDD_HHMMSS`

The parser uses regex pattern matching and extracts:
- Subject ID (Rat number)
- Session (timepoint: p30, p60, p90)
- Cohort number
- Acquisition date

## Conversion Execution

### Commands Run
```bash
# Cohort 1
python scripts/convert_bruker_to_bids.py --cohort-root /mnt/arborea/7T_Scanner_new \
    --output-root /mnt/arborea/7T_Scanner_new --cohorts Cohort1 -v

# Cohorts 2-3
python scripts/convert_bruker_to_bids.py --cohort-root /mnt/arborea/7T_Scanner_new \
    --output-root /mnt/arborea/7T_Scanner_new --cohorts Cohort2 Cohort3 -v

# Cohorts 4-5
python scripts/convert_bruker_to_bids.py --cohort-root /mnt/arborea/7T_Scanner_new \
    --output-root /mnt/arborea/7T_Scanner_new --cohorts Cohort4 Cohort5 -v

# Cohorts 7-8
python scripts/convert_bruker_to_bids.py --cohort-root /mnt/arborea/7T_Scanner_new \
    --output-root /mnt/arborea/7T_Scanner_new --cohorts Cohort7 Cohort8 -v
```

### Conversion Time
- **Total time**: ~25 minutes for all cohorts
- Conversion is CPU-bound (data reading and NIfTI writing)
- Can be run in the background

## Known Issues and Warnings

1. **FLASH metadata extraction**: Some FLASH scans have list-valued parameters that cause warnings:
   ```
   Could not extract all metadata: float() argument must be a string or a real number, not 'list'
   ```
   - These scans still convert successfully with partial metadata

2. **Spectroscopy metadata**: Some spectroscopy parameters cause warnings:
   ```
   Could not extract all metadata: object of type 'int' has no len()
   ```
   - Spectroscopy scans are currently skipped (to be processed in Phase 6)

3. **Six "unknown" sessions**: Parser could not determine timepoint from directory structure
   - These sessions converted successfully but may need manual timepoint assignment

## Quality Control

### Recommended Checks
1. **Verify subject count**: Ensure 141 unique subjects across all sessions
2. **Check DTI gradients**: Confirm all 193 DTI scans have .bvec and .bval files
3. **Inspect metadata**: Random sample of JSON files for completeness
4. **Validate NIfTI headers**: Check orientations and voxel dimensions
5. **Cross-reference with Excel sheets**: Verify subject/session matches cohort Excel files

### Sample Validation Commands
```bash
# Count subjects
ls -d /mnt/arborea/bpa-rat/raw/bids/sub-Rat* | wc -l

# Count sessions by timepoint
find /mnt/arborea/bpa-rat/raw/bids -type d -name "ses-p30" | wc -l
find /mnt/arborea/bpa-rat/raw/bids -type d -name "ses-p60" | wc -l
find /mnt/arborea/bpa-rat/raw/bids -type d -name "ses-p90" | wc -l

# Verify DTI gradients
find /mnt/arborea/bpa-rat/raw/bids -name "*.bvec" | wc -l
find /mnt/arborea/bpa-rat/raw/bids -name "*.bval" | wc -l

# Check for missing JSON sidecars
diff <(find /mnt/arborea/bpa-rat/raw/bids -name "*.nii.gz" | wc -l) \
     <(find /mnt/arborea/bpa-rat/raw/bids -name "*.json" | wc -l)
```

## Next Steps (Phase 3+)

With the Bruker data successfully converted to BIDS format, the project can proceed to:

1. **Phase 3**: Anatomical preprocessing (T2w)
   - Skull stripping
   - Bias field correction
   - Registration to SIGMA atlas

2. **Phase 4**: Diffusion preprocessing (DTI/DWI)
   - Eddy current correction
   - Motion correction
   - DTI fitting (FA, MD, etc.)
   - Registration to SIGMA atlas (slice-specific)

3. **Phase 5**: Functional preprocessing (fMRI)
   - Motion correction
   - Slice timing correction
   - Spatial smoothing
   - Confound regression
   - Registration to SIGMA atlas

4. **Phase 6**: Advanced modalities
   - MSME (T2 mapping)
   - MTR (magnetization transfer)
   - FLASH processing
   - Spectroscopy (using fsl-mrs)

## References

- **BIDS Specification**: https://bids-specification.readthedocs.io/
- **Bruker ParaVision**: 6.0.1
- **Scanner**: Bruker 7T
- **Study**: BPA-Rat (Bisphenol A Rat Cohort Study)
- **Study folder**: `/mnt/arborea/bpa-rat/`
- **Project repository**: `~/sandbox/neurofaune/`
