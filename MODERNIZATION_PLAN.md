# Network Analysis Modernization Plan

All network analysis modules should follow the CovNetAnalysis pattern:
- Class-level `prepare(config_path=, force=)` factory
- `run_*()` methods with `_check_or_clear()` guard
- Config-driven path resolution via `resolve_paths()` helper
- Scripts are thin CLI wrappers (~100 lines)

Reference implementation: `neurofaune/network/covnet/pipeline.py`

## Quick Wins (ALL COMPLETE)

### QW-1: CovNet Territory script ✅
**Files:** `scripts/run_covnet_territory.py`
**Effort:** Small — already uses CovNetAnalysis, just missing --config/--force in the CLI wrapper.
- Add `--config`, `--force` args
- Make `--roi-dir`/`--output-dir` optional
- Use `resolve_covnet_paths()` for path resolution
- Pass `force` to `prepare()`
- Move progress/summary to `_variant_dir()` level

### QW-2: VBM analysis script ✅ (--force added; full refactor in MR-5)
**Files:** `scripts/run_vbm_analysis.py`, `neurofaune/analysis/vbm.py`
**Effort:** Small — module already calls `load_config()` internally.
- Expose `--config` in argparse
- Add `--force` with directory existence check before running

### QW-3: Voxelwise fMRI script ✅ (--force added; full refactor in MR-5)
**Files:** `scripts/run_voxelwise_fmri.py`, `neurofaune/analysis/voxelwise_fmri.py`
**Effort:** Small — same as VBM, module already loads config.
- Expose `--config` in argparse
- Add `--force`

### QW-4: FC Graph Theory script ✅ (--config/--force added)
**Files:** `scripts/run_fc_graph_theory.py`, `neurofaune/network/fc_graph_theory.py`
**Effort:** Small — clean module API, just needs --config/--force in CLI.
- Add `--config`, `--force` args
- Add config path resolution for ROI/output dirs

---

## Major Refactors

### MR-1: ClassificationAnalysis class ✅
**Files:**
- `neurofaune/network/classification/` (module — exists, has functions)
- `scripts/run_classification_analysis.py` (565 lines → ~100)

**Current state:** Script imports 6+ functions and orchestrates manually with nested loops over cohorts, metrics, feature sets. Logic for data preparation, analysis dispatch, and result aggregation lives in the script.

**Target:**
```python
class ClassificationAnalysis:
    @classmethod
    def prepare(cls, config_path=None, roi_dir=None, output_dir=None,
                modality="dwi", metric="FA", cohort="p30",
                feature_set="all", force=False): ...
    
    def run_permanova(self, n_perm=5000): ...
    def run_pca(self): ...
    def run_lda(self): ...
    def run_classifiers(self, n_perm=5000): ...
    def run_all(self, n_perm=5000): ...  # convenience
```

**Steps:**
1. Add `paths.network.classification` to config.yaml and default.yaml
2. Create `ClassificationAnalysis` class in `neurofaune/network/classification/pipeline.py`
3. Move data prep, grouping, feature selection from script to `prepare()`
4. Move PERMANOVA, PCA, LDA, classifier dispatch to `run_*()` methods
5. Add `_check_or_clear()` and `resolve_paths()` pattern
6. Reduce script to thin CLI wrapper
7. Update README.md with new API

### MR-2: RegressionAnalysis class ✅
**Files:**
- `neurofaune/network/regression.py` (module — single function)
- `scripts/run_regression_analysis.py` (475 lines → ~100)

**Current state:** Similar to classification — script orchestrates loops over cohorts/metrics/feature sets with inline result aggregation.

**Target:**
```python
class RegressionAnalysis:
    @classmethod
    def prepare(cls, config_path=None, roi_dir=None, output_dir=None,
                modality="dwi", metric="FA", cohort="p30",
                feature_set="all", target="dose", force=False): ...
    
    def run_svr(self, n_perm=5000): ...
    def run_ridge(self, n_perm=5000): ...
    def run_pls(self, n_perm=5000): ...
    def run_all(self, n_perm=5000): ...
```

**Steps:**
1. Add `paths.network.regression` to config
2. Create `RegressionAnalysis` class (can go in `regression.py` or new `regression/pipeline.py`)
3. Move data prep, target selection, feature selection from script
4. Move SVR/Ridge/PLS dispatch to methods
5. Add force/config pattern
6. Reduce script

### MR-3: MCCAAnalysis class ✅
**Files:**
- `neurofaune/network/mcca.py` (module — multiple functions)
- `scripts/run_mcca_analysis.py` (660 lines → ~150)

**Current state:** `run_cohort_mcca()` in the script is 250+ lines of orchestration. Design discovery, modality combination, and result writing are all in the script.

**Target:**
```python
class MCCAAnalysis:
    @classmethod
    def prepare(cls, config_path=None, roi_dir=None, output_dir=None,
                cohort="p30", design="pooled", force=False): ...
    
    def run(self, n_perm=1000, n_components=5): ...
```

**Steps:**
1. Add `paths.network.mcca` to config
2. Create class with prepare/run pattern
3. Move design discovery and modality combination logic to module
4. Move `run_cohort_mcca()` to class method
5. Add force/config pattern
6. Reduce script

### MR-4: MVPAAnalysis class
**Files:**
- `neurofaune/analysis/mvpa.py` (if exists) or new module
- `scripts/run_mvpa_analysis.py` (667 lines → ~100)

**Current state:** Large script with design discovery, cross-validation setup, and multi-metric orchestration.

**Target:**
```python
class MVPAAnalysis:
    @classmethod
    def prepare(cls, config_path=None, derivatives_dir=None, output_dir=None,
                modality="dwi", metric="FA", cohort="p30", force=False): ...
    
    def run(self, classifier="svm", n_perm=1000): ...
```

**Steps:**
1. Create `neurofaune/analysis/mvpa/pipeline.py` (or add to existing)
2. Move design discovery to module
3. Move cross-validation and classifier dispatch to class methods
4. Add force/config pattern
5. Reduce script

### MR-5: RandomiseAnalysis base class (VBM + Voxelwise fMRI)
**Files:**
- `neurofaune/analysis/randomise_analysis.py` (NEW — shared base class)
- `neurofaune/analysis/vbm.py` or `vbm_analysis.py` (may need creation)
- `neurofaune/analysis/voxelwise_fmri.py` or similar (may need creation)
- `scripts/run_vbm_analysis.py` (622 lines → ~80)
- `scripts/run_voxelwise_fmri_analysis.py` (621 lines → ~80)

**Current state:** Both scripts are ~620 lines and nearly identical. They share:
- `setup_logging()`, `load_subject_list()`, `validate_provenance()`, `subset_4d_volume()`
- `run_single_analysis()` (~200 lines each, identical structure)
- Same FSL randomise + cluster report + reporting registration pattern

Only differences: tissue list (GM/WM vs fALFF/ReHo), TFCE mode labeling, config auto-discovery in VBM.

**Target:**
```python
class RandomiseAnalysis:
    """Base class for FSL randomise-based voxelwise analyses."""
    @classmethod
    def prepare(cls, config_path=None, analysis_dir=None,
                analyses=None, metrics=None, force=False): ...
    
    def run(self, n_permutations=5000, seed=None,
            cluster_threshold=0.95, min_cluster_size=10,
            skip_existing=False): ...

class VBMAnalysis(RandomiseAnalysis):
    DEFAULT_METRICS = ['GM', 'WM']
    ANALYSIS_TYPE = 'vbm'

class VoxelwiseFMRIAnalysis(RandomiseAnalysis):
    DEFAULT_METRICS = ['fALFF', 'ReHo']
    ANALYSIS_TYPE = 'voxelwise_fmri'
```

**Steps:**
1. Create `neurofaune/analysis/randomise_analysis.py` with shared base class
2. Move `subset_4d_volume`, `validate_provenance`, `load_subject_list`, `run_single_analysis` to base
3. Create `VBMAnalysis` and `VoxelwiseFMRIAnalysis` subclasses with metric defaults
4. Add force/config pattern (prepare checks for existing randomise dirs)
5. Reduce both scripts to thin wrappers

### MR-6: EdgeRegressionAnalysis class ✅
**Files:**
- `neurofaune/network/edge_regression.py` (module — exists)
- `scripts/run_edge_regression.py` (~200 lines → ~80)

**Current state:** Moderate script size, module has core logic. Needs config/force pattern.

**Target:**
```python
class EdgeRegressionAnalysis:
    @classmethod
    def prepare(cls, config_path=None, roi_dir=None, output_dir=None,
                modality="dwi", metric="FA", target="log_auc",
                force=False): ...
    
    def run(self, n_perm=1000, threshold=3.0): ...
```

**Steps:**
1. Add `paths.network.edge_regression` to config
2. Create class or add to existing module
3. Add config/force pattern
4. Update script

---

## Config Path Additions

All new paths to add to `config.yaml` and `configs/default.yaml`:

```yaml
paths:
  network:
    roi: ${paths.study_root}/network/roi
    covnet: ${paths.study_root}/network/covnet          # already done
    classification: ${paths.study_root}/network/classification
    regression: ${paths.study_root}/network/regression
    mcca: ${paths.study_root}/network/mcca
    edge_regression: ${paths.study_root}/network/edge_regression
    fc_graph_theory: ${paths.study_root}/network/fc_graph_theory
  analysis:
    tbss: ${paths.study_root}/analysis/tbss
    vbm: ${paths.study_root}/analysis/vbm
    voxelwise_fmri: ${paths.study_root}/analysis/voxelwise_fmri
    mvpa: ${paths.study_root}/analysis/mvpa
```

---

## Execution Order

1. ~~**Quick wins** (QW-1 through QW-4)~~ DONE
2. ~~**Config path additions**~~ DONE
3. **Edge regression** (MR-6) — smallest major refactor, good warmup
4. **Classification** (MR-1) — template for regression
5. **Regression** (MR-2) — mirrors classification pattern
6. **MCCA** (MR-3) — unique but similar pattern
7. **VBM + Voxelwise fMRI** (MR-5) — shared base class, two scripts
8. **MVPA** (MR-4) — most complex, do last

Each step gets its own commit. Config path additions (all at once) get a separate commit before the refactors.

---

## Verification Checklist (per module)

- [ ] `--config` resolves paths from config.yaml
- [ ] `--force` deletes existing results before running
- [ ] Without `--force`, existing results cause FileExistsError
- [ ] `--roi-dir`/`--output-dir` work as backwards-compat overrides
- [ ] Script is <150 lines
- [ ] All orchestration logic is in the module, not the script
- [ ] Summary/progress files go to analysis+variant directory
- [ ] No nested directory creation
- [ ] README.md updated with new API examples
