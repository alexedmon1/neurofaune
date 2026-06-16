# neurofaune capabilities

_Generated from the code by `neurofaune capabilities` (v0.3.0a0)._ Do not edit by hand — run `make capabilities`.

**CLI subcommands:** `bids`, `capabilities`

**Entry points:** 119 across 9 stages.


## analysis

| function | module | summary | config keys |
|---|---|---|---|
| `compute_falff_map` | `neurofaune.analysis.func.falff` | Compute ALFF and fALFF maps for whole brain using vectorized FFT. | — |
| `compute_falff_zscore` | `neurofaune.analysis.func.falff` | Standardize ALFF and fALFF maps to z-scores within the brain mask. | — |
| `run_dual_regression` | `neurofaune.analysis.func.melodic` | Run FSL dual regression to obtain subject-specific IC spatial maps. | — |
| `run_group_melodic` | `neurofaune.analysis.func.melodic` | Run FSL MELODIC group ICA on SIGMA-space BOLD timeseries. | — |
| `build_rsn_mosaic` | `neurofaune.analysis.func.melodic_clean` | Generate a mosaic figure of RSN spatial maps. | — |
| `build_rsn_volume` | `neurofaune.analysis.func.melodic_clean` | Extract RSN components and save as a new 4D NIfTI. | — |
| `select_manual_components` | `neurofaune.analysis.func.melodic_clean` | Build classification result from a manually supplied component list. | — |
| `compute_reho_map` | `neurofaune.analysis.func.reho` | Compute ReHo (Regional Homogeneity) map for whole brain. | — |
| `compute_reho_zscore` | `neurofaune.analysis.func.reho` | Standardize ReHo map to z-scores within the brain mask. | — |
| `run_searchlight` | `neurofaune.analysis.mvpa.searchlight` | Run searchlight analysis with optional FWER correction. | — |
| `run_whole_brain_decoding` | `neurofaune.analysis.mvpa.whole_brain` | Run whole-brain decoding with PCA + permutation testing. | — |
| `extract_clusters` | `neurofaune.analysis.stats.cluster_report` | Extract significant clusters from corrected p-value map. | — |
| `compute_cohens_d_map` | `neurofaune.analysis.stats.effect_size` | Compute Cohen's d map from a t-statistic map. | — |
| `compute_contrast_variance_factors` | `neurofaune.analysis.stats.effect_size` | Compute c'(X'X)^{-1}c for each contrast row. | — |
| `compute_partial_etasq_from_fstat` | `neurofaune.analysis.stats.effect_size` | Compute partial eta-squared map from F-statistic. | — |
| `compute_partial_etasq_from_tstat` | `neurofaune.analysis.stats.effect_size` | Compute partial eta-squared map from t-statistic. | — |
| `run_randomise` | `neurofaune.analysis.stats.randomise_wrapper` | Execute FSL randomise with specified parameters. | — |
| `build_coverage_mask` | `neurofaune.analysis.tbss.prepare_template_tbss` | Intersect WM mask with per-voxel subject coverage. | — |
| `run_tbss_statistical_analysis` | `neurofaune.analysis.tbss.run_tbss_stats` | Run statistical analysis on prepared TBSS data. | `paths.study_root` |
| `compute_jacobian` | `neurofaune.analysis.vbm.prepare_vbm` | Compute Jacobian determinant from a displacement field. | — |

## atlas

| function | module | summary | config keys |
|---|---|---|---|
| `extract_modality_slices` | `neurofaune.atlas.slice_extraction` | Extract slices for a specific modality based on configuration. | — |
| `extract_slices` | `neurofaune.atlas.slice_extraction` | Extract contiguous slices from a 3D image along specified axis. | — |

## network

| function | module | summary | config keys |
|---|---|---|---|
| `run_classification` | `neurofaune.network.classification.classifiers` | LOOCV classification with linear SVM + permutation test. | — |
| `run_lda` | `neurofaune.network.classification.lda` | Run LDA and save diagnostic plots. | — |
| `run_manova` | `neurofaune.network.classification.omnibus` | Parametric MANOVA (optional, requires statsmodels). | — |
| `run_permanova` | `neurofaune.network.classification.omnibus` | PERMANOVA (Permutational Multivariate Analysis of Variance). | — |
| `run_pca` | `neurofaune.network.classification.pca` | Run PCA and save diagnostic plots. | — |
| `run_all_comparisons` | `neurofaune.network.covnet.nbs` | Run NBS for each specified pairwise comparison. | — |
| `build_territory_mapping` | `neurofaune.network.covnet.pipeline` | Map each ROI to a hybrid territory group. | — |
| `compute_territory_means` | `neurofaune.network.covnet.pipeline` | Compute per-subject mean across ROIs within each territory group. | — |
| `run_all_comparisons` | `neurofaune.network.covnet.whole_network` | Run absolute distance test for each pairwise comparison. | — |
| `run_maturation_distance` | `neurofaune.network.covnet.whole_network` | Run relative distance tests for all triplets and distance functions. | — |
| `run_rel_distance` | `neurofaune.network.covnet.whole_network` | Run relative distance tests for all triplets and distance functions. | — |
| `run_subject_rel_distance` | `neurofaune.network.covnet.whole_network` | Run subject-level relative distance tests for all triplets. | — |
| `run_edge_regression` | `neurofaune.network.edge_regression` | Run edge-level regression for one metric and cohort. | — |
| `build_groups` | `neurofaune.network.fc_graph_theory` | Build dose group labels, optionally filtering by cohort. | — |
| `compute_subject_aucs` | `neurofaune.network.fc_graph_theory` | Compute graph metric AUCs for each subject's FC matrix. | — |
| `compute_fc_matrix` | `neurofaune.network.functional` | Compute functional connectivity matrix (Pearson r -> Fisher z). | — |
| `extract_roi_timeseries` | `neurofaune.network.functional` | Extract mean timeseries for each ROI in the atlas. | — |
| `compute_all_metrics` | `neurofaune.network.graph_theory` | Compute all registered metrics across densities. | — |
| `compute_metric_curve` | `neurofaune.network.graph_theory` | Compute a single metric across a range of densities. | — |
| `compute_spearman_matrices` | `neurofaune.network.matrices` | Compute Spearman correlation matrices for each group. | — |
| `run_mcca` | `neurofaune.network.mcca` | Fit regularised Multiset Canonical Correlation Analysis. | — |
| `run_regression` | `neurofaune.network.regression` | LOOCV regression with SVR, Ridge, and PLS + permutation test. | — |
| `compute_territory_means` | `neurofaune.network.roi_extraction` | Aggregate region means into territory-level means, weighted by voxel count. | — |
| `extract_all_subjects` | `neurofaune.network.roi_extraction` | Extract ROI means for all subjects, one DataFrame per metric. | — |
| `extract_roi_means` | `neurofaune.network.roi_extraction` | Compute mean metric value within each labeled ROI. | — |

## preprocess (qc)

| function | module | summary | config keys |
|---|---|---|---|
| `compute_slice_metrics` | `neurofaune.preprocess.qc.batch_summary` | Compute per-slice QC metrics for a DTI FA map. | — |
| `compute_fd_from_confounds` | `neurofaune.preprocess.qc.func.motion_qc` | Compute per-volume framewise displacement from a BIDS confounds TSV. | — |

## preprocess (utils)

| function | module | summary | config keys |
|---|---|---|---|
| `select_best` | `neurofaune.preprocess.utils.bet4animal` | Pick the best candidate (pure function — unit-testable without FSL). | — |
| `convert_5d_to_4d` | `neurofaune.preprocess.utils.dwi_utils` | Convert 5D DWI data to 4D by averaging or selecting across 5th dimension. | — |
| `extract_b0_volume` | `neurofaune.preprocess.utils.dwi_utils` | Extract first b0 volume from DWI data. | — |
| `extract_acompcor_components` | `neurofaune.preprocess.utils.func.acompcor` | Extract aCompCor components from CSF and white matter regions. | — |
| `run_melodic_ica` | `neurofaune.preprocess.utils.func.ica_denoising` | Run FSL MELODIC ICA decomposition. | — |
| `compute_meica_kappa_rho` | `neurofaune.preprocess.utils.func.meica_classify` | Compute kappa (TE-dependence) and rho (TE-independence) for MELODIC components. | — |
| `extract_slice_order_from_json` | `neurofaune.preprocess.utils.func.slice_timing` | Extract slice timing information from BIDS JSON sidecar. | — |
| `run_slice_timing_correction` | `neurofaune.preprocess.utils.func.slice_timing` | Perform slice timing correction using FSL slicetimer. | — |

## preprocess (workflows)

| function | module | summary | config keys |
|---|---|---|---|
| `extract_slices_from_volume` | `neurofaune.preprocess.workflows.anat_preprocess` | Extract specific slices from a 3D volume and merge them. | — |
| `register_to_atlas_ants` | `neurofaune.preprocess.workflows.anat_preprocess` | Register subject to atlas using ANTs. | `anatomical.registration.convergence_threshold`, `anatomical.registration.convergence_window_size`, `anatomical.registration.iterations`, `anatomical.registration.metric_bins`, `anatomical.registration.shrink_factors`, `anatomical.registration.smoothing_sigmas`, `anatomical.registration.syn_params` |
| `run_anatomical_preprocessing` | `neurofaune.preprocess.workflows.anat_preprocess` | Run anatomical T2w preprocessing workflow. | `anatomical.intensity_normalization.factor`, `anatomical.n4.convergence_threshold`, `anatomical.n4.iterations`, `anatomical.n4.shrink_factor`, `anatomical.skull_strip.atropos_convergence`, `anatomical.skull_strip.atropos_iterations`, `anatomical.skull_strip.method`, `anatomical.skull_strip.mrf_radius`, `anatomical.skull_strip.mrf_smoothing_factor`, `anatomical.skull_strip.n_classes`, `anatomical.skull_strip.tissue_confidence_threshold`, `anatomical.tissue_segmentation.convergence`, `anatomical.tissue_segmentation.enabled`, `anatomical.tissue_segmentation.iterations`, `anatomical.tissue_segmentation.mrf_radius`, `anatomical.tissue_segmentation.mrf_smoothing_factor`, `anatomical.tissue_segmentation.n_classes` |
| `segment_brain_tissue` | `neurofaune.preprocess.workflows.anat_preprocess` | Extract tissue probability maps from Atropos skull stripping posteriors. | — |
| `segment_brain_tissue_atropos` | `neurofaune.preprocess.workflows.anat_preprocess` | Standalone Atropos tissue segmentation, decoupled from skull stripping. | — |
| `fit_dti` | `neurofaune.preprocess.workflows.dwi_preprocess` | Fit DTI model and compute FA, MD, AD, RD maps using FSL's dtifit. | — |
| `register_fa_to_t2w` | `neurofaune.preprocess.workflows.dwi_preprocess` | Register FA to T2w within the same subject. | — |
| `register_fa_to_template` | `neurofaune.preprocess.workflows.dwi_preprocess` | Register FA directly to the cohort template. | — |
| `register_to_atlas_slices` | `neurofaune.preprocess.workflows.dwi_preprocess` | Register moving image to fixed atlas slices using ANTs SyN. | — |
| `run_dwi_preprocessing` | `neurofaune.preprocess.workflows.dwi_preprocess` | Run complete DTI/DWI preprocessing workflow. | `diffusion.eddy.data_is_shelled`, `diffusion.eddy.phase_encoding_direction`, `diffusion.eddy.readout_time`, `diffusion.eddy.repol`, `diffusion.skull_strip.method`, `diffusion.skull_strip.n_classes`, `diffusion.topup.readout_time` |
| `extract_brain_from_bold` | `neurofaune.preprocess.workflows.func_preprocess` | Extract brain from BOLD image using BET. | — |
| `extract_confounds` | `neurofaune.preprocess.workflows.func_preprocess` | Extract confound regressors from motion parameters. | — |
| `register_bold_to_t2w` | `neurofaune.preprocess.workflows.func_preprocess` | Register mean BOLD to T2w within the same subject. | — |
| `register_bold_to_template` | `neurofaune.preprocess.workflows.func_preprocess` | Register mean BOLD directly to the cohort template. | — |
| `run_functional_preprocessing` | `neurofaune.preprocess.workflows.func_preprocess` | Run complete functional fMRI preprocessing workflow. | — |
| `run_motion_correction` | `neurofaune.preprocess.workflows.func_preprocess` | Perform motion correction on fMRI timeseries. | — |
| `run_multiecho_motion_correction` | `neurofaune.preprocess.workflows.func_preprocess` | Motion-correct multi-echo data using middle echo as reference. | — |
| `run_optimal_combination` | `neurofaune.preprocess.workflows.func_preprocess` | Optimally combine multi-echo data (T2*-weighted) without ICA denoising. | — |
| `run_tedana` | `neurofaune.preprocess.workflows.func_preprocess` | Run TEDANA multi-echo ICA denoising. | — |
| `register_msme_to_t2w` | `neurofaune.preprocess.workflows.msme_preprocess` | Register MSME first echo to T2w within the same subject. | — |
| `register_msme_to_template` | `neurofaune.preprocess.workflows.msme_preprocess` | Register MSME first echo directly to the cohort template. | `msme.registration.z_anchor`, `msme.registration.z_range` |
| `run_msme_preprocessing` | `neurofaune.preprocess.workflows.msme_preprocess` | Run MSME preprocessing workflow with T2 mapping and MWF calculation. | `msme.geometry.slice_thickness_mm`, `msme.geometry.voxel_scale`, `msme.skull_strip.cog_offset_x`, `msme.skull_strip.cog_offset_y`, `msme.skull_strip.frac_max`, `msme.skull_strip.frac_min`, `msme.skull_strip.frac_step`, `msme.skull_strip.method`, `msme.skull_strip.n_classes`, `msme.skull_strip.target_ratio`, `msme.t2_fitting.intra_extra_cutoff`, `msme.t2_fitting.lambda_reg`, `msme.t2_fitting.myelin_water_cutoff`, `msme.t2_fitting.n_components`, `msme.t2_fitting.t2_range` |
| `fit_dki` | `neurofaune.preprocess.workflows.multishell_models` | Fit Diffusion Kurtosis Imaging model using DIPY. | — |
| `fit_noddi` | `neurofaune.preprocess.workflows.multishell_models` | Fit NODDI model using AMICO. | — |
| `run_multishell_fitting` | `neurofaune.preprocess.workflows.multishell_models` | Run DKI and/or NODDI fitting on preprocessed multi-shell DWI data. | — |

## reporting

| function | module | summary | config keys |
|---|---|---|---|
| `build_provenance` | `neurofaune.reporting.summarize` | Build a provenance metadata dict for embedding in summary JSONs. | — |

## templates

| function | module | summary | config keys |
|---|---|---|---|
| `propagate_atlas_direct` | `neurofaune.templates.anat_registration` | Propagate SIGMA atlas to T2w using direct registration transforms. | — |
| `propagate_atlas_to_anat` | `neurofaune.templates.anat_registration` | Propagate SIGMA atlas to T2w space through the transform chain. | — |
| `register_anat_to_sigma_direct` | `neurofaune.templates.anat_registration` | Register T2w directly to SIGMA (no study template). | — |
| `register_anat_to_template` | `neurofaune.templates.anat_registration` | Register preprocessed T2w to cohort template. | — |
| `build_template` | `neurofaune.templates.builder` | Build template using ANTs multivariate template construction. | — |
| `extract_mean_bold` | `neurofaune.templates.builder` | Extract mean or median timepoint from 4D BOLD data. | — |
| `register_template_to_sigma` | `neurofaune.templates.builder` | Register study template to SIGMA atlas (T2w only). | — |
| `select_subjects_for_template` | `neurofaune.templates.builder` | Select best subjects for template building based on QC metrics. | — |
| `propagate_atlas_to_bold` | `neurofaune.templates.registration` | Propagate SIGMA atlas to BOLD/fMRI space through the transform chain. | — |
| `propagate_atlas_to_bold_direct` | `neurofaune.templates.registration` | Propagate SIGMA atlas to BOLD/fMRI space via direct BOLD→Template registration. | — |
| `propagate_atlas_to_dwi` | `neurofaune.templates.registration` | Propagate SIGMA atlas to DTI/FA space through the transform chain. | — |
| `propagate_atlas_to_dwi_direct` | `neurofaune.templates.registration` | Propagate SIGMA atlas to DTI/FA space via direct FA→Template registration. | — |
| `propagate_atlas_to_msme_direct` | `neurofaune.templates.registration` | Propagate SIGMA atlas to MSME space via direct MSME→Template registration. | — |
| `propagate_labels_to_subject` | `neurofaune.templates.registration` | Propagate atlas labels to subject space. | — |
| `register_subject_to_template` | `neurofaune.templates.registration` | Register subject image to study template. | — |
| `register_within_subject` | `neurofaune.templates.registration` | Register two modalities within the same subject (e.g., T2w ↔ FA). | — |
| `compute_correlation` | `neurofaune.templates.registration_qc` | Compute Pearson correlation between two images. | — |
| `compute_dice_coefficient` | `neurofaune.templates.registration_qc` | Compute Dice coefficient between two binary masks. | — |
| `compute_registration_metrics` | `neurofaune.templates.registration_qc` | Compute comprehensive registration QC metrics. | — |
| `compute_slice_correspondence` | `neurofaune.templates.slice_registration` | Compute which atlas region corresponds to each template slice. | — |
| `extract_coronal_slab_atlas` | `neurofaune.templates.slice_registration` | Extract and average a coronal slab from the atlas. | — |
| `extract_coronal_slice_template` | `neurofaune.templates.slice_registration` | Extract a coronal slice from the template. | — |
| `propagate_labels_slice_wise` | `neurofaune.templates.slice_registration` | Propagate atlas labels to template space using slice-wise transforms. | — |
| `register_2d_slices` | `neurofaune.templates.slice_registration` | Register two 2D slices using affine transformation. | — |

## utils

| function | module | summary | config keys |
|---|---|---|---|
| `convert_scan` | `neurofaune.utils.bids` | Convert one Bruker scan into one-or-more analysis-ready BIDS NIfTIs. | — |
| `convert_session` | `neurofaune.utils.bids` | Convert one session to BIDS; always (re)writes its scans.tsv. | — |
| `convert_study` | `neurofaune.utils.bids` | Discover + convert all (optionally subject-filtered) sessions. | — |
| `convert_bruker_to_nifti` | `neurofaune.utils.bruker_convert` | Convert Bruker scan to NIfTI using brukerapi. | — |
| `extract_bids_metadata` | `neurofaune.utils.bruker_convert` | Extract BIDS-relevant metadata from Bruker scan. | — |
| `extract_bvec_bval` | `neurofaune.utils.bruker_convert` | Extract b-values and b-vectors from DTI scan. | — |
| `select_best_dwi_from_inventory` | `neurofaune.utils.bruker_convert` | Select best DWI scan from an inventory list. | — |
| `select_best_func_from_inventory` | `neurofaune.utils.bruker_convert` | Select best functional (BOLD) scan from an inventory list. | — |
| `select_best_msme_from_inventory` | `neurofaune.utils.bruker_convert` | Select best MSME (multi-echo T2 mapping) scan from an inventory list. | — |
| `select_best_t2w_from_inventory` | `neurofaune.utils.bruker_convert` | Select best T2w scan from an inventory list. | — |
| `compute_orientation_metrics` | `neurofaune.utils.orientation` | Compute orientation metrics between two images. | — |
| `select_best_anatomical` | `neurofaune.utils.select_anatomical` | Select the best anatomical scan from a subject/session. | — |
