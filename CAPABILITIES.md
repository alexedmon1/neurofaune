# neurofaune capabilities

_Generated from the code by `neurofaune capabilities` (v0.7.2a0)._ Do not edit by hand — run `make capabilities`.

**CLI subcommands:** `bids`, `capabilities`, `check-paths`

**Entry points:** 168 across 10 stages.


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
| `warp_metric_to_sigma` | `neurofaune.analysis.tbss.prepare_tbss` | Warp a DTI metric map to SIGMA study-space using the full transform chain. | `atlas.study_space`, `paths.study_root` |
| `build_coverage_mask` | `neurofaune.analysis.tbss.prepare_template_tbss` | Intersect WM mask with per-voxel subject coverage. | — |
| `warp_atlas_to_template` | `neurofaune.analysis.tbss.prepare_template_tbss` | Warp SIGMA atlas assets to per-cohort template space. | — |
| `warp_metric_to_template` | `neurofaune.analysis.tbss.prepare_template_tbss` | Warp one DTI metric to template space via the FA_to_template affine. | — |
| `run_tbss_statistical_analysis` | `neurofaune.analysis.tbss.run_tbss_stats` | Run statistical analysis on prepared TBSS data. | `paths.study_root` |
| `compute_jacobian` | `neurofaune.analysis.vbm.prepare_vbm` | Compute Jacobian determinant from a displacement field. | — |
| `warp_tissue_to_sigma` | `neurofaune.analysis.vbm.prepare_vbm` | Warp a native-space tissue probability map to SIGMA space. | — |

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
| `extract_roi_means` | `neurofaune.network.roi_extraction` | Compute mean metric value within each labeled ROI, over COVERED voxels only. | — |

## preprocess (qc)

| function | module | summary | config keys |
|---|---|---|---|
| `compute_slice_metrics` | `neurofaune.preprocess.qc.batch_summary` | Compute per-slice QC metrics for a DTI FA map. | — |
| `select_absolute_gates` | `neurofaune.preprocess.qc.batch_summary` | Split configured gates into ones that discriminate and ones that don't. | — |
| `select_zscore_metrics` | `neurofaune.preprocess.qc.batch_summary` | Decide which metrics may be z-scored, and in which direction. | — |
| `compute_fd_from_confounds` | `neurofaune.preprocess.qc.func.motion_qc` | Compute per-volume framewise displacement from a BIDS confounds TSV. | — |

## preprocess (utils)

| function | module | summary | config keys |
|---|---|---|---|
| `select_best` | `neurofaune.preprocess.utils.bet4animal` | Pick the best candidate (pure function — unit-testable without FSL). | — |
| `degibbs_dwi` | `neurofaune.preprocess.utils.dwi_denoise` | Remove Gibbs ringing from a 3-D/4-D image (slices along axis 2). Returns ``out_file``. | — |
| `denoise_dwi_mppca` | `neurofaune.preprocess.utils.dwi_denoise` | MP-PCA denoise a 4-D DWI series. Returns ``out_file``. | — |
| `convert_5d_to_4d` | `neurofaune.preprocess.utils.dwi_utils` | Convert 5D DWI data to 4D by averaging or selecting across 5th dimension. | — |
| `extract_b0_volume` | `neurofaune.preprocess.utils.dwi_utils` | Extract first b0 volume from DWI data. | — |
| `normalize_dwi_intensity` | `neurofaune.preprocess.utils.dwi_utils` | Normalize DWI intensity to a consistent range for robust brain extraction. | — |
| `normalize_for_brain_extraction` | `neurofaune.preprocess.utils.dwi_utils` | Range-compress an image so brain extraction behaves. MASKING ONLY. | — |
| `extract_acompcor_components` | `neurofaune.preprocess.utils.func.acompcor` | Extract aCompCor components from CSF and white matter regions. | — |
| `run_melodic_ica` | `neurofaune.preprocess.utils.func.ica_denoising` | Run FSL MELODIC ICA decomposition. | — |
| `compute_meica_kappa_rho` | `neurofaune.preprocess.utils.func.meica_classify` | Compute kappa (TE-dependence) and rho (TE-independence) for MELODIC components. | — |
| `extract_slice_order_from_json` | `neurofaune.preprocess.utils.func.slice_timing` | Extract slice timing information from BIDS JSON sidecar. | — |
| `run_slice_timing_correction` | `neurofaune.preprocess.utils.func.slice_timing` | Perform slice timing correction using FSL slicetimer. | — |
| `build_affine` | `neurofaune.preprocess.utils.mrs.bruker_mrs` | Build the NIfTI affine for the SVS voxel. | — |
| `convert_session` | `neurofaune.preprocess.utils.mrs.bruker_mrs` | Find, read and convert a session's SVS acquisition in one call. | — |
| `select_svs_scan` | `neurofaune.preprocess.utils.mrs.bruker_mrs` | Pick the real SVS acquisition from a session's PRESS scans. | — |
| `fit_with_lcmodel` | `neurofaune.preprocess.utils.mrs.lcmodel` | Fit a preprocessed NIfTI-MRS spectrum with LCModel. | — |
| `run_lcmodel` | `neurofaune.preprocess.utils.mrs.lcmodel` | Run LCModel, feeding it the control file on stdin. | — |
| `fit_mm_lineshape` | `neurofaune.preprocess.utils.mrs.mm_quantify` | Fit MM09 as one complex Lorentzian with a free phase. | — |
| `fit_mm_spline` | `neurofaune.preprocess.utils.mrs.mm_quantify` | Fit a least-squares cubic spline to the metabolite-free signal. | — |
| `compute_tissue_fractions` | `neurofaune.preprocess.utils.mrs.voxel_geometry` | Rasterise the voxel and measure its tissue content in one call. | — |
| `propagate_anat_image` | `neurofaune.preprocess.utils.registration_utils` | Warp any anat-space image into moving space, reusing an existing registration. | — |
| `propagate_anat_mask` | `neurofaune.preprocess.utils.registration_utils` | Derive a brain mask by warping the same-session anat mask into moving space. | — |
| `register_via_anat_composition` | `neurofaune.preprocess.utils.registration_utils` | Register a partial-slab moving reference to template VIA the same-session anat. | — |

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
| `run_dwi_preprocessing` | `neurofaune.preprocess.workflows.dwi_preprocess` | Run complete DTI/DWI preprocessing workflow. | `diffusion.dti.max_bval`, `diffusion.eddy.data_is_shelled`, `diffusion.eddy.phase_encoding_direction`, `diffusion.eddy.readout_time`, `diffusion.eddy.repol`, `diffusion.second_mask.method`, `diffusion.skull_strip.method`, `diffusion.skull_strip.n_classes`, `diffusion.topup.readout_time` |
| `warp_dti_to_sigma` | `neurofaune.preprocess.workflows.dwi_preprocess` | Warp DTI metric maps to SIGMA atlas space. | — |
| `extract_brain_from_bold` | `neurofaune.preprocess.workflows.func_preprocess` | Extract brain from BOLD image using BET. | — |
| `extract_confounds` | `neurofaune.preprocess.workflows.func_preprocess` | Extract confound regressors from motion parameters. | — |
| `register_bold_to_t2w` | `neurofaune.preprocess.workflows.func_preprocess` | Register mean BOLD to T2w within the same subject. | — |
| `register_bold_to_template` | `neurofaune.preprocess.workflows.func_preprocess` | Register mean BOLD directly to the cohort template. | — |
| `run_functional_preprocessing` | `neurofaune.preprocess.workflows.func_preprocess` | Run complete functional fMRI preprocessing workflow. | — |
| `run_motion_correction` | `neurofaune.preprocess.workflows.func_preprocess` | Perform motion correction on fMRI timeseries. | — |
| `run_multiecho_motion_correction` | `neurofaune.preprocess.workflows.func_preprocess` | Motion-correct multi-echo data using middle echo as reference. | — |
| `run_optimal_combination` | `neurofaune.preprocess.workflows.func_preprocess` | Optimally combine multi-echo data (T2*-weighted) without ICA denoising. | — |
| `run_tedana` | `neurofaune.preprocess.workflows.func_preprocess` | Run TEDANA multi-echo ICA denoising. | — |
| `convert_svs` | `neurofaune.preprocess.workflows.mrs_preprocess` | Convert a session's PRESS acquisition to NIfTI-MRS. | — |
| `run_fsl_mrs_fit` | `neurofaune.preprocess.workflows.mrs_preprocess` | Run ``fsl_mrs`` to fit and quantify the preprocessed spectrum. | `spectroscopy.baseline`, `spectroscopy.combine`, `spectroscopy.free_shift`, `spectroscopy.internal_ref`, `spectroscopy.metab_groups`, `spectroscopy.ppmlim` |
| `run_fsl_mrs_preproc` | `neurofaune.preprocess.workflows.mrs_preprocess` | Run ``fsl_mrs_preproc``. | `spectroscopy.align_window`, `spectroscopy.remove_outliers`, `spectroscopy.remove_water` |
| `run_internal_preproc` | `neurofaune.preprocess.workflows.mrs_preprocess` | Run the preprocessing chain directly, skipping the shift/phase steps. | `spectroscopy.align_window`, `spectroscopy.phase_method`, `spectroscopy.remove_outliers`, `spectroscopy.remove_water` |
| `run_lcmodel_fit` | `neurofaune.preprocess.workflows.mrs_preprocess` | Fit with LCModel instead of ``fsl_mrs``. | `spectroscopy.lcmodel.basis`, `spectroscopy.lcmodel.bin`, `spectroscopy.lcmodel.license`, `spectroscopy.ppmlim` |
| `run_mrs_preprocessing` | `neurofaune.preprocess.workflows.mrs_preprocess` | Run the full SVS pipeline for one subject/session. | `spectroscopy.basis`, `spectroscopy.export_curves`, `spectroscopy.fitter`, `spectroscopy.preproc`, `spectroscopy.remove_water` |
| `register_msme_to_t2w` | `neurofaune.preprocess.workflows.msme_preprocess` | Register MSME first echo to T2w within the same subject. | — |
| `register_msme_to_template` | `neurofaune.preprocess.workflows.msme_preprocess` | Register MSME first echo directly to the cohort template. | `msme.registration.z_anchor`, `msme.registration.z_range` |
| `run_msme_preprocessing` | `neurofaune.preprocess.workflows.msme_preprocess` | Run MSME preprocessing workflow with T2 mapping and MWF calculation. | `msme.geometry.slice_thickness_mm`, `msme.geometry.voxel_scale`, `msme.skull_strip.cog_offset_x`, `msme.skull_strip.cog_offset_y`, `msme.skull_strip.erode_voxels`, `msme.skull_strip.frac_max`, `msme.skull_strip.frac_min`, `msme.skull_strip.frac_step`, `msme.skull_strip.method`, `msme.skull_strip.n_classes`, `msme.skull_strip.target_ratio`, `msme.t2_fitting.T1_ms`, `msme.t2_fitting.epg_n_components`, `msme.t2_fitting.epg_n_workers`, `msme.t2_fitting.intra_extra_cutoff`, `msme.t2_fitting.lambda_reg`, `msme.t2_fitting.myelin_water_cutoff`, `msme.t2_fitting.n_components`, `msme.t2_fitting.stimulated_echo_correction`, `msme.t2_fitting.t2_range` |
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
| `run_consistency_qc` | `neurofaune.templates.consistency_qc` | Warp every registered subject brain into atlas space and compute the two measures. | — |
| `warp_brain_to_atlas` | `neurofaune.templates.consistency_qc` | Warp a subject brain mask into atlas (SIGMA) space via subject→template→SIGMA. | — |
| `propagate_atlas_to_bold` | `neurofaune.templates.registration` | Propagate SIGMA atlas to BOLD/fMRI space through the transform chain. | — |
| `propagate_atlas_to_bold_direct` | `neurofaune.templates.registration` | Propagate SIGMA atlas to BOLD/fMRI space via direct BOLD→Template registration. | — |
| `propagate_atlas_to_dwi` | `neurofaune.templates.registration` | Propagate SIGMA atlas to DTI/FA space through the transform chain. | — |
| `propagate_atlas_to_dwi_direct` | `neurofaune.templates.registration` | Propagate SIGMA atlas to DTI/FA space via direct FA→Template registration. | — |
| `propagate_atlas_to_msme_direct` | `neurofaune.templates.registration` | Propagate SIGMA atlas to MSME space via direct MSME→Template registration. | — |
| `propagate_labels_to_subject` | `neurofaune.templates.registration` | Propagate atlas labels to subject space. | — |
| `register_subject_to_template` | `neurofaune.templates.registration` | Register subject image to study template. | — |
| `register_within_subject` | `neurofaune.templates.registration` | Register two modalities within the same subject (e.g., T2w ↔ FA). | — |
| `warp_bold_to_sigma` | `neurofaune.templates.registration` | Warp BOLD-space maps to SIGMA atlas space. | — |
| `compute_correlation` | `neurofaune.templates.registration_qc` | Compute Pearson correlation between two images. | — |
| `compute_dice_coefficient` | `neurofaune.templates.registration_qc` | Compute Dice coefficient between two binary masks. | — |
| `compute_registration_metrics` | `neurofaune.templates.registration_qc` | Compute comprehensive registration QC metrics. | — |
| `build_metric_files` | `neurofaune.templates.sigma_warp` | Expand a metric spec into ``{name: path}``, keeping only what exists. | — |
| `warp_coverage_mask` | `neurofaune.templates.sigma_warp` | Warp a session brain mask into SIGMA as the COVERAGE mask. | — |
| `warp_maps_to_sigma` | `neurofaune.templates.sigma_warp` | Warp scalar maps (or a 4D timeseries) from modality space into SIGMA. | — |
| `compute_slice_correspondence` | `neurofaune.templates.slice_registration` | Compute which atlas region corresponds to each template slice. | — |
| `extract_coronal_slab_atlas` | `neurofaune.templates.slice_registration` | Extract and average a coronal slab from the atlas. | — |
| `extract_coronal_slice_template` | `neurofaune.templates.slice_registration` | Extract a coronal slice from the template. | — |
| `propagate_labels_slice_wise` | `neurofaune.templates.slice_registration` | Propagate atlas labels to template space using slice-wise transforms. | — |
| `register_2d_slices` | `neurofaune.templates.slice_registration` | Register two 2D slices using affine transformation. | — |

## tractography

| function | module | summary | config keys |
|---|---|---|---|
| `build_connectome` | `neurofaune.tractography.connectome` | Build a structural connectivity matrix from a tractogram. | — |
| `compute_node_coverage` | `neurofaune.tractography.connectome` | Fraction of each parcellation node lying inside the DWI field of view. | — |
| `build_5tt_from_probseg` | `neurofaune.tractography.fivett` | Assemble an MRtrix-compatible 5TT image from Atropos tissue posteriors. | — |
| `warp_5tt` | `neurofaune.tractography.fivett` | Resample a 5TT image into another space with ANTs, then renormalise. | — |
| `build_fod_template` | `neurofaune.tractography.fixel` | Build a study-specific FOD template with ``population_template``. | — |
| `compute_fixel_metrics` | `neurofaune.tractography.fixel` | Compute FD, log(FC) and FDC for one subject in template fixel space. | — |
| `compute_group_response` | `neurofaune.tractography.fixel` | Average per-subject response functions into one group response. | — |
| `register_fod_to_template` | `neurofaune.tractography.fixel` | Register one subject's FOD to the template, keeping both warps. | — |
| `run_fixel_stats` | `neurofaune.tractography.fixel` | Run connectivity-based fixel enhancement statistics. | — |
| `build_roi_seed_masks` | `neurofaune.tractography.fsl` | Split a parcellation into one binary mask per node for network tracking. | — |
| `run_bedpostx` | `neurofaune.tractography.fsl` | Fit the ball-and-sticks model with BEDPOSTX. | — |
| `run_probtrackx_connectome` | `neurofaune.tractography.fsl` | Run probtrackx2 in network mode to produce a connectivity matrix. | — |
| `convert_to_mif` | `neurofaune.tractography.mrtrix` | Convert FSL-format DWI (+ gradient table) to a single MRtrix ``.mif``. | — |
| `run_msmt_csd` | `neurofaune.tractography.mrtrix` | Estimate tissue responses and fit FODs for one session. | — |
| `run_tractography` | `neurofaune.tractography.tractogram` | Generate a tractogram from a WM FOD, optionally ACT-constrained. | — |

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
