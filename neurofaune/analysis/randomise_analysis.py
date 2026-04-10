"""
Base class and subclasses for FSL randomise-based voxelwise analyses.

Provides a common framework for VBM and voxelwise fMRI analyses that share
the same pattern: subset 4D volumes to match design subject orders, run
FSL randomise with TFCE, extract clusters, and generate reports.

Subclass-specific differences:
    - VBMAnalysis: GM/WM tissues, 3D TFCE, config auto-discovery
    - VoxelwiseFMRIAnalysis: fALFF/ReHo metrics, 3D TFCE
"""

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

from neurofaune.analysis.stats.cluster_report import generate_reports_for_all_contrasts
from neurofaune.analysis.stats.randomise_wrapper import run_randomise, summarize_results
from neurofaune.config import get_config_value, load_config


def load_subject_list(path: Path) -> List[str]:
    """Load a subject list file (one subject per line)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def validate_provenance(
    analysis_dir: Path,
    analysis_name: str,
    logger: logging.Logger,
) -> None:
    """
    Validate that a design's provenance matches the current subject list.

    Computes SHA256 of subject_list.txt and compares against the hash
    stored in the design's provenance.json.
    """
    provenance_file = analysis_dir / 'designs' / analysis_name / 'provenance.json'

    if not provenance_file.exists():
        logger.warning(
            f"  No provenance.json for {analysis_name} - "
            "skipping hash validation"
        )
        return

    with open(provenance_file) as f:
        provenance = json.load(f)

    expected_hash = provenance.get('subject_list_sha256')
    if not expected_hash:
        logger.warning(
            f"  provenance.json for {analysis_name} has no hash - skipping"
        )
        return

    subject_list_path = analysis_dir / 'subject_list.txt'
    h = hashlib.sha256()
    with open(subject_list_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    current_hash = h.hexdigest()

    if current_hash != expected_hash:
        raise ValueError(
            f"Subject list mismatch for design '{analysis_name}'!\n"
            f"  Current subject_list.txt SHA256:  {current_hash[:16]}...\n"
            f"  Design provenance expected:       {expected_hash[:16]}...\n"
            f"  Re-run design preparation scripts to update."
        )

    logger.info(
        f"  Provenance OK for {analysis_name} "
        f"(hash: {current_hash[:16]}..., "
        f"n_design={provenance.get('n_subjects_in_design', '?')})"
    )


def subset_4d_volume(
    input_4d: Path,
    master_subjects: List[str],
    design_subjects: List[str],
    output_path: Path,
    logger: logging.Logger,
) -> Path:
    """Extract volumes from a 4D NIfTI to match a design's subject order."""
    master_index = {subj: i for i, subj in enumerate(master_subjects)}

    indices = []
    missing = []
    for subj in design_subjects:
        if subj in master_index:
            indices.append(master_index[subj])
        else:
            missing.append(subj)

    if missing:
        raise ValueError(
            f"{len(missing)} design subjects not found in master list: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    img = nib.load(input_4d)
    data = img.get_fdata()

    if data.shape[3] != len(master_subjects):
        raise ValueError(
            f"4D volume has {data.shape[3]} volumes but master list has "
            f"{len(master_subjects)} subjects"
        )

    subset_data = data[:, :, :, indices].astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(subset_data, img.affine, img.header), output_path)

    logger.info(
        f"  Subsetted {input_4d.name}: "
        f"{data.shape[3]} -> {subset_data.shape[3]} volumes"
    )
    return output_path


def _metric_randomise_complete(output_dir: Path, metric: str) -> bool:
    """Check if randomise output already exists for a metric."""
    metric_dir = output_dir / f"randomise_{metric}"
    if not metric_dir.is_dir():
        return False
    corrp_files = list(metric_dir.glob("randomise_tfce_corrp_*.nii.gz"))
    return len(corrp_files) > 0


class RandomiseAnalysis:
    """Base class for FSL randomise-based voxelwise analyses.

    Subclasses must set ANALYSIS_TYPE, DEFAULT_METRICS, TFCE_MODE_LABEL,
    and CONFIG_PATH_KEY, and may override prepare() and
    _resolve_parcellation().
    """

    ANALYSIS_TYPE: str = ""
    DEFAULT_METRICS: List[str] = []
    TFCE_MODE_LABEL: str = "3D"
    CONFIG_PATH_KEY: str = ""

    def __init__(
        self,
        analysis_dir: Path,
        config: Optional[Dict] = None,
        force: bool = False,
    ):
        self.analysis_dir = Path(analysis_dir)
        self.config = config
        self.force = force
        self.logger = logging.getLogger(f"neurofaune.{self.ANALYSIS_TYPE}")

    @classmethod
    def prepare(
        cls,
        config_path: Optional[Path] = None,
        analysis_dir: Optional[Path] = None,
        force: bool = False,
    ) -> "RandomiseAnalysis":
        """Factory: load config, resolve analysis_dir, validate layout.

        Parameters
        ----------
        config_path : Path, optional
            Study config YAML. If given and analysis_dir is None, derives
            the analysis directory from cls.CONFIG_PATH_KEY.
        analysis_dir : Path, optional
            Explicit analysis directory (overrides config-derived path).
        force : bool
            If True, --force semantics (delete existing results before run).

        Returns
        -------
        RandomiseAnalysis (or subclass) instance, ready to call .run().
        """
        config = None
        if config_path is not None:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found: {config_path}")
            config = load_config(config_path)

        if analysis_dir is None and config is not None and cls.CONFIG_PATH_KEY:
            derived = get_config_value(config, cls.CONFIG_PATH_KEY, default=None)
            if derived:
                analysis_dir = Path(derived)

        if analysis_dir is None:
            raise ValueError(
                "analysis_dir is required (either explicitly or via config)"
            )

        analysis_dir = Path(analysis_dir)
        if not analysis_dir.exists():
            raise FileNotFoundError(
                f"Analysis directory not found: {analysis_dir}"
            )

        for subdir in ("stats", "designs"):
            if not (analysis_dir / subdir).exists():
                raise FileNotFoundError(
                    f"Required subdirectory not found: {analysis_dir / subdir}"
                )

        return cls(analysis_dir=analysis_dir, config=config, force=force)

    # ------------------------------------------------------------------
    # Result directory helpers
    # ------------------------------------------------------------------

    def _result_dir(self, analysis_name: str) -> Path:
        return self.analysis_dir / "randomise" / analysis_name

    def _check_or_clear(self, analysis_name: str) -> None:
        """If force is set, delete existing results for this analysis."""
        result_dir = self._result_dir(analysis_name)
        if self.force and result_dir.exists():
            self.logger.info(
                f"  --force: removing existing results in {result_dir}"
            )
            shutil.rmtree(result_dir)

    # ------------------------------------------------------------------
    # Parcellation resolution (subclasses may override)
    # ------------------------------------------------------------------

    def _resolve_parcellation(
        self, parcellation_override: Optional[Path] = None
    ) -> Optional[Path]:
        """Resolve the SIGMA parcellation path for cluster labeling."""
        if parcellation_override and Path(parcellation_override).exists():
            return Path(parcellation_override)

        if self.config:
            study_root = Path(
                get_config_value(self.config, 'paths.study_root', default='')
            )
            parc_path = (
                study_root / 'atlas' / 'SIGMA_study_space'
                / 'SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz'
            )
            if parc_path.exists():
                return parc_path

        return None

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------

    def run(
        self,
        analysis_name: str,
        metrics: Optional[List[str]] = None,
        n_permutations: int = 5000,
        cluster_threshold: float = 0.95,
        min_cluster_size: int = 10,
        seed: Optional[int] = None,
        skip_existing: bool = False,
        parcellation_override: Optional[Path] = None,
    ) -> Dict:
        """Run randomise for a single analysis (design).

        Parameters
        ----------
        analysis_name : str
            Name of the design directory under designs/.
        metrics : list of str, optional
            Metrics to analyze. Defaults to cls.DEFAULT_METRICS.
        n_permutations : int
            Number of permutations for randomise.
        cluster_threshold : float
            1-p threshold for cluster extraction (0.95 = p<0.05).
        min_cluster_size : int
            Minimum cluster size in voxels.
        seed : int, optional
            Random seed for reproducibility.
        skip_existing : bool
            If True, skip metrics whose randomise output already exists.
        parcellation_override : Path, optional
            Override path to SIGMA parcellation NIfTI.

        Returns
        -------
        dict with 'success', 'analysis_name', 'n_subjects', 'output_dir',
        'results'.
        """
        if metrics is None:
            metrics = list(self.DEFAULT_METRICS)

        design_dir = self.analysis_dir / "designs" / analysis_name
        stats_dir = self.analysis_dir / "stats"
        output_dir = self._result_dir(analysis_name)

        self.logger.info("=" * 70)
        self.logger.info(f"ANALYSIS: {analysis_name}")
        self.logger.info("=" * 70)

        # --- Load design subject order ---
        subject_order_file = design_dir / "subject_order.txt"
        if not subject_order_file.exists():
            raise FileNotFoundError(
                f"subject_order.txt not found in {design_dir}"
            )
        design_subjects = load_subject_list(subject_order_file)
        n_design = len(design_subjects)
        self.logger.info(f"Design subjects: {n_design}")

        # --- Validate design matrix dimensions ---
        design_mat = design_dir / "design.mat"
        design_con = design_dir / "design.con"
        if not design_mat.exists():
            raise FileNotFoundError(f"design.mat not found: {design_mat}")
        if not design_con.exists():
            raise FileNotFoundError(f"design.con not found: {design_con}")

        n_points = None
        n_waves = None
        with open(design_mat) as f:
            for line in f:
                if '/NumPoints' in line:
                    n_points = int(line.split()[-1])
                elif '/NumWaves' in line:
                    n_waves = int(line.split()[-1])

        if n_points != n_design:
            raise ValueError(
                f"design.mat has NumPoints={n_points} but subject_order.txt "
                f"has {n_design} subjects"
            )
        self.logger.info(
            f"Design matrix: {n_points} subjects x {n_waves} predictors"
        )

        n_contrasts = None
        with open(design_con) as f:
            for line in f:
                if '/NumContrasts' in line:
                    n_contrasts = int(line.split()[-1])
        self.logger.info(f"Contrasts: {n_contrasts}")

        # --- Load contrast names from design summary ---
        contrast_names = None
        design_summary_file = design_dir / "design_summary.json"
        if design_summary_file.exists():
            with open(design_summary_file) as f:
                design_summary = json.load(f)
            contrast_names = design_summary.get('contrasts', None)

        # --- Load master subject list ---
        master_list_file = self.analysis_dir / "subject_list.txt"
        if not master_list_file.exists():
            raise FileNotFoundError(
                f"Master subject list not found: {master_list_file}"
            )
        master_subjects = load_subject_list(master_list_file)

        # --- Subset 4D volumes ---
        self.logger.info("\nSubsetting 4D volumes...")
        analysis_mask = stats_dir / "analysis_mask.nii.gz"
        if not analysis_mask.exists():
            raise FileNotFoundError(
                f"Analysis mask not found: {analysis_mask}"
            )

        subset_dir = output_dir / "data"
        subset_dir.mkdir(parents=True, exist_ok=True)
        metric_files = {}

        for metric in metrics:
            master_4d = stats_dir / f"all_{metric}.nii.gz"
            if not master_4d.exists():
                raise FileNotFoundError(
                    f"Master 4D not found: {master_4d}"
                )

            subset_file = subset_dir / f"all_{metric}.nii.gz"
            if subset_file.exists():
                existing_shape = nib.load(subset_file).shape
                if len(existing_shape) == 4 and existing_shape[3] == n_design:
                    self.logger.info(
                        f"  {metric}: using existing subset "
                        f"({existing_shape[3]} volumes)"
                    )
                    metric_files[metric] = subset_file
                    continue
                else:
                    self.logger.info(
                        f"  {metric}: existing subset has wrong shape "
                        f"{existing_shape}, re-creating"
                    )

            subset_4d_volume(
                input_4d=master_4d,
                master_subjects=master_subjects,
                design_subjects=design_subjects,
                output_path=subset_file,
                logger=self.logger,
            )
            metric_files[metric] = subset_file

        # --- Run randomise for each metric ---
        tfce_label = self.TFCE_MODE_LABEL
        self.logger.info(
            f"\nRunning FSL randomise "
            f"({n_permutations} permutations, {tfce_label} TFCE)..."
        )
        all_results = {}

        for metric in metrics:
            self.logger.info(f"\n  --- {metric} ---")
            metric_output = output_dir / f"randomise_{metric}"

            if skip_existing and _metric_randomise_complete(output_dir, metric):
                self.logger.info(
                    f"  {metric}: randomise output exists, "
                    "skipping (--skip-existing)"
                )
                all_results[metric] = {
                    'randomise': {'skipped': True},
                    'summary': summarize_results(
                        metric_output, cluster_threshold
                    ),
                }
                continue

            randomise_result = run_randomise(
                input_file=metric_files[metric],
                design_mat=design_mat,
                contrast_con=design_con,
                output_dir=metric_output,
                mask=analysis_mask,
                n_permutations=n_permutations,
                tfce=True,
                tfce_2d=False,  # 3D TFCE for volumetric data
                seed=seed,
            )

            all_results[metric] = {
                'randomise': randomise_result,
                'summary': summarize_results(
                    metric_output, cluster_threshold
                ),
            }

        # --- Extract clusters and generate reports ---
        self.logger.info("\nExtracting clusters...")
        sigma_parcellation = self._resolve_parcellation(parcellation_override)

        for metric in metrics:
            metric_output = output_dir / f"randomise_{metric}"
            reports_dir = output_dir / f"cluster_reports_{metric}"

            cluster_results = generate_reports_for_all_contrasts(
                randomise_output_dir=metric_output,
                output_dir=reports_dir,
                contrast_names=contrast_names,
                sigma_parcellation=sigma_parcellation,
                threshold=cluster_threshold,
                min_cluster_size=min_cluster_size,
            )
            all_results[metric]['clusters'] = cluster_results

        # --- Log summary ---
        self.logger.info(f"\n{analysis_name} results:")
        for metric in metrics:
            summary = all_results[metric]['summary']
            for contrast in summary['contrasts']:
                status = "SIGNIFICANT" if contrast['significant'] else "ns"
                self.logger.info(
                    f"  {metric} {contrast['type']}"
                    f"{contrast['contrast_number']}: "
                    f"{contrast['n_significant_voxels']} voxels ({status})"
                )

        # --- Save analysis summary ---
        summary_file = output_dir / 'analysis_summary.json'
        summary_data = {
            'analysis_name': analysis_name,
            'analysis_type': self.ANALYSIS_TYPE,
            'tfce_mode': tfce_label,
            'date': datetime.now().isoformat(),
            'n_subjects': n_design,
            'n_predictors': n_waves,
            'n_contrasts': n_contrasts,
            'contrast_names': contrast_names,
            'metrics': metrics,
            'n_permutations': n_permutations,
            'results': {
                metric: {
                    'n_significant_contrasts': sum(
                        1 for c in all_results[metric]['summary']['contrasts']
                        if c['significant']
                    ),
                    'contrasts': all_results[metric]['summary']['contrasts'],
                }
                for metric in metrics
            },
        }
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)

        return {
            'success': True,
            'analysis_name': analysis_name,
            'n_subjects': n_design,
            'output_dir': str(output_dir),
            'results': all_results,
        }


class VBMAnalysis(RandomiseAnalysis):
    """VBM voxel-wise analysis with FSL randomise (3D TFCE)."""

    ANALYSIS_TYPE = "vbm"
    DEFAULT_METRICS = ["GM", "WM"]
    TFCE_MODE_LABEL = "3D"
    CONFIG_PATH_KEY = "paths.analysis.vbm"

    @classmethod
    def prepare(
        cls,
        config_path: Optional[Path] = None,
        analysis_dir: Optional[Path] = None,
        force: bool = False,
    ) -> "VBMAnalysis":
        """Prepare a VBMAnalysis with config auto-discovery.

        If config_path is not given, walks parent directories of
        analysis_dir looking for config.yaml.
        """
        # Auto-discover config from analysis_dir parents
        if config_path is None and analysis_dir is not None:
            for parent in Path(analysis_dir).resolve().parents:
                candidate = parent / 'config.yaml'
                if candidate.exists():
                    config_path = candidate
                    break

        return super().prepare(
            config_path=config_path,
            analysis_dir=analysis_dir,
            force=force,
        )

    def detect_metrics(self) -> List[str]:
        """Auto-detect metrics from vbm_config.json if present."""
        config_file = self.analysis_dir / 'vbm_config.json'
        if config_file.exists():
            with open(config_file) as f:
                vbm_cfg = json.load(f)
            return vbm_cfg.get('tissues', list(self.DEFAULT_METRICS))
        return list(self.DEFAULT_METRICS)


class VoxelwiseFMRIAnalysis(RandomiseAnalysis):
    """Whole-brain voxelwise fMRI analysis with FSL randomise (3D TFCE)."""

    ANALYSIS_TYPE = "voxelwise_fmri"
    DEFAULT_METRICS = ["fALFF", "ReHo"]
    TFCE_MODE_LABEL = "3D (-T)"
    CONFIG_PATH_KEY = "paths.analysis.voxelwise_fmri"
