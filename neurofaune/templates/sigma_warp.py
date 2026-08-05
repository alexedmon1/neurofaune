"""Warp modality maps into SIGMA atlas space.

Group analysis consumes ``space-SIGMA_*`` images from derivatives -- see
``neurofaune.network.roi_extraction.discover_sigma_metrics``, which globs for
them rather than warping anything itself. Producing them is therefore a
preprocessing responsibility, and this module is the one place that does it, for
every modality.

The chain is always ``modality -> cohort template -> SIGMA``. ANTs applies
transforms in reverse order (last listed is applied first), which this module
handles so callers pass them in the natural order.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import subprocess

TPL_TO_SIGMA_AFFINE = "tpl-to-SIGMA_0GenericAffine.mat"
TPL_TO_SIGMA_WARP = "tpl-to-SIGMA_1Warp.nii.gz"


def resolve_tpl_to_sigma(
    template_file: Optional[Path] = None,
    candidate_dirs: Optional[Sequence[Path]] = None,
) -> Dict[str, Optional[Path]]:
    """Locate the template->SIGMA transforms.

    Studies do not agree on where these live. The obvious guess is a
    ``transforms/`` directory beside the cohort template, but a study may keep
    them keyed by timepoint instead (e.g. ``templates/anat/1/transforms/`` for
    ses-1) while the template itself sits in ``templates/p60/``. Silently
    skipping when the guess misses is how a whole cohort ends up with no
    SIGMA-space outputs and nobody notices until the analysis stage finds
    nothing, so callers should pass explicit ``candidate_dirs`` and check
    ``found``.

    Parameters
    ----------
    template_file : Path, optional
        Cohort template; ``template_file.parent / 'transforms'`` is tried last.
    candidate_dirs : sequence of Path, optional
        Directories to try first, in order.

    Returns
    -------
    dict
        ``affine``, ``warp`` (None if the registration was affine-only),
        ``found`` (bool), ``searched`` (list of directories tried).
    """
    tried: List[Path] = []
    for d in list(candidate_dirs or []) + (
            [template_file.parent / "transforms"] if template_file else []):
        d = Path(d)
        tried.append(d)
        affine = d / TPL_TO_SIGMA_AFFINE
        if affine.exists():
            warp = d / TPL_TO_SIGMA_WARP
            return {"affine": affine, "warp": warp if warp.exists() else None,
                    "found": True, "searched": tried}
    return {"affine": None, "warp": None, "found": False, "searched": tried}


def warp_maps_to_sigma(
    metric_files: Dict[str, Path],
    moving_to_template: Path,
    sigma_template: Path,
    output_dir: Path,
    subject: str,
    session: str,
    tpl_to_sigma_affine: Path,
    tpl_to_sigma_warp: Optional[Path] = None,
    interpolation: str = "Linear",
    suffix_style: str = "metric",
    force: bool = False,
) -> Dict[str, Path]:
    """Warp scalar maps (or a 4D timeseries) from modality space into SIGMA.

    Parameters
    ----------
    metric_files : dict
        ``{name: path}``. ``name`` becomes part of the output filename.
    moving_to_template : Path
        Modality -> cohort template transform (affine .mat or composite warp).
    sigma_template : Path
        Reference image defining the output geometry.
    output_dir : Path
        Where the ``space-SIGMA_*`` images are written (the modality's
        derivatives directory).
    subject, session : str
        BIDS identifiers.
    tpl_to_sigma_affine, tpl_to_sigma_warp : Path
        From :func:`resolve_tpl_to_sigma`. ``warp`` may be None.
    interpolation : str
        ANTs interpolation. Use ``NearestNeighbor`` for label images.
    suffix_style : {'metric', 'bold'}
        ``metric`` -> ``{sub}_{ses}_space-SIGMA_{name}.nii.gz`` (DWI, MSME).
        ``bold``   -> ``{sub}_{ses}_space-SIGMA_bold.nii.gz`` when name is
        ``bold``, else ``..._space-SIGMA_desc-{name}_bold.nii.gz``, matching
        what ``roi_extraction`` expects for functional data.
    force : bool
        Rewrite outputs that already exist. Default False SKIPS them, which is
        wrong after an upstream refit -- pass True whenever the inputs changed.

    Returns
    -------
    dict
        ``{name: output_path}`` for maps that were produced or already present.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ANTs applies the LAST transform first, so the modality->template step
    # must be listed last.
    chain: List[str] = []
    if tpl_to_sigma_warp is not None:
        chain.append(str(tpl_to_sigma_warp))
    chain.append(str(tpl_to_sigma_affine))
    chain.append(str(moving_to_template))

    out: Dict[str, Path] = {}
    for name, src in metric_files.items():
        src = Path(src)
        if suffix_style == "bold":
            fname = (f"{subject}_{session}_space-SIGMA_bold.nii.gz"
                     if name == "bold"
                     else f"{subject}_{session}_space-SIGMA_desc-{name}_bold.nii.gz")
        else:
            fname = f"{subject}_{session}_space-SIGMA_{name}.nii.gz"
        dst = output_dir / fname

        if not src.exists():
            print(f"  {name}: input not found ({src.name}), skipping")
            continue
        if dst.exists() and not force:
            print(f"  {name}: exists, skipping (pass force=True to rewrite)")
            out[name] = dst
            continue

        cmd = ["antsApplyTransforms", "-d", "3",
               "-i", str(src), "-r", str(sigma_template), "-o", str(dst),
               "-n", interpolation]
        # 4D input needs -e 3 so ANTs treats it as a timeseries of 3D volumes
        import nibabel as nib
        if nib.load(str(src)).ndim == 4:
            cmd = ["antsApplyTransforms", "-d", "3", "-e", "3",
                   "-i", str(src), "-r", str(sigma_template), "-o", str(dst),
                   "-n", interpolation]
        for t in chain:
            cmd += ["-t", t]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  {name}: FAILED\n{res.stderr[-500:]}")
            continue
        print(f"  {name} -> {dst.name}")
        out[name] = dst

    return out


# Metric sets per modality. Kept here so every caller warps the same things and
# a new metric is added in one place.
DWI_SIGMA_METRICS: Dict[str, str] = {
    # tensor
    "FA": "{prefix}_FA.nii.gz",
    "MD": "{prefix}_MD.nii.gz",
    "AD": "{prefix}_AD.nii.gz",
    "RD": "{prefix}_RD.nii.gz",
    # kurtosis
    "MK": "{prefix}_model-DKI_MK.nii.gz",
    "AK": "{prefix}_model-DKI_AK.nii.gz",
    "RK": "{prefix}_model-DKI_RK.nii.gz",
    "KFA": "{prefix}_model-DKI_KFA.nii.gz",
    # NODDI
    "ODI": "{prefix}_model-NODDI_ODI.nii.gz",
    "FICVF": "{prefix}_model-NODDI_FICVF.nii.gz",
    "FISO": "{prefix}_model-NODDI_FISO.nii.gz",
}

MSME_SIGMA_METRICS: Dict[str, str] = {
    "T2": "{prefix}_T2.nii.gz",
    "MWF": "{prefix}_MWF.nii.gz",
    "IWF": "{prefix}_IWF.nii.gz",
    "CSFF": "{prefix}_CSFF.nii.gz",
}


def build_metric_files(derivatives_dir: Path, prefix: str,
                       spec: Dict[str, str]) -> Dict[str, Path]:
    """Expand a metric spec into ``{name: path}``, keeping only what exists."""
    out = {}
    for name, pattern in spec.items():
        p = Path(derivatives_dir) / pattern.format(prefix=prefix)
        if p.exists():
            out[name] = p
    return out


COVERAGE_MASK_NAME = "desc-brain_mask"


def warp_coverage_mask(
    mask_file: Path,
    moving_to_template: Path,
    sigma_template: Path,
    output_dir: Path,
    subject: str,
    session: str,
    tpl_to_sigma_affine: Path,
    tpl_to_sigma_warp: Optional[Path] = None,
    force: bool = False,
) -> Optional[Path]:
    """Warp a session brain mask into SIGMA as the COVERAGE mask.

    These are slab acquisitions -- a 27-slice DWI or an 11-slice MSME does not
    span the atlas -- so a warped map is zero wherever the slab did not reach.
    `network.roi_extraction` needs to tell those voxels from genuine zeros, and
    only an explicit mask can: MWF, for one, returns exact zeros in-slab where
    NNLS finds no short-T2 component. Without this file ROI extraction falls
    back to finite-nonzero and silently mixes the two.

    NearestNeighbor, so the mask stays binary.

    Returns the output path, or None when `mask_file` does not exist.
    """
    mask_file = Path(mask_file)
    if not mask_file.exists():
        print(f"  Coverage mask not found ({mask_file.name}); ROI extraction "
              f"will fall back to finite-nonzero coverage.")
        return None

    out = warp_maps_to_sigma(
        metric_files={COVERAGE_MASK_NAME: mask_file},
        moving_to_template=moving_to_template,
        sigma_template=sigma_template,
        output_dir=output_dir,
        subject=subject,
        session=session,
        tpl_to_sigma_affine=tpl_to_sigma_affine,
        tpl_to_sigma_warp=tpl_to_sigma_warp,
        interpolation="NearestNeighbor",
        suffix_style="metric",
        force=force,
    )
    return out.get(COVERAGE_MASK_NAME)


def sigma_targets_from_config(
    config: Dict[str, Any],
    session: str,
    cohort: Optional[str] = None,
    study_root: Optional[Path] = None,
    template_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Resolve the SIGMA reference and template->SIGMA transforms from config.

    Reads ``atlas.study_space.template`` for the reference image and, optionally,
    ``atlas.study_space.tpl_to_sigma_dir`` for where the transforms live. The
    latter is a path template accepting ``{study_root}``, ``{cohort}`` and
    ``{session_num}`` -- studies key these directories differently, and guessing
    is what caused a whole cohort to be silently skipped.

    Returns a dict with ``sigma_template``, ``affine``, ``warp``, ``ready``
    (bool) and ``reason`` (why not, when not ready).

    Set ``atlas.study_space.required: true`` to make a non-ready result RAISE
    instead of returning. A printed skip is not a safeguard in a cohort run: on
    the cuprizone study it printed on all 52 sessions, every run exited 0, and
    the missing warps were only found later by grepping the log. If a study's
    analysis stage depends on space-SIGMA images, failing the session is the
    honest behaviour.
    """
    study_space = (config.get("atlas", {}) or {}).get("study_space", {}) or {}
    sigma_template = study_space.get("template")
    required = bool(study_space.get("required", False))

    def _not_ready(reason, tpl=None):
        if required:
            raise RuntimeError(
                f"SIGMA warp is required (atlas.study_space.required) but is "
                f"not resolvable: {reason}"
            )
        return {"ready": False, "sigma_template": tpl, "affine": None,
                "warp": None, "reason": reason}

    if not sigma_template or not Path(sigma_template).exists():
        return _not_ready(
            f"atlas.study_space.template not set or missing ({sigma_template!r})"
        )

    candidates: List[Path] = []
    spec = study_space.get("tpl_to_sigma_dir")
    if spec:
        session_num = session.replace("ses-", "")
        candidates.append(Path(str(spec).format(
            study_root=str(study_root or ""), cohort=cohort or "",
            session_num=session_num)))

    res = resolve_tpl_to_sigma(template_file=template_file,
                               candidate_dirs=candidates)
    if not res["found"]:
        searched = ", ".join(str(p) for p in res["searched"]) or "(nowhere)"
        return _not_ready(
            f"template->SIGMA transforms not found; searched: {searched}",
            tpl=Path(sigma_template),
        )

    return {"ready": True, "sigma_template": Path(sigma_template),
            "affine": res["affine"], "warp": res["warp"], "reason": None}
