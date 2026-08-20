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
import shutil
import subprocess
import tempfile

TPL_TO_SIGMA_AFFINE = "tpl-to-SIGMA_0GenericAffine.mat"
TPL_TO_SIGMA_WARP = "tpl-to-SIGMA_1Warp.nii.gz"

#: Timepoints per ``antsApplyTransforms -e 3`` call for a 4-D input.
#:
#: ANTs materialises the WHOLE output time series in memory, so a single call
#: scales with run length, not with anything the caller controls. A rat BOLD run
#: warped onto SIGMA is 128x128x218 per volume; at 360 volumes that is 1.29e9
#: voxels, which ITK holds in double unless told otherwise -- 10.3 GB for the
#: buffer alone, ~24 GB peak measured, on a 31 GB machine. It OOMed at
#: concurrency 1 and the kill left empty stderr, so the failure looked like a
#: data problem rather than a memory one.
#:
#: Chunking makes peak memory a property of THIS constant instead of a property
#: of the acquisition. 60 volumes ~= 3 GB peak with --float.
TIMESERIES_CHUNK = 60


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
    search: List[Path] = [Path(d) for d in (candidate_dirs or [])]
    if template_file:
        template_file = Path(template_file)
        # transforms/ beside the template, and the template's OWN directory.
        # ANTs writes its outputs next to the moving image by default, so a
        # study that registers template->SIGMA with a plain output prefix ends
        # up with them there. Requiring the transforms/ subdirectory meant the
        # cuprizone study bridged the gap with hand-made symlinks, which nobody
        # scripted -- so a from-scratch rebuild silently lost them and every DWI
        # session failed its SIGMA warp.
        search += [template_file.parent / "transforms", template_file.parent]

    tried: List[Path] = []
    for d in search:
        tried.append(d)
        if not d.is_dir():
            continue
        # Exact canonical name first, then any *_to-SIGMA_ prefix. ANTs derives
        # the prefix from the template name (tpl-CPZp60_to-SIGMA_...), so
        # insisting on one spelling rejects the file the tool actually produced.
        affine = d / TPL_TO_SIGMA_AFFINE
        if not affine.exists():
            hits = sorted(d.glob("*to-SIGMA_0GenericAffine.mat"))
            affine = hits[0] if hits else None
        if affine is None:
            continue

        # Pair the warp to the affine by prefix so two registrations sitting in
        # one directory cannot be mixed.
        prefix = affine.name[: -len("0GenericAffine.mat")]
        warp = d / f"{prefix}1Warp.nii.gz"
        if not warp.exists():
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

        import nibabel as nib
        is_4d = nib.load(str(src)).ndim == 4

        try:
            if is_4d:
                _warp_timeseries(src, dst, sigma_template, chain, interpolation)
            else:
                _run_ants(_ants_cmd(src, dst, sigma_template, chain, interpolation))
        except RuntimeError as e:
            print(f"  {name}: FAILED\n{e}")
            continue

        print(f"  {name} -> {dst.name}")
        out[name] = dst

    return out


def _ants_cmd(src: Path, dst: Path, sigma_template: Path, chain: Sequence[str],
              interpolation: str, timeseries: bool = False) -> List[str]:
    """Build an antsApplyTransforms invocation.

    ``--float`` is always passed: the inputs are float32 on disk, so ITK's
    default double precision doubles peak memory to buy accuracy the data does
    not carry.
    """
    cmd = ["antsApplyTransforms", "-d", "3", "--float", "1"]
    if timeseries:
        cmd += ["-e", "3"]
    cmd += ["-i", str(src), "-r", str(sigma_template), "-o", str(dst),
            "-n", interpolation]
    for t in chain:
        cmd += ["-t", t]
    return cmd


def _run_ants(cmd: List[str]) -> None:
    """Run antsApplyTransforms, raising with whatever it told us.

    A process killed by the OOM killer exits nonzero with EMPTY stderr, so
    reporting stderr alone renders the most likely failure invisible. Report the
    exit code always, and say so explicitly when stderr is empty.
    """
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        return
    detail = res.stderr.strip()
    if not detail:
        detail = (f"no stderr (exit {res.returncode}); a silent nonzero exit "
                  "usually means the process was killed -- check for OOM")
    else:
        detail = f"exit {res.returncode}\n{detail[-500:]}"
    raise RuntimeError(detail)


def _warp_timeseries(src: Path, dst: Path, sigma_template: Path,
                     chain: Sequence[str], interpolation: str) -> None:
    """Warp a 4-D series in chunks of :data:`TIMESERIES_CHUNK` volumes.

    Equivalent to one ``-e 3`` call over the whole series -- each volume is
    warped independently either way -- but peak memory is bounded by the chunk
    size rather than by the number of timepoints. See TIMESERIES_CHUNK.
    """
    import nibabel as nib
    import numpy as np

    img = nib.load(str(src))
    n_vols = img.shape[3]
    if n_vols <= TIMESERIES_CHUNK:
        _run_ants(_ants_cmd(src, dst, sigma_template, chain, interpolation,
                            timeseries=True))
        return

    tmpdir = Path(tempfile.mkdtemp(prefix="sigma_warp_", dir=dst.parent))
    try:
        warped_chunks: List[Path] = []
        for start in range(0, n_vols, TIMESERIES_CHUNK):
            stop = min(start + TIMESERIES_CHUNK, n_vols)
            chunk_in = tmpdir / f"in_{start:04d}.nii.gz"
            chunk_out = tmpdir / f"out_{start:04d}.nii.gz"
            # slicer keeps the chunk off the heap between calls; dataobj slicing
            # reads only the requested volumes rather than the whole series.
            nib.save(
                nib.Nifti1Image(np.asarray(img.dataobj[..., start:stop]),
                                img.affine, img.header),
                str(chunk_in))
            _run_ants(_ants_cmd(chunk_in, chunk_out, sigma_template, chain,
                                interpolation, timeseries=True))
            chunk_in.unlink()
            warped_chunks.append(chunk_out)

        merged = nib.concat_images([nib.load(str(c)) for c in warped_chunks],
                                   axis=3)
        # concat_images gives (x,y,z,1,t) when the parts are themselves 4-D.
        data = np.asarray(merged.dataobj)
        if data.ndim == 5:
            data = data.reshape(data.shape[:3] + (-1,))
        ref = nib.load(str(warped_chunks[0]))
        outimg = nib.Nifti1Image(data.astype(np.float32), ref.affine, ref.header)
        outimg.header.set_data_dtype(np.float32)
        nib.save(outimg, str(dst))

        n_out = nib.load(str(dst)).shape[3]
        if n_out != n_vols:
            raise RuntimeError(
                f"chunked warp produced {n_out} volumes, expected {n_vols}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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

#: The subset fitted AFTER run_dwi_preprocessing, by run_multishell_fitting.
#:
#: The DWI workflow warps to SIGMA before these exist and reports them as "not
#: yet fitted"; nothing warped them afterwards, so a normal cohort run produced
#: 4 of the 11 space-SIGMA maps and every run needed backfill_sigma_warps.py or
#: the analysis stage found no kurtosis and no NODDI at all.
#:
#: Derived from DWI_SIGMA_METRICS rather than written out again, so adding a
#: kurtosis or NODDI metric in one place cannot leave it unwarped in the other.
MULTISHELL_SIGMA_METRICS: Dict[str, str] = {
    name: pattern for name, pattern in DWI_SIGMA_METRICS.items()
    if "model-" in pattern
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
