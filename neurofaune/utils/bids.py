#!/usr/bin/env python3
"""Config-driven Bruker (ParaVision) → BIDS/NIfTI conversion.

This is the user-facing converter behind ``neurofaune bids``. It is driven
entirely by a study config (no per-study Python) and fixes the limitations of
the legacy ``bruker_convert.process_all_cohorts`` path:

* **Config-driven discovery + naming** — session directories are matched by a
  configurable regex with named ``subject``/``session`` groups, in either a
  ``flat`` or ``nested`` layout (gh #3).
* **Overridable sequence map** — method→(modality, suffix) defaults can be
  extended/overridden from config; ships ``T2S_EPI`` (BOLD) and ``FISP`` (gh #4).
* **Correct per-modality dimensionality** — frame groups (``VisuFGOrderDesc``)
  are interpreted so DWI is a 4-D ``(x,y,z,vol)`` aligned to bval/bvec and
  multi-echo BOLD is emitted as 4-D ``(x,y,z,t)`` per echo, instead of a blanket
  ``squeeze`` that leaves 5-D arrays (gh #5).
* **Per-session ``scans.tsv``** — every scan (written *and* skipped, with reason)
  plus parameters is listed in each ``ses-`` folder (gh #8).

The legacy functions in :mod:`neurofaune.utils.bruker_convert` are left intact
for back-compat; this module reuses their pure helpers where they are correct
(``get_bruker_method``, ``extract_bids_metadata``, ``extract_bvec_bval``).
"""
from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Defaults (overridable from config)
# --------------------------------------------------------------------------- #
# method -> (modality, suffix). Modalities map to BIDS-ish sub-folders.
DEFAULT_SEQUENCE_MAP: Dict[str, Tuple[str, str]] = {
    "Bruker:RARE": ("anat", "T2w"),
    "Bruker:FLASH": ("anat", "FLASH"),
    "Bruker:MSME": ("anat", "MESE"),       # multi-echo (T2 mapping)
    "Bruker:FieldMap": ("fmap", "fieldmap"),
    "Bruker:DtiEpi": ("dwi", "dwi"),
    "Bruker:EPI": ("func", "bold"),
    "Bruker:T2S_EPI": ("func", "bold"),    # gh #4: T2*-EPI BOLD (was dropped)
    "Bruker:FISP": ("anat", "FISP"),       # gh #4: was dropped
    "Bruker:PRESS": ("spec", "svs"),
}

# Modalities discovered but not written as NIfTI (handled elsewhere).
DEFAULT_SKIP_MODALITIES = {"spec"}

# How a frame-group axis is treated when assembling a modality volume.
_CYCLE_NAMES = {"cycle", "movie", "repetition", "rep"}


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested; no I/O)
# --------------------------------------------------------------------------- #
def clean_fg_name(raw: str) -> str:
    """'<FG_SLICE>' -> 'slice'."""
    return raw.replace("<", "").replace(">", "").replace("FG_", "").strip().lower()


def frame_group_names(visu_fg_order_desc: Optional[List]) -> List[str]:
    """Extract cleaned frame-group names (in axis order) from VisuFGOrderDesc.

    VisuFGOrderDesc rows look like ``[size, '<FG_SLICE>', '<>', start, n]``.
    Returns e.g. ``['slice', 'diffusion', 'cycle']``.
    """
    if not visu_fg_order_desc:
        return []
    out = []
    for row in visu_fg_order_desc:
        try:
            out.append(clean_fg_name(str(row[1])))
        except (IndexError, TypeError):
            out.append("unknown")
    return out


def assemble_modality(
    data: np.ndarray,
    fg_names: List[str],
    modality: str,
    spatial_dim: int = 2,
) -> Dict[str, np.ndarray]:
    """Reorder a raw Bruker array into analysis-ready BIDS volume(s).

    Parameters
    ----------
    data : np.ndarray
        Raw array of shape ``(X, Y[, ...frame-group axes])`` for a 2-D
        acquisition (``spatial_dim=2``). Frame-group axes follow the spatial
        axes in ``fg_names`` order.
    fg_names : list of str
        Cleaned frame-group names aligned to the trailing axes, e.g.
        ``['slice', 'diffusion', 'cycle']``.
    modality : str
        Target modality (``anat``/``dwi``/``func``/``fmap`` ...). Controls which
        trailing axes are kept, collapsed (mean), or split.
    spatial_dim : int
        Number of leading spatial axes in ``data`` (2 for 2-D ParaVision).

    Returns
    -------
    dict
        Maps a BIDS entity suffix ('' for the plain image, or 'echo-<n>') to an
        output array. DWI -> ``(x,y,z,vol)``; multi-echo func -> one
        ``(x,y,z,t)`` per echo; MSME/MESE -> ``(x,y,z,echo)``; anat -> ``(x,y,z)``.
    """
    if data.ndim == spatial_dim:  # no frame groups (already a plain image)
        return {"": data}
    if len(fg_names) != data.ndim - spatial_dim:
        # Fall back: trust array as-is rather than guess.
        logger.warning("FG names (%s) do not match array ndim %s; passing through",
                       fg_names, data.ndim)
        return {"": np.squeeze(data)}

    axis_of = {nm: spatial_dim + i for i, nm in enumerate(fg_names)}

    # Build axis permutation: spatial, then slice (z), then modality-ordered rest.
    perm: List[int] = list(range(spatial_dim))
    if "slice" in axis_of:
        perm.append(axis_of["slice"])
    z_present = "slice" in axis_of

    priority = {
        "dwi": ["diffusion", "cycle", "movie", "repetition"],
        "func": ["echo", "cycle", "movie", "repetition"],
        "anat": ["echo", "cycle"],          # MSME/MESE keep echo
        "fmap": ["echo", "cycle"],
    }.get(modality, ["echo", "diffusion", "cycle", "movie", "repetition"])

    used = set(perm)
    ordered_rest: List[int] = []
    for nm in priority:
        ax = axis_of.get(nm)
        if ax is not None and ax not in used:
            ordered_rest.append(ax)
            used.add(ax)
    for ax in range(spatial_dim, data.ndim):  # any leftover axes
        if ax not in used:
            ordered_rest.append(ax)
            used.add(ax)

    arr = np.transpose(data, perm + ordered_rest)
    base = spatial_dim + (1 if z_present else 0)
    rest_names = [fg_names[ax - spatial_dim] for ax in ordered_rest]

    # Collapse pure repetition/cycle axes (mean) for non-functional modalities;
    # for func, the cycle axis is the BOLD time axis and is kept.
    def _collapse(a: np.ndarray, names: List[str], to_collapse: set) -> Tuple[np.ndarray, List[str]]:
        i = 0
        while i < len(names):
            if names[i] in to_collapse:
                a = a.mean(axis=base + i)
                names = names[:i] + names[i + 1:]
            else:
                i += 1
        return a, names

    if modality == "func":
        # keep echo (split) and cycle (BOLD time axis); collapse anything else
        keep = {"echo"} | _CYCLE_NAMES
        arr, rest_names = _collapse(arr, rest_names, {n for n in rest_names if n not in keep})
        if "echo" not in rest_names:
            return {"": arr}
        echo_axis = base + rest_names.index("echo")
        n_echo = arr.shape[echo_axis]
        return {f"echo-{e + 1}": np.take(arr, e, axis=echo_axis) for e in range(n_echo)}

    if modality == "dwi":
        arr, rest_names = _collapse(arr, rest_names, _CYCLE_NAMES)
        return {"": arr}

    # anat (incl. MSME/MESE keep echo), fmap, flash, etc.: collapse cycles only.
    arr, rest_names = _collapse(arr, rest_names, _CYCLE_NAMES)
    return {"": arr}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class BidsConfig:
    """Resolved ``bids`` settings (from a study config dict or kwargs)."""

    def __init__(self, raw_root: Path, bids_root: Path, *,
                 layout: str = "flat",
                 session_regex: Optional[str] = None,
                 session_relabel: Optional[Dict[str, str]] = None,
                 sequence_map: Optional[Dict[str, Tuple[str, str]]] = None,
                 skip_modalities: Optional[set] = None,
                 voxel_scale: float = 10.0):
        self.raw_root = Path(raw_root)
        self.bids_root = Path(bids_root)
        self.layout = layout
        self.session_regex = re.compile(session_regex) if session_regex else None
        self.session_relabel = session_relabel or {}
        self.sequence_map = {**DEFAULT_SEQUENCE_MAP, **(sequence_map or {})}
        self.skip_modalities = set(skip_modalities) if skip_modalities is not None else set(DEFAULT_SKIP_MODALITIES)
        self.voxel_scale = voxel_scale

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "BidsConfig":
        b = dict(cfg.get("bids", {}))
        if "raw_root" not in b or "bids_root" not in b:
            raise ValueError("config 'bids' block requires 'raw_root' and 'bids_root'")
        seqmap = {k: tuple(v) if isinstance(v, (list, tuple)) else (v["modality"], v["suffix"])
                  for k, v in (b.get("sequence_map") or {}).items()}
        return cls(
            raw_root=b["raw_root"], bids_root=b["bids_root"],
            layout=b.get("layout", "flat"),
            session_regex=b.get("session_regex"),
            session_relabel=b.get("session_relabel"),
            sequence_map=seqmap or None,
            skip_modalities=set(b["skip_modalities"]) if "skip_modalities" in b else None,
            voxel_scale=float(b.get("voxel_scale", 10.0)),
        )


def parse_session_name(name: str, regex: Optional[re.Pattern],
                       relabel: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Apply the configured regex to a session dir name -> {subject, session, ...}."""
    if regex is None:
        return None
    m = regex.search(name)
    if not m:
        return None
    g = {k: v for k, v in m.groupdict().items() if v is not None}
    if "subject" not in g or "session" not in g:
        raise ValueError("session_regex must define named groups 'subject' and 'session'")
    g["subject"] = g["subject"].upper()
    g["session"] = relabel.get(g["session"], g["session"])
    return g


# --------------------------------------------------------------------------- #
# Discovery + conversion (I/O)
# --------------------------------------------------------------------------- #
def _looks_like_session(d: Path) -> bool:
    numbered = [x for x in d.iterdir() if x.is_dir() and x.name.isdigit()]
    return any((x / "method").exists() for x in numbered[:5])


def discover_sessions(cfg: BidsConfig) -> List[Tuple[Path, Dict[str, str]]]:
    """Find + parse session dirs under raw_root for the configured layout."""
    candidates: List[Path] = []
    if cfg.layout == "nested":
        for parent in sorted(p for p in cfg.raw_root.iterdir() if p.is_dir()):
            candidates += [d for d in sorted(parent.iterdir()) if d.is_dir()]
    else:  # flat
        candidates = [d for d in sorted(cfg.raw_root.iterdir()) if d.is_dir()]

    sessions = []
    for d in candidates:
        try:
            if not _looks_like_session(d):
                continue
        except OSError:
            continue
        meta = parse_session_name(d.name, cfg.session_regex, cfg.session_relabel)
        if meta is None:
            logger.warning("unparsed session dir (skipped): %s", d.name)
            continue
        sessions.append((d, meta))
    return sessions


def _numbered_scans(session_dir: Path) -> List[Path]:
    return sorted(
        (d for d in session_dir.iterdir() if d.is_dir() and d.name.isdigit() and (d / "method").exists()),
        key=lambda p: int(p.name),
    )


def convert_scan(scan_dir: Path, modality: str, suffix: str, base_stem: str,
                 out_dir: Path, voxel_scale: float = 10.0) -> List[Path]:
    """Convert one Bruker scan into one-or-more analysis-ready BIDS NIfTIs.

    Returns the list of written .nii.gz paths.
    """
    from brukerapi.dataset import Dataset  # local import (optional dep)
    from neurofaune.utils.fix_bruker_voxel_sizes import parse_bruker_method, update_nifti_header

    pdata = scan_dir / "pdata" / "1" / "2dseq"
    if not pdata.exists():
        logger.error("2dseq not found: %s", pdata)
        return []

    ds = Dataset(str(pdata))
    ds.add_parameter_file("visu_pars")
    visu = ds.parameters["visu_pars"].to_dict()
    data = ds.data
    if np.iscomplexobj(data):
        data = np.abs(data)

    spatial_dim = int(visu.get("VisuCoreDim", {}).get("value", 2)) if isinstance(visu.get("VisuCoreDim"), dict) else 2
    fg = frame_group_names(_get(visu, "VisuFGOrderDesc"))
    volumes = assemble_modality(data, fg, modality, spatial_dim=spatial_dim)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    voxel = None
    try:
        voxel = parse_bruker_method(scan_dir / "method").get("voxel_size")
    except Exception:  # noqa: BLE001
        pass

    import nibabel as nib
    for entity, vol in volumes.items():
        vol = np.squeeze(vol)
        out_file = out_dir / (bids_filename(base_stem, entity, suffix) + ".nii.gz")
        nib.save(nib.Nifti1Image(np.asarray(vol), np.eye(4)), str(out_file))
        if voxel is not None:
            try:
                update_nifti_header(out_file, voxel, scale_factor=voxel_scale)
            except Exception as e:  # noqa: BLE001
                logger.warning("voxel-size fix failed for %s: %s", out_file.name, e)
        written.append(out_file)
    return written


def bids_filename(base_stem: str, entity: str, suffix: str) -> str:
    """``sub-X_ses-Y_run-N`` (+ optional ``echo-M``) + suffix, suffix last.

    >>> bids_filename("sub-1Y_ses-1_run-9", "echo-1", "bold")
    'sub-1Y_ses-1_run-9_echo-1_bold'
    >>> bids_filename("sub-1Y_ses-1_run-5", "", "T2w")
    'sub-1Y_ses-1_run-5_T2w'
    """
    parts = [base_stem] + ([entity] if entity else []) + [suffix]
    return "_".join(parts)


def _get(d: Dict, key: str):
    v = d.get(key)
    return v.get("value") if isinstance(v, dict) else v


# --------------------------------------------------------------------------- #
# scans.tsv (gh #8)
# --------------------------------------------------------------------------- #
SCANS_TSV_COLS = ["filename", "scan_number", "method", "modality", "suffix", "status",
                  "n_slices", "matrix", "n_volumes", "n_repetitions", "n_echoes",
                  "echo_times", "max_bvalue", "fov_mm", "voxel_size_mm", "shape", "zooms_mm"]


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (list, tuple)):
        return "x".join(f"{x:g}" if isinstance(x, (int, float)) else str(x) for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


# --------------------------------------------------------------------------- #
# Session + study drivers
# --------------------------------------------------------------------------- #
def convert_session(session_dir: Path, meta: Dict[str, str], cfg: BidsConfig,
                    convert: bool = True) -> Dict[str, Any]:
    """Convert one session to BIDS; always (re)writes its scans.tsv."""
    from neurofaune.utils.bruker_convert import get_bruker_method, extract_bids_metadata, extract_bvec_bval
    from neurofaune.utils.fix_bruker_voxel_sizes import parse_bruker_method

    sub, ses = meta["subject"], meta["session"]
    ses_dir = cfg.bids_root / f"sub-{sub}" / f"ses-{ses}"
    rows: List[Dict[str, Any]] = []
    written_files: List[Path] = []

    for sd in _numbered_scans(session_dir):
        num = int(sd.name)
        method = get_bruker_method(sd)
        try:
            params = parse_bruker_method(sd / "method")
        except Exception:  # noqa: BLE001
            params = {}
        mapped = cfg.sequence_map.get(method)
        modality, suffix = mapped if mapped else (None, None)

        status, files = "", []
        if modality is None:
            status = f"skipped:unrecognized({method})"
        elif modality in cfg.skip_modalities:
            status = f"skipped:{modality}"
        else:
            base_stem = f"sub-{sub}_ses-{ses}_run-{num}"
            out_dir = ses_dir / modality
            if convert:
                files = convert_scan(sd, modality, suffix, base_stem, out_dir, cfg.voxel_scale)
                status = "written" if files else "failed"
                for f in files:
                    md = extract_bids_metadata(sd, modality)
                    if md:
                        f.with_suffix("").with_suffix(".json").write_text(json.dumps(md, indent=2))
                if modality == "dwi" and files:
                    gi = extract_bvec_bval(sd)
                    if gi is not None:
                        bval, bvec = gi
                        bvec_t = bvec.T if bvec.ndim > 1 else bvec.reshape(-1, 1).T
                        np.savetxt(files[0].with_suffix("").with_suffix(".bvec"), bvec_t, fmt="%.6f")
                        np.savetxt(files[0].with_suffix("").with_suffix(".bval"), bval.reshape(1, -1), fmt="%.1f")
                written_files += files
            else:
                status = "planned"

        shape = zooms = None
        if files:
            try:
                import nibabel as nib
                img = nib.load(str(files[0]))
                shape = "x".join(map(str, img.shape))
                zooms = "x".join(f"{z:g}" for z in img.header.get_zooms()[:5])
            except Exception:  # noqa: BLE001
                pass

        rows.append({
            "filename": ("/".join(files[0].parts[-2:]) if files else "n/a"),
            "scan_number": num, "method": method, "modality": modality or "n/a",
            "suffix": suffix or "n/a", "status": status,
            "n_slices": params.get("n_slices"), "matrix": params.get("matrix"),
            "n_volumes": params.get("n_bvalues") or params.get("n_repetitions"),
            "n_repetitions": params.get("n_repetitions"), "n_echoes": params.get("n_echoes"),
            "echo_times": params.get("echo_times"), "max_bvalue": params.get("max_bvalue"),
            "fov_mm": params.get("fov"), "voxel_size_mm": params.get("voxel_size"),
            "shape": shape, "zooms_mm": zooms,
        })

    ses_dir.mkdir(parents=True, exist_ok=True)
    tsv = ses_dir / f"sub-{sub}_ses-{ses}_scans.tsv"
    with open(tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(SCANS_TSV_COLS)
        for r in rows:
            w.writerow([_fmt(r.get(c)) for c in SCANS_TSV_COLS])

    n_written = sum(1 for r in rows if r["status"] == "written")
    logger.info("sub-%s ses-%s: %d scans, %d written -> %s",
                sub, ses, len(rows), n_written, tsv)
    return {"subject": sub, "session": ses, "rows": rows, "n_written": n_written, "scans_tsv": tsv}


def write_dataset_description(bids_root: Path, name: str = "neurofaune BIDS") -> None:
    bids_root.mkdir(parents=True, exist_ok=True)
    desc = bids_root / "dataset_description.json"
    if not desc.exists():
        desc.write_text(json.dumps(
            {"Name": name, "BIDSVersion": "1.6.0", "DatasetType": "raw"}, indent=2))


def convert_study(cfg: BidsConfig, subjects: Optional[set] = None,
                  convert: bool = True) -> List[Dict[str, Any]]:
    """Discover + convert all (optionally subject-filtered) sessions."""
    write_dataset_description(cfg.bids_root)
    sessions = discover_sessions(cfg)
    if subjects:
        subjects = {s.upper() for s in subjects}
        sessions = [(d, m) for d, m in sessions if m["subject"] in subjects]
    return [convert_session(d, m, cfg, convert=convert) for d, m in sessions]
