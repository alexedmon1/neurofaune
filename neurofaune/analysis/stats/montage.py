"""Montages of an analysis folder's maps: display only, nothing recomputed.

Every voxelwise analysis folder lists its maps in ``analysis.json`` (`readout_results` for
TBSS / VBM / voxelwise randomise, the searchlight writer for decoding): per measure and
contrast a statistic map (``kind: stat``) and a corrected p map (``kind: p_corrected``), with
the mask and the template it sits on. :func:`add_montages` draws, for each measure (and
facet), one figure: a row per contrast, coronal slices evenly spaced through the mask from
posterior to anterior, the template in grey, the statistic coloured only where the corrected p
passes alpha, and a light outline round the uncorrected p < alpha extent where that map is
listed. It writes them into the folder's ``figures/`` and lists them in ``analysis.json``, so
the study's gallery (Lumière) shows them. Moved here from cuprizone's
``analyses/code/voxelwise_montage.py`` (results plan step 6) and made general: orientation
comes from each map's ``axes`` (else its header), never from a study's conventions.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FIGURES = "figures"
#: The spec version from which a figure may say its ``kind`` (Lumière 0.4).
KIND_SINCE = (0, 4)


def _load(path: Path):
    import nibabel as nib

    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32), img.affine


def _axes(entry: dict, affine: np.ndarray) -> str:
    """The direction each voxel axis runs toward (e.g. "LIA"): the map's own say, else its header."""
    given = str(entry.get("axes") or "").upper()
    if len(given) == 3:
        return given
    import nibabel as nib

    return "".join(nib.aff2axcodes(affine))


def _standard(data: np.ndarray, axes: str) -> np.ndarray:
    """``data`` as (left to right, inferior to superior, posterior to anterior)."""
    pairs = {"L": ("R", 0, False), "R": ("R", 0, True), "I": ("S", 1, False), "S": ("S", 1, True),
             "P": ("A", 2, False), "A": ("A", 2, True)}
    order, flips = [0, 0, 0], []
    for i, code in enumerate(axes):
        _, target, toward_positive = pairs[code]
        order[target] = i
        # an axis running toward L/I/P counts down in R/S/A: flip it so indices increase toward R/S/A
        if not toward_positive:
            flips.append(target)
    out = np.transpose(data, order)
    for target in flips:
        out = np.flip(out, axis=target)
    return out


def _significant(p: np.ndarray, values: str, alpha: float) -> np.ndarray:
    """Where a p map passes alpha: randomise writes 1 - p ("one_minus_p"), the searchlight p."""
    return p > 1 - alpha if values == "one_minus_p" else (p > 0) & (p < alpha)


def _version(analysis: dict) -> tuple[int, int]:
    try:
        major, minor = str(analysis.get("spec_version", "0.0")).split(".")[:2]
        return int(major), int(minor)
    except ValueError:
        return (0, 0)


def add_montages(folder: Path, *, alpha: float = 0.05, slices: int = 10) -> list[dict[str, Any]]:
    """Draw the folder's montages into ``figures/`` and list them in its ``analysis.json``.

    Returns the figure entries written (one per measure, or per measure and facet). A folder
    with no statistic and corrected p maps gets none. Re-running replaces earlier montages.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    folder = Path(folder)
    meta_path = folder / "analysis.json"
    analysis = json.loads(meta_path.read_text())
    maps = [m for m in analysis.get("maps") or [] if isinstance(m, dict) and m.get("path")]
    by_kind = {m["kind"]: m for m in maps if m.get("kind") in ("mask", "background")}
    panels: dict[tuple[str, str], dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for m in maps:
        if m.get("kind") in ("stat", "p_corrected", "p_uncorrected") and m.get("measure"):
            panels[(m["measure"], m.get("facet") or "")][m.get("contrast") or ""][m["kind"]] = m
    if "mask" not in by_kind or not panels:
        return []
    mask_entry = by_kind["mask"]
    mask, affine = _load(folder / mask_entry["path"])
    axes = _axes(mask_entry, affine)
    mask = _standard(mask, axes) > 0
    if by_kind.get("background"):
        back = _standard(_load(folder / by_kind["background"]["path"])[0], axes)
    else:
        back = mask.astype(np.float32)
    # the grey scale comes from the whole template (a TBSS mask is the skeleton, its brightest part)
    tissue = back[back > 0]
    lo, hi = np.percentile(tissue, [1, 99.5]) if tissue.size else (0.0, 1.0)
    # a sparse mask is a skeleton: drawn faintly for context, and significant voxels thickened
    # one voxel in-plane so they can be seen (display only, said in the caption)
    sparse = bool(tissue.size) and mask.sum() < 0.2 * (back > 0).sum()
    span = np.nonzero(mask.any(axis=(0, 1)))[0]
    if span.size == 0:
        return []
    cuts = np.unique(np.linspace(span[0], span[-1], slices + 2)[1:-1].round().astype(int))
    out_dir = folder / FIGURES
    out_dir.mkdir(exist_ok=True)
    space = mask_entry.get("space") or analysis.get("space") or "template"
    inference = ((analysis.get("inference") or {}).get("method")) or "the analysis's correction"
    # a figure says it is a brain figure only where the spec allows the term (0.4 on) and the
    # checker's rule holds: a brain figure beside an elements table needs the folder's atlas
    elements = any(isinstance(t, dict) and t.get("role") == "elements" for t in analysis.get("tables") or [])
    with_kind = _version(analysis) >= KIND_SINCE and (not elements or "atlas" in analysis)
    entries = []
    for (measure, facet), contrasts in sorted(panels.items()):
        rows = [(c, kinds) for c, kinds in contrasts.items() if "stat" in kinds and "p_corrected" in kinds]
        if not rows:
            continue
        shown, any_outline = [], False
        for contrast, kinds in rows:
            stat = _standard(_load(folder / kinds["stat"]["path"])[0], axes)
            corr = kinds["p_corrected"]
            sig = _significant(_standard(_load(folder / corr["path"])[0], axes),
                               corr.get("values", "one_minus_p"), alpha) & mask
            outline = None
            if "p_uncorrected" in kinds:
                unc = kinds["p_uncorrected"]
                outline = _significant(_standard(_load(folder / unc["path"])[0], axes),
                                       unc.get("values", "one_minus_p"), alpha) & mask
                any_outline = True
            n_sig = int(sig.sum())
            if sparse and n_sig:
                from scipy.ndimage import grey_dilation

                # thicken in-plane only (each coronal slice on its own), keeping each voxel's sign
                signed = np.where(sig, stat, 0.0)
                pos = grey_dilation(np.clip(signed, 0, None), size=(3, 3, 1))
                neg = -grey_dilation(np.clip(-signed, 0, None), size=(3, 3, 1))
                wide = np.where(pos >= -neg, pos, neg)
                shown.append((contrast, np.where(wide != 0, wide, np.nan), outline, n_sig))
            else:
                shown.append((contrast, np.where(sig, stat, np.nan), outline, n_sig))
        # each row is one one-sided contrast, so what it shows is positive: red to yellow from the
        # smallest significant statistic to the largest (randomise's usual display)
        values = [np.abs(s[np.isfinite(s)]) for _, s, _, n in shown if n]
        top = max([float(v.max()) for v in values] or [1.0])
        bottom = min([float(v.min()) for v in values] or [0.0])
        bottom = bottom if bottom < top else 0.0
        aspect = back.shape[1] / back.shape[0]  # a coronal slice's height over its width
        fig, axs = plt.subplots(len(shown), len(cuts), squeeze=False, layout="compressed",
                                figsize=(1.45 * len(cuts) + 2.6, 1.45 * aspect * len(shown) + 1.1))
        for r, (contrast, stat, outline, n_sig) in enumerate(shown):
            for c, k in enumerate(cuts):
                ax = axs[r][c]
                ax.imshow(back[:, :, k].T, cmap="gray", origin="lower", vmin=lo, vmax=hi, interpolation="nearest")
                if sparse:
                    ax.imshow(np.where(mask[:, :, k], 1.0, np.nan).T, cmap="Greens", origin="lower",
                              vmin=0, vmax=1.6, alpha=0.6, interpolation="nearest")
                ax.imshow(np.abs(stat[:, :, k]).T, cmap="autumn", origin="lower", vmin=bottom, vmax=top,
                          interpolation="nearest")
                if outline is not None and outline[:, :, k].any():
                    ax.contour(outline[:, :, k].T.astype(float), levels=[0.5], colors="0.85", linewidths=0.4)
                ax.set_xticks([]), ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_visible(False)
                if r == 0:
                    ax.set_title(f"slice {k}", fontsize=7, color="0.4")
            label = f"{contrast}\n{n_sig} voxels p < {alpha:g}" if n_sig else f"{contrast}\nnone p < {alpha:g}"
            axs[r][0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center", labelpad=8)
        axs[0][0].text(0.02, 0.98, "L", transform=axs[0][0].transAxes, fontsize=7, color="w", va="top")
        axs[0][0].text(0.98, 0.98, "R", transform=axs[0][0].transAxes, fontsize=7, color="w", va="top", ha="right")
        sm = plt.cm.ScalarMappable(cmap="autumn", norm=Normalize(bottom, top))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axs, shrink=0.7, pad=0.01)
        cbar.set_label("statistic of the row's contrast, where significant", fontsize=8)
        what = f"{measure}" + (f", {facet}" if facet else "")
        fig.suptitle(f"{what}: the statistic where corrected p < {alpha:g}\n({inference}), on the {space} "
                     "template; coronal slices, posterior to anterior"
                     + ("; the skeleton in green, significant voxels thickened for display" if sparse else ""),
                     fontsize=10, y=1.0)
        name = f"montage__{measure}" + (f"__{facet}" if facet else "")
        name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name) + ".png"
        fig.savefig(out_dir / name, dpi=130, bbox_inches="tight")
        plt.close(fig)
        caption = (f"{what}: one row per contrast, the statistic where corrected p < {alpha:g} "
                   f"({inference}) on the {space} template, coronal slices posterior to anterior"
                   + ("; outline: uncorrected p < " + f"{alpha:g}" if any_outline else "")
                   + ("; the skeleton in green, significant voxels thickened one voxel in-plane for "
                      "display" if sparse else "") + ".")
        entry: dict[str, Any] = {"path": f"{FIGURES}/{name}", "caption": caption, "measure": measure}
        if facet:
            entry["facet"] = facet
        if with_kind:
            entry["kind"] = "brain"
        entries.append(entry)
    written = {e["path"] for e in entries}
    kept = [f for f in analysis.get("figures") or []
            if not (isinstance(f, dict) and str(f.get("path", "")).startswith(f"{FIGURES}/montage__")
                    and f["path"] not in written)]
    kept = [f for f in kept if not (isinstance(f, dict) and f.get("path") in written)]
    analysis["figures"] = kept + entries
    meta_path.write_text(json.dumps(analysis, indent=2) + "\n")
    logger.info("montages: %d figure(s) in %s", len(entries), out_dir)
    return entries
