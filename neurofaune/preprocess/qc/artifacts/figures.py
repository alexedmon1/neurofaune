"""Figures for the implant artifact QC: depth profiles and map montages.

Static PNGs for QC reports. Colour roles: categorical slots in fixed order for
sources (validated: CVD-separated; two slots sit below 3:1 on the surface, so
every line is also direct-labelled), a blue <-> gray <-> red diverging map for
signed quantities (Hz, mm) and a one-hue blue ramp for magnitudes (tSNR).
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIVERGING = LinearSegmentedColormap.from_list("nf_div", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#8f1d1c"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "nf_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

METRICS = [
    ("b0_abs_hz", "|B0 offset| (Hz)", "B0 map"),
    ("displacement_abs_mm", "|EPI displacement| (mm)", "topup field x readout"),
    ("ge_se_ratio", "GE / SE signal (deep brain = 1)", "dropout"),
    ("tsnr", "tSNR", "BOLD, motion corrected"),
]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_depth_profiles(profiles: List[Dict], out: Path) -> None:
    """Small multiples: one panel per metric; median + IQR band per source.

    Solid = distance below the dorsal surface (implant side); dashed = height
    above the ventral surface (the no-implant control). Line style is the
    secondary encoding for that pairing; colour is the source.
    """
    by = defaultdict(lambda: defaultdict(list))
    for r in profiles:
        by[r["metric"]][(r["source"], r.get("surface", "dorsal"))].append(r)
    panels = [m for m in METRICS if m[0] in by]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.0), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (metric, ylabel, sub) in zip(axes, panels):
        _style(ax)
        # Fixed slot order within a panel; a panel never holds more than a few sources.
        srcs = sorted({src for src, _ in by[metric]})
        colour = {src: CATEGORICAL[i] for i, src in enumerate(srcs[: len(CATEGORICAL)])}
        for (src, surface), rows in sorted(by[metric].items()):
            if src not in colour:
                continue
            rows = sorted(rows, key=lambda r: r["depth_mm"])
            d = np.array([r["depth_mm"] for r in rows])
            med = np.array([r["median"] for r in rows])
            dorsal = surface == "dorsal"
            if dorsal:
                ax.fill_between(d, [r["p25"] for r in rows], [r["p75"] for r in rows],
                                color=colour[src], alpha=0.15, linewidth=0)
            ax.plot(d, med, color=colour[src], linewidth=2, linestyle="-" if dorsal else "--",
                    label=f"{src} ({'dorsal' if dorsal else 'ventral ctrl'})")
            if dorsal:
                ax.annotate(src, (d[0], med[0]), xytext=(-4, 0), textcoords="offset points",
                            fontsize=8, color=TEXT_2, va="center", ha="right")
        if metric == "ge_se_ratio":
            ax.axhline(1.0, color=TEXT_2, linewidth=0.8, linestyle=":")
        ax.set_xlim(left=-0.9)
        ax.set_xlabel("distance from brain surface (mm)", fontsize=8, color=TEXT_2)
        ax.set_title(f"{ylabel}\n{sub}", fontsize=9, color=TEXT, loc="left")
        ax.legend(fontsize=7, frameon=False, labelcolor=TEXT_2)
    fig.suptitle("Implant artifacts vs distance from the dorsal surface (solid, implant side) "
                 "and the ventral surface (dashed, control). Median; band = IQR (dorsal).",
                 fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_map_montage(maps_dir: Path, ref: nib.Nifti1Image, brain: np.ndarray, out: Path) -> None:
    """Mid-sagittal and mid-coronal slices of each map over the T2w reference."""
    t2 = ref.get_fdata()
    files = [p for p in sorted(Path(maps_dir).glob("*.nii.gz"))
             if p.name.startswith(("b0_", "topup_", "dropout_ratio_", "tsnr_"))]
    if not files:
        return
    xs = int(np.median(np.nonzero(brain)[0]))
    ys = int(np.median(np.nonzero(brain)[1]))
    fig, axes = plt.subplots(len(files), 2, figsize=(9, 2.9 * len(files)), facecolor=SURFACE)
    axes = np.atleast_2d(axes)
    vmax_t2 = np.percentile(t2[brain], 99.5)
    for row, f in zip(axes, files):
        data = nib.load(str(f)).get_fdata()
        name = f.name.replace(".nii.gz", "")
        if name.startswith("tsnr_"):
            cmap, lim = SEQUENTIAL, (0, np.nanpercentile(data[brain & (data > 0)], 98) if (data[brain] > 0).any() else 1)
        elif name.startswith("dropout_ratio_"):
            cmap, lim = DIVERGING, (0, 2)
        else:
            m = np.nanpercentile(np.abs(data[brain & (data != 0)]), 98) if (data[brain] != 0).any() else 1
            cmap, lim = DIVERGING, (-m, m)
        for ax, (sl, label) in zip(row, ((np.s_[xs, :, :], "sagittal"), (np.s_[:, ys, :], "coronal"))):
            ax.imshow(t2[sl].T, origin="lower", cmap="gray", vmax=vmax_t2, aspect="auto")
            ov = np.where(brain[sl] & (data[sl] != 0), data[sl], np.nan)
            im = ax.imshow(ov.T, origin="lower", cmap=cmap, vmin=lim[0], vmax=lim[1], alpha=0.85, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{name} ({label}; dorsal up)", fontsize=8, color=TEXT, loc="left")
        fig.colorbar(im, ax=row.tolist(), fraction=0.025, pad=0.01).ax.tick_params(labelsize=7, colors=TEXT_2)
    fig.savefig(out, dpi=110, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
