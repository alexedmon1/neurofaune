"""Read a randomise run out as rows a results section can be written from.

`extract_clusters` / `generate_cluster_report` answer "which clusters survived";
a result is not reported until its magnitude, direction, extent and location are
stated, and a test that found nothing is still a result. This module returns two
tables for one randomise run:

* **tests** -- one row per t-contrast, significant or not: n, df, what the
  contrast tests in the design's own terms, extent at the corrected and (when
  randomise wrote it) uncorrected threshold, peak t, and the *unselected* effect
  over the whole mask (Cohen's d with a confidence interval, the raw
  standardised difference, and the group means).
* **clusters** -- one row per cluster under a stated definition: extent in voxels
  and mm^3, peak t and coordinates, the named atlas region at the peak and every
  region the cluster covers with voxel counts, corrected and uncorrected minimum
  p, whether it crosses the midline, and the cluster-mean effect.

Nothing here is TBSS-specific -- the mask is the skeleton for TBSS and a brain or
grey-matter mask for VBM -- so it lives in `analysis/stats` beside
`cluster_report` and `effect_size`.

**Effect size.** For contrast c of the GLM fitted to each subject's mean over a
voxel set, d = c'b / s, with s the residual standard deviation on n - rank(X)
degrees of freedom. This is the voxelwise convention of `effect_size`
(d = t * sqrt(c'(X'X)^-1 c)) applied to the voxel-set mean: for two groups with
no covariates it is the pooled-SD Cohen's d, for a one-sample design the mean
over the SD. Covariates (e.g. centred batch terms) leave the residual SD smaller
than the raw SD, so d is also reported unadjusted (`d_raw`) wherever the
contrast compares two groups or one mean. The CI is exact under normal residuals:
the observed t is inverted through the noncentral t distribution (Steiger &
Fouladi 1997) and the noncentrality bounds scaled by sqrt(c'(X'X)^-1 c).

**Selection.** A cluster's effect is computed over voxels chosen because they
were significant, so it is inflated and its CI does not hold its nominal
coverage; cluster rows say so (`effect_selected`). The whole-mask effect on the
test row is never selected.

**Direction** is stated from the design: a contrast of +1/-1 on two indicator
columns tests "<+ column> > <- column>"; a weight on a constant column tests
"mean > 0" or "mean < 0"; one weight on any other column tests the sign of that
column's coefficient; anything else is written out as the weighted sum. Column
names come from neuroaider's design_summary.json when it is there, or are given;
without them columns are EV1..EVp. randomise tests each contrast one-sided
(t > 0), so `tested_direction` is the hypothesis and `observed_direction` the
sign of the whole-mask effect.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, optimize, stats

from neurofaune.analysis.stats.effect_size import read_fsl_vest

logger = logging.getLogger("neurofaune.stats")

CLUSTER_ON = ("fwe", "uncorrected")


# ----------------------------------------------------------------- inputs ---
@dataclass
class Atlas:
    """A parcellation with region names and hemispheres, in the data's space."""

    parcellation: np.ndarray
    names: dict[int, str] = field(default_factory=dict)
    hemispheres: dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_files(cls, parcellation: Path, labels_csv: Path | None = None) -> Atlas:
        """Read a parcellation and, if given, its label table.

        The table is SIGMA's (`Labels`, `Region of interest`, `Hemisphere`), read
        with `network.roi_extraction.load_parcellation` so both places agree.
        """
        if labels_csv is None:
            data = np.asarray(nib.load(str(parcellation)).dataobj, dtype=np.int32)
            return cls(data)
        from neurofaune.network.roi_extraction import load_parcellation
        data, table = load_parcellation(Path(parcellation), Path(labels_csv))
        names = dict(zip(table["Labels"].astype(int), table["Region of interest"].astype(str), strict=False))
        hemi = (dict(zip(table["Labels"].astype(int), table["Hemisphere"].astype(str), strict=False))
                if "Hemisphere" in table else {})
        return cls(np.asarray(data, dtype=np.int32), names, hemi)

    def name(self, label: int) -> str | None:
        if label == 0:
            return None
        return self.names.get(int(label), f"label {int(label)}")


def _record(design_dir: Path) -> dict | None:
    """The design record (design.json) beside the design, when there is one."""
    from neurofaune.analysis.stats.design_record import read_design_record
    try:
        return read_design_record(design_dir)
    except (OSError, ValueError):
        return None


def design_names(design_dir: Path, n_columns: int, n_contrasts: int) -> tuple[list[str], list[str]]:
    """Column and contrast names: the design record (design.json), then
    design_summary.json, then design.con, then EV<i>/C<i>."""
    columns, contrasts = None, None
    record = _record(design_dir)
    if record:
        rc = [c.get("name") for c in record.get("columns", [])]
        rk = [c.get("name") for c in record.get("contrasts", [])]
        columns = rc if len(rc) == n_columns and all(rc) else None
        contrasts = rk if len(rk) == n_contrasts and all(rk) else None
    summary = Path(design_dir) / "design_summary.json"
    if (columns is None or contrasts is None) and summary.exists():
        s = json.loads(summary.read_text())
        columns = columns if columns is not None else s.get("columns")
        contrasts = contrasts if contrasts is not None else s.get("contrasts")
    con = Path(design_dir) / "design.con"
    if contrasts is None and con.exists():
        named = {}
        for line in con.read_text().splitlines():
            if line.startswith("/ContrastName"):
                key, _, value = line.partition(" ")
                named[int(key[len("/ContrastName"):])] = value.strip()
        if named:
            contrasts = [named.get(i + 1, f"C{i + 1}") for i in range(n_contrasts)]
    if not columns or len(columns) != n_columns:
        columns = [f"EV{i + 1}" for i in range(n_columns)]
    if not contrasts or len(contrasts) != n_contrasts:
        contrasts = [f"C{i + 1}" for i in range(n_contrasts)]
    return list(columns), list(contrasts)


# ---------------------------------------------------------------- contrast ---
def describe_contrast(c: np.ndarray, X: np.ndarray, columns: Sequence[str]) -> dict:
    """What a t-contrast tests, in the design's terms (see module docstring)."""
    c = np.asarray(c, float)
    nz = np.flatnonzero(np.abs(c) > 1e-12)
    vals = [np.unique(X[:, j]) for j in range(X.shape[1])]
    indicator = [len(v) == 2 and set(np.round(v, 12)) <= {0.0, 1.0} for v in vals]
    constant = [len(v) == 1 and v[0] != 0 for v in vals]
    out = {"kind": "general", "pos": None, "neg": None, "column": None}
    if len(nz) == 2 and all(indicator[j] for j in nz) and np.isclose(c[nz[0]], -c[nz[1]]):
        pos, neg = (nz[0], nz[1]) if c[nz[0]] > 0 else (nz[1], nz[0])
        out.update(kind="two-group", pos=int(pos), neg=int(neg),
                   tested=f"{columns[pos]} > {columns[neg]}",
                   opposite=f"{columns[neg]} > {columns[pos]}")
    elif len(nz) == 1 and constant[nz[0]]:
        sign = "> 0" if c[nz[0]] > 0 else "< 0"
        out.update(kind="one-sample", column=int(nz[0]), tested=f"mean {sign}",
                   opposite=f"mean {'< 0' if sign == '> 0' else '> 0'}")
    elif len(nz) == 1:
        j = int(nz[0])
        word = "positive" if c[j] > 0 else "negative"
        other = "negative" if word == "positive" else "positive"
        out.update(kind="single-column", column=j,
                   tested=f"{word} coefficient of {columns[j]}",
                   opposite=f"{other} coefficient of {columns[j]}")
    else:
        terms = " ".join(f"{c[j]:+g}*{columns[j]}" for j in nz)
        out.update(tested=f"{terms} > 0", opposite=f"{terms} < 0")
    out["vector"] = " ".join(f"{v:g}" for v in c)
    return out


# ------------------------------------------------------------------ effect ---
def _d_ci(t: float, df: int, v: float, level: float) -> tuple[float, float]:
    """CI on d = delta * sqrt(v) from the noncentral t (exact under normality)."""
    if not (np.isfinite(t) and df > 0 and v > 0):
        return float("nan"), float("nan")
    alpha = (1 - level) / 2

    def bound(q):
        def f(delta):
            # scipy's nct.cdf is NaN far in the tails, where it is 0 (delta >> t) or 1;
            # it is monotone in delta, so the limit stands in without moving the root.
            v = stats.nct.cdf(t, df, delta)
            return (0.0 if delta > t else 1.0) - q if np.isnan(v) else v - q

        lo, hi = t - 10 - abs(t), t + 10 + abs(t)
        while f(lo) < 0:
            lo -= 10 + abs(lo)
        while f(hi) > 0:
            hi += 10 + abs(hi)
        return optimize.brentq(f, lo, hi, xtol=1e-10)

    return float(bound(1 - alpha) * np.sqrt(v)), float(bound(alpha) * np.sqrt(v))


def contrast_effect(y: np.ndarray, X: np.ndarray, c: np.ndarray, desc: dict,
                    level: float = 0.95) -> dict:
    """Effect of contrast c on per-subject values y (see module docstring)."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    y, X = y[ok], X[ok]
    n, rank = len(y), int(np.linalg.matrix_rank(X))
    df = n - rank
    res = {"effect_n": int(n), "effect_df": int(df)}
    if df < 1:
        return {**res, "estimate": np.nan, "d": np.nan, "d_ci_low": np.nan, "d_ci_high": np.nan,
                "d_raw": np.nan}
    pinv = np.linalg.pinv(X.T @ X)
    beta = pinv @ X.T @ y
    resid = y - X @ beta
    s = float(np.sqrt(resid @ resid / df))
    est = float(c @ beta)
    v = float(c @ pinv @ c)
    d = est / s if s > 0 else np.nan
    lo, hi = _d_ci(d / np.sqrt(v), df, v, level) if s > 0 and v > 0 else (np.nan, np.nan)
    res.update(estimate=est, d=float(d), d_ci_low=lo, d_ci_high=hi, d_raw=np.nan)

    if desc["kind"] == "two-group":
        a, b = y[X[:, desc["pos"]] == 1], y[X[:, desc["neg"]] == 1]
        res.update(mean_pos=float(a.mean()) if len(a) else np.nan,
                   mean_neg=float(b.mean()) if len(b) else np.nan, n_pos=len(a), n_neg=len(b))
        if len(a) > 1 and len(b) > 1:
            sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                         / (len(a) + len(b) - 2))
            res["d_raw"] = float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan
    elif desc["kind"] == "one-sample":
        res["mean"] = float(y.mean())
        if n > 1 and y.std(ddof=1) > 0:
            sign = np.sign(c[desc["column"]])
            res["d_raw"] = float(sign * y.mean() / y.std(ddof=1))
    if np.isfinite(d):
        res["observed_direction"] = desc["tested"] if d > 0 else desc["opposite"]
    return res


# ------------------------------------------------------------------- maps ---
def _map(directory: Path, prefix: str, kind: str, c: int) -> Path | None:
    """randomise output for contrast c: kind is tstat | corrp | p (uncorrected)."""
    names = {"tstat": [f"{prefix}_tstat{c}"],
             "corrp": [f"{prefix}_tfce_corrp_tstat{c}", f"{prefix}_vox_corrp_tstat{c}"],
             "p": [f"{prefix}_tfce_p_tstat{c}", f"{prefix}_vox_p_tstat{c}"]}[kind]
    for stem in names:
        for ext in (".nii.gz", ".nii"):
            if (directory / f"{stem}{ext}").exists():
                return directory / f"{stem}{ext}"
    return None


def _contrast_count(directory: Path, prefix: str) -> int:
    k = 0
    while _map(directory, prefix, "tstat", k + 1) is not None:
        k += 1
    return k


# ---------------------------------------------------------------- readout ---
def read_randomise(
    randomise_dir: Path,
    data: Path,
    design_mat: Path,
    mask: Path,
    *,
    prefix: str = "randomise",
    atlas: Atlas | None = None,
    column_names: Sequence[str] | None = None,
    contrast_names: Sequence[str] | None = None,
    design_con: Path | None = None,
    cluster_on: str = "fwe",
    alpha: float = 0.05,
    min_cluster_size: int = 1,
    connectivity: int = 26,
    ci_level: float = 0.95,
    labels: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tests and clusters for one randomise run (see module docstring).

    Args:
        randomise_dir: folder holding `<prefix>_tstat<c>` and the corrected maps.
        data: the 4-D input randomise was given (subjects in design order).
        design_mat, design_con: the design. Names from `column_names` /
            `contrast_names`, else design_summary.json / design.con beside them.
        mask: the analysis mask (TBSS skeleton, VBM brain mask).
        atlas: parcellation in the same space, for region names.
        cluster_on: clusters are connected voxels with corrected ("fwe") or
            uncorrected p below `alpha`; uncorrected needs randomise --uncorrp.
        alpha: the p threshold for clusters and for the extent columns.
        connectivity: 6, 18 or 26 (26 keeps a one-voxel-thick skeleton connected).
        labels: constant columns added to every row (e.g. measure, analysis).

    Returns:
        (tests, clusters) DataFrames.
    """
    if cluster_on not in CLUSTER_ON:
        raise ValueError(f"cluster_on must be one of {CLUSTER_ON}")
    randomise_dir, design_mat = Path(randomise_dir), Path(design_mat)
    design_con = Path(design_con) if design_con else design_mat.with_suffix(".con")
    X = read_fsl_vest(design_mat)
    C = read_fsl_vest(design_con)
    if C.ndim == 1:
        C = C[None, :]
    n_con = _contrast_count(randomise_dir, prefix)
    if n_con == 0:
        raise FileNotFoundError(f"no {prefix}_tstat<c> maps in {randomise_dir}")
    if n_con != len(C):
        raise ValueError(f"{n_con} t-stat maps but {len(C)} contrasts in {design_con}")
    cols, cons = design_names(design_mat.parent, X.shape[1], len(C))
    record = _record(design_mat.parent)
    said = [c.get("tests") for c in record.get("contrasts", [])] if record else []
    said = said if len(said) == len(C) else [None] * len(C)
    cols = list(column_names) if column_names is not None else cols
    cons = list(contrast_names) if contrast_names is not None else cons

    mimg = nib.load(str(mask))
    m = np.asarray(mimg.dataobj) > 0
    affine = mimg.affine
    vox_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    ijk = np.argwhere(m)
    y = np.asarray(nib.load(str(data)).dataobj, dtype=np.float64)[m]   # voxels x subjects
    if y.shape[1] != X.shape[0]:
        raise ValueError(f"{data} has {y.shape[1]} volumes, design has {X.shape[0]} rows")
    if atlas is not None and atlas.parcellation.shape != m.shape:
        raise ValueError(f"atlas shape {atlas.parcellation.shape} != mask shape {m.shape}")
    parc = atlas.parcellation[m] if atlas is not None else None
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    labels = dict(labels or {})
    whole = np.nanmean(y, axis=0)
    n_mask = int(m.sum())

    def load(path):
        return np.asarray(nib.load(str(path)).dataobj, dtype=np.float64)[m]

    tests, clusters = [], []
    for k in range(len(C)):
        c, cn = C[k], cons[k]
        desc = describe_contrast(c, X, cols)
        t = load(_map(randomise_dir, prefix, "tstat", k + 1))
        corrp = _map(randomise_dir, prefix, "corrp", k + 1)
        uncp = _map(randomise_dir, prefix, "p", k + 1)
        p_fwe = 1 - load(corrp) if corrp else np.full(n_mask, np.nan)
        p_unc = 1 - load(uncp) if uncp else np.full(n_mask, np.nan)
        if cluster_on == "uncorrected" and uncp is None:
            raise FileNotFoundError(f"cluster_on='uncorrected' needs {prefix}_*_p_tstat{k + 1} "
                                    "(run randomise with --uncorrp)")
        sel = (p_fwe if cluster_on == "fwe" else p_unc) < alpha
        vol = np.zeros(m.shape, bool)
        vol[m] = sel
        lab = ndimage.label(vol, structure=structure)[0][m]
        sizes = np.bincount(lab)[1:] if lab.max() else np.array([], int)
        keep = [i + 1 for i, s in enumerate(sizes) if s >= min_cluster_size]

        base = {**labels, "contrast": k + 1, "contrast_name": cn, "contrast_vector": desc["vector"],
                "design": desc["kind"], "tested_direction": desc["tested"],
                "contrast_tests": said[k],
                "n": int(X.shape[0]), "df": int(X.shape[0] - np.linalg.matrix_rank(X))}
        whole_eff = contrast_effect(whole, X, c, desc, ci_level)
        tests.append({
            **base,
            "mask_voxels": n_mask,
            "n_vox_fwe": int(np.sum(p_fwe < alpha)), "frac_mask_fwe": float(np.mean(p_fwe < alpha)),
            "min_p_fwe": float(np.nanmin(p_fwe)) if corrp else np.nan,
            "n_vox_uncorr": int(np.sum(p_unc < alpha)) if uncp else np.nan,
            "frac_mask_uncorr": float(np.mean(p_unc < alpha)) if uncp else np.nan,
            "min_p_uncorr": float(np.nanmin(p_unc)) if uncp else np.nan,
            "peak_t": float(np.max(t)),
            "cluster_definition": f"{cluster_on} p < {alpha}, {connectivity}-connected, "
                                  f">= {min_cluster_size} voxels",
            "n_clusters": len(keep), "largest_cluster_vox": int(max((sizes[i - 1] for i in keep), default=0)),
            "significant_fwe": bool(np.any(p_fwe < alpha)),
            **{f"whole_{key}": v for key, v in whole_eff.items()},
        })
        for cid in keep:
            s = lab == cid
            peak = np.flatnonzero(s)[np.argmax(t[s])]
            p_ijk = ijk[peak]
            row = {**base, "cluster": int(cid), "n_voxels": int(s.sum()), "mm3": float(s.sum() * vox_mm3),
                   "peak_t": float(t[peak]), "peak_ijk": " ".join(map(str, p_ijk)),
                   "peak_xyz_mm": " ".join(f"{v:.2f}" for v in nib.affines.apply_affine(affine, p_ijk)),
                   "cog_xyz_mm": " ".join(f"{v:.2f}" for v in
                                          nib.affines.apply_affine(affine, ijk[s].mean(axis=0))),
                   "min_p_fwe": float(np.nanmin(p_fwe[s])) if corrp else np.nan,
                   "min_p_uncorr": float(np.nanmin(p_unc[s])) if uncp else np.nan,
                   "fwe_significant": bool(np.nanmin(p_fwe[s]) < alpha) if corrp else None}
            if parc is not None:
                counts = pd.Series(parc[s]).value_counts()
                inside = counts[counts.index > 0]
                # largest first; ties by label id, so the order does not depend on pandas
                inside = inside.iloc[np.lexsort((inside.index.to_numpy(), -inside.to_numpy()))]
                row.update(peak_region=atlas.name(int(parc[peak])),
                           top_region=atlas.name(int(inside.index[0])) if len(inside) else None,
                           regions="; ".join(f"{atlas.name(int(r))}:{int(n)}" for r, n in inside.items()),
                           n_regions=int(len(inside)), outside_atlas_vox=int(counts.get(0, 0)),
                           crosses_midline=({"L", "R"} <= {atlas.hemispheres.get(int(r)) for r in inside.index})
                           if atlas.hemispheres else None)
            eff = contrast_effect(np.nanmean(y[s], axis=0), X, c, desc, ci_level)
            row.update({**eff, "effect_selected": True})
            clusters.append(row)
    return pd.DataFrame(tests), pd.DataFrame(clusters)
