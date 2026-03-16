"""
Analysis findings summarizer.

Reads per-analysis summary JSONs and produces structured findings
(significant / trending / null) with human-readable text summaries.
Each analysis script calls ``summarize_analysis()`` after completion;
``summarize_all()`` aggregates across the registry for a cross-analysis
convergence report.
"""

import dataclasses
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def build_provenance(script_name: Optional[str] = None) -> Dict[str, Any]:
    """Build a provenance metadata dict for embedding in summary JSONs.

    Call this from run scripts and store the result under a ``_provenance``
    key in the summary JSON so that downstream summarisers can verify
    *when*, *how*, and *on what data* the analysis was run.
    """
    prov: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER", "unknown"),
        "argv": sys.argv,
    }

    if script_name:
        prov["script"] = script_name
    elif sys.argv:
        prov["script"] = Path(sys.argv[0]).name

    # neurofaune git revision
    try:
        nf_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=nf_root, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            prov["neurofaune_version"] = result.stdout.strip()
    except Exception:
        pass

    return prov


def _extract_provenance(data: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """Extract provenance from a summary JSON, falling back to file mtime."""
    prov = data.get("_provenance", {})
    if not prov:
        # Fallback: use file modification time
        try:
            mtime = os.path.getmtime(path)
            prov = {
                "timestamp": datetime.fromtimestamp(mtime).isoformat(),
                "source": "file_mtime",
            }
        except OSError:
            prov = {"source": "unknown"}
    return prov

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Finding:
    """One testable comparison result."""
    comparison: str       # e.g. "FA/per_pnd_p60/H_gt_C"
    test_name: str        # e.g. "TBSS TFCE", "PERMANOVA", "Ridge permutation"
    statistic_name: str   # e.g. "n_significant_voxels", "r_squared"
    statistic_value: Any
    p_value: float
    n_subjects: int = 0
    detail: str = ""


@dataclasses.dataclass
class AnalysisFindings:
    """Output of a single analysis summarizer."""
    analysis_type: str
    source: str
    timestamp: str
    significant: List[Finding]
    trending: List[Finding]
    null: List[Finding]
    summary_text: str
    provenance: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify(p: float) -> str:
    if p < 0.05:
        return "significant"
    if p < 0.10:
        return "trending"
    return "null"


def _best_finding(findings: List[Finding]) -> Optional[Finding]:
    if not findings:
        return None
    return min(findings, key=lambda f: f.p_value)


def _generate_summary_text(
    analysis_type: str,
    significant: List[Finding],
    trending: List[Finding],
    null: List[Finding],
) -> str:
    total = len(significant) + len(trending) + len(null)
    parts = [f"{len(significant)}/{total} significant"]
    if trending:
        parts.append(f"{len(trending)}/{total} trending")
    text = f"{analysis_type}: {', '.join(parts)}."

    best = _best_finding(significant)
    if best:
        text += (
            f" Strongest: {best.comparison}"
            f" {best.statistic_name}={best.statistic_value}"
            f" (p={best.p_value:.4f})."
        )
    elif not significant:
        text += " No significant results."

    return text


def _finding_to_dict(f: Finding) -> Dict[str, Any]:
    return dataclasses.asdict(f)


def _findings_to_dict(af: AnalysisFindings) -> Dict[str, Any]:
    d = {
        "analysis_type": af.analysis_type,
        "source": af.source,
        "timestamp": af.timestamp,
        "counts": {
            "significant": len(af.significant),
            "trending": len(af.trending),
            "null": len(af.null),
            "total": len(af.significant) + len(af.trending) + len(af.null),
        },
        "significant": [_finding_to_dict(f) for f in af.significant],
        "trending": [_finding_to_dict(f) for f in af.trending],
        "null": [_finding_to_dict(f) for f in af.null],
        "summary_text": af.summary_text,
    }
    if af.provenance:
        d["provenance"] = af.provenance
    return d


# ---------------------------------------------------------------------------
# Per-analysis-type summarizers
# ---------------------------------------------------------------------------

def summarize_randomise(path: Path) -> AnalysisFindings:
    """Summarize FSL randomise results (TBSS, VBM, fALFF, ReHo)."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    prov = _extract_provenance(data, path)
    analysis_name = data.get("analysis_name", path.parent.name)
    contrast_names = data.get("contrast_names", [])
    n_subjects = data.get("n_subjects", 0)

    sig, trend, null = [], [], []
    for metric, mdata in data.get("results", {}).items():
        for con in mdata.get("contrasts", []):
            cnum = str(con.get("contrast_number", "?"))
            cidx = int(cnum) - 1 if cnum.isdigit() else 0
            cname = contrast_names[cidx] if cidx < len(contrast_names) else f"tstat{cnum}"
            n_vox = con.get("n_significant_voxels", 0)
            max_corrp = con.get("max_corrp", 0.0)
            # corrp is 1-p in FSL; significant if corrp > 0.95
            # For classification: significant = n_vox > 0
            # No continuous p-value for trending; use corrp thresholds
            if n_vox > 0:
                p_eff = 1.0 - max_corrp  # effective p
            else:
                p_eff = 1.0  # no significant voxels

            finding = Finding(
                comparison=f"{metric}/{cname}",
                test_name="Randomise TFCE",
                statistic_name="n_significant_voxels",
                statistic_value=n_vox,
                p_value=p_eff,
                n_subjects=n_subjects,
                detail=f"{n_vox} voxels (max 1-p={max_corrp:.4f})" if n_vox > 0
                    else "no significant voxels",
            )
            cat = _classify(p_eff)
            {"significant": sig, "trending": trend, "null": null}[cat].append(finding)

    text = _generate_summary_text(f"Randomise {analysis_name}", sig, trend, null)
    return AnalysisFindings(
        analysis_type="randomise",
        source=str(path),
        timestamp=data.get("date", datetime.now().isoformat()),
        significant=sig, trending=trend, null=null,
        summary_text=text,
        provenance=prov,
    )


summarize_tbss = summarize_randomise


def summarize_classification(path: Path) -> AnalysisFindings:
    """Summarize ROI-based classification results."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    prov = _extract_provenance(data, path)
    sig, trend, null = [], [], []
    for metric, combos in data.get("per_metric", {}).items():
        for combo_name, d in combos.items():
            if d.get("status") != "completed":
                continue
            n = d.get("n_samples", 0)

            # PERMANOVA
            perm = d.get("permanova", {})
            if "p_value" in perm:
                p = perm["p_value"]
                finding = Finding(
                    comparison=f"{metric}/{combo_name}",
                    test_name="PERMANOVA",
                    statistic_name="pseudo_F",
                    statistic_value=round(perm.get("pseudo_f", 0), 3),
                    p_value=p,
                    n_subjects=n,
                    detail=f"F={perm.get('pseudo_f', 0):.2f}, R²={perm.get('r_squared', 0):.3f}",
                )
                {"significant": sig, "trending": trend, "null": null}[_classify(p)].append(finding)

            # SVM classification
            svm = d.get("classification_svm", {})
            if "permutation_p_value" in svm:
                p = svm["permutation_p_value"]
                finding = Finding(
                    comparison=f"{metric}/{combo_name}",
                    test_name="SVM classification",
                    statistic_name="balanced_accuracy",
                    statistic_value=round(svm.get("balanced_accuracy", 0), 3),
                    p_value=p,
                    n_subjects=n,
                    detail=f"acc={svm.get('balanced_accuracy', 0):.3f}",
                )
                {"significant": sig, "trending": trend, "null": null}[_classify(p)].append(finding)

    text = _generate_summary_text("Classification", sig, trend, null)
    return AnalysisFindings(
        analysis_type="classification",
        source=str(path),
        timestamp=data.get("timestamp", datetime.now().isoformat()),
        significant=sig, trending=trend, null=null,
        summary_text=text,
        provenance=prov,
    )


def summarize_regression(path: Path) -> AnalysisFindings:
    """Summarize ROI-based regression results."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    prov = _extract_provenance(data, path)
    sig, trend, null = [], [], []
    for metric, combos in data.get("per_metric", {}).items():
        for combo_name, d in combos.items():
            if d.get("status") != "completed":
                continue
            n = d.get("n_samples", 0)

            for method in ("regression_svr", "regression_ridge", "regression_pls"):
                reg = d.get(method, {})
                if "permutation_p_value" not in reg:
                    continue
                r2 = reg.get("r_squared", 0)
                # Skip degenerate PLS fits
                if abs(r2) > 100:
                    continue
                p = reg["permutation_p_value"]
                method_short = method.replace("regression_", "").upper()
                finding = Finding(
                    comparison=f"{metric}/{combo_name}/{method_short}",
                    test_name=f"{method_short} regression",
                    statistic_name="r_squared",
                    statistic_value=round(r2, 3),
                    p_value=p,
                    n_subjects=n,
                    detail=f"R²={r2:.3f}, ρ={reg.get('spearman_rho', 0):.3f}",
                )
                {"significant": sig, "trending": trend, "null": null}[_classify(p)].append(finding)

    text = _generate_summary_text("Regression", sig, trend, null)
    return AnalysisFindings(
        analysis_type="regression",
        source=str(path),
        timestamp=data.get("timestamp", datetime.now().isoformat()),
        significant=sig, trending=trend, null=null,
        summary_text=text,
        provenance=prov,
    )


def summarize_mvpa(path: Path) -> AnalysisFindings:
    """Summarize MVPA whole-brain decoding results."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    prov = _extract_provenance(data, path)

    sig, trend, null = [], [], []
    for metric, combos in data.get("per_metric", {}).items():
        for combo_name, d in combos.items():
            if d.get("status") != "completed":
                continue
            n = d.get("n_subjects", 0)
            atype = d.get("analysis_type", "classification")

            # Whole-brain
            wb = d.get("whole_brain", {})
            if "permutation_p" in wb:
                p = wb["permutation_p"]
                score_label = wb.get("score_label", "score")
                score = wb.get("mean_score", 0)
                finding = Finding(
                    comparison=f"{metric}/{d.get('design', combo_name)}/{atype}",
                    test_name=f"MVPA whole-brain {atype}",
                    statistic_name=score_label,
                    statistic_value=round(score, 3),
                    p_value=p,
                    n_subjects=n,
                    detail=f"{score_label}={score:.3f} ± {wb.get('std_score', 0):.3f}",
                )
                {"significant": sig, "trending": trend, "null": null}[_classify(p)].append(finding)

            # Searchlight
            sl = d.get("searchlight", {})
            if sl.get("n_significant_voxels", 0) > 0:
                finding = Finding(
                    comparison=f"{metric}/{d.get('design', combo_name)}/{atype}/searchlight",
                    test_name="MVPA searchlight",
                    statistic_name="n_significant_voxels",
                    statistic_value=sl["n_significant_voxels"],
                    p_value=0.0,  # FWER-corrected, already thresholded
                    n_subjects=n,
                    detail=f"{sl['n_significant_voxels']} voxels (FWER)",
                )
                sig.append(finding)

    text = _generate_summary_text("MVPA", sig, trend, null)
    return AnalysisFindings(
        analysis_type="mvpa",
        source=str(path),
        timestamp=data.get("timestamp", datetime.now().isoformat()),
        significant=sig, trending=trend, null=null,
        summary_text=text,
        provenance=prov,
    )


def summarize_covnet_nbs(path: Path) -> AnalysisFindings:
    """Summarize NBS results from per-comparison component files."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    prov = _extract_provenance(data, path)

    # Try to find per-comparison detail files relative to summary
    nbs_root = path.parent / "nbs"

    sig, trend, null = [], [], []
    for metric, mdata in data.items():
        if metric.startswith("_"):
            continue
        n = mdata.get("n_subjects", 0)

        # Look for per-comparison components.json files
        metric_nbs_dir = None
        for modality in ("dwi", "msme", "func"):
            candidate = nbs_root / modality / metric
            if candidate.is_dir():
                metric_nbs_dir = candidate
                break

        if metric_nbs_dir and metric_nbs_dir.is_dir():
            for comp_dir in sorted(metric_nbs_dir.iterdir()):
                comp_file = comp_dir / "components.json"
                if not comp_file.exists():
                    continue
                with open(comp_file) as f:
                    comp = json.load(f)
                pair = f"{comp.get('group_a', '?')}_vs_{comp.get('group_b', '?')}"
                components = comp.get("components", [])
                if components:
                    best_p = min(c.get("pvalue", c.get("p_value", 1.0)) for c in components)
                    best_size = max(c.get("size", c.get("n_edges", 0)) for c in components)
                    finding = Finding(
                        comparison=f"{metric}/{pair}",
                        test_name="NBS",
                        statistic_name="component_edges",
                        statistic_value=best_size,
                        p_value=best_p,
                        n_subjects=n,
                        detail=f"{len(components)} component(s), largest={best_size} edges",
                    )
                else:
                    finding = Finding(
                        comparison=f"{metric}/{pair}",
                        test_name="NBS",
                        statistic_name="component_edges",
                        statistic_value=0,
                        p_value=1.0,
                        n_subjects=n,
                        detail="no significant components",
                    )
                {"significant": sig, "trending": trend, "null": null}[
                    _classify(finding.p_value)
                ].append(finding)
        else:
            # Fallback: summary-level counts only
            n_sig = mdata.get("n_significant", 0)
            n_comp = mdata.get("n_comparisons", 0)
            finding = Finding(
                comparison=metric,
                test_name="NBS (summary)",
                statistic_name="n_significant",
                statistic_value=n_sig,
                p_value=0.0 if n_sig > 0 else 1.0,
                n_subjects=n,
                detail=f"{n_sig}/{n_comp} comparisons significant",
            )
            (sig if n_sig > 0 else null).append(finding)

    text = _generate_summary_text("CovNet NBS", sig, trend, null)
    return AnalysisFindings(
        analysis_type="covnet_nbs",
        source=str(path),
        timestamp=datetime.now().isoformat(),
        significant=sig, trending=trend, null=null,
        summary_text=text,
        provenance=prov,
    )


def summarize_covnet_graph_theory(path: Path) -> AnalysisFindings:
    """Summarize graph theory results from AUC comparison CSVs."""
    import pandas as pd

    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    prov = _extract_provenance(data, path)
    graph_metrics_root = path.parent / "graph_metrics"

    sig, trend, null = [], [], []
    for metric, mdata in data.items():
        if metric.startswith("_"):
            continue
        n = mdata.get("n_subjects", 0)

        # Look for auc_comparison.csv
        auc_csv = None
        for modality in ("dwi", "msme", "func"):
            candidate = graph_metrics_root / modality / metric / "auc_comparison.csv"
            if candidate.exists():
                auc_csv = candidate
                break

        if auc_csv:
            df = pd.read_csv(auc_csv)
            for _, row in df.iterrows():
                pair = f"{row['group_a']}_vs_{row['group_b']}"
                gmetric = row["metric"]
                p = row["p_value"]
                finding = Finding(
                    comparison=f"{metric}/{pair}/{gmetric}",
                    test_name="Graph AUC",
                    statistic_name="auc_diff",
                    statistic_value=round(row.get("auc_diff", 0), 4),
                    p_value=p,
                    n_subjects=n,
                    detail=f"Δ={row.get('auc_diff', 0):.4f}",
                )
                {"significant": sig, "trending": trend, "null": null}[_classify(p)].append(finding)
        else:
            # Fallback to summary counts
            n_sig = mdata.get("n_significant", 0)
            n_total = mdata.get("n_total_tests", 0)
            finding = Finding(
                comparison=metric,
                test_name="Graph theory (summary)",
                statistic_name="n_significant",
                statistic_value=n_sig,
                p_value=0.0 if n_sig > 0 else 1.0,
                n_subjects=n,
                detail=f"{n_sig}/{n_total} tests significant",
            )
            (sig if n_sig > 0 else null).append(finding)

    text = _generate_summary_text("Graph Theory", sig, trend, null)
    return AnalysisFindings(
        analysis_type="covnet_graph_theory",
        source=str(path),
        timestamp=datetime.now().isoformat(),
        significant=sig, trending=trend, null=null,
        summary_text=text,
        provenance=prov,
    )


def summarize_mcca(path: Path) -> AnalysisFindings:
    """Summarize MCCA results across cohorts."""
    path = Path(path)

    # mcca_summary.json or walk per-cohort summary.json files
    if path.name == "mcca_summary.json":
        with open(path) as f:
            data = json.load(f)
        cohort_data = data.get("per_cohort", {})
    else:
        with open(path) as f:
            data = json.load(f)
        cohort_data = {"single": data}

    prov = _extract_provenance(data, path)
    sig, trend, null = [], [], []
    for cohort_key, d in cohort_data.items():
        if isinstance(d, str):
            continue
        if d.get("status") != "completed":
            continue
        cohort = d.get("cohort", cohort_key)
        n = d.get("n_samples", 0)

        # Canonical correlation significance
        cc = d.get("canonical_correlations", [])
        pp = d.get("permutation_p_values", [])
        for i, (corr, p) in enumerate(zip(cc, pp)):
            finding = Finding(
                comparison=f"{cohort}/CV{i+1}",
                test_name="MCCA canonical correlation",
                statistic_name="correlation",
                statistic_value=round(corr, 3),
                p_value=p,
                n_subjects=n,
                detail=f"r={corr:.3f}",
            )
            {"significant": sig, "trending": trend, "null": null}[_classify(p)].append(finding)

        # Dose association
        dose = d.get("dose_association", {})
        for cv_key, assoc in dose.items():
            if not isinstance(assoc, dict):
                continue
            p = assoc.get("p_value", 1.0)
            finding = Finding(
                comparison=f"{cohort}/{cv_key}/dose",
                test_name="MCCA dose association",
                statistic_name="spearman_rho",
                statistic_value=round(assoc.get("spearman_rho", 0), 3),
                p_value=p,
                n_subjects=n,
                detail=f"ρ={assoc.get('spearman_rho', 0):.3f}",
            )
            {"significant": sig, "trending": trend, "null": null}[_classify(p)].append(finding)

        # PERMANOVA
        perm = d.get("permanova", {})
        if "p_value" in perm:
            p = perm["p_value"]
            finding = Finding(
                comparison=f"{cohort}/permanova",
                test_name="MCCA PERMANOVA",
                statistic_name="pseudo_F",
                statistic_value=round(perm.get("pseudo_f", 0), 2),
                p_value=p,
                n_subjects=n,
                detail=f"F={perm.get('pseudo_f', 0):.2f}, R²={perm.get('r_squared', 0):.3f}",
            )
            {"significant": sig, "trending": trend, "null": null}[_classify(p)].append(finding)

        # Sex differences
        sex = d.get("sex_differences", {})
        sex_perm = sex.get("permanova", {})
        if "p_value" in sex_perm:
            p = sex_perm["p_value"]
            finding = Finding(
                comparison=f"{cohort}/sex_permanova",
                test_name="MCCA sex PERMANOVA",
                statistic_name="pseudo_F",
                statistic_value=round(sex_perm.get("pseudo_f", 0), 2),
                p_value=p,
                n_subjects=n,
                detail=f"F={sex_perm.get('pseudo_f', 0):.2f}",
            )
            {"significant": sig, "trending": trend, "null": null}[_classify(p)].append(finding)

    text = _generate_summary_text("MCCA", sig, trend, null)
    return AnalysisFindings(
        analysis_type="mcca",
        source=str(path),
        timestamp=datetime.now().isoformat(),
        significant=sig, trending=trend, null=null,
        summary_text=text,
        provenance=prov,
    )


def summarize_covnet_abs_distance(path: Path) -> AnalysisFindings:
    """Summarize absolute-distance covariance comparison results."""
    import pandas as pd

    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    prov = _extract_provenance(data, path)
    covnet_root = path.parent

    sig, trend, null = [], [], []
    for metric, mdata in data.items():
        if metric.startswith("_"):
            continue
        n = mdata.get("n_subjects", 0)

        # Try abs_distance first, fall back to whole_network (old naming)
        results_csv = None
        for subdir, csv_name in [
            ("abs_distance", "abs_distance_results.csv"),
            ("whole_network", "whole_network_results.csv"),
        ]:
            for modality in ("dwi", "msme", "func"):
                candidate = covnet_root / subdir / modality / metric / csv_name
                if candidate.exists():
                    results_csv = candidate
                    break
            if results_csv:
                break

        if results_csv:
            df = pd.read_csv(results_csv)
            for _, row in df.iterrows():
                pair = f"{row['group_a']}_vs_{row['group_b']}"
                for test, p_col, stat_col in [
                    ("Mantel", "mantel_p", "mantel_r"),
                    ("Frobenius", "frobenius_p", "frobenius_d"),
                    ("Spectral", "spectral_p", "spectral_d"),
                ]:
                    p = row[p_col]
                    stat = row[stat_col]
                    finding = Finding(
                        comparison=f"{metric}/{pair}/{test}",
                        test_name=f"Abs-distance {test}",
                        statistic_name=stat_col,
                        statistic_value=round(stat, 4),
                        p_value=p,
                        n_subjects=n,
                        detail=f"{stat_col}={stat:.4f}",
                    )
                    {"significant": sig, "trending": trend, "null": null}[
                        _classify(p)
                    ].append(finding)
        else:
            # Fallback to summary counts
            n_sig = mdata.get("n_significant", 0)
            n_comp = mdata.get("n_comparisons", 0)
            finding = Finding(
                comparison=metric,
                test_name="Abs-distance (summary)",
                statistic_name="n_significant",
                statistic_value=n_sig,
                p_value=0.0 if n_sig > 0 else 1.0,
                n_subjects=n,
                detail=f"{n_sig}/{n_comp} comparisons significant",
            )
            (sig if n_sig > 0 else null).append(finding)

    text = _generate_summary_text("Abs-Distance", sig, trend, null)
    return AnalysisFindings(
        analysis_type="covnet_abs_distance",
        source=str(path),
        timestamp=datetime.now().isoformat(),
        significant=sig, trending=trend, null=null,
        summary_text=text,
        provenance=prov,
    )


summarize_covnet_whole_network = summarize_covnet_abs_distance


def summarize_covnet_abs_distance_sex(path: Path) -> AnalysisFindings:
    """Summarize sex-stratified absolute-distance results."""
    import pandas as pd

    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    prov = _extract_provenance(data, path)
    covnet_root = path.parent

    sig, trend, null = [], [], []
    for entry_key, mdata in data.items():
        if entry_key.startswith("_"):
            continue
        metric = mdata.get("metric", entry_key)
        sex = mdata.get("sex", "")
        n_comp = mdata.get("n_comparisons", 0)

        # Try abs_distance_sex first, fall back to whole_network_sex
        results_csv = None
        for subdir, csv_name in [
            ("abs_distance_sex", "abs_distance_sex_results.csv"),
            ("whole_network_sex", "whole_network_sex_results.csv"),
        ]:
            for modality in ("dwi", "msme", "func"):
                candidate = covnet_root / subdir / modality / metric / csv_name
                if candidate.exists():
                    results_csv = candidate
                    break
            if results_csv:
                break

        if results_csv:
            df = pd.read_csv(results_csv)
            if sex and "sex" in df.columns:
                df = df[df["sex"] == sex]
            for _, row in df.iterrows():
                pair = f"{row['group_a']}_vs_{row['group_b']}"
                sex_label = f"/{sex}" if sex else ""
                for test, p_col, stat_col in [
                    ("Mantel", "mantel_p", "mantel_r"),
                    ("Frobenius", "frobenius_p", "frobenius_d"),
                    ("Spectral", "spectral_p", "spectral_d"),
                ]:
                    if p_col not in row.index:
                        continue
                    p = row[p_col]
                    stat = row[stat_col]
                    finding = Finding(
                        comparison=f"{metric}{sex_label}/{pair}/{test}",
                        test_name=f"Abs-distance {test}",
                        statistic_name=stat_col,
                        statistic_value=round(stat, 4),
                        p_value=p,
                        n_subjects=0,
                        detail=f"{stat_col}={stat:.4f}",
                    )
                    {"significant": sig, "trending": trend, "null": null}[
                        _classify(p)
                    ].append(finding)
        else:
            n_sig = mdata.get("n_significant", 0)
            sex_label = f"/{sex}" if sex else ""
            finding = Finding(
                comparison=f"{metric}{sex_label}",
                test_name="Abs-distance sex (summary)",
                statistic_name="n_significant",
                statistic_value=n_sig,
                p_value=0.0 if n_sig > 0 else 1.0,
                n_subjects=0,
                detail=f"{n_sig}/{n_comp} comparisons significant",
            )
            (sig if n_sig > 0 else null).append(finding)

    text = _generate_summary_text("Abs-Distance Sex", sig, trend, null)
    return AnalysisFindings(
        analysis_type="covnet_abs_distance_sex",
        source=str(path),
        timestamp=datetime.now().isoformat(),
        significant=sig, trending=trend, null=null,
        summary_text=text,
        provenance=prov,
    )


def summarize_covnet_rel_distance(path: Path) -> AnalysisFindings:
    """Summarize relative (maturation) distance results."""
    import pandas as pd

    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    prov = _extract_provenance(data, path)
    covnet_root = path.parent

    sig, trend, null = [], [], []
    for metric, mdata in data.items():
        if metric.startswith("_"):
            continue
        n = mdata.get("n_subjects", 0)

        # Try rel_distance first, then maturation_distance (old naming)
        results_csv = None
        for subdir, csv_name in [
            ("rel_distance", "rel_distance_results.csv"),
            ("maturation_distance", "maturation_distance_results.csv"),
        ]:
            for modality in ("dwi", "msme", "func"):
                candidate = covnet_root / subdir / modality / metric / csv_name
                if candidate.exists():
                    results_csv = candidate
                    break
            if results_csv:
                break

        if results_csv:
            df = pd.read_csv(results_csv)
            for _, row in df.iterrows():
                comparison = row.get("comparison", f"{row.get('dose_group', '?')}")
                dist_fn = row.get("distance_fn", "?")
                p_accel = row.get("p_accelerated", 1.0)
                p_decel = row.get("p_decelerated", 1.0)
                p_eff = min(p_accel, p_decel)
                direction = "accelerated" if p_accel < p_decel else "decelerated"
                delta = row.get("delta", 0)

                finding = Finding(
                    comparison=f"{metric}/{comparison}/{dist_fn}",
                    test_name=f"Rel-distance {dist_fn}",
                    statistic_name="delta",
                    statistic_value=round(delta, 4),
                    p_value=p_eff,
                    n_subjects=n,
                    detail=f"delta={delta:.4f} ({direction},"
                           f" p_accel={p_accel:.4f}, p_decel={p_decel:.4f})",
                )
                {"significant": sig, "trending": trend, "null": null}[
                    _classify(p_eff)
                ].append(finding)
        else:
            n_accel = mdata.get("n_accelerated", 0)
            n_decel = mdata.get("n_decelerated", 0)
            n_total = mdata.get("n_total_tests", 0)
            n_sig = n_accel + n_decel
            finding = Finding(
                comparison=metric,
                test_name="Rel-distance (summary)",
                statistic_name="n_significant",
                statistic_value=n_sig,
                p_value=0.0 if n_sig > 0 else 1.0,
                n_subjects=n,
                detail=f"{n_accel} accelerated, {n_decel} decelerated / {n_total} tests",
            )
            (sig if n_sig > 0 else null).append(finding)

    text = _generate_summary_text("Rel-Distance", sig, trend, null)
    return AnalysisFindings(
        analysis_type="covnet_rel_distance",
        source=str(path),
        timestamp=datetime.now().isoformat(),
        significant=sig, trending=trend, null=null,
        summary_text=text,
        provenance=prov,
    )


def summarize_covnet_rel_distance_sex(path: Path) -> AnalysisFindings:
    """Summarize sex-stratified relative (maturation) distance results."""
    import pandas as pd

    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    prov = _extract_provenance(data, path)
    covnet_root = path.parent

    sig, trend, null = [], [], []
    for entry_key, mdata in data.items():
        if entry_key.startswith("_"):
            continue
        metric = mdata.get("metric", entry_key)
        sex = mdata.get("sex", "")
        n_tests = mdata.get("n_tests", 0)

        # Try rel_distance_sex first, then maturation_distance_sex
        results_csv = None
        for subdir, csv_name in [
            ("rel_distance_sex", "rel_distance_sex_results.csv"),
            ("maturation_distance_sex", "maturation_distance_sex_results.csv"),
        ]:
            for modality in ("dwi", "msme", "func"):
                candidate = covnet_root / subdir / modality / metric / csv_name
                if candidate.exists():
                    results_csv = candidate
                    break
            if results_csv:
                break

        if results_csv:
            df = pd.read_csv(results_csv)
            if sex and "sex" in df.columns:
                df = df[df["sex"] == sex]
            for _, row in df.iterrows():
                comparison = row.get("comparison", f"{row.get('dose_group', '?')}")
                dist_fn = row.get("distance_fn", "?")
                p_accel = row.get("p_accelerated", 1.0)
                p_decel = row.get("p_decelerated", 1.0)
                p_eff = min(p_accel, p_decel)
                direction = "accelerated" if p_accel < p_decel else "decelerated"
                delta = row.get("delta", 0)
                sex_label = f"/{sex}" if sex else ""

                finding = Finding(
                    comparison=f"{metric}{sex_label}/{comparison}/{dist_fn}",
                    test_name=f"Rel-distance {dist_fn}",
                    statistic_name="delta",
                    statistic_value=round(delta, 4),
                    p_value=p_eff,
                    n_subjects=0,
                    detail=f"delta={delta:.4f} ({direction},"
                           f" p_accel={p_accel:.4f}, p_decel={p_decel:.4f})",
                )
                {"significant": sig, "trending": trend, "null": null}[
                    _classify(p_eff)
                ].append(finding)
        else:
            n_accel = mdata.get("n_accelerated", 0)
            n_decel = mdata.get("n_decelerated", 0)
            n_sig = n_accel + n_decel
            sex_label = f"/{sex}" if sex else ""
            finding = Finding(
                comparison=f"{metric}{sex_label}",
                test_name="Rel-distance sex (summary)",
                statistic_name="n_significant",
                statistic_value=n_sig,
                p_value=0.0 if n_sig > 0 else 1.0,
                n_subjects=0,
                detail=f"{n_accel} accelerated, {n_decel} decelerated / {n_tests} tests",
            )
            (sig if n_sig > 0 else null).append(finding)

    text = _generate_summary_text("Rel-Distance Sex", sig, trend, null)
    return AnalysisFindings(
        analysis_type="covnet_rel_distance_sex",
        source=str(path),
        timestamp=datetime.now().isoformat(),
        significant=sig, trending=trend, null=null,
        summary_text=text,
        provenance=prov,
    )


def summarize_covnet_territory(path: Path) -> AnalysisFindings:
    """Summarize territory (edge-level Fisher Z) results."""
    import pandas as pd

    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    prov = _extract_provenance(data, path)
    territory_root = path.parent / "territory"

    sig, trend, null = [], [], []
    for metric, mdata in data.items():
        if metric.startswith("_"):
            continue
        n = mdata.get("n_subjects", 0)

        results_csv = None
        for modality in ("dwi", "msme", "func"):
            candidate = territory_root / modality / metric / "fisher_z_results.csv"
            if candidate.exists():
                results_csv = candidate
                break

        if results_csv:
            df = pd.read_csv(results_csv)
            for _, row in df.iterrows():
                comparison = row.get("comparison", "?")
                roi_a = row.get("roi_a", "?")
                roi_b = row.get("roi_b", "?")
                p_fdr = row.get("p_fdr", 1.0)
                z = row.get("z_stat", 0)

                finding = Finding(
                    comparison=f"{metric}/{comparison}/{roi_a}-{roi_b}",
                    test_name="Territory Fisher-Z",
                    statistic_name="z_stat",
                    statistic_value=round(z, 4),
                    p_value=p_fdr,
                    n_subjects=n,
                    detail=f"z={z:.4f}, p_fdr={p_fdr:.4f}",
                )
                {"significant": sig, "trending": trend, "null": null}[
                    _classify(p_fdr)
                ].append(finding)
        else:
            n_sig = mdata.get("n_fdr_significant_edges", 0)
            n_comp = mdata.get("n_comparisons", 0)
            finding = Finding(
                comparison=metric,
                test_name="Territory (summary)",
                statistic_name="n_fdr_significant",
                statistic_value=n_sig,
                p_value=0.0 if n_sig > 0 else 1.0,
                n_subjects=n,
                detail=f"{n_sig} FDR-significant edges across {n_comp} comparisons",
            )
            (sig if n_sig > 0 else null).append(finding)

    text = _generate_summary_text("Territory", sig, trend, null)
    return AnalysisFindings(
        analysis_type="covnet_territory",
        source=str(path),
        timestamp=datetime.now().isoformat(),
        significant=sig, trending=trend, null=null,
        summary_text=text,
        provenance=prov,
    )


def summarize_covnet_nbs_interaction(path: Path) -> AnalysisFindings:
    """Summarize NBS interaction (dose x sex) results."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    prov = _extract_provenance(data, path)
    nbs_int_root = path.parent / "nbs_interaction"

    sig, trend, null = [], [], []
    for metric, mdata in data.items():
        if metric.startswith("_"):
            continue
        if not isinstance(mdata, dict) or "cohorts" not in mdata:
            continue
        for cohort, cdata in mdata["cohorts"].items():
            n = cdata.get("n_subjects", 0)

            comp_file = None
            for modality in ("dwi", "msme", "func"):
                candidate = nbs_int_root / modality / metric / cohort / "components.json"
                if candidate.exists():
                    comp_file = candidate
                    break

            if comp_file:
                with open(comp_file) as cf:
                    comp = json.load(cf)
                components = comp.get("components", [])
                if components:
                    best_p = min(
                        c.get("pvalue", c.get("p_value", 1.0)) for c in components
                    )
                    best_size = max(
                        c.get("size", c.get("n_edges", 0)) for c in components
                    )
                    finding = Finding(
                        comparison=f"{metric}/{cohort}",
                        test_name="NBS interaction",
                        statistic_name="component_edges",
                        statistic_value=best_size,
                        p_value=best_p,
                        n_subjects=n,
                        detail=f"{len(components)} component(s), largest={best_size} edges",
                    )
                else:
                    finding = Finding(
                        comparison=f"{metric}/{cohort}",
                        test_name="NBS interaction",
                        statistic_name="component_edges",
                        statistic_value=0,
                        p_value=1.0,
                        n_subjects=n,
                        detail="no significant components",
                    )
            else:
                n_sig = cdata.get("n_significant", 0)
                finding = Finding(
                    comparison=f"{metric}/{cohort}",
                    test_name="NBS interaction (summary)",
                    statistic_name="n_significant",
                    statistic_value=n_sig,
                    p_value=0.0 if n_sig > 0 else 1.0,
                    n_subjects=n,
                    detail=f"{n_sig} significant component(s)",
                )
            {"significant": sig, "trending": trend, "null": null}[
                _classify(finding.p_value)
            ].append(finding)

    text = _generate_summary_text("NBS Interaction", sig, trend, null)
    return AnalysisFindings(
        analysis_type="covnet_nbs_interaction",
        source=str(path),
        timestamp=datetime.now().isoformat(),
        significant=sig, trending=trend, null=null,
        summary_text=text,
        provenance=prov,
    )


def summarize_edge_regression(path: Path) -> AnalysisFindings:
    """Summarize edge regression NBS-style results."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    prov = _extract_provenance(data, path)

    # Summary JSON could be in covnet/ or covnet/edge_regression/
    if path.parent.name == "edge_regression":
        edge_root = path.parent
    else:
        edge_root = path.parent / "edge_regression"

    sig, trend, null = [], [], []
    for metric, mdata in data.items():
        if metric.startswith("_"):
            continue
        if not isinstance(mdata, dict):
            continue
        for cohort, cdata in mdata.items():
            if not isinstance(cdata, dict):
                continue
            n = cdata.get("n_subjects", 0)

            comp_file = None
            for modality in ("dwi", "msme", "func"):
                candidate = edge_root / modality / metric / cohort / "components.json"
                if candidate.exists():
                    comp_file = candidate
                    break

            if comp_file:
                with open(comp_file) as cf:
                    comp = json.load(cf)
                components = comp.get(
                    "components", comp.get("significant_components", [])
                )
                if components:
                    best_p = min(
                        c.get("pvalue", c.get("p_value", 1.0)) for c in components
                    )
                    best_size = max(
                        c.get("size", c.get("n_edges", 0)) for c in components
                    )
                    finding = Finding(
                        comparison=f"{metric}/{cohort}",
                        test_name="Edge regression",
                        statistic_name="component_edges",
                        statistic_value=best_size,
                        p_value=best_p,
                        n_subjects=n,
                        detail=f"{len(components)} component(s), largest={best_size} edges",
                    )
                else:
                    finding = Finding(
                        comparison=f"{metric}/{cohort}",
                        test_name="Edge regression",
                        statistic_name="component_edges",
                        statistic_value=0,
                        p_value=1.0,
                        n_subjects=n,
                        detail="no significant components",
                    )
            else:
                n_sig = cdata.get("n_significant_components", 0)
                finding = Finding(
                    comparison=f"{metric}/{cohort}",
                    test_name="Edge regression (summary)",
                    statistic_name="n_significant",
                    statistic_value=n_sig,
                    p_value=0.0 if n_sig > 0 else 1.0,
                    n_subjects=n,
                    detail=f"{n_sig} significant component(s)",
                )
            {"significant": sig, "trending": trend, "null": null}[
                _classify(finding.p_value)
            ].append(finding)

    text = _generate_summary_text("Edge Regression", sig, trend, null)
    return AnalysisFindings(
        analysis_type="edge_regression",
        source=str(path),
        timestamp=datetime.now().isoformat(),
        significant=sig, trending=trend, null=null,
        summary_text=text,
        provenance=prov,
    )


def summarize_fc_graph_theory(path: Path) -> AnalysisFindings:
    """Summarize FC graph theory pairwise test results."""
    import pandas as pd

    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    prov = _extract_provenance(data, path)
    gt_root = path.parent

    sig, trend, null = [], [], []
    for cohort, cdata in data.get("cohorts", {}).items():
        if not isinstance(cdata, dict):
            continue
        n = cdata.get("n_subjects", 0)

        csv_path = gt_root / cohort / "pairwise_tests.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                pair = f"{row['group_a']}_vs_{row['group_b']}"
                gmetric = row["metric"]
                p = row["p_value"]
                diff = row.get("diff", 0)
                finding = Finding(
                    comparison=f"{cohort}/{pair}/{gmetric}",
                    test_name="FC graph theory",
                    statistic_name="diff",
                    statistic_value=round(diff, 4),
                    p_value=p,
                    n_subjects=n,
                    detail=f"diff={diff:.4f}",
                )
                {"significant": sig, "trending": trend, "null": null}[
                    _classify(p)
                ].append(finding)
        else:
            n_sig = cdata.get("n_significant", 0)
            n_tests = cdata.get("n_tests", 0)
            finding = Finding(
                comparison=cohort,
                test_name="FC graph theory (summary)",
                statistic_name="n_significant",
                statistic_value=n_sig,
                p_value=0.0 if n_sig > 0 else 1.0,
                n_subjects=n,
                detail=f"{n_sig}/{n_tests} significant",
            )
            (sig if n_sig > 0 else null).append(finding)

    text = _generate_summary_text("FC Graph Theory", sig, trend, null)
    return AnalysisFindings(
        analysis_type="fc_graph_theory",
        source=str(path),
        timestamp=data.get("timestamp", datetime.now().isoformat()),
        significant=sig, trending=trend, null=null,
        summary_text=text,
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

SUMMARIZERS = {
    "randomise": summarize_randomise,
    "tbss": summarize_tbss,  # backward compat alias
    "classification": summarize_classification,
    "regression": summarize_regression,
    "mvpa": summarize_mvpa,
    "covnet_nbs": summarize_covnet_nbs,
    "covnet_graph_theory": summarize_covnet_graph_theory,
    "covnet_abs_distance": summarize_covnet_abs_distance,
    "covnet_whole_network": summarize_covnet_whole_network,  # backward compat alias
    "covnet_abs_distance_sex": summarize_covnet_abs_distance_sex,
    "covnet_rel_distance": summarize_covnet_rel_distance,
    "covnet_rel_distance_sex": summarize_covnet_rel_distance_sex,
    "covnet_territory": summarize_covnet_territory,
    "covnet_nbs_interaction": summarize_covnet_nbs_interaction,
    "edge_regression": summarize_edge_regression,
    "fc_graph_theory": summarize_fc_graph_theory,
    "mcca": summarize_mcca,
}


def summarize_analysis(
    analysis_type: str,
    summary_json: Path,
    output_dir: Optional[Path] = None,
) -> AnalysisFindings:
    """Summarize a single analysis from its summary JSON.

    If *output_dir* is given, writes ``analysis_findings.json`` and
    ``analysis_findings.txt`` there.
    """
    summarizer = SUMMARIZERS.get(analysis_type)
    if summarizer is None:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    findings = summarizer(Path(summary_json))

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_findings_json(findings, output_dir)
        write_findings_text(findings, output_dir)

    return findings


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_findings_json(findings: AnalysisFindings, output_dir: Path) -> Path:
    """Write analysis_findings.json."""
    output_dir = Path(output_dir)
    out = output_dir / "analysis_findings.json"
    with open(out, "w") as f:
        json.dump(_findings_to_dict(findings), f, indent=2)
    logger.info("Wrote findings JSON: %s", out)
    return out


def write_findings_text(findings: AnalysisFindings, output_dir: Path) -> Path:
    """Write analysis_findings.txt — human-readable summary."""
    output_dir = Path(output_dir)
    out = output_dir / "analysis_findings.txt"

    prov = findings.provenance or {}
    prov_ts = prov.get("timestamp", "")
    prov_script = prov.get("script", "")
    prov_source = prov.get("source", "")  # "file_mtime" if fallback
    prov_ver = prov.get("neurofaune_version", "")

    lines = [
        f"{'=' * 70}",
        f"  {findings.analysis_type.upper()} — Analysis Findings",
        f"{'=' * 70}",
        f"Source: {findings.source}",
        f"Date:   {findings.timestamp}",
    ]

    # Provenance block
    if prov_ts:
        if prov_source == "file_mtime":
            lines.append(f"Run:    {prov_ts}  (from file mtime — no embedded provenance)")
        else:
            lines.append(f"Run:    {prov_ts}")
    if prov_script:
        lines.append(f"Script: {prov_script}")
    if prov_ver:
        lines.append(f"Pipeline version: {prov_ver}")
    if not prov or prov_source == "file_mtime":
        lines.append("WARNING: No embedded provenance — re-run analysis to record provenance")

    lines += [
        "",
        findings.summary_text,
        "",
    ]

    for label, bucket in [
        ("SIGNIFICANT (p < 0.05)", findings.significant),
        ("TRENDING (0.05 ≤ p < 0.10)", findings.trending),
    ]:
        if bucket:
            lines.append(f"{label}:")
            lines.append("-" * 50)
            for f in sorted(bucket, key=lambda x: x.p_value):
                lines.append(
                    f"  {f.comparison:50s}  "
                    f"{f.statistic_name}={f.statistic_value:<10}  "
                    f"p={f.p_value:.4f}  "
                    f"n={f.n_subjects}"
                )
                if f.detail:
                    lines.append(f"    {f.detail}")
            lines.append("")

    n_null = len(findings.null)
    if n_null > 0:
        lines.append(f"NULL (p ≥ 0.10): {n_null} comparisons")
        lines.append("")

    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    logger.info("Wrote findings text: %s", out)
    return out


# ---------------------------------------------------------------------------
# Cross-analysis aggregation
# ---------------------------------------------------------------------------

def summarize_all(
    summary_paths: Dict[str, Path],
    output_dir: Path,
) -> Dict[str, AnalysisFindings]:
    """Summarize multiple analyses and write a cross-analysis report.

    Args:
        summary_paths: Mapping of analysis_type to summary JSON path.
        output_dir: Where to write cross_analysis_findings.{json,txt}.

    Returns:
        Dict of analysis_type to AnalysisFindings.
    """
    all_findings = {}
    for atype, spath in summary_paths.items():
        try:
            findings = summarize_analysis(atype, spath)
            all_findings[atype] = findings
        except Exception as exc:
            logger.warning("Failed to summarize %s: %s", atype, exc)

    if all_findings:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_cross_analysis(all_findings, output_dir)

    return all_findings


def _write_cross_analysis(
    all_findings: Dict[str, AnalysisFindings],
    output_dir: Path,
) -> None:
    """Write cross-analysis convergence report."""
    # JSON
    combined = {
        atype: _findings_to_dict(f) for atype, f in all_findings.items()
    }
    with open(output_dir / "cross_analysis_findings.json", "w") as f:
        json.dump(combined, f, indent=2)

    # Text
    lines = [
        "=" * 70,
        "  CROSS-ANALYSIS FINDINGS SUMMARY",
        "=" * 70,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Analyses: {len(all_findings)}",
        "",
    ]

    # Per-analysis summary lines with provenance
    missing_prov = []
    oldest_ts = None
    newest_ts = None
    for atype, findings in all_findings.items():
        n_sig = len(findings.significant)
        n_trend = len(findings.trending)
        n_total = n_sig + n_trend + len(findings.null)
        prov = findings.provenance or {}
        prov_ts = prov.get("timestamp", "")
        prov_src = prov.get("source", "")
        age_tag = ""
        if prov_ts:
            try:
                ts = datetime.fromisoformat(prov_ts)
                if oldest_ts is None or ts < oldest_ts:
                    oldest_ts = ts
                if newest_ts is None or ts > newest_ts:
                    newest_ts = ts
                age_days = (datetime.now() - ts).days
                if age_days > 14:
                    age_tag = f"  [STALE: {age_days}d old]"
                else:
                    age_tag = f"  [{prov_ts[:10]}]"
            except (ValueError, TypeError):
                age_tag = ""
        if not prov or prov_src == "file_mtime":
            missing_prov.append(atype)
            age_tag += "  [no provenance]"
        lines.append(
            f"  {atype:40s}  {n_sig:3d}/{n_total:<3d} significant  "
            f"{n_trend:3d} trending{age_tag}"
        )
    lines.append("")

    # Provenance warnings
    if missing_prov:
        lines.append(f"WARNING: {len(missing_prov)} analyses lack embedded provenance:")
        for atype in missing_prov:
            lines.append(f"  - {atype}")
        lines.append("  Re-run these analyses with updated scripts to record provenance.")
        lines.append("")
    if oldest_ts and newest_ts:
        span_days = (newest_ts - oldest_ts).days
        if span_days > 7:
            lines.append(
                f"WARNING: Analysis dates span {span_days} days "
                f"({oldest_ts:%Y-%m-%d} to {newest_ts:%Y-%m-%d}). "
                f"Older results may not reflect current data/exclusions."
            )
            lines.append("")

    # Convergence: group significant findings by comparison root
    # (strip method/searchlight suffix, group by metric/cohort pattern)
    convergence = {}
    for atype, findings in all_findings.items():
        for f in findings.significant:
            # Extract a coarse key: metric + cohort/design
            parts = f.comparison.split("/")
            if len(parts) >= 2:
                coarse_key = "/".join(parts[:2])
            else:
                coarse_key = f.comparison
            convergence.setdefault(coarse_key, []).append(
                (atype, f)
            )

    # Report comparisons with convergent evidence (>1 analysis type)
    converging = {
        k: v for k, v in convergence.items()
        if len(set(atype for atype, _ in v)) > 1
    }
    if converging:
        lines.append("CONVERGENT FINDINGS (significant in >1 analysis type):")
        lines.append("-" * 60)
        for key in sorted(converging, key=lambda k: -len(converging[k])):
            entries = converging[key]
            types = sorted(set(atype for atype, _ in entries))
            lines.append(f"  {key}:")
            for atype, f in sorted(entries, key=lambda x: x[1].p_value):
                lines.append(
                    f"    {atype:20s}  {f.test_name:25s}  "
                    f"{f.statistic_name}={f.statistic_value}  p={f.p_value:.4f}"
                )
        lines.append("")

    with open(output_dir / "cross_analysis_findings.txt", "w") as fh:
        fh.write("\n".join(lines))

    logger.info("Wrote cross-analysis findings to %s", output_dir)
