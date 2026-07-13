"""Top-level preprocessing QC index.

Scans a study's ``qc/`` tree and renders one navigable ``index.html`` that ties
every QC layer together:

* per-modality **batch dashboards** (``reports/<mod>/summary.html``) + galleries,
* cohort **montage galleries** (``slicesdir/**/index.html``), and
* a **subject x session x modality matrix** linking every per-session report
  (``subjects/<sub>/<ses>/<mod>/*_qc.html``), with batch-flagged sessions marked.

Idempotent and incremental by design: it renders whatever is present, so it can
be regenerated after each modality's QC completes — each additional modality
simply appears on the next run. All links are relative to ``qc/`` so the tree
stays portable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Modality order + display label. Any modality dir not listed is appended as-is.
_MODALITIES: List[Tuple[str, str]] = [
    ("anat", "Anatomical (T2w)"),
    ("dwi", "Diffusion (DTI/DKI/NODDI)"),
    ("func", "Functional (rs-fMRI)"),
    ("msme", "MSME / MWF"),
]


def _rel(p: Path, base: Path) -> str:
    return p.relative_to(base).as_posix()


def _load_flagged(reports_dir: Path, modality: str) -> Dict[Tuple[str, str], str]:
    """Map (subject, session) -> flag string from a modality's batch exclusions."""
    flagged: Dict[Tuple[str, str], str] = {}
    f = reports_dir / modality / "exclusions_by_reason.json"
    if f.exists():
        try:
            data = json.loads(f.read_text())
            for items in (data.get("by_reason") or {}).values():
                for it in items:
                    flagged[(it["subject"], it["session"])] = it.get("flags", "flagged")
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return flagged


def generate_qc_index(
    qc_dir,
    study_name: Optional[str] = None,
    output_file=None,
) -> Optional[Path]:
    """Render ``qc/index.html`` from whatever QC is present under ``qc_dir``.

    Parameters
    ----------
    qc_dir : Path
        Study QC root (holds ``subjects/``, ``reports/``, ``slicesdir/``).
    study_name : str, optional
        Shown in the header.
    output_file : Path, optional
        Defaults to ``qc_dir/index.html``.

    Returns
    -------
    Path or None
        The written index, or None if ``qc_dir`` does not exist.
    """
    qc_dir = Path(qc_dir)
    if not qc_dir.exists():
        return None
    subjects_dir = qc_dir / "subjects"
    reports_dir = qc_dir / "reports"
    slices_dir = qc_dir / "slicesdir"
    out = Path(output_file) if output_file else qc_dir / "index.html"

    # ---- 1. per-session reports: (sub, ses) -> {mod: [(label, relpath)]} ------
    sessions: Dict[Tuple[str, str], Dict[str, List[Tuple[str, str]]]] = {}
    mods_present: List[str] = []
    if subjects_dir.exists():
        for subdir in sorted(subjects_dir.glob("sub-*")):
            for sesdir in sorted(subdir.glob("ses-*")):
                for moddir in sorted(sesdir.iterdir()):
                    if not moddir.is_dir():
                        continue
                    mod = moddir.name
                    reports = sorted(moddir.glob("*_qc.html"))
                    if not reports:
                        continue
                    prefix = f"{subdir.name}_{sesdir.name}_"
                    labelled = [
                        (r.stem.replace(prefix, "").replace("_qc", "") or "report",
                         _rel(r, qc_dir))
                        for r in reports
                    ]
                    sessions.setdefault((subdir.name, sesdir.name), {})[mod] = labelled
                    if mod not in mods_present:
                        mods_present.append(mod)

    # order modalities: known first (in _MODALITIES order), then any extras
    known = [m for m, _ in _MODALITIES if m in mods_present]
    order = known + [m for m in mods_present if m not in known]
    label_of = dict(_MODALITIES)

    # ---- 2. batch dashboards + galleries + flags per modality ----------------
    dashboards: Dict[str, Dict[str, str]] = {}
    flagged: Dict[str, Dict[Tuple[str, str], str]] = {}
    if reports_dir.exists():
        for mod in order:
            d: Dict[str, str] = {}
            summ = reports_dir / mod / "summary.html"
            gal = reports_dir / mod / "thumbnail_gallery.html"
            if summ.exists():
                d["dashboard"] = _rel(summ, qc_dir)
            if gal.exists():
                d["gallery"] = _rel(gal, qc_dir)
            if d:
                dashboards[mod] = d
            flagged[mod] = _load_flagged(reports_dir, mod)

    # ---- 3. montage galleries (slicesdir/**/slicesdir/index.html) ------------
    montages: List[Tuple[str, str]] = []
    if slices_dir.exists():
        for idx in sorted(slices_dir.glob("*/*/slicesdir/index.html")):
            name = idx.relative_to(slices_dir).parts[:2]  # (group, name)
            montages.append((" / ".join(name), _rel(idx, qc_dir)))

    # ---- 4. render -----------------------------------------------------------
    n_sessions = len(sessions)
    n_subjects = len({s for s, _ in sessions})
    total_flagged = len({k for m in flagged.values() for k in m})
    title = f"Preprocessing QC — {study_name}" if study_name else "Preprocessing QC index"

    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    # modality cards
    cards = []
    for mod in order:
        lbl = label_of.get(mod, mod.upper())
        n = sum(1 for k in sessions if mod in sessions[k])
        nf = len(flagged.get(mod, {}))
        links = []
        db = dashboards.get(mod, {})
        if "dashboard" in db:
            links.append(f'<a href="{db["dashboard"]}">batch dashboard</a>')
        if "gallery" in db:
            links.append(f'<a href="{db["gallery"]}">thumbnail gallery</a>')
        links_html = " · ".join(links) if links else '<span class="muted">no batch rollup</span>'
        flag_badge = f'<span class="flag">{nf} flagged</span>' if nf else '<span class="ok">0 flagged</span>'
        cards.append(
            f'<div class="card"><h3>{esc(lbl)}</h3>'
            f'<div class="count">{n} sessions · {flag_badge}</div>'
            f'<div class="links">{links_html}</div></div>'
        )
    cards_html = "\n".join(cards) or '<p class="muted">No per-session QC found yet.</p>'

    # montage list
    if montages:
        mont_items = "\n".join(f'<li><a href="{p}">{esc(name)}</a></li>' for name, p in montages)
        montages_html = f"<ul class='montages'>{mont_items}</ul>"
    else:
        montages_html = '<p class="muted">No montage galleries yet.</p>'

    # session matrix: rows = (sub, ses), cols = modalities
    head_cells = "".join(f"<th>{esc(label_of.get(m, m.upper()))}</th>" for m in order)
    rows = []
    for (sub, ses) in sorted(sessions):
        cells = [f'<td class="sess">{esc(sub)}<br><span class="ses">{esc(ses)}</span></td>']
        for mod in order:
            reps = sessions[(sub, ses)].get(mod)
            if not reps:
                cells.append('<td class="none">—</td>')
                continue
            is_flag = (sub, ses) in flagged.get(mod, {})
            rlinks = " ".join(f'<a href="{p}">{esc(lbl)}</a>' for lbl, p in reps)
            cls = "cell flagged" if is_flag else "cell"
            title_attr = f' title="{esc(flagged[mod][(sub, ses)])}"' if is_flag else ""
            warn = '<span class="warn">⚠</span> ' if is_flag else ""
            cells.append(f'<td class="{cls}"{title_attr}>{warn}{rlinks}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    matrix_html = (
        f'<table class="matrix"><thead><tr><th>Subject / Session</th>{head_cells}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    ) if rows else '<p class="muted">No sessions yet.</p>'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>
 body {{ font-family: -apple-system, Arial, sans-serif; margin: 24px; color: #222; background:#fafafa; }}
 h1 {{ border-bottom: 3px solid #4CAF50; padding-bottom: 8px; }}
 h2 {{ color:#444; margin-top: 32px; }}
 .sub {{ color:#777; font-size: 0.9em; }}
 .cards {{ display:flex; flex-wrap:wrap; gap:14px; }}
 .card {{ background:#fff; border:1px solid #e0e0e0; border-radius:8px; padding:14px 16px; min-width:220px; box-shadow:0 1px 3px rgba(0,0,0,.05); }}
 .card h3 {{ margin:0 0 6px; }}
 .count {{ color:#555; font-size:.9em; margin-bottom:8px; }}
 .links a {{ color:#1565c0; text-decoration:none; }} .links a:hover {{ text-decoration:underline; }}
 .flag {{ color:#c62828; font-weight:bold; }} .ok {{ color:#2e7d32; }}
 ul.montages {{ columns: 2; }} ul.montages a {{ color:#1565c0; text-decoration:none; }}
 table.matrix {{ border-collapse:collapse; width:100%; background:#fff; }}
 table.matrix th, table.matrix td {{ border:1px solid #e0e0e0; padding:6px 8px; text-align:left; font-size:.86em; vertical-align:top; }}
 table.matrix thead th {{ background:#f0f0f0; position:sticky; top:0; }}
 td.sess {{ white-space:nowrap; font-weight:bold; }} td.sess .ses {{ color:#888; font-weight:normal; }}
 td.cell a {{ color:#1565c0; text-decoration:none; margin-right:6px; display:inline-block; }}
 td.cell a:hover {{ text-decoration:underline; }}
 td.none {{ color:#ccc; text-align:center; }}
 td.flagged {{ background:#fff3e0; }} .warn {{ color:#e65100; }}
 .muted {{ color:#999; }}
</style></head><body>
<h1>{esc(title)}</h1>
<p class="sub">{n_subjects} subjects · {n_sessions} sessions · {total_flagged} flagged by batch rollups.
Regenerate anytime — this index reflects whatever QC is present.</p>

<h2>Modalities</h2>
<div class="cards">{cards_html}</div>

<h2>Cohort montages</h2>
{montages_html}

<h2>Per-session reports</h2>
<p class="sub">Each cell links to that session's QC report(s). ⚠ = flagged by the batch rollup (hover for details).</p>
{matrix_html}
</body></html>"""

    out.write_text(html)
    return out
