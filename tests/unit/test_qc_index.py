"""Unit tests for the top-level preprocessing QC index generator."""
import json
from pathlib import Path

from neurofaune.preprocess.qc import generate_qc_index


def _touch(p: Path, text: str = "<html></html>"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _make_qc_tree(qc: Path):
    # two sessions, partial modality coverage
    _touch(qc / "subjects/sub-1A/ses-1/anat/sub-1A_ses-1_anat_qc.html")
    _touch(qc / "subjects/sub-1A/ses-1/dwi/sub-1A_ses-1_eddy_qc.html")
    _touch(qc / "subjects/sub-1A/ses-1/dwi/sub-1A_ses-1_dti_qc.html")
    _touch(qc / "subjects/sub-1A/ses-1/msme/sub-1A_ses-1_msme_qc.html")
    _touch(qc / "subjects/sub-2B/ses-1/anat/sub-2B_ses-1_anat_qc.html")  # anat only
    # a batch dashboard + gallery + exclusions for dwi (flags sub-1A ses-1)
    _touch(qc / "reports/dwi/summary.html")
    _touch(qc / "reports/dwi/thumbnail_gallery.html")
    (qc / "reports/dwi/exclusions_by_reason.json").write_text(json.dumps({
        "by_reason": {"high_motion": [{"subject": "sub-1A", "session": "ses-1", "flags": "mean_fd high"}]}
    }))
    # montage galleries: the modality arms nest two deep, the anat arm one
    _touch(qc / "slicesdir/dwi/fa/slicesdir/index.html")
    _touch(qc / "slicesdir/brain_extraction/slicesdir/index.html")


def test_index_renders_and_links_resolve(tmp_path):
    qc = tmp_path / "qc"
    _make_qc_tree(qc)
    out = generate_qc_index(qc, study_name="teststudy")
    assert out == qc / "index.html" and out.exists()
    html = out.read_text()

    # sections + modality cards present
    assert "teststudy" in html
    for tok in ("Modalities", "Cohort montages", "Per-session reports"):
        assert tok in html
    assert 'class="card"' in html

    # every href resolves to a real file (relative to qc/)
    import re
    links = set(re.findall(r'href="([^"]+)"', html))
    assert links, "no links emitted"
    for L in links:
        assert (qc / L).exists(), f"broken link: {L}"

    # dwi dashboard + montage linked; per-session reports linked
    assert "reports/dwi/summary.html" in html
    assert "slicesdir/dwi/fa/slicesdir/index.html" in html
    assert "subjects/sub-1A/ses-1/dwi/sub-1A_ses-1_eddy_qc.html" in html

    # flagged session marked (warn glyph + tooltip)
    assert "⚠" in html and "mean_fd high" in html


def test_montage_galleries_found_at_any_depth(tmp_path):
    # A gallery must be listed wherever the arm that wrote it put it. The anat
    # arm nests one level below slicesdir/, the modality arms two; a fixed
    # two-level glob silently drops the anat galleries.
    qc = tmp_path / "qc"
    _make_qc_tree(qc)
    _touch(qc / "slicesdir/templates/slicesdir/index.html")          # one deep
    _touch(qc / "slicesdir/msme/t2/slicesdir/index.html")            # two deep
    _touch(qc / "slicesdir/a/b/c/slicesdir/index.html")              # three deep
    _touch(qc / "slicesdir/dwi/fa/slicesdir/notes.html")             # not an index
    _touch(qc / "slicesdir/stray/index.html")                        # not a gallery

    html = generate_qc_index(qc, study_name="teststudy").read_text()

    for path, label in [
        ("slicesdir/brain_extraction/slicesdir/index.html", "brain_extraction"),
        ("slicesdir/templates/slicesdir/index.html", "templates"),
        ("slicesdir/dwi/fa/slicesdir/index.html", "dwi / fa"),
        ("slicesdir/msme/t2/slicesdir/index.html", "msme / t2"),
        ("slicesdir/a/b/c/slicesdir/index.html", "a / b / c"),
    ]:
        assert path in html, f"gallery not listed: {path}"
        assert f">{label}</a>" in html, f"gallery mislabelled: {label}"

    # an index.html that isn't a slicesdir gallery index is not a montage
    assert "slicesdir/stray/index.html" not in html


def test_incremental_regeneration(tmp_path):
    # index reflects only what's present, and picks up a modality added later
    qc = tmp_path / "qc"
    _touch(qc / "subjects/sub-1A/ses-1/anat/sub-1A_ses-1_anat_qc.html")
    html1 = generate_qc_index(qc).read_text()
    assert "anat_qc.html" in html1 and "msme_qc.html" not in html1

    _touch(qc / "subjects/sub-1A/ses-1/msme/sub-1A_ses-1_msme_qc.html")
    html2 = generate_qc_index(qc).read_text()   # idempotent re-run
    assert "msme_qc.html" in html2  # newly added modality now appears


def test_missing_qc_dir_returns_none(tmp_path):
    assert generate_qc_index(tmp_path / "nope") is None
