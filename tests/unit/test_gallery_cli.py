"""`neurofaune gallery build`: a study's Lumière gallery from its gallery.yaml."""
from click.testing import CliRunner

from neurofaune.preprocess.cli import main


def test_gallery_build_writes_the_study_gallery(tmp_path):
    (tmp_path / "study.toml").write_text('[study]\nname = "demo"\n')
    (tmp_path / "results").mkdir()
    (tmp_path / "gallery.yaml").write_text("name: Demo\npaths:\n  results: ./results\n  gallery: ./gallery\n")
    out = CliRunner().invoke(main, ["gallery", "build", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert out.output.strip().endswith("gallery/index.html")
    assert (tmp_path / "gallery" / ".lumiere-gallery").is_file()
    assert "Lumière - Demo" in (tmp_path / "gallery" / "index.html").read_text(encoding="utf-8")


def test_gallery_build_says_what_is_missing(tmp_path):
    out = CliRunner().invoke(main, ["gallery", "build", str(tmp_path)])
    assert out.exit_code != 0 and "no gallery.yaml or gallery.json" in out.output


def test_gallery_build_never_replaces_another_tools_folder(tmp_path):
    (tmp_path / "gallery.yaml").write_text("name: Demo\n")
    (tmp_path / "gallery").mkdir()
    (tmp_path / "gallery" / "index.html").write_text("someone else's")
    out = CliRunner().invoke(main, ["gallery", "build", str(tmp_path)])
    assert out.exit_code != 0 and "not built by this shell" in out.output
    assert (tmp_path / "gallery" / "index.html").read_text() == "someone else's"
