"""Study configs hold overrides only (generate_config, config slim).

init used to write a full copy of default.yaml into the study config. The study
config is the top layer, so every copied default hid the species profile's value
for that key: a mouse study resolved 21 of the mouse profile's 40 keys to rat
values (skull-strip method, brain-volume QC, smoothing, the whole atlas block).
"""

import pytest
import yaml
from click.testing import CliRunner

from neurofaune.config import (
    get_config_value,
    load_config,
    load_yaml,
    merge_configs,
    slim_config_file,
    slim_study_config,
    strip_defaults,
)
from neurofaune.preprocess.cli import main
from neurofaune.species import SPECIES_DIR, UnknownSpeciesError, load_species_profile
from neurofaune.study_initialization import generate_config

DEFAULTS = load_yaml(SPECIES_DIR.parent / "default.yaml")


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _generate(tmp_path, **kwargs):
    return generate_config(study_root=tmp_path, study_name="t", study_code="T",
                           output_path=tmp_path / "config.yaml", **kwargs)


def _legacy_config(tmp_path, species):
    """A config written the old way: all of default.yaml plus the study block."""
    study = {"study": {"name": "t", "code": "T", "species": species, "strain": "Wistar"},
             "paths": {"study_root": str(tmp_path)}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(merge_configs(DEFAULTS, study), sort_keys=False))
    return path


def test_generated_mouse_config_resolves_every_mouse_profile_value(tmp_path):
    cfg = load_config(_generate(tmp_path, species="mouse"), validate=False)
    profile = _flatten(load_species_profile("mouse"))
    profile.pop("study.strain")  # set below; not a parameter
    wrong = {k: (get_config_value(cfg, k), v) for k, v in profile.items()
             if not str(v).startswith("${") and get_config_value(cfg, k) != v}
    assert wrong == {}
    assert get_config_value(cfg, "study.strain") == "C57BL/6"  # the profile's, not 'Wistar'


def test_generated_config_holds_only_study_and_paths(tmp_path):
    written = load_yaml(_generate(tmp_path, species="mouse"))
    assert set(written) == {"study", "paths"}
    assert written["study"]["species"] == "mouse"
    assert "strain" not in written["study"]


def test_generated_rat_config_loads_the_defaults(tmp_path):
    cfg = load_config(_generate(tmp_path), validate=False)
    for key in ("anatomical.skull_strip.method", "functional.smoothing.fwhm", "atlas.name"):
        assert get_config_value(cfg, key) == get_config_value(DEFAULTS, key)


def test_generated_config_keeps_given_cohorts_and_atlas(tmp_path):
    cohorts = {"baseline": {"age_days": 70}}
    written = load_yaml(_generate(tmp_path, cohorts=cohorts, sigma_atlas_path=tmp_path / "SIGMA"))
    assert written["cohorts"] == cohorts
    assert written["atlas"]["base_path"] == str(tmp_path / "SIGMA")


def test_unknown_species_fails_at_generation(tmp_path):
    with pytest.raises(UnknownSpeciesError):
        _generate(tmp_path, species="hamster")
    assert not (tmp_path / "config.yaml").exists()


def test_strip_defaults_keeps_study_paths_and_real_overrides():
    study = {"study": {"name": "x", "species": "rat"},
             "paths": {"raw": DEFAULTS["paths"]["raw"]},           # equal, but paths are kept
             "functional": {"smoothing": {"fwhm": 4}},             # an override
             "anatomical": DEFAULTS["anatomical"]}                 # a copy
    slim, removed = strip_defaults(study, DEFAULTS)
    assert slim == {"study": study["study"], "paths": study["paths"],
                    "functional": {"smoothing": {"fwhm": 4}}}
    assert "anatomical.skull_strip.method" in removed


def test_slim_legacy_rat_config_changes_nothing(tmp_path):
    result = slim_study_config(load_yaml(_legacy_config(tmp_path, "rat")), DEFAULTS)
    assert result["removed"]
    assert result["changed"] == []


def test_slim_legacy_mouse_config_restores_the_profile(tmp_path):
    path = _legacy_config(tmp_path, "mouse")
    before = load_config(path, validate=False)
    assert get_config_value(before, "anatomical.skull_strip.method") == "atropos_bet"  # the bug

    result = slim_study_config(load_yaml(path), DEFAULTS)
    changed = {k: (a, b) for k, a, b in result["changed"]}
    assert changed["anatomical.skull_strip.method"] == ("atropos_bet", "bet4animal")
    assert changed["functional.smoothing.fwhm"] == (6, 3)
    assert changed["atlas.name"] == ("SIGMA", "Antwerp")


def test_slim_file_dry_run_writes_nothing(tmp_path):
    path = _legacy_config(tmp_path, "rat")
    original = path.read_bytes()
    result = slim_config_file(path)
    assert path.read_bytes() == original
    assert result["backup"] is None


def test_slim_file_write_keeps_a_backup_and_loads_identically(tmp_path):
    path = _legacy_config(tmp_path, "rat")
    original = path.read_bytes()
    before = load_config(path, validate=False)

    result = slim_config_file(path, write=True)
    assert result["backup"].read_bytes() == original
    assert set(load_yaml(path)) == {"study", "paths"}
    assert load_config(path, validate=False) == before

    second = slim_config_file(path, write=True)               # a second backup, not an overwrite
    assert second["backup"].name == "config.yaml.pre-slim.1"
    assert result["backup"].read_bytes() == original


def test_cli_config_slim(tmp_path):
    path = _legacy_config(tmp_path, "mouse")
    run = CliRunner().invoke(main, ["config", "slim", str(path)])
    assert run.exit_code == 0, run.output
    assert "anatomical.skull_strip.method: 'atropos_bet' -> 'bet4animal'" in run.output
    assert "Dry run" in run.output
    assert not (tmp_path / "config.yaml.pre-slim").exists()

    run = CliRunner().invoke(main, ["config", "slim", str(path), "--write"])
    assert run.exit_code == 0, run.output
    assert (tmp_path / "config.yaml.pre-slim").exists()
    cfg = load_config(path, validate=False)
    assert get_config_value(cfg, "anatomical.skull_strip.method") == "bet4animal"
