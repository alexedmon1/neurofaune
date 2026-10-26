"""Unit tests for species profiles (configs/species/*.yaml, neurofaune.species)."""

import pytest
import yaml

from neurofaune.config import get_config_value, load_config, load_yaml, merge_configs
from neurofaune.species import (
    DEFAULT_SPECIES,
    SPECIES_DIR,
    UnknownSpeciesError,
    apply_species_profile,
    available_species,
    load_species_profile,
    species_of,
)

DEFAULTS = SPECIES_DIR.parent / "default.yaml"


def write_study(tmp_path, **study):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"study": {"name": "t", "code": "T", **study}}))
    return path


def test_rat_and_mouse_profiles_exist():
    assert {"rat", "mouse"} <= set(available_species())


def test_species_defaults_to_rat():
    assert DEFAULT_SPECIES == "rat"
    assert species_of({}) == "rat"
    assert species_of({"study": {"species": "  Mouse "}}) == "mouse"


def test_rat_profile_is_empty_so_rat_studies_are_unchanged(tmp_path):
    # The defaults ARE the rat values; selecting rat must not move anything.
    assert load_species_profile("rat") == {}
    defaults = load_yaml(DEFAULTS)
    for study in ({}, {"study": {"species": "rat"}}, {"bids": {"voxel_scale": 10.0}}):
        assert apply_species_profile(defaults, study) == merge_configs(defaults, study)
    cfg = load_config(write_study(tmp_path), validate=False)
    assert cfg["atlas"]["name"] == "SIGMA"
    assert "bids" not in cfg


def test_unknown_species_is_an_error_not_a_silent_rat_fallback(tmp_path):
    with pytest.raises(UnknownSpeciesError, match="Available"):
        load_config(write_study(tmp_path, species="hamster"), validate=False)


def test_mouse_profile_overrides_size_dependent_values(tmp_path):
    cfg = load_config(write_study(tmp_path, species="mouse"), validate=False)
    assert get_config_value(cfg, "study.species") == "mouse"
    assert get_config_value(cfg, "bids.voxel_scale") == 10.0
    assert get_config_value(cfg, "bids.affine") == "scanner"
    assert get_config_value(cfg, "atlas.name") == "Antwerp"
    assert get_config_value(cfg, "anatomical.skull_strip.refine.qc.volume_mm3") == [400, 650]
    assert get_config_value(cfg, "anatomical.skull_strip.bet4animal.radii") == [60]
    # Untouched keys still come from the defaults.
    assert get_config_value(cfg, "anatomical.skull_strip.n_classes") == 5


def test_mouse_atlas_study_space_does_not_inherit_sigma_file_names(tmp_path):
    cfg = load_config(write_study(tmp_path, species="mouse"), validate=False)
    space = cfg["atlas"]["study_space"]
    assert not any("SIGMA" in str(v) for v in space.values())
    assert space["template"].endswith("Antwerp/scaled10/template_T2w.nii.gz")
    assert space["parcellation_eeg"].endswith("allen26_labels.nii.gz")
    # tissue priors are not built for the mouse hub: they stay null, not SIGMA's
    assert space["gm_prob"] is None


def test_study_config_overrides_species_profile(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "study": {"name": "t", "code": "T", "species": "mouse"},
        "bids": {"voxel_scale": 20.0},
    }))
    cfg = load_config(path, validate=False)
    assert cfg["bids"]["voxel_scale"] == 20.0
    assert cfg["bids"]["affine"] == "scanner"     # profile value survives


def test_apply_species_profile_without_defaults():
    # `neurofaune bids` reads the study YAML directly and only needs the
    # profile's bids block under the study's.
    merged = apply_species_profile({}, {"study": {"species": "mouse"},
                                        "bids": {"raw_root": "/r", "bids_root": "/b"}})
    assert merged["bids"] == {"voxel_scale": 10.0, "affine": "scanner",
                              "raw_root": "/r", "bids_root": "/b"}


def test_mouse_profile_anat_preprocessing_settings(tmp_path):
    """Values the mouse anat path needs (validated on the FAC600 pilot 3D RARE)."""
    cfg = load_config(write_study(tmp_path, species="mouse"), validate=False)
    assert get_config_value(cfg, "anatomical.skull_strip.method") == "bet4animal"
    assert get_config_value(cfg, "anatomical.skull_strip.bet4animal.long_axis") == "auto"
    assert get_config_value(cfg, "anatomical.input_already_scaled") is True
    assert get_config_value(cfg, "anatomical.resample_3d_to_2d") is False
    assert get_config_value(cfg, "anatomical.n4.bspline_fitting_distance") == 100
    assert get_config_value(cfg, "anatomical.qc.brain_to_total_range") == [0.05, 0.30]
    assert get_config_value(cfg, "atlas.study_space.brain_extent").endswith("Antwerp/scaled10/brain_mask.nii.gz")
    # the mouse profile no longer sets the dead ants.n4 key (nothing reads ants.*)
    from neurofaune.species import load_species_profile
    assert "ants" not in load_species_profile("mouse")


def test_rat_keeps_its_anat_behaviour(tmp_path):
    cfg = load_config(write_study(tmp_path, species="rat"), validate=False)
    assert get_config_value(cfg, "anatomical.skull_strip.method") == "atropos_bet"
    assert get_config_value(cfg, "anatomical.input_already_scaled", default=False) is False
    assert get_config_value(cfg, "anatomical.resample_3d_to_2d", default=True) is True
    assert get_config_value(cfg, "anatomical.n4.bspline_fitting_distance", default=None) is None
    assert get_config_value(cfg, "atlas.study_space.brain_extent", default=None) is None
