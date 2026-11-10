"""StudyDesign reads who and what each session is, and refuses to guess."""
import numpy as np
import pandas as pd
import pytest

from neurofaune.study_design import load_study_design

PARTICIPANTS = (
    "participant_id\tgroup\tbatch\tsex\tbrain_weight_g\n"
    "sub-1C\tcontrol\tC\tM\t2.3\n"
    "sub-2C\tcontrol\tC\tF\t2.1\n"
    "sub-1X\tcuprizone\tX\tmale\t2.2\n"
    "sub-2X\tn/a\tX\tM\t2.2\n"
)
SESSIONS = (
    "participant_id\tsession_id\tage_days\ttimepoint\n"
    "sub-1C\tses-1\t70\tp60\n"
    "sub-1C\tses-2\t100\tp90\n"
    "sub-2C\tses-1\t71\tp60\n"
    "sub-1X\tses-1\t73\tp60\n"
    "sub-2X\tses-1\t72\tp60\n"
)


def _study(tmp_path, participants=PARTICIPANTS, sessions=SESSIONS, **design):
    bids = tmp_path / "bids"
    bids.mkdir(exist_ok=True)
    (bids / "participants.tsv").write_text(participants)
    (tmp_path / "design").mkdir(exist_ok=True)
    (tmp_path / "design" / "sessions.tsv").write_text(sessions)
    return {
        "paths": {"study_root": str(tmp_path), "bids": str(bids)},
        "design": design,
    }


def test_table_has_the_fixed_fields_and_keeps_study_columns(tmp_path):
    table = load_study_design(_study(tmp_path)).table()
    assert list(table.columns[:6]) == ["animal", "session", "group", "sex", "age_days", "template"]
    assert {"batch", "timepoint", "brain_weight_g"} <= set(table.columns)
    assert "cohort" not in table.columns


def test_a_study_may_have_its_own_cohort_column_and_nothing_reads_it(tmp_path):
    sessions = SESSIONS.replace("timepoint", "cohort")
    design = load_study_design(_study(tmp_path, sessions=sessions, template="adult"))
    assert design.template_of("sub-1C", "ses-2") == "adult"
    assert len(design.where(cohort="p60")) == 3


def test_blank_group_is_a_listed_exclusion(tmp_path):
    design = load_study_design(_study(tmp_path))
    assert design.excluded == ["sub-2X"]
    assert "sub-2X" not in set(design.table()["animal"])


def test_animal_missing_from_participants_is_an_error(tmp_path):
    sessions = SESSIONS + "sub-9Z\tses-1\t70\tp60\n"
    with pytest.raises(ValueError, match=r"no row in participants: \['sub-9Z'\]"):
        load_study_design(_study(tmp_path, sessions=sessions))


def test_duplicates_are_refused(tmp_path):
    with pytest.raises(ValueError, match="duplicate animals"):
        load_study_design(
            _study(tmp_path, participants=PARTICIPANTS + "sub-1C\tcontrol\tC\tM\t2.3\n")
        )
    with pytest.raises(ValueError, match="duplicate session"):
        load_study_design(_study(tmp_path, sessions=SESSIONS + "sub-1C\tses-1\t70\tp60\n"))


def test_levels_are_control_first(tmp_path):
    design = load_study_design(_study(tmp_path, control="control"))
    assert design.levels == ["control", "cuprizone"]
    design = load_study_design(_study(tmp_path, levels=["cuprizone", "control"], control="control"))
    assert design.levels == ["control", "cuprizone"]


def test_group_value_outside_declared_levels_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"\['cuprizone'\] are not in design.levels"):
        load_study_design(_study(tmp_path, levels=["control"]))


def test_control_must_be_a_level(tmp_path):
    with pytest.raises(ValueError, match="design.control"):
        load_study_design(_study(tmp_path, control="vehicle"))


def test_group_column_is_named_by_the_study(tmp_path):
    participants = PARTICIPANTS.replace("\tgroup\t", "\tdose\t")
    design = load_study_design(_study(tmp_path, participants=participants, group_column="dose"))
    assert set(design.table()["group"]) == {"control", "cuprizone"}


def test_second_meaning_of_group_is_refused(tmp_path):
    participants = "participant_id\tgroup\tdose\nsub-1C\ta\tC\nsub-2C\ta\tC\nsub-1X\ta\tX\nsub-2X\ta\tX\n"
    with pytest.raises(ValueError, match="one meaning"):
        load_study_design(_study(tmp_path, participants=participants, group_column="dose"))


@pytest.mark.parametrize("code", ["male", "Male", "M", "m"])
def test_spellings_of_male_are_not_coded_female(tmp_path, code):
    participants = f"participant_id\tgroup\tsex\nsub-1C\tc\t{code}\nsub-2C\tc\tF\nsub-1X\tc\tF\nsub-2X\tc\tF\n"
    table = load_study_design(_study(tmp_path, participants=participants)).table()
    assert table.set_index("animal").loc["sub-1C", "sex"].iloc[0] == "M"


def test_unknown_sex_code_is_refused_until_declared(tmp_path):
    participants = "participant_id\tgroup\tsex\nsub-1C\tc\t1\nsub-2C\tc\t2\nsub-1X\tc\t1\nsub-2X\tc\t2\n"
    with pytest.raises(ValueError, match="design.sex_codes"):
        load_study_design(_study(tmp_path, participants=participants))
    design = load_study_design(
        _study(tmp_path, participants=participants, sex_codes={1: "M", 2: "F"})
    )
    assert design.table().set_index(["animal", "session"]).loc[("sub-2C", "ses-1"), "sex"] == "F"


def test_blank_sex_stays_blank(tmp_path):
    participants = "participant_id\tgroup\tsex\nsub-1C\tc\tn/a\nsub-2C\tc\tF\nsub-1X\tc\tF\nsub-2X\tc\tF\n"
    table = load_study_design(_study(tmp_path, participants=participants)).table()
    assert table.loc[table["animal"] == "sub-1C", "sex"].isna().all()


def test_no_template_source_raises_rather_than_using_the_label(tmp_path):
    design = load_study_design(_study(tmp_path))
    with pytest.raises(ValueError, match="design.template"):
        design.template_of("sub-1C", "ses-1")


def test_template_sources(tmp_path):
    one = load_study_design(_study(tmp_path, template="adult"))
    assert {one.template_of("sub-1C", s) for s in ("ses-1", "ses-2")} == {"adult"}
    by_column = load_study_design(_study(tmp_path, template_column="timepoint"))
    assert by_column.template_of("sub-1C", "ses-2") == "p90"
    by_label = load_study_design(_study(tmp_path, template_column="session_label"))
    assert by_label.template_of("sub-1C", "ses-2") == "2"
    with pytest.raises(ValueError, match="not both"):
        load_study_design(_study(tmp_path, template="adult", template_column="timepoint"))
    with pytest.raises(KeyError, match="not a column"):
        load_study_design(_study(tmp_path, template_column="nope"))


def test_age_is_read_and_never_guessed(tmp_path):
    design = load_study_design(_study(tmp_path))
    assert design.age_of("sub-1C", "ses-2") == 100.0
    with pytest.raises(KeyError, match="not in the study design"):
        design.age_of("sub-1C", "ses-9")

    sessions = SESSIONS.replace("sub-1C\tses-2\t100", "sub-1C\tses-2\tn/a")
    with pytest.raises(ValueError, match="no age_days"):
        load_study_design(_study(tmp_path, sessions=sessions)).age_of("sub-1C", "ses-2")
    bad = SESSIONS.replace("\t100\t", "\tp100\t")
    with pytest.raises(ValueError, match="numeric"):
        load_study_design(_study(tmp_path, sessions=bad))


def test_where_filters_and_refuses_an_empty_result(tmp_path):
    design = load_study_design(_study(tmp_path))
    assert len(design.where(timepoint="p60")) == 3
    assert len(design.where(timepoint=["p60", "p90"], group="control")) == 3
    with pytest.raises(ValueError, match="no session matches"):
        design.where(timepoint="p30")
    with pytest.raises(KeyError, match="no column"):
        design.where(nope="x")


def test_groups_for_returns_the_animal_of_each_row(tmp_path):
    design = load_study_design(_study(tmp_path))
    rows = pd.DataFrame({"subject": ["sub-1C", "sub-1C", "sub-1X"], "session": ["ses-1", "ses-2", "ses-1"]})
    assert np.array_equal(design.groups_for(rows), ["sub-1C", "sub-1C", "sub-1X"])
    with pytest.raises(ValueError, match="not in the study design"):
        design.groups_for(pd.DataFrame({"subject": ["sub-9Z"]}))


def _write_bids_sessions(bids, animal, text):
    (bids / animal).mkdir()
    (bids / animal / f"{animal}_sessions.tsv").write_text(text)


def test_bids_session_files_are_used_when_there_is_no_design_table(tmp_path):
    config = _study(tmp_path)
    (tmp_path / "design" / "sessions.tsv").unlink()
    bids = tmp_path / "bids"
    (bids / "participants.tsv").write_text(
        "participant_id\tgroup\tbatch\nsub-1C\tcontrol\tC\nsub-1X\tcuprizone\tX\n"
    )
    _write_bids_sessions(bids, "sub-1C", "session_id\tage_days\tgroup\tbatch\nses-1\t70\tcontrol\tC\n")
    _write_bids_sessions(bids, "sub-1X", "session_id\tage_days\tgroup\tbatch\nses-1\t73\tcuprizone\tX\n")

    table = load_study_design(config).table()

    assert sorted(table["animal"]) == ["sub-1C", "sub-1X"]
    assert list(table.columns).count("group") == 1


def test_design_table_is_used_ahead_of_bids_files(tmp_path):
    config = _study(tmp_path)
    _write_bids_sessions(tmp_path / "bids", "sub-1C", "session_id\tage_days\nses-7\t1\n")
    assert "ses-7" not in set(load_study_design(config).table()["session"])


def test_tables_that_disagree_are_refused(tmp_path):
    sessions = (
        "participant_id\tsession_id\tgroup\n"
        "sub-1C\tses-1\tcuprizone\n"
        "sub-1X\tses-1\tcuprizone\n"
    )
    with pytest.raises(ValueError, match=r"'group' disagrees .* sub-1C ses-1"):
        load_study_design(_study(tmp_path, sessions=sessions))


def test_missing_sources_say_what_to_set(tmp_path):
    config = _study(tmp_path)
    (tmp_path / "design" / "sessions.tsv").unlink()
    with pytest.raises(FileNotFoundError, match="design.sessions"):
        load_study_design(config)
    (tmp_path / "bids" / "participants.tsv").unlink()
    with pytest.raises(FileNotFoundError, match="design.participants"):
        load_study_design(config)
    config["design"]["sessions"] = str(tmp_path / "nope.tsv")
    with pytest.raises(FileNotFoundError, match="nope.tsv"):
        load_study_design(config)


def test_ids_need_no_prefix(tmp_path):
    participants = "participant_id\tgroup\n1C\tc\n"
    sessions = "participant_id\tsession_id\n1C\t1\n"
    design = load_study_design(_study(tmp_path, participants=participants, sessions=sessions, template="t"))
    assert design.template_of("sub-1C", "ses-1") == "t"
