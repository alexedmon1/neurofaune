"""define_groups is study-agnostic: cells are named by the caller, never assumed.

It previously took grouping="full"|"pnd_dose"|"dose", which required
cohort/dose/sex columns and silently dropped any row whose cohort was not
p30/p60/p90. That encoded one study's design into shared code. The presets are
gone; callers name their factors.
"""
import pandas as pd
import pytest

from neurofaune.network.matrices import define_groups


@pytest.fixture
def dose_frame():
    """A PND x dose x sex design, with a cohort the old whitelist would have cut."""
    rows = []
    for coh in ["p30", "p60", "p90", "p120"]:
        for dose in ["control", "low", "mid", "high"]:
            for sex in ["M", "F"]:
                for _ in range(2):
                    rows.append(dict(subject=f"s{len(rows)}", session="ses-1",
                                     cohort=coh, dose=dose, sex=sex,
                                     roi1=float(len(rows))))
    return pd.DataFrame(rows)


@pytest.fixture
def longitudinal_frame():
    """A design with no dose and no sex -- unusable under the old presets."""
    return pd.DataFrame([
        dict(subject="sub-1C", session="ses-1", timepoint="p60", group="control"),
        dict(subject="sub-2C", session="ses-1", timepoint="p60", group="control"),
        dict(subject="sub-1X", session="ses-1", timepoint="p60", group="treated"),
        dict(subject="sub-1C", session="ses-3", timepoint="p120", group="control"),
        dict(subject="sub-1X", session="ses-3", timepoint="p120", group="treated"),
    ])


class TestFactorsDefineTheCells:
    @pytest.mark.parametrize("factors,n_expected", [
        (["dose"], 4),
        (["cohort", "dose"], 16),
        (["cohort", "dose", "sex"], 32),
        (["sex"], 2),
    ])
    def test_any_number_and_order_of_factors(self, dose_frame, factors, n_expected):
        got = define_groups(dose_frame, factors=factors)
        assert len(got) == n_expected
        assert sum(len(v) for v in got.values()) == len(dose_frame)   # nothing dropped

    def test_labels_join_factor_values_in_order(self, dose_frame):
        got = define_groups(dose_frame, factors=["sex", "dose", "cohort"])
        assert "M_control_p120" in got
        assert all(len(k.split("_")) == 3 for k in got)

    def test_single_factor_labels_are_bare_values(self, longitudinal_frame):
        assert sorted(define_groups(longitudinal_frame, factors=["group"])) == [
            "control", "treated"]

    def test_design_without_dose_or_sex(self, longitudinal_frame):
        got = define_groups(longitudinal_frame, factors=["timepoint", "group"])
        assert sorted(got) == ["p120_control", "p120_treated",
                               "p60_control", "p60_treated"]

    def test_group_contents_are_correct_not_just_counted(self, longitudinal_frame):
        got = define_groups(longitudinal_frame, factors=["timepoint", "group"])
        assert list(got["p60_control"].subject) == ["sub-1C", "sub-2C"]
        assert got["p60_control"].index.tolist() == [0, 1]            # index reset


class TestNoImplicitFiltering:
    def test_nothing_is_dropped_without_include(self, dose_frame):
        got = define_groups(dose_frame, factors=["cohort"])
        assert "p120" in got, "the old p30/p60/p90 whitelist must be gone"
        assert len(got) == 4

    def test_include_filters_explicitly(self, dose_frame):
        got = define_groups(dose_frame, factors=["dose"], include={"cohort": ["p60"]})
        assert len(got) == 4
        assert all(len(v) == 4 for v in got.values())

    def test_include_can_reproduce_the_old_whitelist(self, dose_frame):
        got = define_groups(dose_frame, factors=["cohort", "dose"],
                            include={"cohort": ["p30", "p60", "p90"]})
        assert len(got) == 12
        assert not any("p120" in k for k in got)


class TestErrors:
    def test_factors_is_required(self, dose_frame):
        with pytest.raises(TypeError):
            define_groups(dose_frame)

    def test_empty_factors_rejected(self, dose_frame):
        with pytest.raises(ValueError, match="requires factors"):
            define_groups(dose_frame, factors=[])

    def test_missing_column_names_itself(self, longitudinal_frame):
        with pytest.raises(ValueError, match="not in the DataFrame"):
            define_groups(longitudinal_frame, factors=["dose"])

    def test_include_column_must_exist(self, longitudinal_frame):
        with pytest.raises(ValueError, match="include"):
            define_groups(longitudinal_frame, factors=["group"],
                          include={"nope": ["x"]})

    def test_removed_grouping_kwarg_is_a_hard_error(self, dose_frame):
        """The presets are gone; calling the old way must fail loudly, not
        silently return something plausible."""
        with pytest.raises(TypeError):
            define_groups(dose_frame, grouping="pnd_dose")
