#!/usr/bin/env python3
"""
Unit tests for the morphometry -> CovNet bridge.

The reshaping is bookkeeping; the test that matters is head-size removal. Regional
volumes all scale with the animal, so a covariance matrix built from raw volumes
measures how big the rats are. These tests build data where that is the only signal
and check that each normalisation does or does not remove it.
"""

import numpy as np
import pandas as pd
import pytest

from neurofaune.network.structural_covariance import (
    GLOBAL_STRUCTURES,
    TBV_COLUMN,
    build_covnet_table,
    region_nodes,
    residualise,
    normalise_nodes,
    split_session_stem,
    structure_nodes,
    thickness_nodes,
    total_brain_volume,
)

SESSIONS = [f"sub-R{i:02d}_ses-p60" for i in range(1, 21)]


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def size_driven(rng):
    """Nodes that share ONE global size factor and nothing else.

    Each node is ``k * TBV + independent noise``, so the only thing two nodes have
    in common is the animal's head size. Raw correlations must be high; after
    removing TBV they must collapse.
    """
    tbv = rng.normal(1800.0, 220.0, len(SESSIONS))
    frame = {"subject": SESSIONS, TBV_COLUMN: tbv}
    for i, k in enumerate([0.30, 0.20, 0.12, 0.08]):
        frame[f"node{i}"] = k * tbv + rng.normal(0, 0.02 * k * tbv.mean(), len(SESSIONS))
    return pd.DataFrame(frame)


def mean_offdiag_corr(df, cols):
    corr = df[list(cols)].corr(method="spearman").to_numpy()
    return float(np.abs(corr[np.triu_indices(len(cols), k=1)]).mean())


# --- reshaping ---------------------------------------------------------------

def test_split_session_stem_separates_subject_and_session():
    df = pd.DataFrame({"subject": ["sub-Rat84_ses-p60"], "volume_mm3": [1.0]})
    out = split_session_stem(df)
    assert out.loc[0, "subject"] == "sub-Rat84"
    assert out.loc[0, "session"] == "ses-p60"


def test_split_session_stem_is_idempotent():
    df = pd.DataFrame({"subject": ["sub-Rat84"], "session": ["ses-p60"], "v": [1.0]})
    out = split_session_stem(df)
    assert out.loc[0, "subject"] == "sub-Rat84"
    assert out.loc[0, "session"] == "ses-p60"


def test_split_session_stem_raises_rather_than_dropping():
    """A silently dropped session is indistinguishable from an excluded one."""
    df = pd.DataFrame({"subject": ["sub-Rat84_ses-p60", "Rat85"], "v": [1.0, 2.0]})
    with pytest.raises(ValueError, match="not 'sub-X_ses-Y'"):
        split_session_stem(df)


def test_region_nodes_pivots_to_one_column_per_region():
    long = pd.DataFrame({
        "subject": ["sub-A_ses-p60", "sub-A_ses-p60", "sub-B_ses-p60", "sub-B_ses-p60"],
        "region": ["Motor_Cortex_L", "Motor_Cortex_R"] * 2,
        "volume_GM_mm3": [10.0, 11.0, 12.0, 13.0],
    })
    wide, cols = region_nodes(long)
    assert cols == ["Motor_Cortex_L", "Motor_Cortex_R"]
    assert len(wide) == 2
    assert wide.loc[wide["subject"] == "sub-A", "Motor_Cortex_R"].iloc[0] == 11.0


def test_region_nodes_rejects_a_measure_that_is_not_there():
    long = pd.DataFrame({"subject": ["sub-A_ses-p60"], "region": ["X"], "volume_mm3": [1.0]})
    with pytest.raises(KeyError, match="volume_GM_mm3"):
        region_nodes(long, measure="volume_GM_mm3")


def test_structure_nodes_name_carries_the_tissue():
    long = pd.DataFrame({
        "subject": ["sub-A_ses-p60"] * 3,
        "structure": ["cerebellum", "cerebellum", "thalamus"],
        "tissue": ["GM", "WM", "any"],
        "volume_mm3": [1.0, 2.0, 3.0],
    })
    _, cols = structure_nodes(long)
    assert cols == ["cerebellum_GM", "cerebellum_WM", "thalamus_any"]


def test_structure_nodes_drops_whole_brain_summaries():
    """total_brain is the denominator; correlating it with its own parts is circular."""
    rows = [{"subject": "sub-A_ses-p60", "structure": s, "tissue": "any", "volume_mm3": 1.0}
            for s in (*GLOBAL_STRUCTURES, "hippocampus")]
    _, cols = structure_nodes(pd.DataFrame(rows))
    assert cols == ["hippocampus_any"]


def test_thickness_nodes_use_the_mean_thickness(caplog):
    long = pd.DataFrame({
        "subject": ["sub-A_ses-p60"],
        "region": ["Motor_Cortex_L"],
        "mean_thickness_mm": [1.4],
    })
    with caplog.at_level("WARNING"):
        _, cols = thickness_nodes(long)
    assert cols == ["Motor_Cortex_L"]
    assert "exploratory" in caplog.text.lower()


def test_total_brain_volume_selects_the_unweighted_total():
    long = pd.DataFrame({
        "subject": ["sub-A_ses-p60"] * 3,
        "structure": ["total_brain", "total_brain", "cerebellum"],
        "tissue": ["any", "GM", "any"],
        "volume_mm3": [1800.0, 900.0, 200.0],
    })
    out = total_brain_volume(long)
    assert out.loc[0, TBV_COLUMN] == 1800.0


def test_total_brain_volume_raises_when_absent():
    long = pd.DataFrame({"subject": ["sub-A_ses-p60"], "structure": ["cerebellum"],
                         "tissue": ["any"], "volume_mm3": [200.0]})
    with pytest.raises(ValueError, match="cannot normalise for head size"):
        total_brain_volume(long)


# --- head-size removal (the part that decides whether the network means anything)

def test_raw_volumes_are_dominated_by_a_global_size_factor(size_driven):
    """The premise: without correction every edge is the same size factor."""
    nodes = [c for c in size_driven.columns if c.startswith("node")]
    assert mean_offdiag_corr(size_driven, nodes) > 0.9


def test_residualising_on_tbv_removes_the_size_factor(size_driven):
    nodes = [c for c in size_driven.columns if c.startswith("node")]
    out = residualise(size_driven, nodes, confounds=[TBV_COLUMN])
    assert mean_offdiag_corr(out, nodes) < 0.4


def test_proportion_normalisation_also_removes_it(size_driven):
    nodes = [c for c in size_driven.columns if c.startswith("node")]
    out = normalise_nodes(size_driven, nodes, method="proportion")
    assert mean_offdiag_corr(out, nodes) < 0.4


def test_normalise_none_leaves_the_size_factor_in_place(size_driven, caplog):
    nodes = [c for c in size_driven.columns if c.startswith("node")]
    with caplog.at_level("WARNING"):
        out = normalise_nodes(size_driven, nodes, method="none")
    assert mean_offdiag_corr(out, nodes) > 0.9
    assert "uniformly positive" in caplog.text


def test_residualising_preserves_a_real_association(rng):
    """Removing head size must not remove signal that is independent of it."""
    tbv = rng.normal(1800.0, 220.0, 60)
    shared = rng.normal(0, 1, 60)
    df = pd.DataFrame({
        TBV_COLUMN: tbv,
        "a": 0.3 * tbv + 8.0 * shared + rng.normal(0, 1, 60),
        "b": 0.2 * tbv + 8.0 * shared + rng.normal(0, 1, 60),
    })
    out = residualise(df, ["a", "b"], confounds=[TBV_COLUMN])
    assert out[["a", "b"]].corr(method="spearman").iloc[0, 1] > 0.8


def test_residualising_keeps_the_original_units(size_driven):
    """The column mean is added back, so residuals still read as volumes."""
    nodes = [c for c in size_driven.columns if c.startswith("node")]
    out = residualise(size_driven, nodes, confounds=[TBV_COLUMN])
    for col in nodes:
        assert out[col].mean() == pytest.approx(size_driven[col].mean(), rel=1e-9)


def test_residualise_handles_categorical_confounds(rng):
    df = pd.DataFrame({
        TBV_COLUMN: rng.normal(1800, 200, 40),
        "sex": ["F", "M"] * 20,
        "cohort": ["p60"] * 40,          # constant — must be dropped, not crash
        "a": rng.normal(100, 10, 40),
    })
    out = residualise(df, ["a"], confounds=[TBV_COLUMN, "sex", "cohort"])
    assert out["a"].notna().all()


def test_residualise_marks_rows_with_a_missing_confound(rng):
    df = pd.DataFrame({
        TBV_COLUMN: [1800.0, 1900.0, np.nan] + list(rng.normal(1800, 200, 17)),
        "a": list(rng.normal(100, 10, 20)),
    })
    out = residualise(df, ["a"], confounds=[TBV_COLUMN])
    assert np.isnan(out.loc[2, "a"])
    assert out["a"].notna().sum() == 19


def test_residualise_rejects_an_unknown_confound(size_driven):
    with pytest.raises(KeyError, match="not available"):
        residualise(size_driven, ["node0"], confounds=["weight_g"])


def test_normalise_rejects_an_unknown_method(size_driven):
    with pytest.raises(ValueError, match="unknown normalisation"):
        normalise_nodes(size_driven, ["node0"], method="zscore")


# --- assembly ----------------------------------------------------------------

def test_build_covnet_table_produces_the_layout_prepare_expects():
    nodes = pd.DataFrame({
        "subject": ["sub-A", "sub-B"],
        "session": ["ses-p60", "ses-p90"],
        "hippocampus_any": [50.0, 55.0],
        "cerebellum_GM": [120.0, 130.0],
    })
    structures = pd.DataFrame({
        "subject": ["sub-A_ses-p60", "sub-B_ses-p90"],
        "structure": ["total_brain"] * 2,
        "tissue": ["any"] * 2,
        "volume_mm3": [1800.0, 1900.0],
    })
    pheno = pd.DataFrame({"subject": ["sub-A", "sub-B"], "dose": ["C", "L"], "sex": ["F", "M"]})

    df, cols = build_covnet_table(
        nodes, ["hippocampus_any", "cerebellum_GM"],
        structures=structures, phenotype=pheno, method="proportion",
    )
    assert list(df.columns[:6]) == ["subject", "session", "cohort", "dose", "sex", TBV_COLUMN]
    assert cols == ["hippocampus_any", "cerebellum_GM"]
    assert df.loc[0, "cohort"] == "p60"
    assert df.loc[0, "hippocampus_any"] == pytest.approx(50.0 / 1800.0)


def test_build_covnet_table_averages_bilateral_pairs():
    nodes = pd.DataFrame({
        "subject": ["sub-A"], "session": ["ses-p60"],
        "Motor_Cortex_L": [10.0], "Motor_Cortex_R": [12.0],
    })
    df, cols = build_covnet_table(
        nodes, ["Motor_Cortex_L", "Motor_Cortex_R"], method="none", bilateral=True,
    )
    assert cols == ["Motor_Cortex"]
    assert df.loc[0, "Motor_Cortex"] == 11.0


def test_total_brain_volume_is_not_correlated_as_a_node(tmp_path):
    """The denominator ships with the table for provenance; it must not be a node."""
    from neurofaune.network.matrices import load_and_prepare_data

    df = pd.DataFrame({
        "subject": ["sub-A", "sub-B", "sub-C", "sub-D"],
        "session": ["ses-p60"] * 4,
        "dose": ["C", "C", "L", "L"],
        "sex": ["F", "M", "F", "M"],
        TBV_COLUMN: [1800.0, 1900.0, 1750.0, 1850.0],
        "hippocampus_any": [50.0, 55.0, 48.0, 53.0],
        "cerebellum_GM": [120.0, 130.0, 118.0, 128.0],
    })
    csv = tmp_path / "roi_structures_volume_wide.csv"
    df.to_csv(csv, index=False)

    _, roi_cols = load_and_prepare_data(csv)
    assert TBV_COLUMN not in roi_cols
    assert sorted(roi_cols) == ["cerebellum_GM", "hippocampus_any"]


# --- phenotype ingestion -----------------------------------------------------

def test_load_participants_reads_a_bids_sidecar(tmp_path):
    tsv = tmp_path / "participants.tsv"
    tsv.write_text(
        "participant_id\tgroup\tsex\tbatch\n"
        "sub-10C\tcontrol\tM\tC\n"
        "sub-1X\tcuprizone\tM\tX\n"
    )
    from neurofaune.network.structural_covariance import load_participants

    out = load_participants(tsv)
    assert list(out.columns) == ["subject", "dose", "sex"]
    assert out.loc[0, "subject"] == "sub-10C"
    assert sorted(out["dose"]) == ["control", "cuprizone"]


def test_load_participants_adds_the_sub_prefix(tmp_path):
    csv = tmp_path / "participants.csv"
    csv.write_text("participant_id,group\n10C,control\n")
    from neurofaune.network.structural_covariance import load_participants

    assert load_participants(csv).loc[0, "subject"] == "sub-10C"


def test_load_participants_reports_a_wrong_group_column(tmp_path):
    tsv = tmp_path / "participants.tsv"
    tsv.write_text("participant_id\tarm\nsub-1\tcontrol\n")
    from neurofaune.network.structural_covariance import load_participants

    with pytest.raises(KeyError, match="group"):
        load_participants(tsv)
    assert load_participants(tsv, group_col="arm").loc[0, "dose"] == "control"


def test_confound_error_names_the_way_to_supply_it(size_driven):
    with pytest.raises(KeyError, match="--participants"):
        residualise(size_driven, ["node0"], confounds=[TBV_COLUMN, "sex"])


# --- overlap pruning ---------------------------------------------------------

LABEL_SETS = {
    "hippocampus": [71, 72],
    "amygdala": [31, 32],
    "subcortical": [31, 32, 71, 72, 90],       # contains both of the above
    "corpus_callosum": [891],
    "fiber_tracts": [891, 1121],               # contains corpus_callosum
    "cerebellum_gm": [200, 201],
    "cerebellum_wm": [200, 201],               # same labels, other tissue
    "cerebellum": [200, 201],
    "brainstem": [300],
}


def test_pruning_drops_a_structure_that_contains_another():
    from neurofaune.network.structural_covariance import prune_overlapping_nodes

    kept, dropped = prune_overlapping_nodes(
        ["subcortical_any", "hippocampus_any", "amygdala_any"], LABEL_SETS,
    )
    assert kept == ["hippocampus_any", "amygdala_any"]
    assert "label set contains" in dropped["subcortical_any"]


def test_pruning_drops_the_sum_when_a_tissue_split_survives():
    from neurofaune.network.structural_covariance import prune_overlapping_nodes

    kept, dropped = prune_overlapping_nodes(
        ["cerebellum_any", "cerebellum_gm_GM", "cerebellum_wm_WM"], LABEL_SETS,
    )
    assert kept == ["cerebellum_gm_GM", "cerebellum_wm_WM"]
    assert dropped["cerebellum_any"].startswith("sum of")


def test_pruning_keeps_the_total_for_a_discrete_structure():
    """aseg convention: nuclei get a total, not a tissue-weighted volume."""
    from neurofaune.network.structural_covariance import prune_overlapping_nodes

    kept, dropped = prune_overlapping_nodes(
        ["hippocampus_GM", "hippocampus_any"], LABEL_SETS,
    )
    assert kept == ["hippocampus_any"]
    assert dropped["hippocampus_GM"] == "same label set as hippocampus_any"


def test_pruning_leaves_disjoint_nodes_alone():
    from neurofaune.network.structural_covariance import prune_overlapping_nodes

    nodes = ["hippocampus_any", "amygdala_any", "brainstem_any"]
    kept, dropped = prune_overlapping_nodes(nodes, LABEL_SETS)
    assert kept == nodes and dropped == {}


def test_pruning_ignores_nodes_it_has_no_label_set_for():
    from neurofaune.network.structural_covariance import prune_overlapping_nodes

    kept, _ = prune_overlapping_nodes(["mystery_node", "brainstem_any"], LABEL_SETS)
    assert kept == ["mystery_node", "brainstem_any"]


def test_structure_nodes_prunes_unless_nesting_is_allowed():
    long = pd.DataFrame([
        {"subject": "sub-A_ses-1", "structure": s, "tissue": t, "volume_mm3": v}
        for s, t, v in [("subcortical", "any", 100.0), ("hippocampus", "any", 40.0),
                        ("amygdala", "any", 20.0)]
    ])
    _, pruned = structure_nodes(long, structure_labels=LABEL_SETS)
    assert pruned == ["amygdala_any", "hippocampus_any"]  # _pivot sorts columns

    _, unpruned = structure_nodes(long, structure_labels=LABEL_SETS, allow_nested=True)
    assert "subcortical_any" in unpruned
