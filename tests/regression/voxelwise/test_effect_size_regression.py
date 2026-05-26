"""Gate-tier regression: voxel-based effect-size math (deterministic, hermetic).

Tests the numeric boundary of effect-size computation on a small synthetic
t-map. The full FSL-based VBM/randomise pipeline that *produces* real t-maps
is integration-tier (see tests/regression/integration/).
"""

from __future__ import annotations

import nibabel as nib
import pytest

from neurofaune.analysis.stats.effect_size import compute_partial_etasq_from_tstat
from tests.regression import fixtures
from tests.regression._equivalence import assert_array_matches_golden

pytestmark = pytest.mark.regression


def test_partial_etasq_from_tstat_preserved(tmp_path):
    tstat_file = tmp_path / "synthetic_tstat.nii.gz"
    nib.save(fixtures.tmap_nifti(), tstat_file)

    out_file = compute_partial_etasq_from_tstat(tstat_file, df_error=18)
    etasq = nib.load(out_file).get_fdata()

    # float32 round-trip through NIfTI → slightly looser than default.
    assert_array_matches_golden(etasq, "voxelwise_partial_etasq", rtol=1e-5, atol=1e-6)
